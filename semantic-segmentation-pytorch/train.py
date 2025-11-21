# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
import torch.nn.functional as F
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import TrainDataset, ValDataset
from mit_semseg.metrics.dice_iou import dice_iou_from_logits
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, parse_devices, setup_logger
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback


# train one epoch
def train(segmentation_module, iterator, optimizers, history, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        if isinstance(batch_data, list):
            batch_data = batch_data[0] # unwrap list for single gpu
        
        for k in batch_data:
            if torch.is_tensor(batch_data[k]):
                batch_data[k] = batch_data[k].cuda(non_blocking=True)

        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        # forward pass
        loss, acc = segmentation_module(batch_data)
        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())


def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))


def _conf_from_logits(logits, labels, C, ignore_index=-1):
    pred = torch.argmax(logits, 1)
    lab  = labels
    mask = (lab != ignore_index) & (lab >= 0) & (lab < C)
    pred = pred[mask]; lab = lab[mask]
    if pred.numel() == 0:
        return torch.zeros(C, C, dtype=torch.int64, device=logits.device)
    inds = C * lab + pred
    return torch.bincount(inds, minlength=C*C).reshape(C, C).to(torch.int64)


@torch.no_grad()
def validate(segmentation_module, cfg, epoch):
    print(f"\nRunning validation after epoch {epoch} ...")
    dataset_val = ValDataset(cfg.DATASET.root_dataset, cfg.DATASET.list_val, cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False,
                                             num_workers=cfg.TRAIN.workers, pin_memory=True)

    segmentation_module.eval()
    C = cfg.DATASET.num_class
    conf = torch.zeros(C, C, dtype=torch.int64, device='cuda')

    for i, sample in enumerate(loader_val):
        # --- Unwrap single-GPU list/tuple wrappers ---
        img = sample['img_data']
        while isinstance(img, (list, tuple)):
            img = img[0]
        sample['img_data'] = img

        seg = sample['seg_label']
        while isinstance(seg, (list, tuple)):
            seg = seg[0]
        sample['seg_label'] = seg

        # --- Normalize shapes to [N, C, H, W] / [N, H, W] ---
        img = sample['img_data']
        if img.dim() == 5:                  # [ngpu, batch, C, H, W]
            img = img.flatten(0, 2)         # -> [N, C, H, W]
        if img.dim() == 3:                  # [C, H, W]
            img = img.unsqueeze(0)          # -> [1, C, H, W]
        sample['img_data'] = img.cuda(non_blocking=True)

        lab = sample['seg_label']
        if lab.dim() == 5:                  # [ngpu, batch, 1, H, W]
            lab = lab.flatten(0, 2)         # -> [N, 1, H, W]
        if lab.dim() == 4 and lab.size(1) == 1:
            lab = lab.squeeze(1)            # -> [N, H, W]
        if lab.dim() == 2:                  # [H, W]
            lab = lab.unsqueeze(0)          # -> [1, H, W]
        sample['seg_label'] = lab.cuda(non_blocking=True)

        # --- Forward at label/native resolution ---
        segSize = sample['seg_label'].shape[-2:]
        logits = segmentation_module(sample, segSize=segSize)

        # Align labels to logits if needed
        lab = sample['seg_label']
        if lab.shape[-2:] != logits.shape[-2:]:
            lab = F.interpolate(
                lab.unsqueeze(1).float(), size=logits.shape[-2:], mode='nearest'
            ).squeeze(1).long()
            sample['seg_label'] = lab

        # Update confusion matrix
        conf += _conf_from_logits(logits, lab, C, ignore_index=-1)

        if i % 100 == 0:
            print(f"Val [{i}/{len(loader_val)}]")

    tp = conf.diag().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    denom_iou, denom_dice = tp + fp + fn, 2*tp + fp + fn
    iou  = torch.where(denom_iou  > 0, tp/denom_iou,  torch.zeros_like(tp))
    dice = torch.where(denom_dice > 0, 2*tp/denom_dice, torch.zeros_like(tp))

    support = conf.sum(1).float()
    include = (support > 0); include[0] = False  # drop background
    miou  = iou[include].mean().item() if include.any() else 0.0
    mdice = dice[include].mean().item() if include.any() else 0.0

    print(f"\nEpoch {epoch} (dataset-level) Dice={mdice:.4f}, IoU={miou:.4f}\n")

    # write CSV
    results_path = os.path.join(cfg.DIR, "metrics.csv")
    header_needed = not os.path.exists(results_path)
    with open(results_path, "a") as f:
        if header_needed: f.write("epoch,mean_dice,mean_iou\n")
        f.write(f"{epoch},{mdice:.6f},{miou:.6f}\n")

    segmentation_module.train(not cfg.TRAIN.fix_bn)


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg, gpus):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder)

    crit = nn.NLLLoss(ignore_index=-1)

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        # Re-create iterator each epoch so dataloader reshuffles correctly
        iterator_train = iter(loader_train)
        train(segmentation_module, iterator_train, optimizers, history, epoch+1, cfg)

        # checkpointing
        checkpoint(nets, history, cfg, epoch+1)
        
        # validation
        validate(segmentation_module, cfg, epoch+1)

    print('Training Done!')

    # Save final model for convenience
    final_path = os.path.join(cfg.DIR, "model_final.pth")
    torch.save(segmentation_module.state_dict(), final_path)
    print(f"Saved final model to {final_path}")



if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
