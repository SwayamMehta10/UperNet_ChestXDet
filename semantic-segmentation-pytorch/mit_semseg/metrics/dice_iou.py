import torch

def _flat_preds_labels(logits, labels, num_class, ignore_index=-1):
    # logits: [N,C,H,W]; labels: [N,H,W]
    pred = torch.argmax(logits, dim=1)              # [N,H,W]
    lab  = labels
    # robust mask: ignore_index + bounds
    mask = (lab != ignore_index) & (lab >= 0) & (lab < num_class)
    pred = pred[mask]
    lab  = lab[mask]
    return pred, lab

@torch.no_grad()
def dice_iou_from_logits(logits, labels, num_class, ignore_index=-1):
    pred, lab = _flat_preds_labels(logits, labels, num_class, ignore_index)

    # confusion via bincount (fast)
    conf = torch.zeros((num_class, num_class), device=logits.device, dtype=torch.int64)
    inds = num_class * lab + pred
    conf += torch.bincount(inds, minlength=num_class**2).reshape(num_class, num_class)

    tp = torch.diag(conf).float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp

    denom_iou  = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    iou  = torch.where(denom_iou  > 0, tp / denom_iou,  torch.zeros_like(tp))
    dice = torch.where(denom_dice > 0, 2 * tp / denom_dice, torch.zeros_like(tp))

    # mean over *present* disease classes only (exclude background=0)
    gt_support = conf.sum(1).float()       # per-class GT pixels
    included = (gt_support > 0)
    included[0] = False                    # drop background from means

    if included.any():
        miou  = iou[included].mean()
        mdice = dice[included].mean()
    else:
        miou  = torch.zeros((), device=logits.device)
        mdice = torch.zeros((), device=logits.device)

    return {
        "per_class_iou":  iou,
        "per_class_dice": dice,
        "miou":  miou,
        "mdice": mdice,
        "support": gt_support,
    }

