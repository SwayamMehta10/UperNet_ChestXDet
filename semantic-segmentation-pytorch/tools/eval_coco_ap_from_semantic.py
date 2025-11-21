#!/usr/bin/env python3
import os, sys, json, argparse, glob, itertools, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Repo imports (assumes you run from repo root)
sys.path.append(".")
from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule

# ---------- helpers ----------

def unwrap_to_tensor(x):
    # drill through nested single-element lists/tuples
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


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
def dice_iou_from_logits(logits, labels, num_class, ignore_index=-1):
    # logits: [N,C,H,W], labels: [N,H,W]
    pred = torch.argmax(logits, dim=1)
    lab = labels

    mask = (lab != ignore_index) & (lab >= 0) & (lab < num_class)
    pred = pred[mask]
    lab  = lab[mask]

    conf = torch.zeros((num_class, num_class), device=logits.device, dtype=torch.int64)
    if pred.numel() > 0:
        inds = num_class * lab + pred
        conf += torch.bincount(inds, minlength=num_class**2).reshape(num_class, num_class)

    tp = torch.diag(conf).float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp

    denom_iou  = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    iou  = torch.where(denom_iou  > 0, tp / denom_iou,  torch.zeros_like(tp))
    dice = torch.where(denom_dice > 0, 2 * tp / denom_dice, torch.zeros_like(tp))

    # mean over present disease classes, exclude background 0
    gt_support = conf.sum(1).float()
    included = (gt_support > 0)
    if included.numel() > 0:
        included[0] = False
    if included.any():
        miou  = iou[included].mean()
        mdice = dice[included].mean()
    else:
        miou  = torch.tensor(0.0, device=logits.device)
        mdice = torch.tensor(0.0, device=logits.device)

    return {
        "per_class_iou":  iou,
        "per_class_dice": dice,
        "miou":  miou,
        "mdice": mdice,
        "support": gt_support,
    }

def tensor_to_instances(logits, class_ids, prob_thr=0.5, min_area=10):
    """
    logits: [1,C,H,W] tensor
    Returns: (bbox_results, segm_results) lists of COCO-format dicts for all classes in class_ids.
    """
    H, W = logits.shape[-2:]
    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()  # [C,H,W]
    bboxes, segms = [], []
    for cid in class_ids:
        prob = probs[cid]
        mask_bin = (prob > prob_thr).astype(np.uint8)
        if mask_bin.sum() == 0:
            continue
        # connected components
        num_labels, cc = cv_connected_components(mask_bin)
        for idx in range(1, num_labels):  # 0 is background
            m = (cc == idx).astype(np.uint8)
            area = int(m.sum())
            if area < min_area:
                continue
            ys, xs = np.where(m)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
            score = float(prob[m.astype(bool)].max())  # max score within component

            rle = maskUtils.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("ascii")

            bboxes.append({
                "category_id": int(cid),
                "bbox": bbox,
                "score": score,
            })
            segms.append({
                "category_id": int(cid),
                "segmentation": rle,
                "score": score,
            })
    return bboxes, segms

def cv_connected_components(binary_mask):
    """
    Returns (num_labels, labels_img) using OpenCV if available, else scipy.
    """
    try:
        import cv2
        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
        return num_labels, labels
    except Exception:
        from scipy.ndimage import label as cc_label
        labels, num_labels = cc_label(binary_mask)
        return num_labels + 1, labels  # align semantics with OpenCV

def build_coco_gt_from_masks(odgt_list):
    images, annotations = [], []
    ann_id = 1
    for rec in odgt_list:
        image_id = int(rec["id"]) if "id" in rec else int(len(images))
        H, W = int(rec["height"]), int(rec["width"])
        images.append({
            "id": image_id, "width": W, "height": H,
            "file_name": os.path.basename(rec.get("fpath_img","")),
        })
        mpath = rec["fpath_segm"]
        mask = np.load(mpath) if mpath.lower().endswith(".npy") else _imread_gray(mpath)
        for cid in np.unique(mask):
            if cid <= 0:
                continue
            binm = (mask == cid).astype(np.uint8)
            if binm.sum() == 0:
                continue
            num_labels, cc = cv_connected_components(binm)
            for idx in range(1, num_labels):
                inst = (cc == idx).astype(np.uint8)
                if inst.sum() == 0:
                    continue
                ys, xs = np.where(inst)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                bbox = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
                rle = maskUtils.encode(np.asfortranarray(inst))
                rle["counts"] = rle["counts"].decode("ascii")
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(cid),
                    "bbox": bbox,
                    "area": float(inst.sum()),
                    "iscrowd": 0,
                    "segmentation": rle,
                })
                ann_id += 1

    categories = [{"id": int(c), "name": f"class_{int(c)}"} for c in range(1, 100)]
    used = sorted({a["category_id"] for a in annotations})
    categories = [c for c in categories if c["id"] in used]

    return {
        "info": {"description": "ChestX-Det val (from masks)", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _imread_gray(path):
    try:
        import cv2
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise ValueError(f"cv2.imread failed for {path}")
        return m
    except Exception:
        import imageio.v3 as iio
        return iio.imread(path)

def load_odgt(list_path):
    recs = []
    with open(list_path, "r") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def latest_epoch_pair(ckpt_dir):
    enc = sorted(glob.glob(os.path.join(ckpt_dir, "encoder_epoch_*.pth")))
    dec = sorted(glob.glob(os.path.join(ckpt_dir, "decoder_epoch_*.pth")))
    if not enc or not dec:
        return None, None, None
    # pick latest by epoch number
    def ep(x):
        base = os.path.basename(x)
        return int(base.split("_epoch_")[-1].split(".")[0])
    enc.sort(key=ep); dec.sort(key=ep)
    e = min(ep(enc[-1]), ep(dec[-1]))
    encp = os.path.join(ckpt_dir, f"encoder_epoch_{e}.pth")
    decp = os.path.join(ckpt_dir, f"decoder_epoch_{e}.pth")
    return e, encp, decp

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser("Evaluate COCO AP (bbox & segm) from semantic logits")
    ap.add_argument("--cfg", required=True, type=str, help="path to YAML config")
    ap.add_argument("--gpus", default="0", type=str)
    ap.add_argument("--ckpt_dir", default=None, type=str, help="directory with encoder/decoder_epoch_*.pth")
    ap.add_argument("--encoder", default=None, type=str)
    ap.add_argument("--decoder", default=None, type=str)
    ap.add_argument("--prob_thr", default=0.5, type=float)
    ap.add_argument("--min_area", default=10, type=int)
    ap.add_argument("--out_dir", default=None, type=str)
    args = ap.parse_args()

    # Load config
    cfg.merge_from_file(args.cfg)

    # Determine output directory
    out_dir = args.out_dir or os.path.join(cfg.DIR, "ap_eval")
    os.makedirs(out_dir, exist_ok=True)

    # Find checkpoints
    if args.encoder and args.decoder:
        enc_w, dec_w = args.encoder, args.decoder
        epoch = None
    else:
        ckpt_dir = args.ckpt_dir or cfg.DIR
        epoch, enc_w, dec_w = latest_epoch_pair(ckpt_dir)
        if not enc_w or not dec_w:
            raise FileNotFoundError("Could not find encoder/decoder_epoch_*.pth in " + ckpt_dir)
    print(f"Using encoder: {enc_w}")
    print(f"Using decoder: {dec_w}")

    # Build model
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=enc_w)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=dec_w)
    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit).cuda().eval()

    # Dataset / loader
    ds = ValDataset(cfg.DATASET.root_dataset, cfg.DATASET.list_val, cfg.DATASET)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    # Prepare COCO GT from val masks
    print("Building COCO GT from val masks...")
    odgt_val = load_odgt(cfg.DATASET.list_val)
    coco_gt_dict = build_coco_gt_from_masks(odgt_val)
    gt_path = os.path.join(out_dir, "gt.json")
    with open(gt_path, "w") as f:
        json.dump(coco_gt_dict, f)
    coco_gt = COCO(gt_path)

    # Inference + collect predictions and Dice/IoU
    preds_bbox, preds_segm = [], []
    num_class = cfg.DATASET.num_class
    class_ids = list(range(1, num_class))  # keep 1..13 for instances (background=0)
    conf = torch.zeros(num_class, num_class, dtype=torch.int64, device='cuda')


    print("Running inference and collecting predictions...")
    for i, sample in enumerate(loader):
        # --- unwrap nested wrappers; if list has >1 items, take the first (single-scale eval) ---
        img = sample["img_data"]
        while isinstance(img, (list, tuple)):
            img = img[0]
        sample["img_data"] = img

        if "seg_label" in sample:
            seg = sample["seg_label"]
            while isinstance(seg, (list, tuple)):
                seg = seg[0]
            sample["seg_label"] = seg

        # --- normalize shapes ---
        img = sample["img_data"]
        if img.dim() == 5:            # [ngpu, batch, C, H, W]
            img = img.flatten(0, 2)   # -> [N, C, H, W]
        if img.dim() == 3:            # [C, H, W]
            img = img.unsqueeze(0)    # -> [1, C, H, W]
        sample["img_data"] = img.cuda(non_blocking=True)

        if "seg_label" in sample:
            seg = sample["seg_label"]
            if seg.dim() == 5:                    # [ngpu, batch, 1, H, W]
                seg = seg.flatten(0, 2)           # -> [N, 1, H, W]
            if seg.dim() == 4 and seg.size(1) == 1:
                seg = seg.squeeze(1)              # -> [N, H, W]
            if seg.dim() == 2:                    # [H, W]
                seg = seg.unsqueeze(0)            # -> [1, H, W]
            sample["seg_label"] = seg.cuda(non_blocking=True)

        # --- forward at label/native resolution ---
        segSize = sample["seg_label"].shape[-2:] if "seg_label" in sample else sample["img_data"].shape[-2:]
        logits = segmentation_module(sample, segSize=segSize)

        # Upsample logits to original image size for COCO eval
        H, W = int(odgt_val[i]["height"]), int(odgt_val[i]["width"])
        logits_inst = logits
        if logits.shape[-2:] != (H, W):
            logits_inst = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


        # metrics (semantic): accumulate one dataset-level confusion matrix
        if "seg_label" in sample:
            lab = sample["seg_label"]
            if lab.shape[-2:] != logits.shape[-2:]:
                lab = F.interpolate(lab.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").squeeze(1).long()
            conf += _conf_from_logits(logits, lab, C=num_class, ignore_index=-1)


        # instances from logits → COCO preds
        bbs, segs = tensor_to_instances(
            logits_inst, class_ids=class_ids, prob_thr=args.prob_thr, min_area=args.min_area
        )

        # attach image_id (use odgt id if present; else use index)
        rec = odgt_val[i]
        image_id = int(rec.get("id", i))
        for d in bbs:
            d["image_id"] = image_id
        for d in segs:
            d["image_id"] = image_id

        preds_bbox.extend(bbs)
        preds_segm.extend(segs)

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(loader)} images...")

    # Write predictions
    pred_bbox_path = os.path.join(out_dir, "pred_bbox.json")
    pred_segm_path = os.path.join(out_dir, "pred_segm.json")
    with open(pred_bbox_path, "w") as f:
        json.dump(preds_bbox, f)
    with open(pred_segm_path, "w") as f:
        json.dump(preds_segm, f)

    # COCO eval: bbox
    print("\n== COCO bbox AP ==")
    coco_dt = coco_gt.loadRes(pred_bbox_path)
    eval_bbox = COCOeval(coco_gt, coco_dt, iouType="bbox")
    eval_bbox.evaluate(); eval_bbox.accumulate(); eval_bbox.summarize()

    # COCO eval: segm
    print("\n== COCO segm AP ==")
    coco_dt = coco_gt.loadRes(pred_segm_path)
    eval_segm = COCOeval(coco_gt, coco_dt, iouType="segm")
    eval_segm.evaluate(); eval_segm.accumulate(); eval_segm.summarize()

    # Semantic Dice/IoU (dataset-level, disease-only)
    tp = conf.diag().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    den_iou  = tp + fp + fn
    den_dice = 2*tp + fp + fn
    iou  = torch.where(den_iou  > 0, tp/den_iou,  torch.zeros_like(tp))
    dice = torch.where(den_dice > 0, 2*tp/den_dice, torch.zeros_like(tp))

    support = conf.sum(1).float()
    include = (support > 0)
    # background is class 0 now; exclude it from the mean:
    if include.numel() > 0:
        include[0] = False

    if include.any():
        mdice = float(dice[include].mean().item())
        miou  = float(iou[include].mean().item())
    else:
        mdice = 0.0
        miou  = 0.0

    print(f"\n== Semantic (disease-only, dataset-level) ==")
    print(f"Mean Dice (classes 1..{num_class-1}): {mdice:.4f}")
    print(f"Mean IoU  (classes 1..{num_class-1}): {miou:.4f}")


    # Save a small report json
    pred_bbox_path = os.path.join(out_dir, "pred_bbox.json")
    pred_segm_path = os.path.join(out_dir, "pred_segm.json")
    report = {
        "pred_bbox_json": pred_bbox_path,
        "pred_segm_json": pred_segm_path,
        "gt_json": gt_path,
        "dice_mean": mdice,   # dataset-level, disease-only
        "iou_mean":  miou,    # dataset-level, disease-only
        "num_preds_bbox": len(preds_bbox),
        "num_preds_segm": len(preds_segm),
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    # Try to import cv2 lazily for speed; fallback to scipy if not available.
    try:
        import cv2  # noqa: F401
    except Exception:
        pass
    main()

