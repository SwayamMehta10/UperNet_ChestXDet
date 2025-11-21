#!/usr/bin/env python3
"""
Split ChestX-Det train set into train/val subsets for segmentation.
Produces .odgt files for UPerNet, reading image sizes from actual files.
"""

import os, os.path as osp, json, random, argparse, cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=osp.abspath(osp.join("..","..")))
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)

    chest_dir = osp.join(args.root, "chestxdet")
    img_dir = osp.join(chest_dir, "train")

    with open(osp.join(chest_dir, "ChestX_Det_train.json")) as f:
        ann_train = json.load(f)
    print(f"Loaded {len(ann_train)} train samples")

    # shuffle & split
    random.shuffle(ann_train)
    n_val = int(len(ann_train) * args.val_ratio)
    ann_val = ann_train[:n_val]
    ann_train = ann_train[n_val:]
    print(f"Split: {len(ann_train)} train / {len(ann_val)} val")

    # output directory
    out_root = osp.join(args.root, "semantic-segmentation-pytorch", "data/chestxdet/lists")
    os.makedirs(out_root, exist_ok=True)

    def get_size(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        h, w = img.shape[:2]
        return w, h

    def write_odgt(items, filename):
        odgt_path = osp.join(out_root, filename)
        with open(odgt_path, "w") as f:
            for it in items:
                img_path = osp.join(img_dir, it["file_name"])
                mask_path = osp.join(
                    args.root,
                    "semantic-segmentation-pytorch",
                    "data/chestxdet/masks_train",
                    it["file_name"]
                )
                w, h = get_size(img_path)
                rec = {
                    "fpath_img": img_path,
                    "fpath_segm": mask_path,
                    "width": w,
                    "height": h,
                    "id": osp.splitext(it["file_name"])[0]
                }
                f.write(json.dumps(rec) + "\n")
        print(f"Saved {odgt_path}")

    write_odgt(ann_train, "train.odgt")
    write_odgt(ann_val, "val.odgt")
    print(f"Done. Files saved to {out_root}")

if __name__ == "__main__":
    main()

