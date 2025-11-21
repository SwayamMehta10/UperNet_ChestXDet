#!/usr/bin/env python3
"""
Create a test.odgt file for ChestX-Det test split.
Reads ChestX_Det_test.json and outputs .odgt compatible with UPerNet.
"""

import os, os.path as osp, json, cv2

def main():
    root = osp.abspath(osp.join(".."))
    chest_dir = osp.join(root, "chestxdet")
    img_dir = osp.join(chest_dir, "test")

    out_root = osp.join(root, "semantic-segmentation-pytorch", "data/chestxdet/lists")
    os.makedirs(out_root, exist_ok=True)

    with open(osp.join(chest_dir, "ChestX_Det_test.json")) as f:
        ann_test = json.load(f)
    print(f"Loaded {len(ann_test)} test samples")

    def get_size(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        h, w = img.shape[:2]
        return w, h

    out_file = osp.join(out_root, "test.odgt")
    with open(out_file, "w") as f:
        for it in ann_test:
            img_path = osp.join(img_dir, it["file_name"])
            mask_path = osp.join(
                root,
                "semantic-segmentation-pytorch",
                "data/chestxdet/masks_test",
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

    print(f"Saved test.odgt → {out_file}")

if __name__ == "__main__":
    main()

