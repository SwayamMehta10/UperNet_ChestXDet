#!/usr/bin/env python3
"""
Convert ChestX-Det JSON annotations to ODGT format with BINARY segmentation masks.
All 13 disease classes are merged into a single 'disease' class (ID=1).
Background remains class 0.
"""
import json, os, os.path as osp, argparse, cv2, numpy as np
from tqdm import tqdm

# Binary segmentation: all diseases -> class 1
BINARY_DISEASE_CLASS = 1

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def poly_to_mask(shape, polys, cls_id):
    """Convert list of polygons to binary mask with given class ID."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    for poly in polys:
        cnt = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(m, [cnt], color=cls_id)
    return m

def save_binary_mask_from_item(item, img_dir, out_mask_dir):
    """
    Generate binary segmentation mask from ChestX-Det annotation item.
    Any disease annotation -> class 1, background -> class 0.
    """
    img_path = osp.join(img_dir, item["file_name"])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    
    # Initialize background mask
    mask = np.zeros((h, w), np.uint8)
    
    # Merge all disease polygons into single binary mask
    for sym, poly in zip(item["syms"], item["polygons"]):
        # Any disease gets class 1
        disease_mask = poly_to_mask((h, w), [poly], BINARY_DISEASE_CLASS)
        mask = np.maximum(mask, disease_mask)
    
    # Save binary mask
    out_path = osp.join(out_mask_dir, item["file_name"])
    cv2.imwrite(out_path, mask)
    return img_path, out_path, (w, h)

def build_odgt(items, img_root, mask_root, odgt_path):
    """Build ODGT file with image paths and binary mask paths."""
    with open(odgt_path, "w") as f:
        for it in tqdm(items, desc=f"Writing {osp.basename(odgt_path)}"):
            im, seg, (w, h) = save_binary_mask_from_item(it, img_root, mask_root)
            rec = {
              "fpath_img": im,
              "fpath_segm": seg,
              "width": w, "height": h, "id": osp.splitext(osp.basename(im))[0]
            }
            f.write(json.dumps(rec) + "\n")

def main():
    ap = argparse.ArgumentParser(
        description="Convert ChestX-Det to binary segmentation ODGT format"
    )
    ap.add_argument("--root", default=osp.abspath(".."))
    ap.add_argument("--chestx", default="chestxdet")
    ap.add_argument("--out", default="data/chestxdet_binary")
    args = ap.parse_args()

    proj = args.root
    chest = osp.join(proj, args.chestx)
    img_train = osp.join(chest, "train")
    img_test  = osp.join(chest, "test")
    ann_train = json.load(open(osp.join(chest, "ChestX_Det_train.json")))
    ann_test  = json.load(open(osp.join(chest, "ChestX_Det_test.json")))

    print(f"Train samples: {len(ann_train)}")
    print(f"Test samples: {len(ann_test)}")

    out_root = osp.join(proj, "semantic-segmentation-pytorch", args.out)
    mtrain = osp.join(out_root, "masks_train")
    mtest  = osp.join(out_root, "masks_test")
    ensure_dir(mtrain); ensure_dir(mtest)

    ensure_dir(osp.join(out_root, "lists"))
    print("\nGenerating binary segmentation masks...")
    build_odgt(ann_train, img_train, mtrain, osp.join(out_root, "lists", "train.odgt"))
    build_odgt(ann_test,  img_test,  mtest,  osp.join(out_root, "lists", "val.odgt"))

    # Write class names file
    with open(osp.join(out_root, "classes.txt"), "w") as f:
        f.write("background\ndisease\n")
    
    print(f"\nDone! Binary masks saved to {out_root}")
    print("Classes: 0=background, 1=disease (any of 13 disease types)")

if __name__ == "__main__":
    main()
