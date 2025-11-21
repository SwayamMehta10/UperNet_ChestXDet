#!/usr/bin/env python3
import json, os, os.path as osp, argparse, cv2, numpy as np
from tqdm import tqdm

CLASSES = [
  "Atelectasis","Calcification","Cardiomegaly","Consolidation","Diffuse Nodule",
  "Effusion","Emphysema","Fibrosis","Fracture","Mass","Nodule",
  "Pleural Thickening","Pneumothorax"
]
CLS2ID = {c:i+1 for i,c in enumerate(CLASSES)}  # 0 = background

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def poly_to_mask(shape, polys, cls_id):
    h, w = shape
    m = np.zeros((h,w), np.uint8)
    for poly in polys:
        cnt = np.array(poly, dtype=np.int32).reshape(-1,2)
        cv2.fillPoly(m, [cnt], color=cls_id)
    return m

def save_mask_from_item(item, img_dir, out_mask_dir):
    img_path = osp.join(img_dir, item["file_name"])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    mask = np.zeros((h,w), np.uint8)
    for sym, poly in zip(item["syms"], item["polygons"]):
        cls_id = CLS2ID.get(sym, None)
        if cls_id is None: 
            continue
        mask = np.maximum(mask, poly_to_mask((h,w), [poly], cls_id))
    out_path = osp.join(out_mask_dir, item["file_name"])
    cv2.imwrite(out_path, mask)
    return img_path, out_path, (w, h)

def build_odgt(items, img_root, mask_root, odgt_path):
    with open(odgt_path, "w") as f:
        for it in tqdm(items, desc=f"Writing {osp.basename(odgt_path)}"):
            im, seg, (w,h) = save_mask_from_item(it, img_root, mask_root)
            rec = {
              "fpath_img": im,
              "fpath_segm": seg,
              "width": w, "height": h, "id": osp.splitext(osp.basename(im))[0]
            }
            f.write(json.dumps(rec) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=osp.abspath(osp.join("..","..")))
    ap.add_argument("--chestx", default="chestxdet")
    ap.add_argument("--out", default="data/chestxdet")
    args = ap.parse_args()

    proj = args.root
    chest = osp.join(proj, args.chestx)
    img_train = osp.join(chest, "train")
    img_test  = osp.join(chest, "test")
    ann_train = json.load(open(osp.join(chest, "ChestX_Det_train.json")))
    ann_test  = json.load(open(osp.join(chest, "ChestX_Det_test.json")))

    out_root = osp.join(proj, "semantic-segmentation-pytorch", args.out)
    mtrain = osp.join(out_root, "masks_train")
    mtest  = osp.join(out_root, "masks_test")
    ensure_dir(mtrain); ensure_dir(mtest)

    ensure_dir(osp.join(out_root, "lists"))
    build_odgt(ann_train, img_train, mtrain, osp.join(out_root, "lists", "train.odgt"))
    build_odgt(ann_test,  img_test,  mtest,  osp.join(out_root, "lists", "val.odgt"))

    # write a label names file for convenience
    with open(osp.join(out_root, "classes.txt"), "w") as f:
        f.write("background\n"); f.write("\n".join(CLASSES) + "\n")

if __name__ == "__main__":
    main()

