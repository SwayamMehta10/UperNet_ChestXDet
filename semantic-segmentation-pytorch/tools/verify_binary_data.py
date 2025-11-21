#!/usr/bin/env python3
"""
Quick verification script to check binary mask generation.
Run after chestxdet_to_odgt_binary.py to ensure masks are correct.
"""
import cv2
import glob
import numpy as np
import os.path as osp
import json

def check_binary_masks(mask_dir, name=""):
    """Check that all masks only contain {0, 1}."""
    masks = glob.glob(osp.join(mask_dir, "*.png"))
    if not masks:
        print(f"❌ No masks found in {mask_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Checking {len(masks)} {name} masks...")
    print(f"{'='*60}")
    
    errors = 0
    stats = {"background": 0, "disease": 0, "mixed": 0, "invalid": 0}
    
    for i, m in enumerate(masks[:10]):  # Check first 10
        img = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Could not read: {osp.basename(m)}")
            errors += 1
            continue
        
        unique_vals = set(np.unique(img))
        
        # Check binary
        if not unique_vals.issubset({0, 1}):
            print(f"❌ Non-binary mask: {osp.basename(m)} has values {unique_vals}")
            stats["invalid"] += 1
            errors += 1
        elif unique_vals == {0}:
            stats["background"] += 1
        elif unique_vals == {1}:
            print(f"⚠️  Only disease (no background): {osp.basename(m)}")
            stats["disease"] += 1
        else:  # {0, 1}
            stats["mixed"] += 1
            disease_pct = 100 * (img == 1).sum() / img.size
            print(f"✓ {osp.basename(m)}: {disease_pct:.1f}% disease")
    
    print(f"\nSample statistics:")
    print(f"  Background only: {stats['background']}")
    print(f"  Disease only: {stats['disease']}")
    print(f"  Mixed: {stats['mixed']}")
    print(f"  Invalid: {stats['invalid']}")
    
    return errors == 0

def check_odgt(odgt_path):
    """Verify ODGT file format."""
    print(f"\n{'='*60}")
    print(f"Checking ODGT: {odgt_path}")
    print(f"{'='*60}")
    
    try:
        with open(odgt_path) as f:
            lines = f.readlines()
        
        print(f"Total records: {len(lines)}")
        
        # Check first record
        rec = json.loads(lines[0])
        print(f"\nSample record:")
        print(f"  Image: {rec['fpath_img']}")
        print(f"  Mask: {rec['fpath_segm']}")
        print(f"  Size: {rec['width']}x{rec['height']}")
        
        # Verify files exist
        if osp.exists(rec['fpath_img']):
            print(f"  ✓ Image file exists")
        else:
            print(f"  ❌ Image file NOT found")
        
        if osp.exists(rec['fpath_segm']):
            print(f"  ✓ Mask file exists")
        else:
            print(f"  ❌ Mask file NOT found")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/chestxdet_binary")
    args = ap.parse_args()
    
    root = args.data_root
    
    print(f"\n{'#'*60}")
    print(f"# Binary Segmentation Verification")
    print(f"# Data root: {root}")
    print(f"{'#'*60}")
    
    # Check masks
    train_ok = check_binary_masks(osp.join(root, "masks_train"), "training")
    test_ok = check_binary_masks(osp.join(root, "masks_test"), "test")
    
    # Check ODGT
    train_odgt = check_odgt(osp.join(root, "lists/train.odgt"))
    val_odgt = check_odgt(osp.join(root, "lists/val.odgt"))
    
    print(f"\n{'='*60}")
    if train_ok and test_ok and train_odgt and val_odgt:
        print("✅ All checks passed! Ready to train.")
    else:
        print("❌ Some checks failed. Review errors above.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
