#!/usr/bin/env python3
"""
Create proper 80/20 train/validation split from the original training data.
This fixes the data leakage issue where test set was used as validation.
"""
import json
import random
import os.path as osp

def main():
    random.seed(42)  # For reproducibility
    
    # Load original train data
    train_odgt = 'data/chestxdet_binary/lists/train.odgt'
    
    print("Loading original training data...")
    with open(train_odgt, 'r') as f:
        all_train = [json.loads(line.strip()) for line in f]
    
    print(f"Total training samples: {len(all_train)}")
    
    # Shuffle
    random.shuffle(all_train)
    
    # 80/20 split
    split_idx = int(0.8 * len(all_train))
    train_split = all_train[:split_idx]
    val_split = all_train[split_idx:]
    
    print(f"\nSplit created:")
    print(f"  Train: {len(train_split)} images (80%)")
    print(f"  Val:   {len(val_split)} images (20%)")
    
    # Save train split
    train_out = 'data/chestxdet_binary/lists/train_split.odgt'
    with open(train_out, 'w') as f:
        for item in train_split:
            f.write(json.dumps(item) + '\n')
    print(f"\nSaved: {train_out}")
    
    # Save val split
    val_out = 'data/chestxdet_binary/lists/val_split.odgt'
    with open(val_out, 'w') as f:
        for item in val_split:
            f.write(json.dumps(item) + '\n')
    print(f"Saved: {val_out}")
    
    print("\n✅ Train/Val split created successfully!")
    print("\nNote: Test set (val.odgt with 553 images) remains unchanged as holdout set.")

if __name__ == '__main__':
    main()
