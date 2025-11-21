#!/usr/bin/env python3
"""
Calculate per-image Dice and IoU scores and find best/worst predictions.
"""
import os
import os.path as osp
import json
import numpy as np
import cv2
from tqdm import tqdm

def calculate_dice(pred, gt):
    """Calculate Dice coefficient."""
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return dice

def calculate_iou(pred, gt):
    """Calculate IoU (Jaccard index)."""
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin) - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou

def main():
    # Paths
    data_root = "data/chestxdet_binary/lists/val.odgt"
    
    # Load test set info
    print("Loading test set...")
    with open(data_root, 'r') as f:
        test_data = [json.loads(line.strip()) for line in f]
    
    print(f"Found {len(test_data)} test images")
    
    # Calculate metrics for each image
    results = []
    dice_scores = []
    iou_scores = []
    
    print("\nCalculating per-image metrics...")
    for item in tqdm(test_data):
        img_id = item['id']
        gt_path = item['fpath_segm']
        
        # Load ground truth
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # For now, we'll use ground truth as prediction since eval_multipro doesn't save them
        # In a real scenario, you'd load predictions from TEST.result directory
        # pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate metrics
        # NOTE: This is a placeholder - you need actual predictions
        # For demonstration, using GT (which gives perfect score)
        dice = calculate_dice(gt, gt)
        iou = calculate_iou(gt, gt)
        
        results.append({
            'id': img_id,
            'img_path': item['fpath_img'],
            'gt_path': gt_path,
            'dice': dice,
            'iou': iou
        })
        
        dice_scores.append(dice)
        iou_scores.append(iou)
    
    # Calculate statistics
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    std_dice = np.std(dice_scores)
    std_iou = np.std(iou_scores)
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Number of images: {len(test_data)}")
    print(f"\nDice Coefficient:")
    print(f"  Mean: {mean_dice:.4f} ({mean_dice*100:.2f}%)")
    print(f"  Std:  {std_dice:.4f}")
    print(f"  Min:  {min(dice_scores):.4f}")
    print(f"  Max:  {max(dice_scores):.4f}")
    print(f"\nIoU (Jaccard Index):")
    print(f"  Mean: {mean_iou:.4f} ({mean_iou*100:.2f}%)")
    print(f"  Std:  {std_iou:.4f}")
    print(f"  Min:  {min(iou_scores):.4f}")
    print(f"  Max:  {max(iou_scores):.4f}")
    print("="*60)
    
    # Sort by Dice score
    results_sorted = sorted(results, key=lambda x: x['dice'], reverse=True)
    
    # Best and worst
    print("\n" + "="*60)
    print("BEST 10 PREDICTIONS (by Dice)")
    print("="*60)
    for i, r in enumerate(results_sorted[:10], 1):
        print(f"{i}. {r['id']}: Dice={r['dice']:.4f}, IoU={r['iou']:.4f}")
    
    print("\n" + "="*60)
    print("WORST 10 PREDICTIONS (by Dice)")
    print("="*60)
    for i, r in enumerate(results_sorted[-10:], 1):
        print(f"{i}. {r['id']}: Dice={r['dice']:.4f}, IoU={r['iou']:.4f}")
    
    # Save results to file
    output_file = "../results/per_image_metrics.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'mean_dice': mean_dice,
                'mean_iou': mean_iou,
                'std_dice': std_dice,
                'std_iou': std_iou,
                'num_images': len(test_data)
            },
            'per_image': results_sorted
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Return best/worst for visualization
    return results_sorted[:5], results_sorted[-5:]

if __name__ == '__main__':
    best, worst = main()
    print("\nNote: This script currently uses ground truth as predictions.")
    print("To get actual metrics, you need to save predictions during eval_multipro.py")
