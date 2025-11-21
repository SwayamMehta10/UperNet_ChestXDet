#!/usr/bin/env python3
"""
Visualize best and worst predictions with overlays on X-ray images.
Creates side-by-side comparisons: Original | Ground Truth | Prediction | Overlay
"""
import os
import os.path as osp
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
from mit_semseg.config import cfg
from mit_semseg.dataset import TestDataset
import torch.nn.functional as F

def calculate_dice(pred, gt):
    """Calculate Dice coefficient."""
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / union

def visualize(img_path, gt_path, pred_mask, output_path, dice_score):
    """
    Create visualization with:
    - Original X-ray
    - Ground truth mask overlay
    - Prediction mask overlay  
    - Comparison overlay (TP=green, FP=blue, FN=red)
    """
    # Load original image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Load ground truth
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_bin = (gt > 0).astype(np.uint8)
    
    # Binarize prediction
    pred_bin = (pred_mask > 0).astype(np.uint8)
    
    # Create overlays
    gt_overlay = img_rgb.copy()
    gt_overlay[gt_bin == 1] = [0, 255, 0]  # Green for GT
    gt_overlay = cv2.addWeighted(img_rgb, 0.7, gt_overlay, 0.3, 0)
    
    pred_overlay = img_rgb.copy()
    pred_overlay[pred_bin == 1] = [255, 255, 0]  # Yellow for prediction
    pred_overlay = cv2.addWeighted(img_rgb, 0.7, pred_overlay, 0.3, 0)
    
    # Comparison overlay
    comparison = img_rgb.copy()
    tp_mask = (gt_bin == 1) & (pred_bin == 1)  # True Positive
    fp_mask = (gt_bin == 0) & (pred_bin == 1)  # False Positive
    fn_mask = (gt_bin == 1) & (pred_bin == 0)  # False Negative
    
    comparison[tp_mask] = [0, 255, 0]    # Green: correct
    comparison[fp_mask] = [0, 0, 255]    # Blue: false alarm
    comparison[fn_mask] = [255, 0, 0]    # Red: missed
    comparison = cv2.addWeighted(img_rgb, 0.6, comparison, 0.4, 0)
    
    # Combine into single image
    h, w = img.shape[:2]
    combined = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    combined[0:h, 0:w] = img_rgb
    combined[0:h, w:2*w] = gt_overlay
    combined[h:2*h, 0:w] = pred_overlay
    combined[h:2*h, w:2*w] = comparison
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, f"Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, f"Ground Truth", (w+10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, f"Prediction", (10, h+30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, f"Comparison (Dice={dice_score:.3f})", (w+10, h+30), font, 1, (255, 255, 255), 2)
    
    # Add legend for comparison
    legend_y = h + 60
    cv2.putText(combined, "Green=Correct", (w+10, legend_y), font, 0.6, (0, 255, 0), 2)
    cv2.putText(combined, "Blue=FalseAlarm", (w+10, legend_y+30), font, 0.6, (0, 0, 255), 2)
    cv2.putText(combined, "Red=Missed", (w+10, legend_y+60), font, 0.6, (255, 0, 0), 2)
    
    # Save
    cv2.imwrite(output_path, combined)
    return combined

def load_model(cfg_file, checkpoint):
    """Load trained model."""
    # Load config
    cfg.merge_from_file(cfg_file)
    
    # Build model
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=''  # Don't load ImageNet weights
    )
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=''
    )
    
    # Load checkpoint
    checkpoint_dir = cfg.DIR
    encoder_path = osp.join(checkpoint_dir, f"encoder_{checkpoint}")
    decoder_path = osp.join(checkpoint_dir, f"decoder_{checkpoint}")
    
    net_encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    net_decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    
    # Create segmentation module
    segmentation_module = SegmentationModule(net_encoder, net_decoder, cfg.DATASET.num_class)
    segmentation_module.eval()
    segmentation_module.cuda()
    
    return segmentation_module

def predict(segmentation_module, img_path):
    """Generate prediction for single image."""
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original size
    orig_h, orig_w = img.shape[:2]
    
    # Prepare input (simplified - using fixed size)
    img_resized = cv2.resize(img, (512, 512))
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float()
    img_tensor = img_tensor / 255.0
    
    # Normalize  
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.25, 0.25, 0.25]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    # Add batch dimension and move to GPU
    img_tensor = img_tensor.unsqueeze(0).cuda()
    
    # Predict at original resolution
    with torch.no_grad():
        pred = segmentation_module({'img_data': img_tensor}, segSize=(orig_h, orig_w))
    
    # Get prediction mask
    _, pred_mask = torch.max(pred, dim=1)
    pred_mask = pred_mask.cpu().numpy()[0]
    
    return pred_mask

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='Config file')
    parser.add_argument('--checkpoint', default='epoch_10.pth', help='Checkpoint to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of best/worst to visualize')
    parser.add_argument('--output-dir', default='../results/visualizations', help='Output directory')
    args = parser.parse_args()
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    
    # Load model
    print("Loading model...")
    segmentation_module = load_model(args.cfg, args.checkpoint)
    
    # Load test set
    print("Loading test data...")
    with open('data/chestxdet_binary/lists/val.odgt', 'r') as f:
        test_data = [json.loads(line.strip()) for line in f]
    
    # Calculate metrics for all images
    print(f"\nGenerating predictions for {len(test_data)} images...")
    results = []
    
    for item in tqdm(test_data):
        img_id = item['id']
        img_path = item['fpath_img']
        gt_path = item['fpath_segm']
        
        # Generate prediction
        pred_mask = predict(segmentation_module, img_path)
        
        # Load GT
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate Dice
        dice = calculate_dice(pred_mask, gt)
        
        results.append({
            'id': img_id,
            'img_path': img_path,
            'gt_path': gt_path,
            'pred_mask': pred_mask,
            'dice': dice
        })
    
    # Sort by Dice
    results_sorted = sorted(results, key=lambda x: x['dice'], reverse=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'best'), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'worst'), exist_ok=True)
    
    # Visualize best
    print(f"\nVisualizing {args.num_samples} best predictions...")
    for i, r in enumerate(results_sorted[:args.num_samples], 1):
        output_path = osp.join(args.output_dir, 'best', f"best_{i}_{r['id']}.png")
        visualize(r['img_path'], r['gt_path'], r['pred_mask'], output_path, r['dice'])
        print(f"  {i}. {r['id']}: Dice={r['dice']:.4f} -> {output_path}")
    
    # Visualize worst
    print(f"\nVisualizing {args.num_samples} worst predictions...")
    for i, r in enumerate(results_sorted[-args.num_samples:], 1):
        output_path = osp.join(args.output_dir, 'worst', f"worst_{i}_{r['id']}.png")
        visualize(r['img_path'], r['gt_path'], r['pred_mask'], output_path, r['dice'])
        print(f"  {i}. {r['id']}: Dice={r['dice']:.4f} -> {output_path}")
    
    # Print summary
    dice_scores = [r['dice'] for r in results]
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images: {len(results)}")
    print(f"Mean Dice: {np.mean(dice_scores):.4f} ({np.mean(dice_scores)*100:.2f}%)")
    print(f"Std Dice:  {np.std(dice_scores):.4f}")
    print(f"Min Dice:  {min(dice_scores):.4f}")
    print(f"Max Dice:  {max(dice_scores):.4f}")
    print(f"\nVisualizations saved to: {args.output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
