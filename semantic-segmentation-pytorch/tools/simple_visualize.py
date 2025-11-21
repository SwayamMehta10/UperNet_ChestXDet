#!/usr/bin/env python3
"""
Simple visualization: Show original, GT mask, and prediction mask side-by-side.
Only for 2 best and 2 worst Dice scores.
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
from mit_semseg.config import cfg

def calculate_dice(pred, gt):
    """Calculate Dice coefficient."""
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / union

def create_simple_visualization(img_path, gt_path, pred_mask, output_path, dice_score, img_id):
    """
    Create simple 1x3 visualization:
    [Original | Ground Truth | Prediction]
    """
    # Load original image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # Load ground truth
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure same size
    if pred_mask.shape != gt.shape:
        pred_mask = cv2.resize(pred_mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert masks to binary 0/255 for visualization
    gt_vis = ((gt > 0) * 255).astype(np.uint8)
    pred_vis = ((pred_mask > 0) * 255).astype(np.uint8)
    
    # Create side-by-side image
    combined = np.zeros((h, w * 3), dtype=np.uint8)
    combined[:, 0:w] = img
    combined[:, w:2*w] = gt_vis
    combined[:, 2*w:3*w] = pred_vis
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 3
    color = 255  # White text
    
    # Title at the top
    title = f"ID: {img_id}, Dice: {dice_score:.4f}"
    cv2.putText(combined, title, (10, 50), font, font_scale, color, thickness)
    
    # Column labels
    cv2.putText(combined, "Original", (10, h-20), font, font_scale*0.7, color, thickness-1)
    cv2.putText(combined, "Ground Truth", (w+10, h-20), font, font_scale*0.7, color, thickness-1)
    cv2.putText(combined, "Prediction", (2*w+10, h-20), font, font_scale*0.7, color, thickness-1)
    
    # Save
    cv2.imwrite(output_path, combined)
    print(f"Saved: {output_path}")
    return combined

def load_model(cfg_file, checkpoint):
    """Load trained model."""
    cfg.merge_from_file(cfg_file)
    
    # Build model
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=''
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
    
    # Prepare input
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
    parser.add_argument('--output-dir', default='../results/simple_viz', help='Output directory')
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
    print(f"\nCalculating Dice scores for {len(test_data)} images...")
    results = []
    
    for item in tqdm(test_data):
        img_id = item['id']
        img_path = item['fpath_img']
        gt_path = item['fpath_segm']
        
        # Generate prediction
        pred_mask = predict(segmentation_module, img_path)
        
        # Load GT
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure prediction and GT have same size
        if pred_mask.shape != gt.shape:
            pred_mask = cv2.resize(pred_mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        
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
    
    # Print summary
    dice_scores = [r['dice'] for r in results]
    print("\n" + "="*70)
    print("DICE SCORE SUMMARY")
    print("="*70)
    print(f"Total images: {len(results)}")
    print(f"Mean Dice: {np.mean(dice_scores):.4f} ({np.mean(dice_scores)*100:.2f}%)")
    print(f"Std Dice:  {np.std(dice_scores):.4f}")
    print(f"Min Dice:  {min(dice_scores):.4f}")
    print(f"Max Dice:  {max(dice_scores):.4f}")
    print("="*70)
    
    # Visualize best 2
    print("\nCreating visualizations for 2 BEST predictions...")
    for i, r in enumerate(results_sorted[:2], 1):
        output_path = osp.join(args.output_dir, f"best_{i}_dice{r['dice']:.3f}_id{r['id']}.png")
        create_simple_visualization(r['img_path'], r['gt_path'], r['pred_mask'], 
                                   output_path, r['dice'], r['id'])
        print(f"  Best #{i}: ID={r['id']}, Dice={r['dice']:.4f}")
    
    # Visualize worst 2
    print("\nCreating visualizations for 2 WORST predictions...")
    for i, r in enumerate(results_sorted[-2:], 1):
        output_path = osp.join(args.output_dir, f"worst_{i}_dice{r['dice']:.3f}_id{r['id']}.png")
        create_simple_visualization(r['img_path'], r['gt_path'], r['pred_mask'],
                                   output_path, r['dice'], r['id'])
        print(f"  Worst #{i}: ID={r['id']}, Dice={r['dice']:.4f}")
    
    print(f"\n✅ Visualizations saved to: {args.output_dir}")
    print("="*70)

if __name__ == '__main__':
    main()

