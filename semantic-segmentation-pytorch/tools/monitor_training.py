#!/usr/bin/env python3
"""
Monitor training progress and estimate time to completion.
Useful for knowing when to resubmit jobs on 4-hour windows.
"""
import argparse
import os.path as osp
import pandas as pd
from datetime import datetime, timedelta

def parse_metrics_csv(csv_path):
    """Parse metrics.csv and print progress."""
    if not osp.exists(csv_path):
        print(f"Metrics file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*70}")
    print(f"Training Progress")
    print(f"{'='*70}")
    print(f"Total epochs completed: {len(df)}")
    print(f"Latest epoch: {df['epoch'].iloc[-1]}")
    print(f"\nLatest metrics:")
    print(f"  Mean Dice: {df['mean_dice'].iloc[-1]:.4f} ({df['mean_dice'].iloc[-1]*100:.2f}%)")
    print(f"  Mean IoU:  {df['mean_iou'].iloc[-1]:.4f} ({df['mean_iou'].iloc[-1]*100:.2f}%)")
    
    # Show trend
    if len(df) >= 5:
        recent_dice = df['mean_dice'].iloc[-5:].values
        recent_iou = df['mean_iou'].iloc[-5:].values
        print(f"\nLast 5 epochs:")
        print(f"  Dice: {', '.join([f'{x:.3f}' for x in recent_dice])}")
        print(f"  IoU:  {', '.join([f'{x:.3f}' for x in recent_iou])}")
        
        # Trend arrows
        dice_trend = "📈" if recent_dice[-1] > recent_dice[0] else "📉"
        iou_trend = "📈" if recent_iou[-1] > recent_iou[0] else "📉"
        print(f"  Trend: Dice {dice_trend}, IoU {iou_trend}")
    
    # Best metrics
    best_dice = df['mean_dice'].max()
    best_iou = df['mean_iou'].max()
    best_dice_epoch = df.loc[df['mean_dice'].idxmax(), 'epoch']
    best_iou_epoch = df.loc[df['mean_iou'].idxmax(), 'epoch']
    
    print(f"\nBest results so far:")
    print(f"  Dice: {best_dice:.4f} ({best_dice*100:.2f}%) at epoch {best_dice_epoch}")
    print(f"  IoU:  {best_iou:.4f} ({best_iou*100:.2f}%) at epoch {best_iou_epoch}")
    
    print(f"{'='*70}\n")

def estimate_time(log_file, target_epochs=30):
    """Estimate time to completion from log file."""
    if not osp.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    # Parse epoch times from log
    # This is a simplified version - actual implementation would parse timestamps
    print(f"\n{'='*70}")
    print(f"Time Estimation")
    print(f"{'='*70}")
    print(f"Target: {target_epochs} epochs")
    print(f"\n⚠️  Estimate: ~3.5-4 hours per job window")
    print(f"Estimated jobs needed: {(target_epochs / 7):.1f} (≈7-8 epochs per 4hr window)")
    print(f"Estimated total time: ~{(target_epochs / 7 * 4):.0f} hours")
    print(f"{'='*70}\n")

def main():
    ap = argparse.ArgumentParser(description="Monitor UperNet training progress")
    ap.add_argument("--ckpt_dir", default="ckpt/chestxdet-resnet50-upernet-binary")
    ap.add_argument("--target_epochs", type=int, default=30)
    args = ap.parse_args()
    
    metrics_path = osp.join(args.ckpt_dir, "metrics.csv")
    
    parse_metrics_csv(metrics_path)
    estimate_time(None, args.target_epochs)
    
    # Recommendation
    if osp.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        current_epoch = int(df['epoch'].iloc[-1])
        remaining = args.target_epochs - current_epoch
        
        if remaining > 0:
            print(f"📋 Next steps:")
            print(f"   {remaining} epochs remaining")
            print(f"   Resubmit with: sbatch train_upernet_binary.sbatch")
            print(f"   (Script will auto-resume from epoch {current_epoch})")
        else:
            print(f"✅ Training complete!")
            print(f"   Final Dice: {df['mean_dice'].iloc[-1]*100:.2f}%")
            print(f"   Final IoU: {df['mean_iou'].iloc[-1]*100:.2f}%")

if __name__ == "__main__":
    main()
