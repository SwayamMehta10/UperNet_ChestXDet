# Evaluation Guide - Binary Segmentation UperNet on ChestX-Det

## Quick Summary

**Best Model**: Epoch 10
- **mDice**: 53.28%
- **mIoU**: 36.31%
- **Improvement**: 3.2x over 14-class baseline (11.30% → 36.31%)

---

## Evaluation Options

### Option 1: Validation Metrics (Already Done ✅)

The model was validated during training. Best results from logs:

```
Epoch 10 (dataset-level) Dice=0.5328, IoU=0.3631
```

**No action needed** - these are your final metrics!

---

### Option 2: Detailed Per-Image Evaluation

Generate detailed metrics for each test image:

```bash
cd /scratch/$USER/upernet_chestxdet/semantic-segmentation-pytorch

# Load environment
module load mamba/latest
source activate upernet_env

# Run evaluation
python eval_multipro.py \
  --cfg config/chestxdet-resnet50-upernet-binary.yaml \
  --gpus 0 \
  TEST.checkpoint epoch_10.pth \
  TEST.result ../results/eval_epoch10
```

**Output**:
- `accuracy.txt`: Per-class and mean metrics
- Per-image Dice and IoU scores
- Confusion matrix

**Time**: ~5-10 minutes

---

### Option 3: Generate Prediction Visualizations

Create visual comparisons (ground truth vs. predictions):

```bash
cd /scratch/$USER/upernet_chestxdet/semantic-segmentation-pytorch

# Generate predictions with optional visualization
python test.py \
  --cfg config/chestxdet-resnet50-upernet-binary.yaml \
  --gpu 0 \
  TEST.checkpoint epoch_10.pth \
  TEST.result ../results/visualizations_epoch10
```

**Output**: Predicted masks saved as PNG files in results directory

**Time**: ~10-15 minutes for 553 test images

---

### Option 4: Visual Comparison Script

Create overlays showing ground truth vs. predictions:

```python
#!/usr/bin/env python3
"""
Compare ground truth and predictions visually
"""
import cv2
import numpy as np
import os.path as osp

# Paths
img_dir = "/scratch/$USER/upernet_chestxdet/chestxdet/test"
gt_dir = "/scratch/$USER/upernet_chestxdet/semantic-segmentation-pytorch/data/chestxdet_binary/masks_test"
pred_dir = "../results/visualizations_epoch10"
output_dir = "../results/comparisons"

os.makedirs(output_dir, exist_ok=True)

# Process first 20 images
for i, filename in enumerate(sorted(os.listdir(img_dir))[:20]):
    # Load
    img = cv2.imread(osp.join(img_dir, filename), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(osp.join(gt_dir, filename), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(osp.join(pred_dir, filename), cv2.IMREAD_GRAYSCALE)
    
    # Binarize (predictions might be 0-255 or 0-1)
    gt_bin = (gt > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)
    
    # Create RGB overlay
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay = img_rgb.copy()
    
    # Green: True Positive (correct disease prediction)
    tp_mask = (gt_bin == 1) & (pred_bin == 1)
    # Red: False Negative (missed disease)
    fn_mask = (gt_bin == 1) & (pred_bin == 0)
    # Blue: False Positive (incorrect disease prediction)
    fp_mask = (gt_bin == 0) & (pred_bin == 1)
    
    overlay[tp_mask] = [0, 255, 0]    # Green
    overlay[fn_mask] = [255, 0, 0]    # Red  
    overlay[fp_mask] = [0, 0, 255]    # Blue
    
    # Blend with original
    result = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
    
    # Save
    cv2.imwrite(osp.join(output_dir, filename), result)
    print(f"Processed {i+1}/20: {filename}")

print(f"\\nDone! Overlays saved to {output_dir}")
print("Legend: Green=Correct, Red=Missed, Blue=False Alarm")
```

**Save as**: `semantic-segmentation-pytorch/tools/visualize_results.py`

**Run**:
```bash
cd semantic-segmentation-pytorch
python tools/visualize_results.py
```

---

## Metrics Explanation

### Dice Coefficient (F1 Score)
```
Dice = 2 × (Prediction ∩ Ground Truth) / (Prediction + Ground Truth)
```
- Range: 0.0 - 1.0 (0% - 100%)
- **53.28%** = moderate overlap
- Higher is better

### Intersection over Union (IoU / Jaccard Index)
```
IoU = (Prediction ∩ Ground Truth) / (Prediction ∪ Ground Truth)
```
- Range: 0.0 - 1.0 (0% - 100%)
- **36.31%** = fair segmentation quality
- Higher is better
- Stricter than Dice (penalizes both false positives and negatives)

### What Do These Numbers Mean?

| IoU Range | Quality | Description |
|-----------|---------|-------------|
| 0-25% | Poor | Barely overlapping |
| 25-40% | Fair | **← You are here (36.3%)** |
| 40-60% | Good | Reasonable segmentation |
| 60-80% | Very Good | High-quality segmentation |
| 80-100% | Excellent | Near-perfect |

**Your performance (36.3%)**: Fair quality, significant improvement from baseline (11.3%)!

---

## Quick SBATCH Script for Evaluation

Create `eval_best_model.sbatch`:

```bash
#!/usr/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p physicsgpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smehta90@asu.edu

module load mamba/latest
source activate upernet_env

cd $SLURM_SUBMIT_DIR/semantic-segmentation-pytorch

echo "Running evaluation on best model (epoch 10)..."

python eval_multipro.py \
  --cfg config/chestxdet-resnet50-upernet-binary.yaml \
  --gpus 0 \
  TEST.checkpoint epoch_10.pth \
  TEST.result ../results/eval_best_model

echo "Evaluation complete!"
echo "Results saved to: ../results/eval_best_model/"
```

**Submit**:
```bash
sbatch eval_best_model.sbatch
```

---

## What to Present

### Slide 1: Problem
- Baseline: 11.3% mIoU (very poor)
- 14-class segmentation too hard

### Slide 2: Solution  
- Binary segmentation (all diseases → 1 class)
- Fixed X-ray normalization
- Unfroze BatchNorm for adaptation
- Reduced downsampling for detail

### Slide 3: Results
- **53.3% Dice, 36.3% mIoU**
- **3.2x improvement!**
- Converges in 10 epochs (~1.5 hours)

### Slide 4: Visualizations
- Show 3-5 example predictions
- Green: correct, Red: missed, Blue: false alarm

### Slide 5: Future Work
- U-Net architecture (better for medical)
- Medical pretrained weights
- Early stopping (overfitting after epoch 10)
- Instance segmentation

---

## Files You Need

**Model Checkpoint** (Best):
```
ckpt/chestxdet-resnet50-upernet-binary/encoder_epoch_10.pth
ckpt/chestxdet-resnet50-upernet-binary/decoder_epoch_10.pth
```

**Config**:
```
config/chestxdet-resnet50-upernet-binary.yaml
```

**Training Log**:
```
logs/upernet_binary_39980395.out
```

---

## Summary Commands

```bash
# 1. Simple: Use training validation metrics (already done)
#    Best: Epoch 10 → mDice=53.28%, mIoU=36.31%

# 2. Detailed: Run full evaluation
cd semantic-segmentation-pytorch
python eval_multipro.py --cfg config/chestxdet-resnet50-upernet-binary.yaml TEST.checkpoint epoch_10.pth

# 3. Visual: Generate predictions
python test.py --cfg config/chestxdet-resnet50-upernet-binary.yaml TEST.checkpoint epoch_10.pth TEST.result ../results/preds

# 4. Compare: Create overlays
python tools/visualize_results.py
```

Good luck with your presentation! 🎉
