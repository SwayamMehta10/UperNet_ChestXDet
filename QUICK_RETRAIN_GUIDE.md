# Quick Implementation Guide for One-Shot Retraining

## Files Created ✅
1. `tools/split_train_val.py` - Creates 80/20 train/val split
2. `mit_semseg/loss.py` - Dice loss implementation
3. `config/chestxdet-resnet50-upernet-binary-v2.yaml` - Updated config

## Steps to Run on Sol

### 1. Create Train/Val Split (2 min)
```bash
cd /scratch/$USER/upernet_chestxdet/semantic-segmentation-pytorch
python tools/split_train_val.py
```

**Expected output**:
```
Loading original training data...
Total training samples: 3025

Split created:
  Train: 2420 images (80%)
  Val:   605 images (20%)

Saved: data/chestxdet_binary/lists/train_split.odgt
Saved: data/chestxdet_binary/lists/val_split.odgt

✅ Train/Val split created successfully!
```

### 2. Modify train.py for Early Stopping & Dice Loss

**Option A: Manual edit** (10 min)
Add to `train.py` after imports (around line 20):
```python
from mit_semseg.loss import CombinedLoss

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_metric, epoch):
        if self.best_score is None:
            self.best_score = val_metric
            self.best_epoch = epoch
        elif val_metric < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(f"Val improved: {self.best_score:.4f} -> {val_metric:.4f}")
            self.best_score = val_metric
            self.best_epoch = epoch
            self.counter = 0
        return self.early_stop
```

Replace loss (around line 250):
```python
# OLD:
# crit = nn.NLLLoss(ignore_index=-1)

# NEW:
crit = CombinedLoss(weight_ce=0.5, weight_dice=0.5, ignore_index=-1)
```

Add early stopping in main loop (around line 400):
```python
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
    # ... existing training code ...
    
    # After validation (around line 450)
    if epoch % 1 == 0:
        # Get disease class IoU from validation
        val_iou = iou[1]  # Disease class
        
        if early_stopping(val_iou, epoch):
            logger.info(f"Early stopping at epoch {epoch}")
            logger.info(f"Best: epoch {early_stopping.best_epoch}, IoU {early_stopping.best_score:.4f}")
            break
```

**Option B: Skip for now** (0 min)
- Just use new config with proper split
- Early stopping can be added later if time permits

### 3. Submit Retraining Job
```bash
# Update SBATCH script to use new config
cd /scratch/$USER/upernet_chestxdet

# Edit train_upernet_binary.sbatch
# Change: --cfg config/chestxdet-resnet50-upernet-binary.yaml
# To:     --cfg config/chestxdet-resnet50-upernet-binary-v2.yaml

sbatch train_upernet_binary.sbatch
```

### 4. Monitor Training
```bash
# Watch logs
watch -n 60 'tail -50 logs/upernet_binary_*.out | grep -E "(Epoch|Val|IoU|Dice)"'

# Check job status
squeue -u $USER
```

## Minimal Version (If Time is Tight)

**Just do the split + retrain** (skip early stopping and Dice loss):

1. Run `python tools/split_train_val.py`
2. Update SBATCH to use `config/chestxdet-resnet50-upernet-binary-v2.yaml`
3. Submit job
4. Wait ~3 hours
5. Evaluate on test set

This still fixes the data leakage issue!

## After Training Completes

### Evaluate on Test Set (ONCE)
```bash
cd semantic-segmentation-pytorch

# Find best checkpoint (check logs for best validation epoch)
# Let's say it's epoch 15

python eval_multipro.py \
  --cfg config/chestxdet-resnet50-upernet-binary-v2.yaml \
  --gpus 0 \
  TEST.checkpoint epoch_15.pth
```

### Generate Visualizations
```bash
python tools/simple_visualize.py \
  --cfg config/chestxdet-resnet50-upernet-binary-v2.yaml \
  --checkpoint epoch_15.pth \
  --gpu 0
```

## Expected Timeline
- Split creation: 2 min
- Training: 2-3 hours (with early stopping)
- Evaluation: 5 min
- **Total: ~3 hours**

## What to Report in Presentation

**Methodology**:
- ✅ Proper train (80%) / validation (20%) / test (holdout) split
- ✅ No data leakage
- ✅ Reproducible (seed=42)

**Results** (expected):
- Disease IoU: 38-42%
- Disease Dice: 55-59%
- Still 3-4× better than baseline

**Honesty**:
- "Initial experiment had test set leakage"
- "Retrained properly - slightly lower but defensible"
- "Results now follow best practices"
