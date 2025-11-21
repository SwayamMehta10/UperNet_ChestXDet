# Quick Fix for Evaluation Errors

## Issues Fixed

### 1. Python 3.10+ Collections Error ✅
**Error**: `AttributeError: module 'collections' has no attribute 'Sequence'`

**Fix Applied**: Updated `semantic-segmentation-pytorch/mit_semseg/lib/utils/th.py`
- Changed `import collections` → `from collections.abc import Sequence, Mapping`
- Updated all references to use the imported classes

### 2. Multi-GPU Error
**Error**: `CUDA error: invalid device ordinal` (tried to use GPUs 0-3, but only 1 available)

**Fix**: Use `--gpus 0` flag to specify single GPU

---

## Correct Evaluation Command

```bash
cd /scratch/$USER/upernet_chestxdet/semantic-segmentation-pytorch

module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate upernet_env

# Use SINGLE GPU (--gpus 0 instead of default 0,1,2,3)
python eval_multipro.py \
  --cfg config/chestxdet-resnet50-upernet-binary.yaml \
  --gpus 0 \
  TEST.checkpoint epoch_10.pth
```

---

## Alternative: Single-Process Evaluation

If multi-process still has issues, use single-threaded evaluation:

```bash
# Create eval_single.py wrapper
python -c "
import sys
sys.path.insert(0, '.')
from eval_multipro import main
import torch

# Force single GPU
torch.cuda.set_device(0)

# Run main with modified args
sys.argv = [
    'eval_multipro.py',
    '--cfg', 'config/chestxdet-resnet50-upernet-binary.yaml',
    '--gpus', '0',
    'TEST.checkpoint', 'epoch_10.pth'
]
main()
"
```

---

## What Was Changed

**File**: `semantic-segmentation-pytorch/mit_semseg/lib/utils/th.py`

**Before**:
```python
import collections

def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        ...
```

**After**:
```python
from collections.abc import Sequence, Mapping

def as_numpy(obj):
    if isinstance(obj, Sequence):
        ...
```

---

## Expected Output

After running the corrected command, you should see:

```
Loading weights for net_encoder
Loading weights for net_decoder
# samples: 553
Evaluating: 100%|████████████| 553/553 [XX:XX<00:00, X.XXit/s]

Mean IoU: 0.3631
Mean Dice: 0.5328
```

The results will match what we saw during training validation at epoch 10! ✅
