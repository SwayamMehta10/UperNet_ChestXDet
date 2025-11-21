#!/bin/bash
#SBATCH --job-name=upernet_binary
#SBATCH --output=logs/upernet_binary_%j.out
#SBATCH --error=logs/upernet_binary_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=publicgpu
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smehta90@asu.edu

# ChestX-Det Binary Segmentation Training with UperNet
# Time-optimized for 4-hour A100 windows with checkpoint resumption

module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate upernet_env

cd $SLURM_SUBMIT_DIR/semantic-segmentation-pytorch

# Ensure log directory exists
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Step 1: Generate binary masks (only first time)
if [ ! -d "data/chestxdet_binary/masks_train" ]; then
    echo "================================================"
    echo "Generating binary segmentation masks..."
    echo "================================================"
    python tools/chestxdet_to_odgt_binary.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate masks"
        exit 1
    fi
    echo "Masks generated successfully!"
fi

# Step 2: Determine start epoch from existing checkpoints
CFG="config/chestxdet-resnet50-upernet-binary-v2.yaml"
CKPT_DIR="ckpt/chestxdet-resnet50-upernet-binary"

START_EPOCH=0
if [ -d "$CKPT_DIR" ]; then
    # Find latest epoch checkpoint
    LATEST=$(ls -1 $CKPT_DIR/encoder_epoch_*.pth 2>/dev/null | sort -V | tail -1)
    if [ ! -z "$LATEST" ]; then
        START_EPOCH=$(echo $LATEST | grep -oP 'epoch_\K[0-9]+')
        echo "================================================"
        echo "Resuming from epoch $START_EPOCH"
        echo "================================================"
    fi
fi

# Step 3: Run training
echo "================================================"
echo "Starting training (Epoch $START_EPOCH onwards)..."
echo "================================================"
echo "Config: $CFG"
echo "GPUs: 1 x A100"
echo "Batch size per GPU: 2"
echo "Workers: 16"
echo "Target epochs: 30"
echo "================================================"

python train.py \
    --cfg $CFG \
    --gpus 0 \
    TRAIN.start_epoch $START_EPOCH

EXIT_CODE=$?

echo "================================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "================================================"

# Print latest metrics if available
if [ -f "$CKPT_DIR/metrics.csv" ]; then
    echo ""
    echo "Latest metrics:"
    tail -5 $CKPT_DIR/metrics.csv
fi

exit $EXIT_CODE
