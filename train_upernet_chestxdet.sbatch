#!/bin/bash
#SBATCH -J chestxdet_upp
#SBATCH -p public
#SBATCH -q class
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 16:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smehta90@asu.edu

module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate upernet_env

cd "$SLURM_SUBMIT_DIR/semantic-segmentation-pytorch"

# deps (first run only; safe to re-run)
pip install -r requirements.txt yacs tqdm opencv-python

python -u train.py \
  --gpus 0 \
  --cfg config/chestxdet-resnet50-upernet.yaml
