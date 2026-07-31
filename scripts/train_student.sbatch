#!/usr/bin/env bash
#SBATCH --job-name=adresso_student
#SBATCH --output=/home/usuaris/veu/joan.lapeyra/knowledge_distillation/adresso_student/logs/slurm/%x_%j.txt
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --nodelist=veuc12

set -euo pipefail

REPO_DIR="/home/usuaris/veu/joan.lapeyra/knowledge_distillation/adresso_student"
KD_DIR="/home/usuaris/veu/joan.lapeyra/knowledge_distillation"
cd "$REPO_DIR"

source "$KD_DIR/.venv/bin/activate"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$REPO_DIR/.cache/huggingface"

mkdir -p "$HF_HOME" "$REPO_DIR/logs/slurm"

date
nvidia-smi
echo "Starting ADReSSo Student training..."

python -u train.py \
  --ablation all \
  --cv \
  --n-folds 5 \
  --epochs 50 \
  --batch-size 8 \
  --lr 5e-5 \
  --lr-pretrained 1e-5 \
  --weight-decay 1e-2 \
  --patience 20 \
  --grad-accum 2 \
  --temperature 1.1861923153571683 \
  --alpha 0.6 \
  --beta 0.4 \
  --num-workers 4 \
  --augment-train \
  --aug-factor 2

date
echo "Student training completed."
