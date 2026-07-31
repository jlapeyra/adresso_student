#!/usr/bin/env bash
#SBATCH --job-name=adresso_softlabel_source
#SBATCH --output=/home/usuaris/veu/joan.lapeyra/knowledge_distillation/adresso_student/logs/slurm/%x_%j.txt
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --nodelist=veuc12

# Soft-label source ablation: 21 teacher variants x {with,without} clinical x 5 folds
# = 210 runs total. Supports resume — re-running is safe.
#
# To chunk across multiple SLURM jobs, override SOURCES / CLINICAL / FOLDS via
# environment variables before sbatch, e.g.:
#   SOURCES="b0_random b1_mri_link" sbatch train_softlabel_source.sbatch
#   FOLDS="0 1" sbatch train_softlabel_source.sbatch
#   AGGREGATE_ONLY=1 sbatch train_softlabel_source.sbatch

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
echo "Starting soft-label source ablation..."

EXTRA_ARGS=()
if [[ -n "${SOURCES:-}" ]];        then EXTRA_ARGS+=(--sources ${SOURCES}); fi
if [[ -n "${CLINICAL:-}" ]];       then EXTRA_ARGS+=(--clinical-modes ${CLINICAL}); fi
if [[ -n "${FOLDS:-}" ]];          then EXTRA_ARGS+=(--folds ${FOLDS}); fi
if [[ -n "${AGGREGATE_ONLY:-}" ]]; then EXTRA_ARGS+=(--aggregate-only); fi

python -u train_softlabel_source_ablation.py \
  --n-folds 5 \
  --epochs 50 \
  --batch-size 8 \
  --lr 5e-5 \
  --lr-pretrained 1e-5 \
  --weight-decay 1e-2 \
  --patience 20 \
  --grad-accum 2 \
  --temperature 0.5804632194973941 \
  --alpha 0.6 \
  --beta 0.4 \
  --num-workers 4 \
  --augment-train \
  --aug-factor 2 \
  "${EXTRA_ARGS[@]}"

date
echo "Soft-label source ablation completed."
