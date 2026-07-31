#!/usr/bin/env bash
#SBATCH --job-name=adresso_sl_src
#SBATCH --output=/home/usuaris/veu/joan.lapeyra/knowledge_distillation/adresso_student/logs/slurm/%x_%A_%a.txt
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --array=0-41

# Array job: 42 elements = 21 soft-label sources x 2 clinical modes.
# Each element trains 5 folds for one (source, clinical) combination
# (~5 x 27min = ~2.5h per element). Resume-safe: existing JSONs are skipped.
#
# Submit:   sbatch scripts/train_softlabel_source_array.sbatch
# Status:   squeue -u $USER -t RUNNING,PENDING --array
# Cancel:   scancel <jobid>     # cancels whole array
#           scancel <jobid>_5   # cancels just element 5

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

# Map SLURM_ARRAY_TASK_ID -> (source, clinical)
SOURCES=(\
  b0_random b1_mri_link b2_xattn b3_priv b4_qcond \
  b5_gated b6_dropout b7_full b8_no_mmse \
  full link_only no_priv no_mri no_pretrain no_mmse \
  no_query_cond no_gated_fusion mean_pool no_vision_head \
  no_priv_dropout no_distill \
)
CLINS=(with without)

IDX="${SLURM_ARRAY_TASK_ID}"
SRC_IDX=$((IDX / 2))
CLIN_IDX=$((IDX % 2))
SOURCE="${SOURCES[$SRC_IDX]}"
CLIN="${CLINS[$CLIN_IDX]}"

date
hostname
nvidia-smi -L || true
echo "Array task ${IDX}: source=${SOURCE} clinical=${CLIN}"

python -u train_softlabel_source_ablation.py \
  --sources "${SOURCE}" \
  --clinical-modes "${CLIN}" \
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
  --aug-factor 2

date
echo "Array task ${IDX} done."
