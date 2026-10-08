#!/bin/bash


KD_DIR="/home/usuaris/veu/joan.lapeyra/knowledge_distillation"
RESULTS_DIR="$KD_DIR/adresso_student/outputs/results_oct5"

SCRIPT="$KD_DIR/adresso_student/scripts/train_student_generic.sbatch"


sbatch --job-name=multimodal_no_kd_both "$SCRIPT" --ablation multimodal_no_kd  --results-dir "$RESULTS_DIR/mm_sl" --link-features all
sbatch --job-name=multimodal_kd_both "$SCRIPT" --ablation multimodal_kd  --results-dir "$RESULTS_DIR/mm_no_sl" --link-features all

# - audio only
# - text only
# - audio + text
# - audio + text + soft labels
# - audio + text + embeddings
# - audio + text + soft labels + embeddings

