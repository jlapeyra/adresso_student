#!/bin/bash


KD_DIR="/home/usuaris/veu/joan.lapeyra/knowledge_distillation"
RESULTS_DIR="$KD_DIR/adresso_student/outputs/results_oct4"

SCRIPT="$KD_DIR/adresso_student/scripts/train_student_generic.sbatch"

sbatch --job-name=audio_only "$SCRIPT" --ablation audio_only  --results-dir "$RESULTS_DIR/audio_only"
sbatch --job-name=text_only "$SCRIPT" --ablation text_only   --results-dir "$RESULTS_DIR/text_only"
sbatch --job-name=multimodal_no_kd "$SCRIPT" --ablation multimodal_no_kd  --results-dir "$RESULTS_DIR/mm"
sbatch --job-name=multimodal_kd "$SCRIPT" --ablation multimodal_kd  --results-dir "$RESULTS_DIR/mm_sl"
sbatch --job-name=multimodal_no_kd_emb "$SCRIPT" --ablation multimodal_no_kd  --results-dir "$RESULTS_DIR/mm_emb" --link-features emb
sbatch --job-name=multimodal_kd_emb "$SCRIPT" --ablation multimodal_kd  --results-dir "$RESULTS_DIR/mm_sl_emb" --link-features emb
sbatch --job-name=multimodal_no_kd_both "$SCRIPT" --ablation multimodal_no_kd  --results-dir "$RESULTS_DIR/mm_both" --link-features both
sbatch --job-name=multimodal_kd_both "$SCRIPT" --ablation multimodal_kd  --results-dir "$RESULTS_DIR/mm_sl_both" --link-features both

# - audio only
# - text only
# - audio + text
# - audio + text + soft labels
# - audio + text + embeddings
# - audio + text + soft labels + embeddings

