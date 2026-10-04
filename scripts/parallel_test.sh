#!/bin/bash


KD_DIR="/home/usuaris/veu/joan.lapeyra/knowledge_distillation"
RESULTS_DIR="$KD_DIR/adresso_student/outputs/results_oct4"

SCRIPT="$KD_DIR/adresso_student/scripts/train_student_generic.sbatch"

sbatch "$SCRIPT" --ablition audio_only  --results-dir "$RESULTS_DIR/audio_only"
sbatch "$SCRIPT" --ablation text_only   --results-dir "$RESULTS_DIR/text_only"
sbatch "$SCRIPT" --ablation multimodal_no_kd  --results-dir "$RESULTS_DIR/mm"
sbatch "$SCRIPT" --ablation multimodal_kd  --results-dir "$RESULTS_DIR/mm_sl"
sbatch "$SCRIPT" --ablation multimodal_no_kd  --results-dir "$RESULTS_DIR/mm_emb" --link-features emb
sbatch "$SCRIPT" --ablation multimodal_kd  --results-dir "$RESULTS_DIR/mm_sl_emb" --link-features emb
sbatch "$SCRIPT" --ablation multimodal_no_kd  --results-dir "$RESULTS_DIR/mm_both" --link-features both
sbatch "$SCRIPT" --ablation multimodal_kd  --results-dir "$RESULTS_DIR/mm_sl_both" --link-features both

# - audio only
# - text only
# - audio + text
# - audio + text + soft labels
# - audio + text + embeddings
# - audio + text + soft labels + embeddings

