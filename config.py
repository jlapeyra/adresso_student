"""
config.py — Central configuration for adresso_student.

Student multimodal (Wav2Vec2 + RoBERTa + Clinical MLP + Fusion Transformer)
trained with knowledge distillation from adni_teacher soft labels.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

# Teacher soft labels — sibling folder within knowledge_distillation/
KD_ROOT         = BASE_DIR.parent   # knowledge_distillation/
SOFT_LABELS_DIR = KD_ROOT / "adresso_softlabels" / "outputs"
SOFT_LABELS_T1_CSV = SOFT_LABELS_DIR / "adresso_soft_labels_T1.csv"
SOFT_LABELS_T2_CSV = SOFT_LABELS_DIR / "adresso_soft_labels_T2.csv"
SOFT_LABELS_T3_CSV = SOFT_LABELS_DIR / "adresso_soft_labels_T3.csv"

# ADReSSo enriched CSV from tfm_alzheimer (has audio_path + normalised clinical features)
TFM_DIR              = BASE_DIR.parent.parent / "tfm_alzheimer"
ADRESSO_ENRICHED_CSV = TFM_DIR / "data_processed" / "adresso" / "adresso_enriched.csv"
# Whisper transcriptions from tfm_alzheimer
TRANSCRIPTIONS_CSV   = TFM_DIR / "data_processed" / "adresso" / "transcriptions_whisper.csv"
# ADReSSo audio root (Roger's directory)
ADRESSO_AUDIO_ROOT   = Path(
    "/home/usuaris/veu/roger.esteve.sanchez/adresso/ADReSSo21/diagnosis"
)

ADNI_TEACHER_EMBEDDINGS_CSV = KD_ROOT / "adni_teacher" / "outputs" / "adni_teacher_embeddings.csv"

LOGS_DIR     = BASE_DIR / "logs"
OUTPUTS_DIR  = BASE_DIR / "outputs"
CKPT_DIR     = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR  = OUTPUTS_DIR / "results"

# ---------------------------------------------------------------------------
# Clinical features (minimal: shared between ADNI and ADReSSo)
# ---------------------------------------------------------------------------
CLINICAL_FEATURE_COLS = [
    "MMSE_norm",
    "AGE_norm",
    "PTGENDER_enc",
]

# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS  = 60.0
MAX_AUDIO_SAMPLES  = int(TARGET_SAMPLE_RATE * MAX_AUDIO_SECONDS)

# Audio augmentation (train only)
AUDIO_AUG_GAIN_DB_RANGE    = 3.0
AUDIO_AUG_SNR_DB_MIN       = 30.0
AUDIO_AUG_SNR_DB_MAX       = 40.0
AUDIO_AUG_RIR_DELAY_MS_MIN = 1.0
AUDIO_AUG_RIR_DELAY_MS_MAX = 8.0
AUDIO_AUG_RIR_DECAY_MIN    = 0.10
AUDIO_AUG_RIR_DECAY_MAX    = 0.25

# ---------------------------------------------------------------------------
# Dataset splits — applied on the 166 subjects WITH audio
# (the challenge test-dist 71 subjects have no audio and are excluded)
# ---------------------------------------------------------------------------
RANDOM_SEED       = 42
TRAIN_SPLIT       = 0.60   # 60 %  (~100 subjects, per CV fold ~80)
VAL_SPLIT         = 0.15   # 15 %  (~25 subjects, per CV fold ~25)
TEST_SPLIT        = 0.25   # 25 %  (~41 subjects, fixed holdout)

# ---------------------------------------------------------------------------
# Knowledge distillation
# ---------------------------------------------------------------------------
SOFT_LABEL_DEFAULT_T = 3.0   # temperature used when generating the soft labels
KD_ALPHA             = 0.7   # weight for hard CE loss
KD_BETA              = 0.3   # weight for KL-div distillation loss
KD_TEMPERATURE       = 3.0   # temperature applied to Student logits during KL

FEATURE_KD = False
FEATURE_KD_WEIGHT = 0.1
FEATURE_KD_PROJ_DIM = 128
FEATURE_KD_LOSS = "cosine"   # o "l2"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
EPOCHS          = 50
BATCH_SIZE      = 8
LR_PRETRAINED   = 1e-5   # Wav2Vec2 + RoBERTa (last 4 layers)
LR_NEW          = 5e-5   # fusion, classifiers, clinical MLP
WEIGHT_DECAY    = 1e-2
GRAD_CLIP       = 1.0
GRAD_ACCUM      = 2      # effective batch = 8 * 2 = 16
WARMUP_EPOCHS   = 10
PATIENCE        = 20
LABEL_SMOOTHING = 0.05

# Modality dropout probability per modality (train only)
MODALITY_DROPOUT_P = 0.2

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
WAV2VEC2_MODEL  = "/home/usuaris/veu/joan.lapeyra/knowledge_distillation/pretrained/wav2vec2-base"
ROBERTA_MODEL   = "roberta-base"
FUSION_DIM      = 256
FREEZE_AUDIO_N  = 12   # freeze all 12 transformer layers of Wav2Vec2 (linear probe)
FREEZE_TEXT_N   = 12   # freeze all 12 transformer layers of RoBERTa (linear probe)
FUSION_LAYERS   = 2
FUSION_HEADS    = 4
MAX_TEXT_LEN    = 512

# ---------------------------------------------------------------------------
# Ablation experiment definitions
# ---------------------------------------------------------------------------
ABLATION_MODES = {
    "clinical_only":    {"mode": "clinical_only", "use_kd": False},
    "audio_only":       {"mode": "audio_only",    "use_kd": False},
    "text_only":        {"mode": "text_only",     "use_kd": False},
    "multimodal_no_kd": {"mode": "full",          "use_kd": False},
    "multimodal_kd":    {"mode": "full",          "use_kd": True},
}
