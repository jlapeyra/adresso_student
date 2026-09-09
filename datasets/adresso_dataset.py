"""
datasets/adresso_dataset.py
ADReSSo multimodal dataset for the Student.

Data sources:
  - Soft labels:     knowledge_distillation/adresso_softlabels/outputs/adresso_soft_labels_T*.csv
                     Columns: subject_id, dx, mmse, age, sex, prob_CN, prob_MCI, prob_AD
  - Clinical feats:  tfm_alzheimer/data_processed/adresso/adresso_enriched.csv
                     Columns: subject_id, MMSE_norm, AGE_norm, PTGENDER_enc, audio_path, ...
  - Transcriptions:  tfm_alzheimer/data_processed/adresso/transcriptions_whisper.csv
                     Columns: subject_id, transcript

Split strategy:
  The challenge test-dist (71 subjects) has NO audio and is excluded.
  We work with the 166 subjects that have audio and do our own 70/15/15
  stratified split.

Original: tfm_alzheimer/datasets/adresso_audio_dataset.py (adapted for new Teacher CSV format)
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

log = logging.getLogger(__name__)


# ============================================================================
# Audio loading
# ============================================================================

def load_audio(path: str) -> Optional[torch.Tensor]:
    """
    Load a .wav file, convert to mono 16 kHz, truncate and normalise to [-1, 1].
    Returns a float32 Tensor (T,) or None on error.
    """
    try:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != cfg.TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, cfg.TARGET_SAMPLE_RATE)
        waveform = waveform.squeeze(0)
        if waveform.shape[0] > cfg.MAX_AUDIO_SAMPLES:
            waveform = waveform[:cfg.MAX_AUDIO_SAMPLES]
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform.float()
    except Exception as e:
        log.warning(f"Error loading audio {path}: {e}")
        return None


# ============================================================================
# Dataset
# ============================================================================

class ADReSSoDataset(Dataset):
    """
    Multimodal ADReSSo dataset for the Student.

    Args:
        df:            merged DataFrame with soft labels + clinical features + audio_path
        transcripts:   dict {subject_id: transcript_text}
        text_tokenizer: RoBERTa tokenizer
        feature_cols:  list of clinical feature column names
        mode:          "full" | "audio_only" | "text_only"
        max_text_len:  max tokens for RoBERTa
    """

    VALID_MODES = {"full", "audio_only", "text_only", "clinical_only"}

    def __init__(
        self,
        df:             pd.DataFrame,
        transcripts:    Dict[str, str],
        text_tokenizer,
        feature_cols:   List[str],
        mode:           str = "full",
        max_text_len:   int = 512,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.df           = df.reset_index(drop=True)
        self.transcripts  = transcripts
        self.tokenizer    = text_tokenizer
        self.feature_cols = feature_cols
        self.mode         = mode
        self.max_text_len = max_text_len

        # 2-class CSVs (adni_teacher_ablation outputs) lack prob_MCI; mass goes to 0
        # so the KD-loss collapse stays equivalent to the raw (CN, AD) distribution.
        self.has_mci = "prob_MCI" in self.df.columns

        log.info(
            f"ADReSSoDataset: {len(self.df)} subjects | mode={mode} | "
            f"HC={int((self.df['dx'] == 0).sum())} AD={int((self.df['dx'] == 1).sum())} | "
            f"soft_label_classes={'3 (CN/MCI/AD)' if self.has_mci else '2 (CN/AD)'}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        sid = str(row.get("subject_id", idx))

        # Hard label
        label = int(max(0, min(1, int(row.get("dx", 0)))))

        # Soft label [P(CN), P(MCI), P(AD)]. For 2-class CSVs prob_MCI defaults to 0
        # so the renormalised collapse in the KD loss equals the raw (CN, AD) input.
        p_mci_default = 1/3 if self.has_mci else 0.0
        sl = torch.tensor(
            [float(row.get("prob_CN", 1/3)),
             float(row.get("prob_MCI", p_mci_default)),
             float(row.get("prob_AD", 1/3))],
            dtype=torch.float32,
        )
        if sl.sum() > 0:
            sl = sl / sl.sum()

        # Clinical features
        clin_vals = []
        for col in self.feature_cols:
            v = row.get(col, 0.0)
            clin_vals.append(float(v) if not pd.isna(v) else 0.0)
        clinical = torch.tensor(clin_vals, dtype=torch.float32)

        # Audio — not loaded in text_only or clinical_only
        audio_values = None
        if self.mode in ("full", "audio_only"):
            apath = str(row.get("audio_path", ""))
            if apath and Path(apath).exists():
                audio_values = load_audio(apath)
            if audio_values is None:
                audio_values = torch.zeros(cfg.TARGET_SAMPLE_RATE, dtype=torch.float32)

        # Text — not loaded in audio_only or clinical_only
        text_input_ids = text_attention_mask = None
        if self.mode in ("full", "text_only"):
            transcript = self.transcripts.get(sid, "")
            if not transcript or not isinstance(transcript, str):
                transcript = "[UNK]"
            enc = self.tokenizer(
                transcript,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            text_input_ids      = enc["input_ids"].squeeze(0)
            text_attention_mask = enc["attention_mask"].squeeze(0)


        return {
            "subject_id":          sid,
            "label":               torch.tensor(label, dtype=torch.long),
            "soft_label":          sl,
            "clinical":            clinical,
            "audio_values":        audio_values,
            "text_input_ids":      text_input_ids,
            "text_attention_mask": text_attention_mask,
        }


# ============================================================================
# Audio augmentation wrapper (train only)
# ============================================================================

class AugmentedDataset(Dataset):
    """
    Wraps ADReSSoDataset and applies online audio augmentation.

    Safe augmentations for Alzheimer detection (preserve speech rate and pauses):
      1. Random gain ±3 dB
      2. Gaussian noise (SNR 30-40 dB)
      3. Synthetic room impulse response (short delay + decay)
    """

    def __init__(self, base: ADReSSoDataset, aug_factor: int = 1):
        self.base       = base
        self.aug_factor = aug_factor

    def __len__(self) -> int:
        return len(self.base) * self.aug_factor

    def __getitem__(self, idx: int) -> Dict:
        item = dict(self.base[idx % len(self.base)])
        if item["audio_values"] is not None:
            item["audio_values"] = self._augment(item["audio_values"])
            item["subject_id"]   = f"{item['subject_id']}_aug{idx // len(self.base)}"
        return item

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        # 1. Gain
        gain_db  = (torch.rand(1).item() * 2 - 1) * cfg.AUDIO_AUG_GAIN_DB_RANGE
        waveform = waveform * (10 ** (gain_db / 20.0))

        # 2. Gaussian noise
        sig_pwr  = waveform.pow(2).mean().clamp(min=1e-10)
        snr_db   = cfg.AUDIO_AUG_SNR_DB_MIN + torch.rand(1).item() * (
            cfg.AUDIO_AUG_SNR_DB_MAX - cfg.AUDIO_AUG_SNR_DB_MIN
        )
        noise    = torch.randn_like(waveform) * (sig_pwr / 10 ** (snr_db / 10)).sqrt()
        waveform = waveform + noise

        # 3. Synthetic RIR (early reflection)
        delay_ms     = cfg.AUDIO_AUG_RIR_DELAY_MS_MIN + torch.rand(1).item() * (
            cfg.AUDIO_AUG_RIR_DELAY_MS_MAX - cfg.AUDIO_AUG_RIR_DELAY_MS_MIN
        )
        delay_samp   = int(cfg.TARGET_SAMPLE_RATE * delay_ms / 1000.0)
        decay        = cfg.AUDIO_AUG_RIR_DECAY_MIN + torch.rand(1).item() * (
            cfg.AUDIO_AUG_RIR_DECAY_MAX - cfg.AUDIO_AUG_RIR_DECAY_MIN
        )
        if 0 < delay_samp < waveform.shape[0]:
            reflection = torch.zeros_like(waveform)
            reflection[delay_samp:] = waveform[:-delay_samp] * decay
            waveform = waveform + reflection

        # Renormalise
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform.float()


# ============================================================================
# Collator
# ============================================================================

class MultimodalCollator:
    """
    Pads variable-length audio sequences within a batch.
    Handles None for missing modalities.
    """

    def __call__(self, batch: List[Dict]) -> Dict:
        labels      = torch.stack([b["label"]      for b in batch])
        soft_labels = torch.stack([b["soft_label"] for b in batch])
        clinical    = torch.stack([b["clinical"]   for b in batch])
        subject_ids = [b["subject_id"] for b in batch]

        # Audio padding
        audio_list = [b["audio_values"] for b in batch]
        if all(a is not None for a in audio_list):
            max_len    = max(a.shape[0] for a in audio_list)
            padded     = torch.zeros(len(batch), max_len)
            audio_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
            for i, a in enumerate(audio_list):
                L = a.shape[0]
                padded[i, :L]     = a
                audio_mask[i, :L] = 1
            audio_values = padded
        else:
            audio_values = audio_mask = None

        # Text (already padded to max_length in __getitem__)
        if batch[0]["text_input_ids"] is not None:
            text_input_ids      = torch.stack([b["text_input_ids"]      for b in batch])
            text_attention_mask = torch.stack([b["text_attention_mask"] for b in batch])
        else:
            text_input_ids = text_attention_mask = None

        teacher_embedding = torch.stack([b["teacher_embedding"] for b in batch])

        return {
            "subject_id":           subject_ids,
            "label":                labels,
            "soft_label":           soft_labels,
            "clinical":             clinical,
            "audio_values":         audio_values,
            "audio_attention_mask": audio_mask,
            "text_input_ids":       text_input_ids,
            "text_attention_mask":  text_attention_mask,
            "teacher_embedding":    teacher_embedding,
        }


# ============================================================================
# Data loading helpers
# ============================================================================

def load_transcripts(csv_path: Path) -> Dict[str, str]:
    """Load Whisper transcriptions CSV -> {subject_id: transcript}."""
    df = pd.read_csv(csv_path)
    return dict(zip(df["subject_id"].astype(str), df["transcript"].fillna("").astype(str)))


def build_merged_df(
    soft_labels_csv: Path,
    enriched_csv:    Path,
    adni_teacher_embeddings_csv: Path,
    link_features: str
) -> pd.DataFrame:
    """
    Merge the new Teacher soft labels CSV with the enriched ADReSSo CSV.

    The enriched CSV has audio_path + normalised clinical features (MMSE_norm, etc.).
    The soft labels CSV has prob_CN, prob_MCI, prob_AD + dx/mmse/age/sex.

    Only subjects with audio (has_audio=True in soft labels CSV) are kept.
    """
    soft_labels_df  = pd.read_csv(soft_labels_csv)
    enriched_df = pd.read_csv(enriched_csv)
    teacher_embeddings_df = pd.read_csv(adni_teacher_embeddings_csv) if adni_teacher_embeddings_csv else None

    # Keep only subjects that have audio
    soft_labels_df = soft_labels_df[soft_labels_df["has_audio"] == True].copy()

    to_merge = [
        soft_labels_df, 
        enriched_df[["subject_id", "audio_path"]]
    ]
    if link_features in ("plain", "both"):
        to_merge.append(enriched_df[["subject_id"] + cfg.CLINICAL_FEATURE_COLS])
    if link_features in ("emb", "both") and teacher_embeddings_df is not None:
        to_merge.append(teacher_embeddings_df)

    # Merge on subject_id — keep soft label columns and clinical norms from enriched
    merged = to_merge[0].copy()
    for df in to_merge[1:]:
        merged = merged.merge(df, on="subject_id", how="inner")

    log.info(
        f"Merged dataset: {len(merged)} subjects with audio "
        f"(HC={int((merged['dx'] == 0).sum())} AD={int((merged['dx'] == 1).sum())})"
    )
    return merged


# ============================================================================
# DataLoader factory
# ============================================================================


def get_feature_cols(link_features: str) -> List[str]:
    """
    Return the list of clinical feature columns based on link_features option.
    """
    if link_features == "plain":
        return cfg.CLINICAL_FEATURE_COLS
    elif link_features == "emb":
        return cfg.EMBEDDED_CLINICAL_FEATURE_COLS
    elif link_features == "both":
        return cfg.CLINICAL_FEATURE_COLS + cfg.EMBEDDED_CLINICAL_FEATURE_COLS
    else:
        raise ValueError(f"Invalid link_features: {link_features}")


def build_dataloaders(
    soft_labels_csv:  Optional[Path] = None,
    enriched_csv:     Optional[Path] = None,
    transcripts_csv:  Optional[Path] = None,
    adni_teacher_embeddings_csv: Optional[Path] = None,
    roberta_model:    str = "roberta-base",
    temperature:      float = 3.0,
    batch_size:       int = 8,
    num_workers:      int = 4,
    mode:             str = "full",
    fold:             Optional[int] = None,
    n_folds:          int = 5,
    augment_train:    bool = False,
    aug_factor:       int = 1,
    link_features:    str = "plain",
) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders.

    Split strategy:
      - If fold is None: single 70/15/15 stratified split.
      - If fold is int:  5-fold CV. Test set is always the same 15 % held-out;
                         val rotates across the remaining 85 % in 5 folds.

    Args:
        fold:       None for single split, 0-4 for CV fold.
        aug_factor: extra augmented copies per training sample (requires augment_train=True).
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import WeightedRandomSampler
    from transformers import RobertaTokenizerFast

    sl_csv  = soft_labels_csv  or cfg.SOFT_LABELS_T3_CSV
    enc_csv = enriched_csv     or cfg.ADRESSO_ENRICHED_CSV
    tr_csv  = transcripts_csv  or cfg.TRANSCRIPTIONS_CSV
    emb_csv = adni_teacher_embeddings_csv or cfg.ADNI_TEACHER_EMBEDDINGS_CSV

    df          = build_merged_df(sl_csv, enc_csv, emb_csv, link_features=link_features)
    transcripts = load_transcripts(tr_csv)

    log.info(f"Loading tokenizer ({roberta_model})...")
    tokenizer = RobertaTokenizerFast.from_pretrained(roberta_model)

    labels = df["dx"].fillna(0).astype(int).values
    all_idx = np.arange(len(df))

    # Fixed held-out test set (same regardless of fold)
    tv_idx, test_idx = train_test_split(
        all_idx, test_size=cfg.TEST_SPLIT,
        stratify=labels, random_state=cfg.RANDOM_SEED,
    )

    if fold is not None:
        # CV: val rotates among the tv_idx pool
        skf = __import__("sklearn.model_selection", fromlist=["StratifiedKFold"]).StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=cfg.RANDOM_SEED
        )
        splits = list(skf.split(tv_idx, labels[tv_idx]))
        train_rel, val_rel = splits[fold % n_folds]
        train_idx = tv_idx[train_rel]
        val_idx   = tv_idx[val_rel]
    else:
        val_frac  = cfg.VAL_SPLIT / (1.0 - cfg.TEST_SPLIT)
        train_idx, val_idx = train_test_split(
            tv_idx, test_size=val_frac,
            stratify=labels[tv_idx], random_state=cfg.RANDOM_SEED,
        )

    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    log.info(
        f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"(fold={fold})"
    )

    collator = MultimodalCollator()
    loaders  = {}


    for split_name, idx in split_map.items():
        split_df = df.iloc[idx].copy()
        is_train = (split_name == "train")

        ds = ADReSSoDataset(
            df=split_df,
            transcripts=transcripts,
            text_tokenizer=tokenizer,
            feature_cols=get_feature_cols(link_features),
            mode=mode,
            max_text_len=512,
        )

        if is_train and augment_train and len(ds) > 0:
            aug_ds       = AugmentedDataset(ds, aug_factor=aug_factor)
            ds_for_loader = ConcatDataset([ds, aug_ds])
            orig_dx = split_df["dx"].fillna(0).astype(int).values
            all_dx  = np.concatenate([orig_dx, np.tile(orig_dx, aug_factor)])
            log.info(
                f"[train] augmentation: {len(ds)} + {len(aug_ds)} aug "
                f"(factor×{aug_factor}) = {len(ds_for_loader)}"
            )
        else:
            ds_for_loader = ds
            all_dx = split_df["dx"].fillna(0).astype(int).values

        sampler = None
        if is_train and len(ds_for_loader) > 0:
            class_cnt = np.bincount(all_dx, minlength=2)
            class_wt  = 1.0 / (class_cnt + 1e-8)
            sw        = torch.tensor([class_wt[d] for d in all_dx], dtype=torch.float32)
            sampler   = WeightedRandomSampler(sw, len(sw), replacement=True)

        loaders[split_name] = DataLoader(
            ds_for_loader,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=False,
        )

    return loaders
