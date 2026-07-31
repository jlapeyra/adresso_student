"""
train.py — Training entry point for the ADReSSo Student.

Experiments:
  A) Full KD training (proposed):
       python train.py

  B) 5-fold cross-validation:
       python train.py --cv

  C) Run all ablations:
       python train.py --ablation all

  D) Single ablation:
       python train.py --ablation audio_only
       python train.py --ablation text_only
       python train.py --ablation multimodal_no_kd
       python train.py --ablation multimodal_kd    (default, proposed)

  E) Dry-run (architecture check without real data):
       python train.py --dry-run

Ablation scheme:
  Baseline 1: audio_only       -> Wav2Vec2 only, no KD
  Baseline 2: text_only        -> RoBERTa/Whisper only, no KD
  Baseline 3: multimodal_no_kd -> Audio+Text+Clinical, hard CE only
  Proposed:   multimodal_kd    -> Audio+Text+Clinical + KD

Original: tfm_alzheimer/train_student.py (adapted for new soft-label CSV format)
"""

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(cfg.LOGS_DIR / "train_student.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ============================================================================
# CLI arguments
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="ADReSSo Student — Knowledge Distillation training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--soft-labels-csv",  type=Path, default=None,
                   help="Soft-labels CSV. Auto-derived from --temperature if not set.")
    p.add_argument("--enriched-csv",     type=Path, default=cfg.ADRESSO_ENRICHED_CSV)
    p.add_argument("--transcripts-csv",  type=Path, default=cfg.TRANSCRIPTIONS_CSV)
    p.add_argument("--temperature",      type=float, default=cfg.KD_TEMPERATURE)

    # Model
    p.add_argument("--wav2vec2",         default=cfg.WAV2VEC2_MODEL)
    p.add_argument("--roberta",          default=cfg.ROBERTA_MODEL)
    p.add_argument("--fusion-dim",       type=int,   default=cfg.FUSION_DIM)
    p.add_argument("--freeze-audio",     type=int,   default=cfg.FREEZE_AUDIO_N)
    p.add_argument("--freeze-text",      type=int,   default=cfg.FREEZE_TEXT_N)
    p.add_argument("--fusion-layers",    type=int,   default=cfg.FUSION_LAYERS)
    p.add_argument("--fusion-heads",     type=int,   default=cfg.FUSION_HEADS)

    # KD loss
    p.add_argument("--alpha",            type=float, default=cfg.KD_ALPHA)
    p.add_argument("--beta",             type=float, default=cfg.KD_BETA)

    # Training
    p.add_argument("--epochs",           type=int,   default=cfg.EPOCHS)
    p.add_argument("--batch-size",       type=int,   default=cfg.BATCH_SIZE)
    p.add_argument("--lr",               type=float, default=cfg.LR_NEW)
    p.add_argument("--lr-pretrained",    type=float, default=cfg.LR_PRETRAINED)
    p.add_argument("--weight-decay",     type=float, default=cfg.WEIGHT_DECAY)
    p.add_argument("--grad-clip",        type=float, default=cfg.GRAD_CLIP)
    p.add_argument("--grad-accum",       type=int,   default=cfg.GRAD_ACCUM)
    p.add_argument("--no-amp",           action="store_true")
    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--patience",         type=int,   default=cfg.PATIENCE)
    p.add_argument("--label-smoothing",  type=float, default=cfg.LABEL_SMOOTHING)

    # Experiment mode
    p.add_argument("--ablation",         default="multimodal_kd",
                   choices=list(cfg.ABLATION_MODES.keys()) + ["all"])
    p.add_argument("--cv",               action="store_true", help="5-fold cross-validation")
    p.add_argument("--n-folds",          type=int, default=5)
    p.add_argument("--fold",             type=int, default=None,
                   help="Run a specific fold (0-indexed). Only with --cv.")
    p.add_argument("--augment-train",    action="store_true")
    p.add_argument("--aug-factor",       type=int, default=1)
    p.add_argument("--no-clinical",      action="store_true",
                   help="Exclude clinical features (MMSE/age/sex) from the Student input.")

    # Misc
    p.add_argument("--seed",             type=int, default=cfg.RANDOM_SEED)
    p.add_argument("--dry-run",          action="store_true")
    p.add_argument("--no-tb",            action="store_true", help="Disable TensorBoard")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================================
# Optimizer (differential LR)
# ============================================================================

def build_optimizer(model, args):
    pretrained, new = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "wav2vec2" in name or "roberta" in name:
            pretrained.append(param)
        else:
            new.append(param)
    optimizer = torch.optim.AdamW([
        {"params": pretrained, "lr": args.lr_pretrained, "name": "pretrained"},
        {"params": new,        "lr": args.lr,            "name": "new_layers"},
    ], weight_decay=args.weight_decay, eps=1e-8)
    log.info(f"  AdamW: lr_pretrained={args.lr_pretrained} lr_new={args.lr} wd={args.weight_decay}")
    return optimizer


# ============================================================================
# Training loop
# ============================================================================

def train_epoch(model, loader, optimizer, loss_fn, scaler, scheduler,
                device, args, tracker, epoch) -> Dict:
    model.train()
    tracker.reset()
    optimizer.zero_grad()
    running_loss = 0.0

    for step, batch in enumerate(tqdm(loader, desc=f"E{epoch:03d} [train]", leave=False, ncols=110)):
        label       = batch["label"].to(device)
        soft_label  = batch["soft_label"].to(device)
        clinical    = batch["clinical"].to(device)

        audio_values = batch.get("audio_values")
        audio_mask   = batch.get("audio_attention_mask")
        text_ids     = batch.get("text_input_ids")
        text_mask    = batch.get("text_attention_mask")

        if audio_values is not None: audio_values = audio_values.to(device)
        if audio_mask   is not None: audio_mask   = audio_mask.to(device)
        if text_ids     is not None: text_ids     = text_ids.to(device)
        if text_mask    is not None: text_mask    = text_mask.to(device)

        if args.no_clinical:
            clinical = None

        # Modality dropout (never leave all modalities absent)
        p = cfg.MODALITY_DROPOUT_P
        drop_audio    = audio_values is not None and torch.rand(1).item() < p
        drop_text     = text_ids     is not None and torch.rand(1).item() < p
        drop_clinical = clinical     is not None and torch.rand(1).item() < p
        # Count modalities that would survive — ensure at least one remains
        n_surviving = (
            (audio_values is not None and not drop_audio) +
            (text_ids     is not None and not drop_text) +
            (clinical     is not None and not drop_clinical)
        )
        if n_surviving == 0:
            # Undo the drop on whichever single active modality exists
            if audio_values is not None: drop_audio    = False
            elif text_ids   is not None: drop_text     = False
            else:                        drop_clinical = False
        if drop_audio:    audio_values = None;   audio_mask = None
        if drop_text:     text_ids     = None;   text_mask  = None
        if drop_clinical: clinical = torch.zeros_like(clinical)

        use_amp = not args.no_amp and device.type == "cuda"
        with autocast("cuda", enabled=use_amp):
            out = model(
                audio_values=audio_values, audio_mask=audio_mask,
                text_input_ids=text_ids,   text_attention_mask=text_mask,
                clinical=clinical,
            )
            loss_dict = loss_fn(
                logits_hard=out["logits_hard"], logits_soft=out["logits_soft"],
                labels_hard=label, soft_labels=soft_label,
            )
            loss = loss_dict["loss"] / args.grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        real_loss = loss.item() * args.grad_accum
        running_loss += real_loss
        tracker.update(out["logits_hard"].detach(), label.detach(), loss=real_loss)

    return tracker.compute()


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, args, tracker, split="val") -> Dict:
    model.eval()
    tracker.reset()

    for batch in tqdm(loader, desc=f"[{split}]", leave=False, ncols=100):
        label      = batch["label"].to(device)
        soft_label = batch["soft_label"].to(device)
        clinical   = batch["clinical"].to(device)

        audio_values = batch.get("audio_values")
        audio_mask   = batch.get("audio_attention_mask")
        text_ids     = batch.get("text_input_ids")
        text_mask    = batch.get("text_attention_mask")

        if audio_values is not None: audio_values = audio_values.to(device)
        if audio_mask   is not None: audio_mask   = audio_mask.to(device)
        if text_ids     is not None: text_ids     = text_ids.to(device)
        if text_mask    is not None: text_mask    = text_mask.to(device)

        if args.no_clinical:
            clinical = None

        use_amp = not args.no_amp and device.type == "cuda"
        with autocast("cuda", enabled=use_amp):
            out = model(
                audio_values=audio_values, audio_mask=audio_mask,
                text_input_ids=text_ids,   text_attention_mask=text_mask,
                clinical=clinical,
            )
            loss_dict = loss_fn(
                logits_hard=out["logits_hard"], logits_soft=out["logits_soft"],
                labels_hard=label, soft_labels=soft_label,
            )
        tracker.update(out["logits_hard"].detach(), label.detach(),
                       loss=loss_dict["loss"].item())

    return tracker.compute()


# ============================================================================
# Single experiment
# ============================================================================

def train_experiment(
    args,
    ablation_name: str,
    mode:          str,
    use_kd:        bool,
    fold:          Optional[int] = None,
) -> Dict:
    fold_str = f"_fold{fold}" if fold is not None else ""
    exp_name = f"{ablation_name}{fold_str}"

    log.info(f"\n{'='*70}")
    log.info(f"EXPERIMENT: {exp_name}  (mode={mode}, kd={use_kd})")
    log.info(f"{'='*70}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    log.info(f"  Device: {device}")

    from datasets.adresso_dataset import build_dataloaders
    loaders = build_dataloaders(
        soft_labels_csv=args.soft_labels_csv,
        enriched_csv=args.enriched_csv,
        transcripts_csv=args.transcripts_csv,
        roberta_model=args.roberta,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=mode,
        fold=fold,
        n_folds=args.n_folds,
        augment_train=args.augment_train,
        aug_factor=args.aug_factor,
    )

    from models.student import build_student
    model = build_student(
        num_classes_hard=2,
        num_classes_soft=2,
        fusion_dim=args.fusion_dim,
        wav2vec2_model=args.wav2vec2,
        roberta_model=args.roberta,
        n_clinical=len(cfg.CLINICAL_FEATURE_COLS),
        freeze_audio_n=args.freeze_audio,
        freeze_text_n=args.freeze_text,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
    ).to(device)

    for k, v in model.count_parameters().items():
        log.info(f"    {k:40s}: {v:>10,}")

    from distillation.kd_loss import build_kd_loss
    loss_fn = build_kd_loss(
        use_distillation=use_kd,
        alpha=args.alpha, beta=args.beta,
        temperature=args.temperature,
        label_smoothing=args.label_smoothing,
    )

    optimizer = build_optimizer(model, args)

    steps_per_epoch = len(loaders["train"])
    warmup_steps    = cfg.WARMUP_EPOCHS * steps_per_epoch // args.grad_accum
    total_steps     = args.epochs * steps_per_epoch // args.grad_accum

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler    = GradScaler("cuda", enabled=(not args.no_amp and device.type == "cuda"))

    from training.metrics import MetricTracker, EarlyStopping
    early_stop  = EarlyStopping(patience=args.patience, mode="max", restore_best=True)
    class_names = ["HC", "AD"]
    tr_tracker  = MetricTracker(num_classes=2, class_names=class_names)
    val_tracker = MetricTracker(num_classes=2, class_names=class_names)
    te_tracker  = MetricTracker(num_classes=2, class_names=class_names)

    # TensorBoard (optional)
    tb_writer = None
    if not args.no_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(str(cfg.LOGS_DIR / "tb" / exp_name))
        except ImportError:
            pass

    # CSV logger
    csv_path = cfg.RESULTS_DIR / f"{exp_name}_curves.csv"
    csv_f    = open(csv_path, "w", newline="")
    csv_w    = None

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tr_m  = train_epoch(model, loaders["train"], optimizer, loss_fn,
                            scaler, scheduler, device, args, tr_tracker, epoch)
        val_m = eval_epoch(model, loaders["val"], loss_fn,
                           device, args, val_tracker)

        elapsed = time.time() - t_ep
        lr_now  = optimizer.param_groups[-1]["lr"]
        log.info(
            f"  E{epoch:03d}/{args.epochs} ({elapsed:.0f}s) | "
            f"Tr: loss={tr_m.get('loss',0):.4f} acc={tr_m.get('accuracy',0):.4f} "
            f"f1={tr_m.get('f1_macro',0):.4f} | "
            f"Val: loss={val_m.get('loss',0):.4f} acc={val_m.get('accuracy',0):.4f} "
            f"f1={val_m.get('f1_macro',0):.4f} auroc={val_m.get('auroc_macro',0):.4f} | "
            f"lr={lr_now:.1e} ES={early_stop.counter}/{args.patience}"
        )

        row = {"epoch": epoch,
               **{f"train_{k}": v for k, v in tr_m.items() if isinstance(v, float)},
               **{f"val_{k}":   v for k, v in val_m.items() if isinstance(v, float)}}
        if csv_w is None:
            csv_w = csv.DictWriter(csv_f, fieldnames=list(row.keys()), extrasaction="ignore")
            csv_w.writeheader()
        csv_w.writerow(row)
        csv_f.flush()

        if tb_writer:
            for k, v in tr_m.items():
                if isinstance(v, float): tb_writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_m.items():
                if isinstance(v, float): tb_writer.add_scalar(f"val/{k}", v, epoch)
            tb_writer.add_scalar("lr", lr_now, epoch)

        monitor    = val_m.get("balanced_accuracy", 0.0)
        should_stop = early_stop(monitor, model, epoch)

        if early_stop.best_epoch == epoch:
            ckpt_dir = cfg.CKPT_DIR / exp_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch":         epoch,
                "model":         early_stop.best_state,
                "metrics":       val_m,
                "ablation":      ablation_name,
                "fold":          fold,
                "best_val_bacc": early_stop.best_score,
            }, ckpt_dir / "best_student.pt")
            log.info(f"  Saved checkpoint (epoch={epoch}, val_bal_acc={early_stop.best_score:.4f})")

        if should_stop:
            log.info(
                f"  Early stopping at epoch {epoch} "
                f"(best: epoch={early_stop.best_epoch}, val_bal_acc={early_stop.best_score:.4f})"
            )
            break

    csv_f.close()
    if tb_writer: tb_writer.close()

    if early_stop.best_state is not None:
        model.load_state_dict(early_stop.best_state)

    test_m = eval_epoch(model, loaders["test"], loss_fn,
                        device, args, te_tracker, "test")
    te_tracker.print_report("test")

    elapsed_total = time.time() - t_start
    log.info(
        f"\n  {exp_name} TEST: "
        f"acc={test_m.get('accuracy',0):.4f} "
        f"bal_acc={test_m.get('balanced_accuracy',0):.4f} "
        f"f1={test_m.get('f1_macro',0):.4f} "
        f"auroc={test_m.get('auroc_macro',0):.4f} "
        f"kappa={test_m.get('kappa',0):.4f} "
        f"({elapsed_total/60:.1f} min)"
    )
    return {"ablation": ablation_name, "fold": fold,
            "best_val_bacc": early_stop.best_score, **test_m}


# ============================================================================
# Ablation study
# ============================================================================

def run_ablations(args) -> pd.DataFrame:
    ablations = list(cfg.ABLATION_MODES.keys()) if args.ablation == "all" \
                else [args.ablation]
    all_results = []

    for abl_name in ablations:
        abl_cfg = cfg.ABLATION_MODES[abl_name]

        if args.no_clinical and abl_cfg["mode"] == "clinical_only":
            log.warning(f"Skipping '{abl_name}': clinical_only mode is incompatible with --no-clinical.")
            continue

        set_seed(args.seed)

        if args.cv:
            fold_range = [args.fold] if args.fold is not None else range(args.n_folds)
            fold_results = []
            for fold in fold_range:
                set_seed(args.seed + fold)
                res = train_experiment(args, abl_name, abl_cfg["mode"], abl_cfg["use_kd"], fold=fold)
                fold_results.append(res)
                all_results.append(res)

            metrics = ["accuracy", "balanced_accuracy", "f1_macro", "auroc_macro",
                       "kappa", "sensitivity_AD", "specificity_AD"]
            log.info(f"\n  {abl_name} — CV summary:")
            for m in metrics:
                vals = [r[m] for r in fold_results if m in r]
                if vals:
                    log.info(f"    {m:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        else:
            res = train_experiment(args, abl_name, abl_cfg["mode"], abl_cfg["use_kd"])
            all_results.append(res)

    df = pd.DataFrame(all_results)

    table_cols = ["ablation", "accuracy", "balanced_accuracy", "f1_macro",
                  "auroc_macro", "kappa", "sensitivity_AD", "specificity_AD"]
    table_cols = [c for c in table_cols if c in df.columns]
    if len(df) > 1:
        table = df.groupby("ablation")[[c for c in table_cols if c != "ablation"]].agg(
            ["mean", "std"]
        ).round(4)
        log.info("\n" + "=" * 70)
        log.info("ABLATION TABLE")
        log.info("=" * 70)
        log.info(table.to_string())
        table.to_csv(cfg.RESULTS_DIR / "ablation_summary.csv")

    df.to_csv(cfg.RESULTS_DIR / "all_results.csv", index=False)
    with open(cfg.RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, default=str)

    log.info(f"\nResults saved in: {cfg.RESULTS_DIR}")
    return df


# ============================================================================
# Dry run
# ============================================================================

def dry_run(args):
    log.info("DRY RUN — verifying architecture with synthetic data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.student import ModalityFusionTransformer
    from distillation.kd_loss import build_kd_loss

    B = args.batch_size
    D = args.fusion_dim
    fusion = ModalityFusionTransformer(d_model=D, nhead=4, num_layers=2).to(device)
    tokens = torch.randn(B, 3, D, device=device)
    fused  = fusion(tokens)
    log.info(f"  Fusion Transformer: {tuple(tokens.shape)} -> {tuple(fused.shape)}")
    assert fused.shape == (B, D)

    loss_fn = build_kd_loss(use_distillation=True, temperature=args.temperature)
    lh = torch.randn(B, 2, device=device, requires_grad=True)
    ls = torch.randn(B, 2, device=device, requires_grad=True)
    y  = torch.randint(0, 2, (B,), device=device)
    sl = torch.softmax(torch.randn(B, 3, device=device), dim=-1)

    out = loss_fn(lh, ls, y, sl)
    log.info(
        f"  KD loss: total={out['loss'].item():.4f} "
        f"hard={out['loss_hard'].item():.4f} "
        f"distill={out['loss_distill'].item():.4f}"
    )
    out["loss"].backward()
    log.info("  Backward OK ✓")
    log.info(f"\n  Binary collapse check (renormalised, MCI discarded):")
    log.info(f"    P(CN_bin) = P(CN)/(P(CN)+P(AD))  |  P(AD_bin) = P(AD)/(P(CN)+P(AD))")
    denom = (sl[0,0] + sl[0,2]).clamp(min=1e-8)
    log.info(f"    soft_labels_bin[0] = {float(sl[0,0]/denom):.3f} | {float(sl[0,2]/denom):.3f}")


# ============================================================================
# Main
# ============================================================================

def _fmt_temp(T: float) -> str:
    return str(int(T)) if T == int(T) else str(T).replace(".", "_")


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.soft_labels_csv is None:
        args.soft_labels_csv = cfg.SOFT_LABELS_DIR / f"adresso_soft_labels_T{_fmt_temp(args.temperature)}.csv"

    log.info("\n" + "=" * 70)
    log.info("ADReSSo Student — Knowledge Distillation Training")
    log.info("=" * 70)
    log.info(f"  Ablation:    {args.ablation}")
    log.info(f"  CV:          {args.cv}")
    log.info(f"  No clinical: {args.no_clinical}")
    log.info(f"  Temperature: {args.temperature}")
    log.info(f"  alpha={args.alpha}  beta={args.beta}")
    log.info(f"  Soft labels: {args.soft_labels_csv}")

    if args.dry_run:
        dry_run(args)
        return

    run_ablations(args)

    log.info("\n" + "=" * 70)
    log.info("Training completed.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
