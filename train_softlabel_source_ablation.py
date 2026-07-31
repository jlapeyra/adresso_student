"""
train_softlabel_source_ablation.py — Soft-label source ablation for the Student.

Compares 21 teacher soft-label sources from adni_teacher_ablation/
(9 buildup stages + 12 subtractive ablations), each evaluated with the full
multimodal_kd student under two clinical settings (with / without clinical
features), in 5-fold CV. Total: 21 sources x 2 clinical x 5 folds = 210 runs.

Examples:
  Full sweep:
    python train_softlabel_source_ablation.py

  Subset of sources / folds (useful for chunked SLURM array jobs):
    python train_softlabel_source_ablation.py --sources b0_random b7_full \
                                              --clinical-modes with \
                                              --folds 0 1

  Re-aggregate only (no training):
    python train_softlabel_source_ablation.py --aggregate-only
"""

import argparse
import datetime as _dt
import json
import logging
import sys
import time
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# Redirect outputs to a dedicated subdir before importing train.py — it caches
# nothing from cfg at import time but make_dirs runs against these paths.
ABLATION_SUBDIR = "softlabel_source"
cfg.RESULTS_DIR = cfg.OUTPUTS_DIR / ABLATION_SUBDIR / "results"
cfg.CKPT_DIR    = cfg.OUTPUTS_DIR / ABLATION_SUBDIR / "checkpoints"
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
(cfg.LOGS_DIR / "slurm").mkdir(parents=True, exist_ok=True)

from train import train_experiment, set_seed, parse_args as train_parse_args  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(cfg.LOGS_DIR / "softlabel_source_ablation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Soft-label source catalogue
# ----------------------------------------------------------------------------
KD_ABL_ROOT = cfg.KD_ROOT / "adni_teacher_ablation"

BUILDUP_SOURCES = [
    "b0_random", "b1_mri_link", "b2_xattn", "b3_priv",
    "b4_qcond", "b5_gated", "b6_dropout", "b7_full", "b8_no_mmse",
]
NORMAL_SOURCES = [
    "full", "link_only", "no_priv", "no_mri", "no_pretrain", "no_mmse",
    "no_query_cond", "no_gated_fusion", "mean_pool", "no_vision_head",
    "no_priv_dropout", "no_distill",
]
ALL_SOURCES = BUILDUP_SOURCES + NORMAL_SOURCES

CLINICAL_MODES = ["with", "without"]   # "with" = use MMSE/AGE/SEX, "without" = --no-clinical


def soft_labels_path(name: str) -> Path:
    """Resolve the soft-labels CSV path for a buildup or subtractive source."""
    if name in BUILDUP_SOURCES:
        return KD_ABL_ROOT / "buildup" / "outputs" / name / "soft_labels.csv"
    if name in NORMAL_SOURCES:
        return KD_ABL_ROOT / "outputs" / name / "soft_labels.csv"
    raise ValueError(f"Unknown soft-label source: {name}")


def source_family(name: str) -> str:
    return "buildup" if name in BUILDUP_SOURCES else "normal"


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Student ablation across teacher soft-label sources",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sources", nargs="+", default=None,
                   help="Subset of soft-label sources (default: all 21)")
    p.add_argument("--clinical-modes", nargs="+", default=CLINICAL_MODES,
                   choices=CLINICAL_MODES,
                   help='Clinical settings to run ("with", "without", or both)')
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--folds", nargs="+", type=int, default=None,
                   help="Specific folds to run (0-indexed, default: all)")

    # Training hyperparameters (pull defaults from train.py / cfg)
    p.add_argument("--epochs",         type=int,   default=cfg.EPOCHS)
    p.add_argument("--batch-size",     type=int,   default=cfg.BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=cfg.LR_NEW)
    p.add_argument("--lr-pretrained",  type=float, default=cfg.LR_PRETRAINED)
    p.add_argument("--weight-decay",   type=float, default=cfg.WEIGHT_DECAY)
    p.add_argument("--patience",       type=int,   default=cfg.PATIENCE)
    p.add_argument("--grad-accum",     type=int,   default=cfg.GRAD_ACCUM)
    p.add_argument("--temperature",    type=float, default=cfg.KD_TEMPERATURE)
    p.add_argument("--alpha",          type=float, default=cfg.KD_ALPHA)
    p.add_argument("--beta",           type=float, default=cfg.KD_BETA)
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--augment-train",  action="store_true")
    p.add_argument("--aug-factor",     type=int,   default=1)
    p.add_argument("--no-amp",         action="store_true")
    p.add_argument("--no-tb",          action="store_true")
    p.add_argument("--seed",           type=int,   default=cfg.RANDOM_SEED)

    # Run management
    p.add_argument("--force", action="store_true",
                   help="Re-run combinations that already have a result on disk")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip training; only re-aggregate existing fold results")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Args plumbing — build a train.py-compatible Namespace from our CLI options
# ----------------------------------------------------------------------------
def build_train_args(opts) -> argparse.Namespace:
    """
    Construct an argparse.Namespace identical to train.py's parse_args output,
    populated from our CLI plus train.py defaults. Per-iteration overrides
    (soft_labels_csv, no_clinical) are applied later via deepcopy.
    """
    saved = sys.argv
    sys.argv = ["train.py"]
    try:
        args = train_parse_args()
    finally:
        sys.argv = saved

    # Apply our hyperparameter overrides
    args.epochs           = opts.epochs
    args.batch_size       = opts.batch_size
    args.lr               = opts.lr
    args.lr_pretrained    = opts.lr_pretrained
    args.weight_decay     = opts.weight_decay
    args.patience         = opts.patience
    args.grad_accum       = opts.grad_accum
    args.temperature      = opts.temperature
    args.alpha            = opts.alpha
    args.beta             = opts.beta
    args.num_workers      = opts.num_workers
    args.augment_train    = opts.augment_train
    args.aug_factor       = opts.aug_factor
    args.no_amp           = opts.no_amp
    args.no_tb            = opts.no_tb
    args.seed             = opts.seed

    # CV settings — train_experiment is called per fold, so no_folds is informative only
    args.ablation         = "multimodal_kd"
    args.cv               = True
    args.n_folds          = opts.n_folds
    args.fold             = None
    args.dry_run          = False
    return args


# ----------------------------------------------------------------------------
# Per-combination runner
# ----------------------------------------------------------------------------
def fold_result_path(src: str, clin: str, fold: int) -> Path:
    return cfg.RESULTS_DIR / f"{src}_{clin}_fold{fold}.json"


def run_combination(opts, base_args: argparse.Namespace,
                    src: str, clin: str, fold: int) -> Optional[Dict]:
    """Train one (source, clinical, fold) combination. Skip if already done."""
    out_path = fold_result_path(src, clin, fold)
    if out_path.exists() and not opts.force:
        log.info(f"[skip] {src} | clin={clin} | fold={fold} (already done)")
        with open(out_path) as f:
            return json.load(f)

    args = deepcopy(base_args)
    args.soft_labels_csv = soft_labels_path(src)
    args.no_clinical     = (clin == "without")
    args.fold            = fold

    if not args.soft_labels_csv.exists():
        log.error(f"Missing soft labels file: {args.soft_labels_csv}")
        return None

    ablation_name = f"{src}_{clin}"
    set_seed(opts.seed + fold)

    t0 = time.time()
    res = train_experiment(
        args,
        ablation_name=ablation_name,
        mode="full",
        use_kd=True,
        fold=fold,
    )
    elapsed = (time.time() - t0) / 60

    # Augment with metadata for downstream aggregation
    res["softlabel_source"] = src
    res["family"]           = source_family(src)
    res["clinical"]         = clin
    res["fold"]             = fold
    res["elapsed_min"]      = round(elapsed, 2)

    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    log.info(f"[done] {src} | clin={clin} | fold={fold} ({elapsed:.1f} min)")
    return res


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
METRIC_COLS = [
    "accuracy", "balanced_accuracy", "f1_macro", "auroc_macro",
    "kappa", "sensitivity_AD", "specificity_AD",
]


def aggregate_results(out_csv: Path = None) -> pd.DataFrame:
    rows = []
    for src in ALL_SOURCES:
        for clin in CLINICAL_MODES:
            for fold in range(10):  # tolerate any fold count we might find
                p = fold_result_path(src, clin, fold)
                if p.exists():
                    with open(p) as f:
                        rows.append(json.load(f))
    if not rows:
        log.warning("No fold results found to aggregate.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(cfg.RESULTS_DIR / "all_fold_results.csv", index=False)

    # Mean ± std per (source, clinical)
    keep = [c for c in METRIC_COLS if c in df.columns]
    summary = (
        df.groupby(["softlabel_source", "family", "clinical"])[keep]
          .agg(["mean", "std"])
          .round(4)
    )
    summary.to_csv(cfg.RESULTS_DIR / "softlabel_source_summary.csv")

    # Wide table: rows = source, columns = (metric, clinical), values = mean
    means_only = (
        df.groupby(["softlabel_source", "family", "clinical"])[keep]
          .mean()
          .round(4)
          .reset_index()
    )
    wide = means_only.pivot(index=["family", "softlabel_source"],
                            columns="clinical", values=keep)
    wide.to_csv(cfg.RESULTS_DIR / "softlabel_source_wide.csv")

    log.info("\n" + "=" * 70)
    log.info("SOFT-LABEL SOURCE ABLATION — SUMMARY")
    log.info("=" * 70)
    log.info("\n" + summary.to_string())
    log.info(f"\nWrote: {cfg.RESULTS_DIR / 'softlabel_source_summary.csv'}")
    log.info(f"Wrote: {cfg.RESULTS_DIR / 'softlabel_source_wide.csv'}")
    log.info(f"Wrote: {cfg.RESULTS_DIR / 'all_fold_results.csv'}")

    write_results_md(df, keep)
    return summary


# ----------------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------------
REFERENCE_SOURCE = "link_only"   # current student baseline (link head soft labels)


def _mean_std(df: pd.DataFrame, src: str, clin: str, metric: str):
    sub = df[(df["softlabel_source"] == src) & (df["clinical"] == clin)]
    if len(sub) == 0 or metric not in sub.columns:
        return None, None
    vals = sub[metric].dropna()
    if len(vals) == 0:
        return None, None
    return float(vals.mean()), float(vals.std()) if len(vals) > 1 else 0.0


def _fmt_ms(m, s):
    if m is None:
        return "—"
    return f"{m:.3f} ± {s:.3f}" if s is not None else f"{m:.3f}"


def write_results_md(df: pd.DataFrame, metrics: List[str]) -> Path:
    """Write a Markdown report in the style of adni_teacher_ablation/results.md."""
    md_path = cfg.RESULTS_DIR / "results_softlabels.md"
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Available sources in this run (preserve catalogue order)
    present = [s for s in ALL_SOURCES if s in set(df["softlabel_source"].unique())]
    have_clin = sorted(set(df["clinical"].unique()))   # ["with", "without"]

    primary_metric = "balanced_accuracy" if "balanced_accuracy" in metrics else metrics[0]
    extra_metrics  = [m for m in ["accuracy", "f1_macro", "auroc_macro", "kappa",
                                  "sensitivity_AD", "specificity_AD"]
                      if m in metrics]

    lines: List[str] = []
    lines.append("# ADReSSo Student — Soft-Label Source Ablation Results\n")
    lines.append(
        f"_Generated {now} from "
        f"`outputs/{ABLATION_SUBDIR}/results/<source>_<clin>_fold<k>.json`._\n"
    )
    lines.append(
        "Each row corresponds to one teacher soft-label source "
        "(buildup stage or subtractive ablation from `adni_teacher_ablation/`). "
        "The student is the standard `multimodal_kd` model (audio + text + clinical "
        f"+ KD), trained from scratch in {df['fold'].nunique() if 'fold' in df else 5}-fold CV. "
        f"Reference: **`{REFERENCE_SOURCE}` with clinical** (the current production "
        "soft-label source used by the student).\n"
    )

    # ----- 1. Primary table: balanced_accuracy by clinical mode + Δ vs reference
    lines.append("## 1. Primary results (balanced accuracy)\n")
    lines.append(
        "Δ columns show change vs reference — negative means worse than baseline.\n"
    )

    ref_with, _ = _mean_std(df, REFERENCE_SOURCE, "with", primary_metric)

    header  = "| Source | Family |"
    divider = "|---|---|"
    for clin in have_clin:
        header  += f" {clin} clinical |"
        divider += "---|"
    if "with" in have_clin and ref_with is not None:
        header  += " Δ (with vs ref) |"
        divider += "---|"
    lines.append(header)
    lines.append(divider)

    for src in present:
        family = source_family(src)
        is_ref = (src == REFERENCE_SOURCE)
        src_tag = f"**`{src}`**" if is_ref else f"`{src}`"
        row = f"| {src_tag} | {family} |"
        for clin in have_clin:
            m, s = _mean_std(df, src, clin, primary_metric)
            row += f" {_fmt_ms(m, s)} |"
        if "with" in have_clin and ref_with is not None:
            m, _ = _mean_std(df, src, "with", primary_metric)
            if m is None:
                row += " — |"
            else:
                d = m - ref_with
                row += f" {d:+.3f} |"
        lines.append(row)
    lines.append("")

    # ----- 2. Full metrics tables: one per clinical mode
    for clin in have_clin:
        lines.append(f"## 2.{have_clin.index(clin) + 1} Full metrics — clinical = `{clin}`\n")
        header = "| Source | Family |" + "".join(f" {m} |" for m in [primary_metric, *extra_metrics])
        div    = "|---|---|" + "---|" * (1 + len(extra_metrics))
        lines.append(header)
        lines.append(div)
        for src in present:
            row = f"| `{src}` | {source_family(src)} |"
            for met in [primary_metric, *extra_metrics]:
                m, s = _mean_std(df, src, clin, met)
                row += f" {_fmt_ms(m, s)} |"
            lines.append(row)
        lines.append("")

    # ----- 3. Δ vs reference per metric (with clinical only)
    if "with" in have_clin:
        lines.append("## 3. Δ vs reference (`link_only` with clinical)\n")
        lines.append(
            "Negative Δ means worse than reference. Useful to spot soft-label sources "
            "that hurt or improve the student vs the production setup.\n"
        )
        header = "| Source | " + " | ".join(f"Δ {m}" for m in [primary_metric, *extra_metrics]) + " |"
        div    = "|---|" + "---|" * (1 + len(extra_metrics))
        lines.append(header)
        lines.append(div)
        for src in present:
            row = f"| `{src}` |"
            for met in [primary_metric, *extra_metrics]:
                m, _   = _mean_std(df, src, "with", met)
                rm, _  = _mean_std(df, REFERENCE_SOURCE, "with", met)
                if m is None or rm is None:
                    row += " — |"
                else:
                    row += f" {m - rm:+.3f} |"
            lines.append(row)
        lines.append("")

    # ----- 4. Effect of removing clinical features
    if {"with", "without"}.issubset(have_clin):
        lines.append("## 4. Effect of removing clinical features\n")
        lines.append(
            "Δ = `without` − `with` per source (negative = clinical features help).\n"
        )
        lines.append("| Source | with | without | Δ (without − with) |")
        lines.append("|---|---|---|---|")
        for src in present:
            mw, sw = _mean_std(df, src, "with", primary_metric)
            mn, sn = _mean_std(df, src, "without", primary_metric)
            d = None if (mw is None or mn is None) else mn - mw
            lines.append(
                f"| `{src}` | {_fmt_ms(mw, sw)} | {_fmt_ms(mn, sn)} | "
                f"{('—' if d is None else f'{d:+.3f}')} |"
            )
        lines.append("")

    # ----- 5. Key findings
    lines.append("## 5. Key findings\n")
    for clin in have_clin:
        rows = []
        for src in present:
            m, _ = _mean_std(df, src, clin, primary_metric)
            if m is not None:
                rows.append((src, m))
        rows.sort(key=lambda r: r[1], reverse=True)
        if not rows:
            continue
        lines.append(f"**Top-3 sources by {primary_metric} (clinical = `{clin}`):**")
        for src, m in rows[:3]:
            lines.append(f"- `{src}`: {m:.3f}")
        lines.append("")
        lines.append(f"**Bottom-3 sources (clinical = `{clin}`):**")
        for src, m in rows[-3:]:
            lines.append(f"- `{src}`: {m:.3f}")
        lines.append("")

    # ----- 6. Runtime
    if "elapsed_min" in df.columns:
        total_min = df["elapsed_min"].sum()
        per_run   = df["elapsed_min"].mean()
        lines.append("## 6. Runtime\n")
        lines.append(f"- Completed runs: {len(df)}")
        lines.append(f"- Mean per run:   {per_run:.1f} min")
        lines.append(f"- Total compute:  {total_min/60:.1f} h")
        lines.append("")

    lines.append("---\n")
    lines.append(
        "_Generated by `adresso_student/train_softlabel_source_ablation.py`._\n"
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Wrote: {md_path}")
    return md_path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    opts = parse_args()

    if opts.aggregate_only:
        aggregate_results()
        return

    sources = opts.sources if opts.sources else ALL_SOURCES
    unknown = [s for s in sources if s not in ALL_SOURCES]
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}. Valid: {ALL_SOURCES}")

    folds = opts.folds if opts.folds else list(range(opts.n_folds))

    log.info("\n" + "=" * 70)
    log.info("SOFT-LABEL SOURCE ABLATION")
    log.info("=" * 70)
    log.info(f"  Sources       : {sources}")
    log.info(f"  Clinical modes: {opts.clinical_modes}")
    log.info(f"  Folds         : {folds}")
    log.info(f"  Total runs    : {len(sources) * len(opts.clinical_modes) * len(folds)}")
    log.info(f"  Output dir    : {cfg.RESULTS_DIR}")

    base_args = build_train_args(opts)
    combos = list(product(sources, opts.clinical_modes, folds))

    for i, (src, clin, fold) in enumerate(combos, 1):
        log.info(f"\n[{i}/{len(combos)}] source={src} clinical={clin} fold={fold}")
        try:
            run_combination(opts, base_args, src, clin, fold)
        except Exception as e:
            log.exception(f"Combination failed: src={src} clin={clin} fold={fold}: {e}")

    aggregate_results()
    log.info("\nDone.")


if __name__ == "__main__":
    main()
