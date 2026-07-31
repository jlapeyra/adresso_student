"""
training/metrics.py
Metric tracking and early stopping for the Student training loop.

Original: tfm_alzheimer/training/metrics.py (copied verbatim)
"""

from typing import Dict, List, Optional
import copy
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)


# ============================================================================
# MetricTracker
# ============================================================================

class MetricTracker:
    """
    Accumulates logits and labels over an epoch, then computes all metrics.

    Usage:
        tracker = MetricTracker(num_classes=2, class_names=["HC", "AD"])
        for batch in loader:
            tracker.update(logits, labels, loss=loss_val)
        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(
        self,
        num_classes:  int = 2,
        class_names:  Optional[List[str]] = None,
        device:       str = "cpu",
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        self._all_probs  = []
        self._all_preds  = []
        self._all_labels = []
        self._losses     = []
        self._n_samples  = 0

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss:   Optional[float] = None,
    ) -> None:
        with torch.no_grad():
            probs = torch.softmax(logits.detach().float(), dim=-1)
            preds = probs.argmax(dim=-1)
        self._all_probs.append(probs.cpu().numpy())
        self._all_preds.append(preds.cpu().numpy())
        self._all_labels.append(labels.cpu().numpy())
        self._n_samples += labels.shape[0]
        if loss is not None:
            self._losses.append(float(loss))

    def compute(self) -> Dict[str, float]:
        if self._n_samples == 0:
            return {"error": "no_samples"}

        probs  = np.vstack(self._all_probs)
        preds  = np.concatenate(self._all_preds)
        labels = np.concatenate(self._all_labels)
        m      = {}

        if self._losses:
            m["loss"] = float(np.mean(self._losses))

        m["accuracy"]          = float(accuracy_score(labels, preds))
        m["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))
        m["f1_macro"]          = float(f1_score(labels, preds, average="macro",    zero_division=0))
        m["f1_weighted"]       = float(f1_score(labels, preds, average="weighted", zero_division=0))

        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        for i, name in enumerate(self.class_names):
            if i < len(per_class_f1):
                m[f"f1_{name.lower()}"] = float(per_class_f1[i])

        try:
            if len(np.unique(labels)) >= 2:
                if self.num_classes == 2:
                    m["auroc"]       = float(roc_auc_score(labels, probs[:, 1]))
                    m["auroc_macro"] = m["auroc"]
                else:
                    m["auroc_macro"] = float(
                        roc_auc_score(labels, probs, multi_class="ovr", average="macro")
                    )
        except Exception:
            pass

        try:
            m["kappa"] = float(cohen_kappa_score(labels, preds))
        except Exception:
            pass

        # Per-class sensitivity / specificity (binary)
        if self.num_classes == 2 and len(np.unique(labels)) == 2:
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                m["sensitivity_AD"] = float(tp / (tp + fn + 1e-8))
                m["specificity_AD"] = float(tn / (tn + fp + 1e-8))

        return m

    def print_report(self, split: str = "test") -> None:
        preds  = np.concatenate(self._all_preds)
        labels = np.concatenate(self._all_labels)
        import logging
        logging.getLogger(__name__).info(
            f"\n[{split}] classification report:\n"
            + classification_report(labels, preds, target_names=self.class_names, zero_division=0)
        )


# ============================================================================
# EarlyStopping
# ============================================================================

class EarlyStopping:
    """
    Stops training when the monitored metric stops improving.

    Args:
        patience:     epochs without improvement before stopping
        min_delta:    minimum change to count as improvement
        mode:         "min" (loss) or "max" (F1, AUROC)
        restore_best: if True, restores best weights when stopping
    """

    def __init__(
        self,
        patience:     int   = 20,
        min_delta:    float = 1e-4,
        mode:         str   = "min",
        restore_best: bool  = True,
    ):
        self.patience     = patience
        self.min_delta    = min_delta
        self.mode         = mode
        self.restore_best = restore_best

        self.best_score: Optional[float] = None
        self.best_state: Optional[dict]  = None
        self.best_epoch: int = 0
        self.counter:    int = 0

    def __call__(self, score: float, model: torch.nn.Module, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return False

        improved = (
            (self.mode == "max" and score >= self.best_score + self.min_delta) or
            (self.mode == "min" and score <= self.best_score - self.min_delta)
        )

        if improved:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter    = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best and self.best_state is not None:
                model.load_state_dict(self.best_state)
            return True

        return False
