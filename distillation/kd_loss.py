"""
distillation/kd_loss.py
Knowledge Distillation loss for the ADReSSo Student.

L_total = alpha * L_hard + beta * L_distill * T^2

L_hard    = CrossEntropy(logits_hard, y_real)          [alpha=0.6]
L_distill = KL-div(log_softmax(logits_soft/T), P_bin)  [beta=0.4]

Teacher soft labels are 3-class [P(CN), P(MCI), P(AD)].
They are collapsed to binary before the KL-div via renormalisation:

    P(CN_bin) = P(CN) / (P(CN) + P(AD))
    P(AD_bin) = P(AD) / (P(CN) + P(AD))   [MCI mass discarded]

Rationale: summing P(MCI) into P(AD) inflates P(AD_bin) to ~0.43 for true-CN
subjects (the LUPI head assigns high P(MCI) under clinical uncertainty), which
reduces CN/AD separability by ~31%. Renormalising over CN and AD only preserves
the relative teacher confidence between the two target classes.

Reference: Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
Original: tfm_alzheimer/distillation/kd_loss.py  (collapse direction corrected)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Dual loss: hard CrossEntropy + KL-divergence distillation.

    Args:
        alpha:           weight for L_hard          [default: 0.6]
        beta:            weight for L_distill        [default: 0.4]
        temperature:     T applied to Student logits before KL
        label_smoothing: smoothing for CrossEntropy
    """

    def __init__(
        self,
        alpha:           float = 0.6,
        beta:            float = 0.4,
        temperature:     float = 3.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        assert alpha >= 0 and beta >= 0
        self.alpha = alpha
        self.beta  = beta
        self.T     = temperature
        self.ce    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits_hard: torch.Tensor,   # (B, 2)
        logits_soft: torch.Tensor,   # (B, 2)
        labels_hard: torch.Tensor,   # (B,)  int64
        soft_labels: torch.Tensor,   # (B, 3) float  [P(CN), P(MCI), P(AD)]
    ) -> dict:
        # Hard loss
        L_hard = self.ce(logits_hard, labels_hard)

        # Collapse 3-class Teacher -> binary via renormalisation over CN and AD only.
        # P(MCI) is discarded rather than added to AD: summing it inflates P(AD_bin)
        # to ~0.43 for true-CN subjects (where prob_MCI is high due to uncertainty),
        # reducing CN/AD separability by ~31% vs. the normalised form.
        p_cn = soft_labels[:, 0]
        p_ad = soft_labels[:, 2]
        denom = (p_cn + p_ad).clamp(min=1e-8)
        soft_labels_bin = torch.stack([
            p_cn / denom,   # P(CN_bin) — renormalised, MCI mass discarded
            p_ad / denom,   # P(AD_bin) — renormalised, MCI mass discarded
        ], dim=1).to(logits_soft.dtype)               # (B, 2)

        log_student = F.log_softmax(logits_soft / self.T, dim=-1)
        L_distill   = F.kl_div(
            log_student,
            soft_labels_bin,
            reduction="batchmean",
            log_target=False,
        )
        L_distill_scaled = L_distill * (self.T ** 2)

        L_total = self.alpha * L_hard + self.beta * L_distill_scaled

        return {
            "loss":         L_total,
            "loss_hard":    L_hard.detach(),
            "loss_distill": L_distill_scaled.detach(),
        }


class HardOnlyLoss(nn.Module):
    """Standard CrossEntropy without distillation (for ablation baselines)."""

    def __init__(self, label_smoothing: float = 0.05):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits_hard, logits_soft, labels_hard, soft_labels):
        L = self.ce(logits_hard, labels_hard)
        return {
            "loss":         L,
            "loss_hard":    L.detach(),
            "loss_distill": torch.zeros(1, device=L.device),
        }


def build_kd_loss(
    use_distillation: bool  = True,
    alpha:            float = 0.6,
    beta:             float = 0.4,
    temperature:      float = 3.0,
    label_smoothing:  float = 0.05,
) -> nn.Module:
    if use_distillation:
        return KnowledgeDistillationLoss(alpha, beta, temperature, label_smoothing)
    return HardOnlyLoss(label_smoothing)
