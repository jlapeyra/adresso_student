"""
distillation/kd_loss.py
Knowledge Distillation losses for the ADReSSo Student.

Supported KD modes:
  response: hard CE + KL-divergence against Teacher probabilities.
  feature:  hard CE + alignment between Student and Teacher embeddings.
  both:     hard CE + response KD + feature alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Dual/combined KD loss.

    Feature KD projects the Student embedding and the frozen Teacher embedding
    into a shared space, then aligns them with cosine distance or L2/MSE.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        temperature: float = 3.0,
        label_smoothing: float = 0.05,
        method: str = "response",
        feature_weight: float = 0.3,
        student_dim: int = 256,
        teacher_dim: int = 256,
        projection_dim: int = 128,
        feature_loss: str = "cosine",
    ):
        super().__init__()
        if method not in {"response", "feature", "both"}:
            raise ValueError("method must be one of: response, feature, both")
        if feature_loss not in {"cosine", "l2"}:
            raise ValueError("feature_loss must be one of: cosine, l2")
        assert alpha >= 0 and beta >= 0 and feature_weight >= 0

        self.alpha = alpha
        self.beta = beta
        self.T = temperature
        self.method = method
        self.feature_weight = feature_weight
        self.feature_loss = feature_loss
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.student_proj = nn.Linear(student_dim, projection_dim)
        self.teacher_proj = nn.Linear(teacher_dim, projection_dim)

    def _response_loss(
        self,
        logits_soft: torch.Tensor,
        soft_labels: torch.Tensor,
    ) -> torch.Tensor:
        # Teacher labels may be 3-class [CN, MCI, AD] or 2-class [CN, AD]
        # represented as [CN, 0, AD]. We renormalise over CN/AD for binary KD.
        p_cn = soft_labels[:, 0]
        p_ad = soft_labels[:, 2]
        denom = (p_cn + p_ad).clamp(min=1e-8)
        soft_labels_bin = torch.stack([p_cn / denom, p_ad / denom], dim=1)
        soft_labels_bin = soft_labels_bin.to(logits_soft.dtype)

        log_student = F.log_softmax(logits_soft / self.T, dim=-1)
        return F.kl_div(
            log_student,
            soft_labels_bin,
            reduction="batchmean",
            log_target=False,
        ) * (self.T ** 2)

    def _feature_loss(
        self,
        student_embedding: torch.Tensor | None,
        teacher_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        if student_embedding is None or teacher_embedding is None or teacher_embedding.numel() == 0:
            raise ValueError("Feature-based KD requires student_embedding and teacher_embedding")

        teacher_embedding = teacher_embedding.to(
            device=student_embedding.device,
            dtype=student_embedding.dtype,
        )
        student_z = self.student_proj(student_embedding)
        teacher_z = self.teacher_proj(teacher_embedding)

        if self.feature_loss == "cosine":
            student_z = F.normalize(student_z, dim=-1)
            teacher_z = F.normalize(teacher_z, dim=-1)
            return (1.0 - F.cosine_similarity(student_z, teacher_z, dim=-1)).mean()
        return F.mse_loss(student_z, teacher_z)

    def forward(
        self,
        logits_hard: torch.Tensor,
        logits_soft: torch.Tensor,
        labels_hard: torch.Tensor,
        soft_labels: torch.Tensor,
        student_embedding: torch.Tensor | None = None,
        teacher_embedding: torch.Tensor | None = None,
    ) -> dict:
        L_hard = self.ce(logits_hard, labels_hard)

        L_response = torch.zeros((), device=logits_hard.device)
        if self.method in {"response", "both"}:
            L_response = self._response_loss(logits_soft, soft_labels)

        L_feature = torch.zeros((), device=logits_hard.device)
        if self.method in {"feature", "both"}:
            L_feature = self._feature_loss(student_embedding, teacher_embedding)

        L_total = self.alpha * L_hard + self.beta * L_response + self.feature_weight * L_feature

        return {
            "loss": L_total,
            "loss_hard": L_hard.detach(),
            "loss_distill": L_response.detach(),
            "loss_feature": L_feature.detach(),
        }


class HardOnlyLoss(nn.Module):
    """Standard CrossEntropy without distillation (for ablation baselines)."""

    def __init__(self, label_smoothing: float = 0.05):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits_hard, logits_soft, labels_hard, soft_labels, **kwargs):
        L = self.ce(logits_hard, labels_hard)
        return {
            "loss": L,
            "loss_hard": L.detach(),
            "loss_distill": torch.zeros(1, device=L.device),
            "loss_feature": torch.zeros(1, device=L.device),
        }


def build_kd_loss(
    use_distillation: bool = True,
    alpha: float = 0.6,
    beta: float = 0.4,
    temperature: float = 3.0,
    label_smoothing: float = 0.05,
    method: str = "response",
    feature_weight: float = 0.3,
    student_dim: int = 256,
    teacher_dim: int = 256,
    projection_dim: int = 128,
    feature_loss: str = "cosine",
) -> nn.Module:
    if use_distillation:
        return KnowledgeDistillationLoss(
            alpha=alpha,
            beta=beta,
            temperature=temperature,
            label_smoothing=label_smoothing,
            method=method,
            feature_weight=feature_weight,
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            projection_dim=projection_dim,
            feature_loss=feature_loss,
        )
    return HardOnlyLoss(label_smoothing)
