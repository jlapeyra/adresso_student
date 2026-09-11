"""
models/student.py
Multimodal Student model for ADReSSo knowledge distillation.

Architecture (adapted from tfm_alzheimer/models/student.py):
  - Audio branch:    Wav2Vec2-base, last 4 layers trainable, mean-pool -> Linear(768->256)
  - Text branch:     RoBERTa-base, last 4 layers trainable, CLS -> Linear(768->256)
  - Clinical branch: MLP(3->64->128->256)
  - Fusion:          TransformerEncoder (2L, 4H, d=256, Pre-LN), masked mean-pool
  - Dual heads:      hard (256->128->2) + soft (256->128->2)

Soft head outputs 2 classes (binary) because Teacher soft labels are collapsed
to binary space before the KL-div loss via renormalisation (MCI discarded):
  P(CN_bin) = P(CN)/(P(CN)+P(AD)),  P(AD_bin) = P(AD)/(P(CN)+P(AD))

Original: tfm_alzheimer/models/student.py
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer

def _find_transformer_layers(model: nn.Module, num_layers: int) -> nn.ModuleList:
    """
    Find the Transformer layer stack in a Hugging Face model.

    Supports common architectures such as:
        - BERT / RoBERTa: encoder.layer
        - DeBERTa: encoder.layer
        - Wav2Vec2: encoder.layers
        - HuBERT: encoder.layers
        - WavLM: encoder.layers

    Returns:
        The ModuleList containing the Transformer layers.
    """
    candidates = []

    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == num_layers:
            candidates.append((name, module))

    if not candidates:
        raise ValueError(
            f"Could not find a Transformer layer stack with "
            f"{num_layers} layers in {model.__class__.__name__}."
        )

    # Prefer modules whose name suggests that they are the encoder layers.
    preferred = [
        (name, module)
        for name, module in candidates
        if name.endswith("encoder.layer")
        or name.endswith("encoder.layers")
    ]

    if preferred:
        return preferred[0][1]

    # Otherwise use the first matching ModuleList.
    return candidates[0][1]


def _freeze_model_except_last_layers(
    model: nn.Module,
    trainable_layers: int,
) -> None:
    """
    Freeze the entire model and unfreeze only the last `trainable_layers`
    Transformer layers.

    Everything outside those layers remains frozen.
    """
    if trainable_layers < 0:
        raise ValueError("trainable_layers must be >= 0")

    num_layers = getattr(model.config, "num_hidden_layers", None)

    if num_layers is None:
        raise ValueError(
            f"{model.__class__.__name__} does not expose "
            "`config.num_hidden_layers`."
        )

    if trainable_layers > num_layers:
        raise ValueError(
            f"trainable_layers={trainable_layers}, but the model only has "
            f"{num_layers} Transformer layers."
        )

    # Freeze everything first.
    for param in model.parameters():
        param.requires_grad_(False)

    # Find the Transformer layers.
    layers = _find_transformer_layers(model, num_layers)

    # Unfreeze only the last N layers.
    if trainable_layers > 0:
        for layer in layers[-trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad_(True)
# ============================================================================
# Audio encoder (Wav2Vec2)
# ============================================================================

class AudioEncoder(nn.Module):
    """
    Hugging Face audio encoder with partial fine-tuning.

    The entire pretrained model is frozen except for the last
    `trainable_layers` Transformer layers.

    A trainable projection maps the model's hidden representation
    to `output_dim`.

    Compatible with common Hugging Face speech encoders such as:
        - facebook/wav2vec2-base
        - facebook/hubert-base-ls960
        - microsoft/wavlm-base-plus
        - facebook/data2vec-audio-base
    """

    def __init__(
        self,
        model_name:    str   = "facebook/wav2vec2-base",
        output_dim:    int   = 256,
        trainable_layers: int = 4,
        dropout_p:     float = 0.1,
        spec_augment:  bool  = True,
    ):
        super().__init__()

        self.output_dim   = output_dim
        self.spec_augment = spec_augment

        self.config = AutoConfig.from_pretrained(model_name)
        self.audio_encoder = AutoModel.from_pretrained(model_name)

        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers

        _freeze_model_except_last_layers(
            self.audio_encoder,
            trainable_layers=trainable_layers,
        )

        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

        self._init_proj()

    def _init_proj(self):
        for module in self.proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def _spec_augment(
        self,
        hidden: torch.Tensor,
        mask_prob: float = 0.1,
    ) -> torch.Tensor:
        if not self.training:
            return hidden
        B, T, D = hidden.shape
        mask_len = max(1, int(T * mask_prob))
        mask = torch.ones(B, T, 1, dtype=hidden.dtype, device=hidden.device)
        for b in range(B):
            if torch.rand(1).item() > 0.5:
                max_start = max(1, T - mask_len + 1)

                start = torch.randint(
                    0,
                    max_start,
                    (1,),
                    device=hidden.device,
                ).item()

                mask[b, start:start + mask_len] = 0.0

        return hidden * mask

    def forward(
        self,
        input_values:   torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        outputs = self.audio_encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        hidden = outputs.last_hidden_state
        # (B, T_feat, hidden_size)

        if self.spec_augment:
            hidden = self._spec_augment(hidden)

        if attention_mask is not None:
            feat_len = hidden.shape[1]

            mask_float = attention_mask.float()

            mask_interp = F.interpolate(
                mask_float.unsqueeze(1),
                size=feat_len,
                mode="nearest",
            ).squeeze(1).unsqueeze(-1)
            # (B, T_feat, 1)

            pooled = (
                (hidden * mask_interp).sum(dim=1)
                / (mask_interp.sum(dim=1) + 1e-8)
            )
        else:
            pooled = hidden.mean(dim=1)

        return self.proj(pooled)


# ============================================================================
# Text encoder (RoBERTa)
# ============================================================================
class TextEncoder(nn.Module):
    """
    Hugging Face text encoder with partial fine-tuning.

    The entire pretrained model is frozen except for the last
    `trainable_layers` Transformer layers.

    The first token ([CLS] for BERT/RoBERTa-like models) is used
    as the sentence representation.

    Compatible with models such as:
        - roberta-base
        - roberta-large
        - microsoft/deberta-v3-base
        - microsoft/deberta-v3-large
        - bert-base-uncased
        - FacebookAI/xlm-roberta-base
    """

    def __init__(
        self,
        model_name:    str   = "roberta-base",
        output_dim:    int   = 256,
        trainable_layers: int = 4,
        dropout_p:     float = 0.1,
        max_length:    int   = 512,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.max_length = max_length

        self.config = AutoConfig.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers

        _freeze_model_except_last_layers(
            self.text_model,
            trainable_layers=trainable_layers,
        )

        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

        self._init_proj()

    def _init_proj(self):
        for module in self.proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def tokenize(
        self,
        texts: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            key: value.to(device)
            for key, value in encoding.items()
        }

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.proj(cls_token)
# ============================================================================
# Clinical encoder (MLP)
# ============================================================================

class ClinicalEncoder(nn.Module):
    """
    MLP for clinical features aligned to ADNI space.
    Default input_dim=3: [MMSE_norm, AGE_norm, PTGENDER_enc].
    """

    def __init__(
        self,
        input_dim:  int   = 3,
        output_dim: int   = 256,
        dropout_p:  float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# Modality gating (learned per-sample modality weights)
# ============================================================================

class ModalityGating(nn.Module):
    """
    Learns per-sample, per-modality scalar gates in [0,1].

    Aggregates all modality tokens into a global context, then predicts
    how much each modality should contribute before fusion. This lets the
    model down-weight noisy modalities (e.g. audio) on a per-sample basis.

    Input:  (B, M, D)  — M modality tokens of dimension D
    Output: (B, M, D)  — gated tokens (element-wise scaled)
    """

    def __init__(self, fusion_dim: int, n_modalities: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, n_modalities),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, M, D)
        context = tokens.mean(dim=1)               # (B, D)  — mean over modalities
        gates   = self.gate(context).unsqueeze(-1)  # (B, M, 1)
        return tokens * gates


# ============================================================================
# Modality fusion (Transformer Encoder over 3 modality tokens)
# ============================================================================

class ModalityFusionTransformer(nn.Module):
    """
    Fuses audio, text and clinical via a Transformer Encoder (Pre-LN).

    Input:  (B, 3, d_model) — three modality tokens
    Output: (B, d_model)    — masked mean-pool over present tokens

    No positional encoding: modalities have no inherent order.
    """

    def __init__(
        self,
        d_model:    int   = 256,
        nhead:      int   = 4,
        num_layers: int   = 2,
        dim_ffn:    int   = 512,
        dropout_p:  float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ffn,
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable on small datasets
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        for m in self.transformer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        tokens:               torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            mask_float = (~src_key_padding_mask).float().unsqueeze(-1)   # (B, 3, 1)
            return (x * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)
        return x.mean(dim=1)


# ============================================================================
# Full Student model
# ============================================================================

class StudentModel(nn.Module):
    """
    Multimodal Student: audio + text + clinical + Fusion Transformer.

    Dual-head design:
      hard_classifier: (256->128->2) trained with CrossEntropy on real HC/AD labels
      soft_classifier: (256->128->2) trained with KL-div against Teacher soft labels
                       Teacher's 3-class probs are renormalised to binary before KL:
                         P(CN_bin) = P(CN)/(P(CN)+P(AD)),  P(AD_bin) = P(AD)/(P(CN)+P(AD))

    Separate pre-classifier paths isolate gradients from hard vs KD losses.
    """

    def __init__(
        self,
        num_classes_hard: int   = 2,
        num_classes_soft: int   = 2,
        fusion_dim:       int   = 256,
        audio_encoder:   str   = "facebook/wav2vec2-base",
        text_encoder:    str   = "roberta-base",
        n_clinical:       Optional[int] = None,
        freeze_audio_n:   int   = 8,
        freeze_text_n:    int   = 8,
        audio_dropout:    float = 0.1,
        text_dropout:     float = 0.1,
        clinical_dropout: float = 0.2,
        fusion_dropout:   float = 0.1,
        fusion_layers:    int   = 2,
        fusion_heads:     int   = 4,
    ):
        super().__init__()
        self.num_classes_hard = num_classes_hard
        self.num_classes_soft = num_classes_soft
        self.fusion_dim       = fusion_dim

        if n_clinical is None:
            import sys as _sys, pathlib as _pl
            _sys.path.insert(0, str(_pl.Path(__file__).parent.parent))
            import config as _cfg
            n_clinical = len(_cfg.CLINICAL_FEATURE_COLS)

        self.audio_encoder    = AudioEncoder(audio_encoder, fusion_dim, freeze_audio_n, audio_dropout)
        self.text_encoder     = TextEncoder(text_encoder, fusion_dim, freeze_text_n, text_dropout)
        self.clinical_encoder = ClinicalEncoder(n_clinical, fusion_dim, clinical_dropout)

        self.modality_gate = ModalityGating(fusion_dim, n_modalities=3)
        self.fusion = ModalityFusionTransformer(
            d_model=fusion_dim, nhead=fusion_heads,
            num_layers=fusion_layers, dim_ffn=fusion_dim * 2,
            dropout_p=fusion_dropout,
        )

        # Learned missing-modality embeddings
        self.missing_audio_emb = nn.Parameter(torch.zeros(1, fusion_dim))
        self.missing_text_emb  = nn.Parameter(torch.zeros(1, fusion_dim))
        self.missing_clin_emb  = nn.Parameter(torch.zeros(1, fusion_dim))
        nn.init.normal_(self.missing_audio_emb, std=0.02)
        nn.init.normal_(self.missing_text_emb,  std=0.02)
        nn.init.normal_(self.missing_clin_emb,  std=0.02)

        # Separate pre-classifier paths
        self.pre_classifier_hard = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(fusion_dropout)
        )
        self.pre_classifier_soft = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(fusion_dropout)
        )

        self.hard_classifier = nn.Linear(128, num_classes_hard)
        self.soft_classifier = nn.Linear(128, num_classes_soft)

        nn.init.xavier_uniform_(self.hard_classifier.weight, gain=0.1)
        nn.init.zeros_(self.hard_classifier.bias)
        nn.init.xavier_uniform_(self.soft_classifier.weight, gain=0.1)
        nn.init.zeros_(self.soft_classifier.bias)

    def forward(
        self,
        audio_values:        Optional[torch.Tensor] = None,
        audio_mask:          Optional[torch.Tensor] = None,
        text_input_ids:      Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        clinical:            Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _present_list = [t for t in [audio_values, text_input_ids, clinical] if t is not None]
        if not _present_list:
            raise ValueError(
                "StudentModel.forward: all modalities are None. "
                "At least one of audio_values, text_input_ids, or clinical must be provided."
            )
        _present = _present_list[0]
        B   = _present.shape[0]
        dev = _present.device

        tokens: List[torch.Tensor] = []
        pad_mask: List[torch.Tensor] = []

        if audio_values is not None:
            tokens.append(self.audio_encoder(audio_values, audio_mask))
            pad_mask.append(torch.zeros(B, dtype=torch.bool, device=dev))
        else:
            tokens.append(self.missing_audio_emb.expand(B, -1))
            pad_mask.append(torch.ones(B, dtype=torch.bool, device=dev))

        if text_input_ids is not None:
            tokens.append(self.text_encoder(text_input_ids, text_attention_mask))
            pad_mask.append(torch.zeros(B, dtype=torch.bool, device=dev))
        else:
            tokens.append(self.missing_text_emb.expand(B, -1))
            pad_mask.append(torch.ones(B, dtype=torch.bool, device=dev))

        if clinical is not None:
            tokens.append(self.clinical_encoder(clinical))
            pad_mask.append(torch.zeros(B, dtype=torch.bool, device=dev))
        else:
            tokens.append(self.missing_clin_emb.expand(B, -1))
            pad_mask.append(torch.ones(B, dtype=torch.bool, device=dev))

        token_stack = torch.stack(tokens, dim=1)          # (B, 3, 256)
        pad_mask_t  = torch.stack(pad_mask, dim=1)        # (B, 3)

        token_stack = self.modality_gate(token_stack)
        fused = self.fusion(token_stack, src_key_padding_mask=pad_mask_t)

        shared_hard = self.pre_classifier_hard(fused)
        shared_soft = self.pre_classifier_soft(fused)

        return {
            "logits_hard": self.hard_classifier(shared_hard),
            "logits_soft": self.soft_classifier(shared_soft),
            "embedding":   shared_hard,
        }

    def count_parameters(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "audio_encoder":   count(self.audio_encoder),
            "text_encoder":    count(self.text_encoder),
            "clinical_encoder": count(self.clinical_encoder),
            "fusion":          count(self.fusion),
            "classifiers":     count(self.pre_classifier_hard) + count(self.pre_classifier_soft)
                               + count(self.hard_classifier) + count(self.soft_classifier),
            "total_trainable": count(self),
        }

