"""DistilBERT text encoder for CLIP-style retrieval.

Design goals:
- Config-driven construction from `configs/model.yaml`
- Efficient inference helpers with low CPU overhead
- Reuse of existing data/config utilities
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertConfig, DistilBertModel, PreTrainedTokenizerBase

from src.data_logic.transforms import build_text_preprocessor, get_data_config
from src.models.runtime import resolve_device, resolve_dtype
from src.utils.config import load_yaml
from src.utils.logger import setup_logger
from src.utils.paths import CONFIGS_DIR
from src.utils.registry import Registry


LOGGER = setup_logger(
    name="distilbert_encoder",
    level="INFO",
    use_console=True,
    use_file=True,
)

TEXT_ENCODER_REGISTRY: Registry[type[nn.Module]] = Registry("text_encoder")


class _SimpleWhitespaceTokenizer:
    """Offline-safe fallback tokenizer.

    This tokenizer is intentionally simple and deterministic:
    - split on whitespace
    - map token to id via stable hash mod vocab size
    - reserve 0 for PAD and 1 for UNK
    """

    def __init__(self, vocab_size: int = 30522) -> None:
        self.vocab_size = int(vocab_size)
        self.pad_token_id = 0
        self.unk_token_id = 1

    def _token_to_id(self, token: str) -> int:
        # Keep IDs in [2, vocab_size-1] to reserve 0/1 for PAD/UNK.
        if not token:
            return self.unk_token_id
        usable = max(2, self.vocab_size - 2)
        return 2 + (abs(hash(token)) % usable)

    def __call__(
        self,
        texts: list[str],
        *,
        truncation: bool = True,
        max_length: int = 32,
        padding: str | bool = "longest",
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        if not texts:
            raise ValueError("Tokenizer received empty texts list.")

        encoded: list[list[int]] = []
        for text in texts:
            tokens = str(text).strip().split()
            ids = [self._token_to_id(token) for token in tokens]
            if truncation:
                ids = ids[:max_length]
            if not ids:
                ids = [self.unk_token_id]
            encoded.append(ids)

        if padding in (True, "longest"):
            target_len = min(max(len(ids) for ids in encoded), max_length)
        elif padding == "max_length":
            target_len = max_length
        else:
            target_len = max(len(ids) for ids in encoded)

        padded_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for ids in encoded:
            ids = ids[:target_len]
            attn = [1] * len(ids)
            pad_needed = target_len - len(ids)
            if pad_needed > 0:
                ids = ids + ([self.pad_token_id] * pad_needed)
                attn = attn + ([0] * pad_needed)
            padded_ids.append(ids)
            attention_masks.append(attn)

        input_ids = torch.tensor(padded_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long)

        if return_tensors != "pt":
            raise ValueError("Fallback tokenizer only supports return_tensors='pt'.")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


@dataclass(frozen=True)
class DistilBertEncoderConfig:
    architecture: str
    pretrained: bool
    trainable: bool
    freeze_backbone: bool
    freeze_until: str | None
    out_dim: int
    dropout: float
    max_length: int
    dynamic_padding: bool
    lowercase: bool
    device: str
    dtype: str
    normalize_output: bool


def _parse_freeze_until_layer(value: str | None) -> int | None:
    """Parse layer-freeze marker.

    Supported forms:
    - None
    - "3"
    - "layer_3"
    - "layers.3"
    """
    if value is None:
        return None
    raw = value.strip().lower()
    if raw.isdigit():
        return int(raw)
    match = re.search(r"(\d+)$", raw)
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Unsupported freeze_until='{value}'. Use integer-like value (e.g., '3', 'layer_3')."
    )


def load_distilbert_encoder_config(
    config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> DistilBertEncoderConfig:
    """Load text encoder settings from model/data config files."""
    model_cfg_path = Path(config_path) if config_path is not None else (CONFIGS_DIR / "model.yaml")
    model_raw = load_yaml(model_cfg_path)
    data_cfg_path = (
        Path(data_config_path)
        if data_config_path is not None
        else (CONFIGS_DIR / "data.yaml")
    )
    data_raw = load_yaml(data_cfg_path)

    model_section = model_raw.get("model", {})
    text_raw = model_raw.get("text_encoder", {})
    data_text_raw = data_raw.get("text", {})

    architecture = str(text_raw.get("architecture", "distilbert-base-uncased")).strip()
    if "distilbert" not in architecture.lower():
        raise ValueError(
            f"This encoder expects a DistilBERT architecture. Got '{architecture}'."
        )

    if text_raw.get("max_length") is not None:
        LOGGER.warning(
            "Ignoring text_encoder.max_length from model config. "
            "Using text.max_length from data config instead."
        )

    cfg = DistilBertEncoderConfig(
        architecture=architecture,
        pretrained=bool(text_raw.get("pretrained", True)),
        trainable=bool(text_raw.get("trainable", True)),
        freeze_backbone=bool(text_raw.get("freeze_backbone", False)),
        freeze_until=(str(text_raw["freeze_until"]).strip() if text_raw.get("freeze_until") else None),
        out_dim=int(text_raw.get("out_dim", 768)),
        dropout=float(text_raw.get("dropout", 0.0)),
        max_length=int(data_text_raw.get("max_length", 40)),
        dynamic_padding=bool(data_text_raw.get("dynamic_padding", True)),
        lowercase=bool(data_text_raw.get("lowercase", True)),
        device=str(model_section.get("device", "cpu")),
        dtype=str(model_section.get("dtype", "float32")),
        normalize_output=bool(text_raw.get("normalize_output", False)),
    )
    return cfg


@TEXT_ENCODER_REGISTRY.register("distilbert")
@TEXT_ENCODER_REGISTRY.register("distilbert-base-uncased")
class DistilBertTextEncoder(nn.Module):
    """Config-driven DistilBERT text encoder with efficient inference helpers."""

    def __init__(
        self,
        config: DistilBertEncoderConfig | None = None,
        *,
        model_config_path: str | Path | None = None,
        data_config_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config or load_distilbert_encoder_config(
            model_config_path,
            data_config_path,
        )

        self.device_obj = resolve_device(self.config.device, logger=LOGGER)
        self.dtype_obj = resolve_dtype(self.config.dtype)
        self.max_length = int(self.config.max_length)
        self.dynamic_padding = bool(self.config.dynamic_padding)

        self.text_preprocessor = build_text_preprocessor(get_data_config(data_config_path))

        self.tokenizer = self._build_tokenizer(self.config.architecture)
        self.backbone = self._build_backbone()

        hidden_size = int(self.backbone.config.dim)
        if self.config.out_dim != hidden_size:
            raise ValueError(
                f"text_encoder.out_dim must be {hidden_size} for DistilBERT, got {self.config.out_dim}."
            )

        self.dropout = nn.Dropout(self.config.dropout) if self.config.dropout > 0 else nn.Identity()
        self.output_dim = hidden_size

        self._apply_trainability_controls()
        self.to(device=self.device_obj, dtype=self.dtype_obj)

        LOGGER.info(
            "Initialized DistilBertTextEncoder | device=%s dtype=%s pretrained=%s "
            "architecture=%s trainable=%s freeze_backbone=%s freeze_until=%s "
            "normalize_output=%s max_length=%d dynamic_padding=%s",
            self.device_obj,
            self.dtype_obj,
            self.config.pretrained,
            self.config.architecture,
            self.config.trainable,
            self.config.freeze_backbone,
            self.config.freeze_until,
            self.config.normalize_output,
            self.max_length,
            self.dynamic_padding,
        )

    def _build_tokenizer(self, architecture: str) -> PreTrainedTokenizerBase | _SimpleWhitespaceTokenizer:
        try:
            return AutoTokenizer.from_pretrained(architecture, use_fast=True)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load tokenizer for '%s' (%s). Retrying with use_fast=False.",
                architecture,
                exc,
            )
            try:
                return AutoTokenizer.from_pretrained(architecture, use_fast=False)
            except Exception as inner_exc:
                LOGGER.warning(
                    "Failed to load tokenizer for '%s' in offline mode (%s). "
                    "Falling back to SimpleWhitespaceTokenizer.",
                    architecture,
                    inner_exc,
                )
                return _SimpleWhitespaceTokenizer(vocab_size=30522)

    def _build_backbone(self) -> DistilBertModel:
        if self.config.pretrained:
            try:
                return DistilBertModel.from_pretrained(self.config.architecture)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to load pretrained DistilBERT '%s' (%s). Falling back to random init.",
                    self.config.architecture,
                    exc,
                )
        try:
            base_cfg = DistilBertConfig.from_pretrained(self.config.architecture)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load DistilBERT config for '%s' (%s). Falling back to default DistilBertConfig.",
                self.config.architecture,
                exc,
            )
            base_cfg = DistilBertConfig()
        return DistilBertModel(base_cfg)

    def _apply_trainability_controls(self) -> None:
        if not self.config.trainable or self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            return

        freeze_until_layer = _parse_freeze_until_layer(self.config.freeze_until)
        if freeze_until_layer is None:
            return

        # Freeze embeddings and encoder blocks [0..freeze_until_layer] inclusive.
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        max_layer_index = len(self.backbone.transformer.layer) - 1
        if freeze_until_layer > max_layer_index:
            raise ValueError(
                f"freeze_until layer {freeze_until_layer} exceeds max DistilBERT layer {max_layer_index}."
            )

        for idx, layer in enumerate(self.backbone.transformer.layer):
            if idx <= freeze_until_layer:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids with shape [B, L], got {tuple(input_ids.shape)}")

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_features = outputs.last_hidden_state[:, 0, :]
        cls_features = self.dropout(cls_features)

        if self.config.normalize_output:
            cls_features = F.normalize(cls_features, p=2, dim=-1)
        return cls_features

    def preprocess_text(self, text: str) -> str:
        return self.text_preprocessor(text)

    def tokenize_texts(
        self,
        texts: Iterable[str],
        *,
        padding: str | bool | None = None,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        processed = [self.preprocess_text(text) for text in texts]
        if not processed:
            raise ValueError("Expected at least one text in tokenize_texts.")

        if padding is None:
            padding = "longest" if self.dynamic_padding else "max_length"

        tokens = self.tokenizer(
            processed,
            truncation=True,
            max_length=self.max_length,
            padding=padding,
            return_tensors=return_tensors,
        )
        return tokens

    @torch.inference_mode()
    def encode_tokens(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        target_device = torch.device(device) if device is not None else self.device_obj
        ids = input_ids.to(target_device, non_blocking=non_blocking)
        mask = (
            attention_mask.to(target_device, non_blocking=non_blocking)
            if attention_mask is not None
            else None
        )

        was_training = self.training
        self.eval()
        embeddings = self.forward(input_ids=ids, attention_mask=mask)
        if was_training:
            self.train()
        return embeddings

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: Iterable[str],
        *,
        device: torch.device | str | None = None,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        tokens = self.tokenize_texts(texts)
        return self.encode_tokens(
            input_ids=tokens["input_ids"],
            attention_mask=tokens.get("attention_mask"),
            device=device,
            non_blocking=non_blocking,
        )

    @classmethod
    def from_config_paths(
        cls,
        *,
        model_config_path: str | Path | None = None,
        data_config_path: str | Path | None = None,
    ) -> "DistilBertTextEncoder":
        cfg = load_distilbert_encoder_config(model_config_path, data_config_path)
        return cls(config=cfg, model_config_path=model_config_path, data_config_path=data_config_path)


__all__ = [
    "TEXT_ENCODER_REGISTRY",
    "DistilBertEncoderConfig",
    "DistilBertTextEncoder",
    "load_distilbert_encoder_config",
]
