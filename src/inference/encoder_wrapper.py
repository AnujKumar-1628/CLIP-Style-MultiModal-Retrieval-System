"""Model-centric inference wrappers for CLIP-style retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from src.models.clip_model import CLIPModel, create_clip_model
from src.utils.config import (
    InferenceModelConfig,
    create_inference_model_config,
)
from src.utils.logger import setup_logger

LOGGER = setup_logger(
    name="inference_encoder_wrapper",
    level="INFO",
    use_console=True,
    use_file=True,
)

class CLIPEncoderWrapper:
    """Thin inference-focused wrapper around `CLIPModel`."""

    def __init__(
        self,
        *,
        config: InferenceModelConfig | None = None,
        model: CLIPModel | None = None,
    ) -> None:
        default_cfg = create_inference_model_config()
        self.config = config or default_cfg
        self.model = model or create_clip_model(
            model_config_path=self.config.model_config_path,
            data_config_path=self.config.data_config_path,
        )
        self._load_checkpoint_if_needed()
        self.model.eval()
        LOGGER.info(
            "Initialized CLIPEncoderWrapper | device=%s embed_dim=%d",
            self.model.device_obj,
            int(self.model.embed_dim),
        )

    def _load_checkpoint_if_needed(self) -> None:
        ckpt_path = self.config.checkpoint_path
        if ckpt_path is None:
            return
        resolved_path = Path(ckpt_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Inference checkpoint not found: {resolved_path}")
        payload = torch.load(resolved_path, map_location=self.model.device_obj)
        if isinstance(payload, dict):
            state_dict = payload.get(self.config.checkpoint_state_dict_key, payload)
        else:
            state_dict = payload
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint payload does not contain a valid state_dict mapping.")
        self.model.load_state_dict(state_dict, strict=self.config.checkpoint_strict)
        LOGGER.info("Loaded checkpoint for inference: %s", resolved_path)

    @staticmethod
    def _to_image_batch(images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            return images.unsqueeze(0)
        if images.ndim == 4:
            return images
        raise ValueError(
            f"Expected image tensor shape [C,H,W] or [B,C,H,W], got {tuple(images.shape)}."
        )

    @staticmethod
    def _materialize_images(images: Iterable[Image.Image]) -> list[Image.Image]:
        batch = list(images)
        if not batch:
            raise ValueError("images must contain at least one item.")
        return batch

    @property
    def device(self) -> torch.device:
        return self.model.device_obj

    @property
    def embed_dim(self) -> int:
        return int(self.model.embed_dim)

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: Iterable[str],
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.model.encode_texts(texts, normalize=normalize)

    @torch.inference_mode()
    def encode_text_tokens(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.model.encode_text_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            normalize=normalize,
        )

    @torch.inference_mode()
    def encode_images_tensor(
        self,
        images: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        image_batch = self._to_image_batch(images)
        return self.model.encode_images(image_batch, normalize=normalize)

    @torch.inference_mode()
    def encode_images_pil(
        self,
        images: Iterable[Image.Image],
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        materialized = self._materialize_images(images)
        batch = self.model.image_encoder.preprocess_images(materialized)
        return self.model.encode_images(batch, normalize=normalize)

    @torch.inference_mode()
    def similarity_from_tensors(
        self,
        *,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.similarity_from_tensors(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    @torch.inference_mode()
    def similarity_from_raw(
        self,
        *,
        images: Iterable[Image.Image],
        texts: Iterable[str],
    ) -> torch.Tensor:
        image_batch = self._materialize_images(images)
        text_batch = list(texts)
        if not text_batch:
            raise ValueError("texts must contain at least one item.")
        return self.model.similarity_from_raw(images=image_batch, texts=text_batch)


def create_inference_wrapper(
    *,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_strict: bool = True,
    checkpoint_state_dict_key: str = "model_state_dict",
) -> CLIPEncoderWrapper:
    cfg = create_inference_model_config(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_strict=checkpoint_strict,
        checkpoint_state_dict_key=checkpoint_state_dict_key,
    )
    return CLIPEncoderWrapper(config=cfg)


__all__ = [
    "CLIPEncoderWrapper",
    "InferenceModelConfig",
    "create_inference_wrapper",
]
