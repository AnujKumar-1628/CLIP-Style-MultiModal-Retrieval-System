"""High-level prediction helpers for CLIP inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

from src.inference.encoder_wrapper import CLIPEncoderWrapper, create_inference_wrapper
from src.utils.config import load_retrieval_config
from src.utils.logger import setup_logger
from src.utils.paths import CONFIGS_DIR

LOGGER = setup_logger(
    name="inference_predictor",
    level="INFO",
    use_console=True,
    use_file=True,
)


@dataclass(frozen=True)
class RankedItem:
    rank: int
    index: int
    score: float
    payload: Any | None = None


@dataclass(frozen=True)
class SimilarityResult:
    logits: torch.Tensor
    image_count: int
    text_count: int


class CLIPPredictor:
    """Task-oriented inference API built on `CLIPEncoderWrapper`."""

    def __init__(
        self,
        *,
        wrapper: CLIPEncoderWrapper | None = None,
        model_config_path: str | Path | None = None,
        data_config_path: str | Path | None = None,
        retrieval_config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.wrapper = wrapper or create_inference_wrapper(
            model_config_path=model_config_path or (CONFIGS_DIR / "model.yaml"),
            data_config_path=data_config_path or (CONFIGS_DIR / "data.yaml"),
            checkpoint_path=checkpoint_path,
        )
        retrieval_cfg = load_retrieval_config(
            retrieval_config_path or (CONFIGS_DIR / "retrieval.yaml")
        )
        self.default_top_k = int(retrieval_cfg.search.default_top_k)
        self.max_top_k = int(retrieval_cfg.search.max_top_k)

    def _resolve_top_k(self, top_k: int | None, total: int) -> int:
        if total <= 0:
            raise ValueError("No candidates available for ranking.")
        k = self.default_top_k if top_k is None else int(top_k)
        if k <= 0:
            raise ValueError("top_k must be > 0.")
        k = min(k, self.max_top_k)
        return min(k, total)

    @staticmethod
    def _rank_scores(
        *,
        scores: torch.Tensor,
        k: int,
        payloads: list[Any] | None = None,
    ) -> list[RankedItem]:
        vals, idx = torch.topk(scores, k=k, largest=True, sorted=True)
        results: list[RankedItem] = []
        payload_values = payloads or []
        for rank_idx, (score_t, idx_t) in enumerate(zip(vals, idx), start=1):
            item_idx = int(idx_t.item())
            payload = payload_values[item_idx] if payload_values else None
            results.append(
                RankedItem(
                    rank=rank_idx,
                    index=item_idx,
                    score=float(score_t.item()),
                    payload=payload,
                )
            )
        return results

    @torch.inference_mode()
    def similarity_matrix_from_raw(
        self,
        *,
        images: Iterable[Image.Image],
        texts: Iterable[str],
    ) -> SimilarityResult:
        image_batch = list(images)
        text_batch = list(texts)
        if not image_batch:
            raise ValueError("images must contain at least one item.")
        if not text_batch:
            raise ValueError("texts must contain at least one item.")
        logits = self.wrapper.similarity_from_raw(images=image_batch, texts=text_batch)
        return SimilarityResult(
            logits=logits,
            image_count=int(logits.shape[0]),
            text_count=int(logits.shape[1]),
        )

    @torch.inference_mode()
    def similarity_matrix_from_tensors(
        self,
        *,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimilarityResult:
        logits = self.wrapper.similarity_from_tensors(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return SimilarityResult(
            logits=logits,
            image_count=int(logits.shape[0]),
            text_count=int(logits.shape[1]),
        )

    @torch.inference_mode()
    def rank_texts_for_image(
        self,
        *,
        image: Image.Image | torch.Tensor,
        candidate_texts: list[str],
        top_k: int | None = None,
    ) -> list[RankedItem]:
        if not candidate_texts:
            raise ValueError("candidate_texts must be non-empty.")

        if isinstance(image, torch.Tensor):
            image_emb = self.wrapper.encode_images_tensor(
                image,
                normalize=True,
            )
        else:
            image_emb = self.wrapper.encode_images_pil([image], normalize=True)
        text_emb = self.wrapper.encode_texts(candidate_texts, normalize=True)

        scores = (image_emb @ text_emb.t()).squeeze(0)
        k = self._resolve_top_k(top_k, len(candidate_texts))
        return self._rank_scores(scores=scores, k=k, payloads=[*candidate_texts])

    @torch.inference_mode()
    def rank_images_for_text(
        self,
        *,
        text: str,
        candidate_images: list[Image.Image] | torch.Tensor,
        top_k: int | None = None,
    ) -> list[RankedItem]:
        if isinstance(candidate_images, list):
            if not candidate_images:
                raise ValueError("candidate_images must be non-empty.")
            image_emb = self.wrapper.encode_images_pil(candidate_images, normalize=True)
            payloads: list[Any] = candidate_images
        else:
            if candidate_images.ndim != 4:
                raise ValueError(
                    "candidate_images tensor must have shape [B,C,H,W]."
                )
            image_emb = self.wrapper.encode_images_tensor(candidate_images, normalize=True)
            payloads = [None] * int(candidate_images.shape[0])

        text_emb = self.wrapper.encode_texts([text], normalize=True)
        scores = (text_emb @ image_emb.t()).squeeze(0)
        k = self._resolve_top_k(top_k, int(image_emb.shape[0]))
        return self._rank_scores(scores=scores, k=k, payloads=payloads)


def create_predictor(
    *,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    retrieval_config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> CLIPPredictor:
    return CLIPPredictor(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        retrieval_config_path=retrieval_config_path,
        checkpoint_path=checkpoint_path,
    )


__all__ = [
    "CLIPPredictor",
    "RankedItem",
    "SimilarityResult",
    "create_predictor",
]
