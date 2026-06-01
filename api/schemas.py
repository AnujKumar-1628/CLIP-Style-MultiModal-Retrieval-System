"""Pydantic schemas for API request/response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class APIErrorResponse(BaseModel):
    detail: str


class APIRootResponse(BaseModel):
    service: str
    status: str
    version: str
    docs_url: str


class APIHealthResponse(BaseModel):
    status: str
    service: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text query for retrieval search.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Optional top-k override. Falls back to configured default.",
    )


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(..., ge=1)
    item_id: str
    score: float
    metadata: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHit]
