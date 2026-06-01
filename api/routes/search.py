"""Search endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_retrieval_server
from api.schemas import SearchRequest, SearchResponse
from src.retrieval.serving import RetrievalServer

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "/text-to-image",
    response_model=SearchResponse,
    responses={400: {"description": "Bad request"}},
)
def search_text_to_image(
    payload: SearchRequest,
    server: RetrievalServer = Depends(get_retrieval_server),
) -> SearchResponse:
    try:
        response = server.search_text_to_image(query=payload.query, top_k=payload.top_k)
        return SearchResponse(hits=response.hits)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
