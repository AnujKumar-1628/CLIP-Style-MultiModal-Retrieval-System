"""Tests for retrieval/api config loading."""

from __future__ import annotations

from src.utils.config import load_api_config, load_retrieval_config


def test_load_retrieval_config_success(write_yaml) -> None:
    cfg_path = write_yaml(
        "retrieval.yaml",
        {
            "embed": {"use_amp": True, "normalize": True, "image_batch_size": 8, "text_batch_size": 8},
            "index": {
                "backend": "flat",
                "metric": "cosine",
                "nlist": 128,
                "nprobe": 16,
                "pq_m": 16,
                "pq_nbits": 8,
                "sq_qtype": "8bit",
                "train_sample_size": 1000,
            },
            "storage": {
                "embeddings_dir": "artifacts/embeddings",
                "index_dir": "artifacts/indices",
                "metadata_dir": "artifacts/metadata",
                "use_mmap": True,
                "embedding_dtype": "float32",
            },
            "search": {"default_top_k": 5, "max_top_k": 10, "return_metadata": True},
            "reranker": {
                "enabled": False,
                "name": "none",
                "candidate_multiplier": 5,
                "max_candidates": 100,
                "blend_alpha": 0.5,
                "mmr_lambda": 0.7,
            },
            "serving": {"host": "0.0.0.0", "port": 8001, "route": "/search"},
        },
    )
    cfg = load_retrieval_config(cfg_path)
    assert cfg.search.default_top_k == 5
    assert cfg.index.backend == "flat"


def test_load_api_config_success(write_yaml) -> None:
    cfg_path = write_yaml(
        "api.yaml",
        {
            "app": {"title": "Test API", "version": "1.2.3"},
            "server": {
                "host": "127.0.0.1",
                "port": 9000,
                "root_path": "",
                "docs_url": "/docs",
                "redoc_url": "/redoc",
            },
            "paths": {
                "retrieval_config": "configs/retrieval.yaml",
                "model_config": "configs/model.yaml",
                "data_config": "configs/data.yaml",
            },
            "index": {"image_index_name": "image_index", "text_index_name": "text_index"},
        },
    )
    cfg = load_api_config(cfg_path)
    assert cfg.app.title == "Test API"
    assert cfg.server.port == 9000


def test_load_api_config_rejects_bad_docs_url(write_yaml) -> None:
    cfg_path = write_yaml(
        "api.yaml",
        {
            "app": {"title": "Test API", "version": "1.0.0"},
            "server": {"port": 8001, "docs_url": "docs", "redoc_url": "/redoc"},
            "paths": {},
            "index": {},
        },
    )
    try:
        load_api_config(cfg_path)
        raise AssertionError("Expected ValueError for invalid docs_url.")
    except ValueError as exc:
        assert "docs_url" in str(exc)
