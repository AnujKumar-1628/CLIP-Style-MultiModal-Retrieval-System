"""
Repository scaffold for the CLIP-style multimodal retrieval project.

"""

from pathlib import Path

PROJECT_STRUCTURE = {
    "configs": [
        "model.yaml",
        "training.yaml",
        "data.yaml",
        "retrieval.yaml",
        "api.yaml",
    ],
    "data": {
        "raw": [
            ".gitkeep",
        ],
        "processed": [
            ".gitkeep",
        ],
        "splits": [
            ".gitkeep",
        ],
    },
    "artifacts": {
        "checkpoints": [
            ".gitkeep",
        ],
        "embeddings": [
            ".gitkeep",
        ],
        "indices": [
            ".gitkeep",
        ],
        "metadata": [
            "run_id.json",
        ],
    },
    "experiments": {
        "runs": [
            ".gitkeep",
        ],
        "logs": [
            ".gitkeep",
        ],
    },
    "src": {
        "__init__.py": None,
        "data_logic": [
            "__init__.py",
            "base_datamodule.py",
            "dataset.py",
            "transforms.py",
            "datamodule.py",
            "eval_datamodule.py",
            "sampler.py",
        ],
        "models": {
            "__init__.py": None,
            "encoders": {
                "image": [
                    "__init__.py",
                    "resnet50.py",
                ],
                "text": [
                    "__init__.py",
                    "distilbert.py",
                ],
            },
            "projection.py": None,
            "clip_model.py": None,
            "runtime.py": None,
        },
        "loss": [
            "__init__.py",
            "contrastive_loss.py",
        ],
        "training": [
            "__init__.py",
            "trainer.py",
            "train.py",
            "callbacks.py",
            "backbone_unfreeze_callback.py",
        ],
        "evaluation": [
            "__init__.py",
            "metrics.py",
            "evaluate.py",
        ],
        "retrieval": [
            "__init__.py",
            "embed.py",
            "index.py",
            "search.py",
            "pipeline.py",
            "storage.py",
            "rerankers.py",
            "serving.py",
        ],
        "inference": [
            "__init__.py",
            "encoder_wrapper.py",
            "predictor.py",
        ],
        "pipelines": [
            "__init__.py",
            "base.py",
            "train_pipeline.py",
            "eval_pipeline.py",
            "indexing_pipeline.py",
        ],
        "utils": [
            "__init__.py",
            "config.py",
            "logger.py",
            "seed.py",
            "registry.py",
            "paths.py",
        ],
    },
    "api": {
        "main.py": None,
        "dependencies.py": None,
        "schemas.py": None,
        "routes": [
            "search.py",
            "index.py",
            "health.py",
        ],
    },
    "scripts": [
        "train.sh",
        "evaluate.sh",
        "build_index.sh",
        "run_api.sh",
    ],
    "tests": [
        "conftest.py",
        "test_data.py",
        "test_model.py",
        "test_loss.py",
        "test_retrieval.py",
        "test_metrics.py",
        "test_api.py",
    ],
    "notebooks": [
        "01_data_exploration.ipynb",
        "02_embedding_visualization.ipynb",
        "03_retrieval_demo.ipynb",
        "04_evaluation_results.ipynb",
        "colab.ipynb",
    ],
    "docker": [
        "Dockerfile",
        "docker-compose.yml",
    ],
}


ROOT_FILES = [
    ".env.example",
    "Makefile",
    "requirements.txt",
    ".gitignore",
    "README.md",
    "setup.py",
    "LICENSE",
    "template.py",
]


def create_structure(base: Path, structure):
    if isinstance(structure, list):
        for item in structure:
            path = base / item
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)

    elif isinstance(structure, dict):
        for name, content in structure.items():
            path = base / name
            if content is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)
                create_structure(path, content)


def main():
    root = Path.cwd()

    for file in ROOT_FILES:
        (root / file).touch(exist_ok=True)

    create_structure(root, PROJECT_STRUCTURE)

    print("OK: Multimodal Retrieval repository scaffold created successfully.")


if __name__ == "__main__":
    main()
