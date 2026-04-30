"""Centralized filesystem paths for the project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Configs
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
FLICKR30K_IMAGES_DIR = RAW_DATA_DIR / "flickr30k_images"
CAPTIONS_FILE = RAW_DATA_DIR / "captions.txt"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
INDICES_DIR = ARTIFACTS_DIR / "indices"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

# Experiments
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
LOGS_DIR = EXPERIMENTS_DIR / "logs"


def ensure_base_dirs() -> None:
    """Create the core writable directories if they do not exist."""
    for path in (
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        CHECKPOINTS_DIR,
        EMBEDDINGS_DIR,
        INDICES_DIR,
        METADATA_DIR,
        RUNS_DIR,
        LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
