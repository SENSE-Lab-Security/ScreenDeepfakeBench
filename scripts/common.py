"""Shared public paths and model names."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("xception", "ffd", "spsl", "coc")
DIGITAL = ("digital_d1", "digital_d2")


def environment(data_root, index_root, weights_dir):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SCREEN_")}
    env.update(
        SCREEN_DATA_ROOT=str(Path(data_root).resolve()),
        SCREEN_JSON_ROOT=str(Path(index_root).resolve()),
        SCREEN_CHECKPOINT_ROOT=str(Path(weights_dir).resolve()),
        NO_ALBUMENTATIONS_UPDATE="1",
    )
    return env


def require_file(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing resource: {path}. See README.md, Downloads.")
    return path
