"""Portable paths passed by the public command-line entry points."""

import os
from pathlib import Path


def apply_runtime(config):
    for key, env in [
        ("rgb_dir", "SCREEN_DATA_ROOT"),
        ("dataset_json_folder", "SCREEN_JSON_ROOT"),
        ("log_dir", "SCREEN_TRAIN_OUTPUT"),
    ]:
        if os.environ.get(env):
            config[key] = os.environ[env]
    if os.environ.get("SCREEN_CHECKPOINT_ROOT"):
        config["pretrained"] = str(
            Path(os.environ["SCREEN_CHECKPOINT_ROOT"]) / Path(config["pretrained"]).name
        )
    for key, env in [
        ("workers", "SCREEN_WORKERS"),
        ("test_batchSize", "SCREEN_EVAL_BATCH"),
        ("manualSeed", "SCREEN_SEED"),
    ]:
        if os.environ.get(env):
            config[key] = int(os.environ[env])
    if os.environ.get("SCREEN_EPOCHS"):
        config["start_epoch"], config["nEpochs"] = (
            0,
            int(os.environ["SCREEN_EPOCHS"]) - 1,
        )
    if os.environ.get("SCREEN_LR"):
        config["optimizer"][config["optimizer"]["type"]]["lr"] = float(
            os.environ["SCREEN_LR"]
        )
    if os.environ.get("SCREEN_DETERMINISTIC") == "1":
        config["deterministic"], config["cudnn"] = True, False
    return config
