"""Optional reproducible RNG streams for single-GPU training."""

import os
import random
import numpy as np
import torch


def configure(config):
    if not config.get("deterministic", False):
        return
    if config.get("ddp", False):
        raise ValueError("Deterministic mode currently supports one GPU")
    seed = config["manualSeed"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or os.environ.get(
        "PYTHONHASHSEED"
    ) != str(seed):
        raise ValueError(
            "Use scripts/train.py --deterministic to initialize the process environment"
        )
    import cv2

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.utils.data.get_worker_info().dataset.transform.set_random_seed(seed)


def loader_options(config, dataset, stream=0):
    if not config.get("deterministic", False):
        return {}
    seed = config["manualSeed"] + stream * 1000
    dataset.transform.set_random_seed(seed + 3)
    options = dict(
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed + 1)
    )
    if int(config["workers"]) > 0:
        options.update(multiprocessing_context="fork", timeout=300)
    return options


def sampler_options(config):
    if not config.get("deterministic", False):
        return {}
    return {"generator": torch.Generator().manual_seed(config["manualSeed"] + 2)}
