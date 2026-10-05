"""Train a detector from ImageNet initialization using the paper's training loop."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from common import ROOT, MODELS, environment, require_file


def main(finetune=False):
    p = argparse.ArgumentParser(
        description="Fine-tune a detector." if finetune else __doc__
    )
    p.add_argument("--model", choices=MODELS, default="coc")
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="training_data directory with images and indexes",
    )
    p.add_argument("--index-root", type=Path, help="Defaults to DATA_ROOT/indexes")
    p.add_argument("--weights-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Fine-tune detector weights; new optimizer state",
    )
    p.add_argument("--epochs", type=int, default=5 if finetune else 11)
    p.add_argument("--lr", type=float, default=2e-5 if finetune else None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--train-dataset", nargs="+", default=["celebdf_train"])
    p.add_argument("--val-dataset", nargs="+", default=["digital_d1", "digital_d2"])
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.epochs < 1 or a.workers < 0 or (a.lr is not None and a.lr <= 0):
        p.error("Invalid epochs, workers or learning rate")
    indexes = (a.index_root or a.data_root / "indexes").resolve()
    for name in a.train_dataset + a.val_dataset:
        require_file(indexes / (name + ".json"))
    require_file(a.weights_dir / "imagenet_xception.pth")
    init = a.init_checkpoint or (
        a.weights_dir / (a.model + ".pth") if finetune else None
    )
    if init:
        init = require_file(init)
    out = (
        a.output
        or ROOT
        / "outputs/training"
        / (a.model + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    ).resolve()
    if out.exists():
        p.error("Choose a new --output directory; existing runs are not overwritten")
    env = environment(a.data_root, indexes, a.weights_dir)
    env.update(
        SCREEN_TRAIN_OUTPUT=str(out),
        SCREEN_WORKERS=str(a.workers),
        SCREEN_EPOCHS=str(a.epochs),
        SCREEN_SEED=str(a.seed),
    )
    if a.lr is not None:
        env["SCREEN_LR"] = str(a.lr)
    if a.deterministic:
        env.update(
            SCREEN_DETERMINISTIC="1",
            PYTHONHASHSEED=str(a.seed),
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
    cmd = [
        sys.executable,
        "training/train.py",
        "--detector_path",
        f"configs/{a.model}.yaml",
        "--task_target",
        a.model,
        "--train_dataset",
        *a.train_dataset,
        "--test_dataset",
        *a.val_dataset,
    ]
    if init:
        cmd += ["--init_weights", str(init)]
    print(
        f'{a.model}: {a.epochs} epochs; initialization={init or "ImageNet"}; output={out}',
        flush=True,
    )
    if not a.dry_run:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
