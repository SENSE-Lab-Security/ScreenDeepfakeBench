"""Evaluate paper or user-provided weights on the public recaptured benchmark."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT, MODELS, DIGITAL, environment, require_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=(*MODELS, "all"), default="coc")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--index-root", type=Path, default=ROOT / "outputs/indexes")
    p.add_argument("--weights-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument(
        "--weights", type=Path, help="Custom detector checkpoint; requires one model"
    )
    p.add_argument(
        "--group",
        choices=("all", "controlled", "multifactor", "digital"),
        default="all",
        help="all means the 103 recaptured configurations; digital is separate",
    )
    p.add_argument("--dataset", nargs="+", help="Exact public configuration IDs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/eval")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.weights and a.model == "all":
        p.error("--weights requires one model")
    if a.workers < 0 or a.batch_size < 1:
        p.error("workers must be nonnegative and batch-size positive")
    configs = json.loads((ROOT / "metadata/configurations.json").read_text())
    names = a.dataset or (
        list(DIGITAL)
        if a.group == "digital"
        else [
            d["configuration_id"]
            for d in configs
            if a.group == "all" or d["group"] == a.group
        ]
    )
    if len(names) != len(set(names)):
        p.error("Dataset IDs must be unique")
    if not set(names) <= ({d["configuration_id"] for d in configs} | set(DIGITAL)):
        p.error("Unknown public configuration ID")
    for name in names:
        require_file(a.index_root / (name + ".json"))
    require_file(a.weights_dir / "imagenet_xception.pth")
    env = environment(a.data_root, a.index_root, a.weights_dir)
    env.update(SCREEN_WORKERS=str(a.workers), SCREEN_EVAL_BATCH=str(a.batch_size))
    if a.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if a.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            p.error("CUDA was requested but is unavailable")
    for model in MODELS if a.model == "all" else [a.model]:
        weight = require_file(a.weights or a.weights_dir / (model + ".pth"))
        out = a.output.resolve() / model
        if (out / "metrics.json").exists() and not a.dry_run:
            p.error(f"Results already exist in {out}; choose a new --output")
        cmd = [
            sys.executable,
            "training/test.py",
            "--detector_path",
            f"configs/{model}.yaml",
            "--weights_path",
            str(weight),
            "--test_dataset",
            *names,
            "--output_dir",
            str(out),
        ]
        if a.save_predictions:
            cmd.append("--save_predictions")
        print(f"{model}: {len(names)} configurations; weights={weight}", flush=True)
        if not a.dry_run:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
