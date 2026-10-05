"""Check required files, optional training images, and detector-weight loading."""

import argparse
import json
from pathlib import Path
import sys
from common import ROOT, MODELS, require_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--training-root", type=Path)
    p.add_argument("--load-models", action="store_true")
    a = p.parse_args()
    for name in ["imagenet_xception", *MODELS]:
        path = require_file(a.weights_dir / (name + ".pth"))
        print(f"Found {path.name} ({path.stat().st_size:,} bytes)")
    for i in range(10):
        require_file(ROOT / f"training/lib/component/MCT/template{i}.png")
    if a.training_root:
        count = 0
        for name, mode in [
            ("celebdf_train", "train"),
            ("digital_d1", "test"),
            ("digital_d2", "test"),
        ]:
            obj = json.loads(
                require_file(a.training_root / "indexes" / (name + ".json")).read_text()
            )[name]
            for splits in obj.values():
                for video in splits[mode].values():
                    for frame in video["frames"]:
                        rel = Path(frame)
                        if rel.is_absolute() or ".." in rel.parts:
                            raise ValueError("Expected relative training paths")
                        require_file(a.training_root / rel)
                        count += 1
        print(
            f"Training and validation indexes: all {count:,} referenced images found."
        )
    if a.load_models:
        import os
        import torch
        import yaml

        sys.path.insert(0, str(ROOT / "training"))
        from detectors import DETECTOR

        os.chdir(ROOT)
        for name in MODELS:
            config = yaml.safe_load((ROOT / "configs" / (name + ".yaml")).read_text())
            config["pretrained"] = str(
                a.weights_dir.resolve() / "imagenet_xception.pth"
            )
            model = DETECTOR[config["model_name"]](config)
            model.load_state_dict(
                torch.load(
                    a.weights_dir / (name + ".pth"),
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            print("Loaded:", name)


if __name__ == "__main__":
    main()
