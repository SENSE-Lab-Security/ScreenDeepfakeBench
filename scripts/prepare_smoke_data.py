"""Generate synthetic images for checking the training/evaluation workflow only."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    (output / "indexes").mkdir(parents=True)
    rng = np.random.default_rng(1024)
    for name, mode, frames in [("smoke_train", "train", 32), ("smoke_val", "test", 5)]:
        entries = {}
        for label in ("real", "fake"):
            videos = {}
            for number in range(2):
                video = f"{name}_{label}_{number}"
                paths = []
                for frame in range(frames):
                    pixels = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
                    pixels[64:192, 64:192, 0 if label == "real" else 2] = 255
                    rel = Path("images") / video / f"{frame:03d}.png"
                    (output / rel).parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(pixels).save(output / rel)
                    paths.append(rel.as_posix())
                videos[video] = {"label": label, "frames": paths}
            entries[label] = {mode: videos}
        (output / "indexes" / f"{name}.json").write_text(
            json.dumps({name: entries}, indent=2) + "\n"
        )
    (output / "README.txt").write_text(
        "Synthetic workflow fixture: 128 training and 20 validation images.\n"
        "Not faces, not Celeb-DF, and not benchmark performance data.\n"
    )
    print(f"Ready: {output}; 128 synthetic training images, 20 validation images.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/smoke"))
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
