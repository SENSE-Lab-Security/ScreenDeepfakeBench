"""Validate the public detector inputs and generate portable evaluation indexes."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from common import ROOT


def prepare(data_root, output):
    with (data_root / "metadata/samples.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    with (data_root / "metadata/configurations.csv").open(newline="") as f:
        configs = list(csv.DictReader(f))
    expected = {
        d["configuration_id"]
        for d in json.loads((ROOT / "metadata/configurations.json").read_text())
    }
    if len(rows) != 29900 or {d["configuration_id"] for d in configs} != expected:
        raise ValueError("Expected the 29,900-input / 103-configuration public release")
    by_key = {}
    labels = Counter()
    for row in rows:
        rel = Path(row["benchmark_input_path"])
        if rel.is_absolute() or ".." in rel.parts or not (data_root / rel).is_file():
            raise FileNotFoundError(f"Invalid or missing input: {rel}")
        key = (
            row["configuration_id"],
            row["label"],
            row["source_video_id"],
            row["frame_name"],
        )
        if key in by_key:
            raise ValueError("Duplicate sample: " + str(key))
        by_key[key] = row["benchmark_input_path"]
        labels[row["label"]] += 1
    if labels != {"real": 14950, "fake": 14950}:
        raise ValueError("Unexpected class balance")
    output.mkdir(parents=True, exist_ok=True)
    used = set()
    for cid in sorted(expected):
        obj = json.loads((ROOT / "metadata/indexes" / (cid + ".json")).read_text())
        for label, groups in obj[cid].items():
            for video, entry in groups["test"].items():
                new_paths = []
                for old in entry["frames"]:
                    key = (cid, label, Path(old).parent.name, Path(old).name)
                    new_paths.append(by_key[key])
                    used.add(key)
                entry["frames"] = new_paths
        (output / (cid + ".json")).write_text(json.dumps(obj, indent=2) + "\n")
    if len(used) != 29900:
        raise ValueError("Some release samples are missing from evaluation indexes")
    print(
        f"Ready: 103 configurations, 29,900 images (14,950 per class). Indexes: {output}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Extracted CoC_Dataset_zenodo directory",
    )
    p.add_argument("--output", type=Path, default=ROOT / "outputs/indexes")
    a = p.parse_args()
    prepare(a.data_root.resolve(), a.output.resolve())


if __name__ == "__main__":
    main()
