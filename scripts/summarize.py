"""Summarize equal-weight configuration metrics, never pooled frame accuracy."""

import argparse
import csv
import json
from pathlib import Path
from common import ROOT, MODELS, DIGITAL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "outputs/eval")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/summary")
    a = p.parse_args()
    configs = json.loads((ROOT / "metadata/configurations.json").read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    groups = {
        "recaptured": [d["configuration_id"] for d in configs],
        "controlled": [
            d["configuration_id"] for d in configs if d["group"] == "controlled"
        ],
        "multifactor": [
            d["configuration_id"] for d in configs if d["group"] == "multifactor"
        ],
        "digital": list(DIGITAL),
    }
    for field in ["camera", "display"]:
        for value in sorted({d[field] for d in configs if d["group"] == "multifactor"}):
            groups[field + ":" + value] = [
                d["configuration_id"]
                for d in configs
                if d["group"] == "multifactor" and d[field] == value
            ]
    for model in MODELS:
        path = a.results / model / "metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text())
        known = {d["configuration_id"] for d in configs} | set(DIGITAL)
        if set(metrics) - known:
            raise ValueError("Unexpected configuration IDs in " + str(path))
        for group, expected in groups.items():
            names = [n for n in expected if n in metrics]
            if not names:
                continue
            row = dict(
                model=model,
                group=group,
                configurations=len(names),
                complete=len(names) == len(expected),
            )
            row.update(
                {
                    f"{level}_{key}": sum(
                        metrics[n][level + "_level"][key] for n in names
                    )
                    / len(names)
                    for level in ["frame", "video"]
                    for key in ["acc", "auc"]
                }
            )
            rows.append(row)
    if not rows:
        raise SystemExit("No evaluation metrics found")
    with (a.output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    if any(not r["complete"] for r in rows):
        print("Partial evaluation: incomplete rows are not full benchmark results.")


if __name__ == "__main__":
    main()
