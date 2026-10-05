"""Small resource-independent checks for benchmark indexes and summaries."""

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_index_coverage(self):
        configs = json.loads((ROOT / "metadata/configurations.json").read_text())
        self.assertEqual(len(configs), 103)
        self.assertEqual(
            Counter(c["group"] for c in configs), {"controlled": 19, "multifactor": 84}
        )
        total = Counter()
        for config in configs:
            cid = config["configuration_id"]
            obj = json.loads((ROOT / "metadata/indexes" / (cid + ".json")).read_text())
            self.assertEqual(set(obj), {cid})
            counts = Counter()
            paths = set()
            for label, groups in obj[cid].items():
                for video in groups["test"].values():
                    self.assertEqual(video["label"], label)
                    self.assertEqual(len(video["frames"]), 5)
                    for frame in video["frames"]:
                        path = Path(frame)
                        self.assertFalse(path.is_absolute())
                        self.assertNotIn("..", path.parts)
                        self.assertEqual(
                            path.parts[:4], ("data", "benchmark_inputs", cid, label)
                        )
                        self.assertNotIn(frame, paths)
                        paths.add(frame)
                        counts[label] += 1
            self.assertEqual(sum(counts.values()), int(config["frame_count"]))
            self.assertEqual(counts["real"], int(config["real_count"]))
            self.assertEqual(counts["fake"], int(config["fake_count"]))
            total.update(counts)
        self.assertEqual(total, {"real": 14950, "fake": 14950})

    def test_reference_coverage(self):
        names = {
            c["configuration_id"]
            for c in json.loads((ROOT / "metadata/configurations.json").read_text())
        }
        for model in ["xception", "ffd", "spsl", "coc"]:
            results = json.loads(
                (ROOT / "reference_results" / (model + ".json")).read_text()
            )
            self.assertEqual(set(results), names)
            for row in results.values():
                self.assertEqual(
                    set(row), {"frame_acc", "frame_auc", "video_acc", "video_auc"}
                )
                self.assertTrue(all(0 <= value <= 1 for value in row.values()))

    def test_partial_summary(self):
        configs = json.loads((ROOT / "metadata/configurations.json").read_text())
        cid = next(
            c["configuration_id"] for c in configs if c["group"] == "multifactor"
        )
        row = {
            "frame_level": {"acc": 0.75, "auc": 0.9},
            "video_level": {"acc": 0.8, "auc": 0.95},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "coc").mkdir()
            (root / "coc/metrics.json").write_text(
                json.dumps({cid: row, "digital_d1": row})
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/summarize.py"),
                    "--results",
                    str(root),
                    "--output",
                    str(root / "summary"),
                ],
                check=True,
                capture_output=True,
            )
            rows = json.loads((root / "summary/summary.json").read_text())
            self.assertTrue(all(not r["complete"] for r in rows))
            self.assertEqual(
                next(r for r in rows if r["group"] == "recaptured")["configurations"], 1
            )
            self.assertEqual(
                next(r for r in rows if r["group"] == "digital")["configurations"], 1
            )

    def test_command_help(self):
        for command in [
            "prepare_data",
            "evaluate",
            "train",
            "fine_tune",
            "summarize",
            "check_resources",
        ]:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / (command + ".py")), "--help"],
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
