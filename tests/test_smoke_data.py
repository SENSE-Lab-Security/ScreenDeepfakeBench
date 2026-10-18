import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("smoke", ROOT / "scripts/prepare_smoke_data.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def test_fixture_layout_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke"
            smoke.prepare(out)
            seen = set()
            for name, mode, count in [("smoke_train", "train", 128), ("smoke_val", "test", 20)]:
                obj = json.loads((out / "indexes" / f"{name}.json").read_text())[name]
                self.assertEqual(set(obj), {"real", "fake"})
                paths = [p for splits in obj.values() for v in splits[mode].values() for p in v["frames"]]
                self.assertEqual(len(paths), count)
                self.assertEqual(len(set(paths)), count)
                self.assertFalse(seen.intersection(paths))
                seen.update(paths)
                for p in paths:
                    with Image.open(out / p) as image:
                        self.assertEqual(image.size, (256, 256))
                        self.assertEqual(image.mode, "RGB")
            with self.assertRaises(FileExistsError):
                smoke.prepare(out)

    def test_custom_evaluation_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "smoke_val.json").write_text("{}")
            for name in ("imagenet_xception.pth", "coc.pth"):
                (root / name).touch()
            command = [sys.executable, str(ROOT / "scripts/evaluate.py"),
                       "--data-root", tmp, "--index-root", tmp, "--weights-dir", tmp,
                       "--model", "coc", "--dry-run", "--dataset"]
            valid = subprocess.run(command + ["smoke_val"], capture_output=True, text=True)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            invalid = subprocess.run(command + ["../smoke_val"], capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("without directory components", invalid.stderr)

            train = subprocess.run(
                [sys.executable, str(ROOT / "scripts/train.py"), "--model", "coc",
                 "--data-root", tmp, "--index-root", tmp, "--weights-dir", tmp,
                 "--train-dataset", "smoke_val", "--val-dataset", "smoke_val",
                 "--output", str(root / "run")],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                capture_output=True, text=True,
            )
            self.assertNotEqual(train.returncode, 0)
            self.assertIn("Training requires an NVIDIA CUDA GPU", train.stderr)
            self.assertNotIn("Traceback", train.stderr)


if __name__ == "__main__":
    unittest.main()
