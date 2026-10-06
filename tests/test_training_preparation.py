"""Selection, copy/resume and raw-frame processing checks without private data."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_training_data as prep


def example():
    obj = {"celebdf_train": {"real": {"train": {}, "val": {}, "test": {}},
                            "fake": {"train": {}, "val": {}, "test": {}}}}
    for label, category, video in [("real", "Celeb-real", "id0_0000"),
                                    ("fake", "Celeb-synthesis", "id0_id1_0000")]:
        obj["celebdf_train"][label]["train"][video] = {
            "label": label,
            "frames": [f"Celeb-DF-v2-clean/{category}/frames/{video}/{i:03d}.png" for i in (0, 2)]}
    return obj


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps(example()))
        self.jobs = prep.read_selection(self.index)

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_uses_literal_indices(self):
        self.assertEqual(self.jobs[0]["raw_video"], "Celeb-real/id0_0000.mp4")
        self.assertEqual(self.jobs[0]["frame_indices"], [0, 2])

    def test_bad_indexes_rejected(self):
        for variant in ("duplicate", "traversal", "label", "category", "frame_number"):
            obj = copy.deepcopy(example())
            item = obj["celebdf_train"]["real"]["train"]["id0_0000"]
            if variant == "duplicate":
                item["frames"].append(item["frames"][0])
            elif variant == "traversal":
                item["frames"][0] = "../" + item["frames"][0]
            elif variant == "label":
                item["label"] = "fake"
            elif variant == "category":
                item["frames"][0] = item["frames"][0].replace("Celeb-real", "YouTube-real")
            else:
                item["frames"][0] = item["frames"][0].replace("000.png", "00.png")
            self.index.write_text(json.dumps(obj))
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                prep.read_selection(self.index)
        with self.assertRaises(ValueError):
            json.loads('{"a": 1, "a": 2}', object_pairs_hook=prep.unique_keys)

    def test_copy_resume_verify_and_missing(self):
        job = self.jobs[0]
        source, output = self.root / "source", self.root / "output"
        for name in job["output_paths"]:
            path = source.joinpath(*Path(name).parts[1:])
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.full((256, 256, 3), 23, dtype=np.uint8))
        task = (job, "frames", source, output, False)
        result = prep.process_video(task)
        self.assertEqual(result["written"], 2)
        for name in job["output_paths"]:
            self.assertEqual((output / name).read_bytes(), source.joinpath(*Path(name).parts[1:]).read_bytes())
        self.assertEqual(prep.process_video(task)["existing"], 2)
        missing = output / job["output_paths"][0]
        missing.write_bytes(b"broken")
        self.assertEqual(len(prep.process_video((job, "verify", source, output, False))["failures"]), 1)
        self.assertEqual(prep.process_video(task)["written"], 1)
        source.joinpath(*Path(job["output_paths"][0]).parts[1:]).unlink()
        missing.unlink()
        self.assertEqual(len(prep.process_video(task)["failures"]), 1)

    def test_raw_video_selection_no_fallback(self):
        job = self.jobs[0]
        source, output = self.root / "source", self.root / "output"
        path = source / job["raw_video"]
        path.parent.mkdir(parents=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (256, 256))
        self.assertTrue(writer.isOpened())
        for value in (10, 50, 100):
            writer.write(np.full((256, 256, 3), value, dtype=np.uint8))
        writer.release()
        cap = cv2.VideoCapture(str(path))
        decoded = [cap.read()[1] for _ in range(3)]
        cap.release()
        fake = types.ModuleType("dlib_alignment")
        fake.extract_aligned_face_dlib = lambda d, p, image: (image, np.zeros((81, 2)), None)
        prep.DETECTOR, prep.PREDICTOR = None, None
        with patch.dict(sys.modules, {"dlib_alignment": fake}):
            result = prep.process_video((job, "videos", source, output, False))
            self.assertFalse(result["failures"])
            for number, name in zip(job["frame_indices"], job["output_paths"]):
                np.testing.assert_array_equal(cv2.imread(str(output / name)), decoded[number])
            self.assertEqual(len(list(output.rglob("*.png"))), 2)
            fake.extract_aligned_face_dlib = lambda *a: (None, None, None)
            failed_root = self.root / "failed"
            result = prep.process_video((job, "videos", source, failed_root, False))
            self.assertEqual(len(result["failures"]), 2)
            self.assertFalse(list(failed_root.rglob("*.png")))

    def test_video_layout_ambiguity(self):
        job = self.jobs[0]
        first = self.root / job["raw_video"]
        first.parent.mkdir()
        first.touch()
        self.assertEqual(prep.resolve_video(self.root, job), first)
        second = first.parent / "videos" / first.name
        second.parent.mkdir()
        second.touch()
        with self.assertRaises(ValueError):
            prep.resolve_video(self.root, job)


if __name__ == "__main__":
    unittest.main()
