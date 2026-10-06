"""Reconstruct the fixed Celeb-DF training selection without changing its split."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {"Celeb-real": "real", "YouTube-real": "real", "Celeb-synthesis": "fake"}
EXPECTED = {"Celeb-real": (555, 17609), "YouTube-real": (283, 8983),
            "Celeb-synthesis": (5589, 178177)}


def unique_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def read_selection(index):
    obj = json.loads(Path(index).read_text(), object_pairs_hook=unique_keys)
    if set(obj) != {"celebdf_train"} or set(obj["celebdf_train"]) != {"real", "fake"}:
        raise ValueError("Expected the released celebdf_train real/fake index")
    jobs, seen, video_ids = [], set(), set()
    for label, splits in obj["celebdf_train"].items():
        if ("train" not in splits or set(splits) - {"train", "val", "test"}
                or any(splits.get(s) for s in ("val", "test"))):
            raise ValueError("Only the train split may contain training-index entries")
        for video, item in splits["train"].items():
            frames = item["frames"]
            if item["label"] != label or not 1 <= len(frames) <= 32:
                raise ValueError(f"Invalid label/frame count: {video}")
            category, indices = None, []
            for name in frames:
                path = PurePosixPath(name)
                parts = path.parts
                if (path.is_absolute() or len(parts) != 5 or ".." in parts
                        or "\\" in name or path.as_posix() != name
                        or parts[0] != "Celeb-DF-v2-clean" or parts[2] != "frames"
                        or parts[3] != video or CATEGORIES.get(parts[1]) != label
                        or path.suffix != ".png" or not path.stem.isascii()
                        or not path.stem.isdigit() or name in seen):
                    raise ValueError(f"Invalid or duplicate training path: {name}")
                if category is not None and category != parts[1]:
                    raise ValueError(f"Mixed source categories: {video}")
                category = parts[1]
                number = int(path.stem)
                if path.name != f"{number:03d}.png" or number in indices:
                    raise ValueError(f"Noncanonical or repeated frame number: {name}")
                indices.append(number)
                seen.add(name)
            if video in video_ids:
                raise ValueError(f"Ambiguous video ID: {video}")
            video_ids.add(video)
            jobs.append({"category": category, "video_id": video, "label": label,
                         "raw_video": f"{category}/{video}.mp4",
                         "frame_indices": indices, "output_paths": frames})
    return jobs


def validate_paper_selection(jobs, index_dir):
    counts = {c: (sum(j["category"] == c for j in jobs),
                  sum(len(j["output_paths"]) for j in jobs if j["category"] == c))
              for c in CATEGORIES}
    if counts != EXPECTED:
        raise ValueError(f"Selection differs from the released recipe: {counts}")
    train_ids = {j["video_id"] for j in jobs}
    controls = {}
    for name, expected_videos, expected_frames in [("digital_d1", 100, 500),
                                                   ("digital_d2", 20, 100)]:
        obj = json.loads((index_dir / f"{name}.json").read_text(), object_pairs_hook=unique_keys)[name]
        ids, paths = set(), set()
        for splits in obj.values():
            for item in splits["test"].values():
                for frame in item["frames"]:
                    ids.add(PurePosixPath(frame).parent.name)
                    paths.add(frame)
        if len(ids) != expected_videos or len(paths) != expected_frames:
            raise ValueError(f"Unexpected digital-control coverage: {name}")
        overlap = train_ids & ids
        if overlap:
            raise ValueError(f"Training/digital source-video overlap: {sorted(overlap)}")
        controls[name] = paths
    if not controls["digital_d2"] <= controls["digital_d1"]:
        raise ValueError("D2 should be a subset of D1")
    return counts


def resolve_video(root, job):
    category, video = job["category"], job["video_id"]
    candidates = [root / category / f"{video}.mp4",
                  root / category / "videos" / f"{video}.mp4"]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one source video, found {len(found)}: {candidates}")
    return found[0]


def valid_image(path):
    import cv2
    image = cv2.imread(str(path)) if path.is_file() else None
    return image is not None and image.shape == (256, 256, 3)


def atomic_copy(source, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temp)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


def init_worker(predictor):
    import cv2
    cv2.setNumThreads(1)
    global DETECTOR, PREDICTOR
    if predictor:
        import dlib
        DETECTOR = dlib.get_frontal_face_detector()
        PREDICTOR = dlib.shape_predictor(predictor)


def process_video(task):
    job, mode, source_root, output_root, overwrite = task
    source_root, output_root = Path(source_root), Path(output_root)
    result = {"video_id": job["video_id"], "written": 0, "existing": 0, "failures": []}
    pending = {}
    for number, name in zip(job["frame_indices"], job["output_paths"]):
        output = output_root / name
        if not overwrite and valid_image(output):
            result["existing"] += 1
        elif mode == "verify":
            result["failures"].append({"path": name, "reason": "missing or invalid 256x256 image"})
        else:
            pending[number] = name
    if not pending:
        return result
    if mode == "frames":
        for name in pending.values():
            source = source_root.joinpath(*PurePosixPath(name).parts[1:])
            if not valid_image(source):
                result["failures"].append({"path": name, "reason": f"missing or invalid source: {source}"})
                continue
            atomic_copy(source, output_root / name)
            result["written"] += 1
        return result

    import cv2
    from dlib_alignment import extract_aligned_face_dlib
    cap = None
    try:
        source = resolve_video(source_root, job)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        # Sequential decoding follows the upstream implementation; indices are zero-based.
        for number in range(max(pending) + 1):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Video decoding stopped at frame {number}")
            if number not in pending:
                continue
            name = pending.pop(number)
            try:
                face, landmarks, _ = extract_aligned_face_dlib(DETECTOR, PREDICTOR, image)
                if face is None or landmarks is None:
                    raise ValueError("dlib face/aligned-face detection failed")
                if len(landmarks) != 81:
                    raise ValueError("Expected the DeepfakeBench 81-point predictor")
                ok, encoded = cv2.imencode(".png", face)
                if not ok:
                    raise ValueError("PNG encoding failed")
                output = output_root / name
                output.parent.mkdir(parents=True, exist_ok=True)
                temp = output.with_name(output.name + f".{os.getpid()}.tmp")
                try:
                    temp.write_bytes(encoded.tobytes())
                    temp.replace(output)
                finally:
                    temp.unlink(missing_ok=True)
                result["written"] += 1
            except Exception as exc:
                result["failures"].append({"path": name, "reason": str(exc)})
    except Exception as exc:
        result["failures"].extend({"path": name, "reason": str(exc)} for name in pending.values())
    finally:
        if cap is not None:
            cap.release()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["audit", "frames", "videos", "verify"], required=True)
    parser.add_argument("--index", type=Path, default=ROOT / "data/training_data/indexes/celebdf_train.json")
    parser.add_argument("--source-root", type=Path, help="Celeb-DF-v2 root containing the three source categories")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/training_data")
    parser.add_argument("--predictor", type=Path, help="DeepfakeBench's 81-point dlib predictor (videos mode only)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-videos", type=int, help="Smoke test only; never reports a complete training set")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/training_preparation.json")
    parser.add_argument("--manifest", type=Path, help="Optionally export the selected raw video paths and frame indices")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit_videos is not None and args.limit_videos < 1):
        parser.error("workers and limit-videos must be positive")
    if args.mode in {"videos", "frames"} and not args.source_root:
        parser.error("--source-root is required for frames/videos mode")
    if args.mode in {"videos", "frames"} and not args.source_root.is_dir():
        parser.error("--source-root must be an existing directory")
    if args.mode == "videos" and (args.predictor is None or not args.predictor.is_file()):
        parser.error("videos mode requires --predictor pointing to the 81-point .dat file")
    if args.mode == "verify" and args.overwrite:
        parser.error("--overwrite is not meaningful in verify mode")
    jobs = read_selection(args.index)
    counts = validate_paper_selection(jobs, args.index.parent)
    print("PASS: 6,427 training videos, 204,769 frames; no D1/D2 source-video overlap.", flush=True)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({"frame_index_base": 0, "videos": jobs}, indent=2) + "\n")
    report = {"mode": args.mode, "index": str(args.index.resolve()), "category_counts": counts,
              "scope": "limited_smoke_test" if args.limit_videos else "full", "complete": False}
    report["environment"] = {"python": sys.version.split()[0]}
    for package in ("numpy", "opencv-python-headless", "opencv-python", "dlib", "scikit-image", "imutils"):
        try:
            report["environment"][package] = version(package)
        except PackageNotFoundError:
            pass
    report["source_root"] = str(args.source_root.resolve()) if args.source_root else None
    report["predictor"] = str(args.predictor.resolve()) if args.predictor else None
    if args.mode == "audit":
        report["index_valid"] = True
    else:
        selected = jobs[:args.limit_videos] if args.limit_videos else jobs
        predictor = str(args.predictor.resolve()) if args.mode == "videos" else None
        tasks = [(j, args.mode, str(args.source_root or "."), str(args.output_root), args.overwrite)
                 for j in selected]
        results = []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                 initargs=(predictor,)) as pool:
            for i, result in enumerate(pool.map(process_video, tasks), 1):
                results.append(result)
                if i % 50 == 0 or i == len(tasks):
                    print(f"Processed {i}/{len(tasks)} videos", flush=True)
        failures = [f for r in results for f in r["failures"]]
        totals = Counter()
        for result in results:
            totals.update({k: result[k] for k in ("written", "existing")})
        report.update(totals)
        report["failures"] = failures
        report["processed_videos"] = len(selected)
        report["complete"] = not failures and not args.limit_videos
        if report["complete"]:
            destination = args.output_root / "indexes/celebdf_train.json"
            if destination.resolve() != args.index.resolve():
                atomic_copy(args.index, destination)
        report["output_valid_for_scope"] = not failures
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.report}")
    if report.get("failures"):
        print(f"INCOMPLETE: {len(report['failures'])} missing/failed frames. Index unchanged.", file=sys.stderr)
        return 1
    if args.mode != "audit":
        print("PASS: complete indexed training set." if report["complete"] else "PASS: limited smoke test only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
