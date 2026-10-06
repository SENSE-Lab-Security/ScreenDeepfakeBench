# Preparing the Celeb-DF Training Selection

The paper's training selection is supplied in
`ScreenDeepfakeBench_training_support.zip`, under
`training_data/indexes/celebdf_train.json`. The preprocessed training-image archive
is approximately **12.15 GB** and is not hosted on Zenodo due to its size.
To request these prepared training images, contact **Shuhao Zhang** at
[szhang515@connect.hkust-gz.edu.cn](mailto:szhang515@connect.hkust-gz.edu.cn).
Alternatively, prepare the same indexed selection locally using either workflow below.
Evaluation of the released paper weights needs only the benchmark inputs and
model resources, not Celeb-DF training data.

## Fixed Selection

| Source category | Label | Videos | Indexed images |
| --- | --- | ---: | ---: |
| Celeb-real | real | 555 | 17,609 |
| YouTube-real | real | 283 | 8,983 |
| Celeb-synthesis | fake | 5,589 | 178,177 |
| Total | | 6,427 | 204,769 |

The index specifies the paper's training selection, with unique frame paths and
source videos separate from D1/D2. It uses a video-level split rather than the
official Celeb-DF split; identities may appear across sets. Use the listed frames:
6,007 videos have 32 indexed images and 420 have fewer.

From the repository root:

```bash
unzip ScreenDeepfakeBench_training_support.zip -d data
python scripts/prepare_training_data.py --mode audit \
  --manifest outputs/celebdf_train_manifest.json
```

The optional manifest lists each selected original video (for example,
`Celeb-real/id0_0000.mp4`), its label, zero-based frame indices, and destination
paths. The JSON is also the training loader's index. Audit checks the
index structure, class/category agreement, coverage, unique paths, and D1/D2
source-video exclusion. Image checks are available through `--mode verify` below.

## Option A: Original Videos

Obtain Celeb-DF-v2 from its [official authors](https://github.com/yuezunli/celeb-deepfakeforensics)
under their terms. Point `--source-root` at a directory with:

```text
Celeb-DF-v2/
  Celeb-real/*.mp4
  YouTube-real/*.mp4
  Celeb-synthesis/*.mp4
```

The alternative `<category>/videos/*.mp4` layout is also supported. If both
locations contain the same selected video, keep one source copy to resolve the
ambiguity. Retain the original video basenames.

Install the optional CPU preprocessing dependencies. Building dlib may require
CMake and a C++ compiler; a CUDA GPU is not required for its HOG detector.

```bash
pip install -r requirements-preprocessing.txt
mkdir -p checkpoints
curl --fail --location --retry 5 \
  https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat \
  --output checkpoints/shape_predictor_81_face_landmarks.dat
```

This is the 81-point predictor linked in the
[DeepfakeBench preprocessing instructions](https://github.com/SCLBD/DeepfakeBench#3-preprocessing-optional).
Use this 81-point predictor with the dependencies in `requirements-preprocessing.txt`.
The predictor's terms are separate from the code license.

Start with a small run:

```bash
python scripts/prepare_training_data.py --mode videos \
  --source-root /path/to/Celeb-DF-v2 \
  --predictor checkpoints/shape_predictor_81_face_landmarks.dat \
  --workers 2 --limit-videos 3 --report outputs/training_smoke.json
```

Then process the complete selection:

```bash
python scripts/prepare_training_data.py --mode videos \
  --source-root /path/to/Celeb-DF-v2 \
  --predictor checkpoints/shape_predictor_81_face_landmarks.dat \
  --workers 4
```

The script decodes videos sequentially and processes the frame indices in the
JSON. The upstream sampling rule is
`numpy.linspace(0, frame_count - 1, 32, dtype=int)`; the released JSON specifies
the frames retained for training.

For each selected frame, the DeepfakeBench routine uses dlib's frontal
face detector with one upsampling step, the largest detected face, landmarks
37/44/30/49/55, a similarity transform with scale 1.3, and 256x256 output. Its
second detection operates on the aligned BGR crop. Failed detections are listed
in the preparation report for resolution before training. The training pipeline
requires face crops; the recaptured evaluation inputs include the full-frame
fallbacks described in the dataset documentation.

## Option B: Existing Preprocessed Faces

If you already have DeepfakeBench's RGB-format Celeb-DF-v2 faces, select the
indexed files directly from that tree, without repeating face alignment.

```text
Celeb-DF-v2/
  Celeb-real/frames/<video>/<frame>.png
  YouTube-real/frames/<video>/<frame>.png
  Celeb-synthesis/frames/<video>/<frame>.png
```

```bash
python scripts/prepare_training_data.py --mode frames \
  --source-root /path/to/preprocessed/Celeb-DF-v2 --workers 4
```

This mode copies the selected files without re-encoding, under
`data/training_data/Celeb-DF-v2-clean/`. It reports missing or non-256x256 images.
Obtain compatible preprocessed faces through
the [DeepfakeBench data instructions](https://github.com/SCLBD/DeepfakeBench#2-download-data)
and the original data-provider terms.

## Verify and Resume

```bash
python scripts/prepare_training_data.py --mode verify --workers 4
python scripts/check_resources.py --training-root data/training_data
```

The preparation report lists failed paths and reasons, written/existing counts,
and whether the full selection is complete. Limited runs are labelled as smoke
tests. Re-run the same command to retry missing/invalid
outputs; valid existing 256x256 images are skipped. Use `--overwrite` when changing
source data, predictor, or preprocessing environment, or use a new output root.

## Reproducibility Notes

Preprocessing follows the DeepfakeBench dlib pipeline and the released frame
selection. Reconstructed crops may vary with video decoding and dependency
versions; pixel-level equivalence to the original prepared images has not been
established. The original prepared images are available upon request.

For evaluation of the paper checkpoints, use the supplied digital-control and
recaptured inputs directly.
