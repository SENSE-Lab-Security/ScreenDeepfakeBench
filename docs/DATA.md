# Data Layout and Protocol

After extracting the resource archives and preparing the training images:

```text
checkpoints/
  imagenet_xception.pth
  xception.pth
  ffd.pth
  spsl.pth
  coc.pth
data/
  CoC_Dataset_zenodo/
    data/benchmark_inputs/<configuration>/<real-or-fake>/<video>/<frame>
    metadata/
  training_data/
    Celeb-DF-v2-clean/
    digital/<real-or-fake>/<video>/<frame>
    indexes/
      celebdf_train.json
      digital_d1.json
      digital_d2.json
```

The recaptured dataset is sufficient for the headline four-model evaluation.
Training images and digital controls are separate resources, not extra recaptured
benchmark configurations. Do not include them in the 103-configuration mean.

The public configuration IDs encode device and capture conditions, for example:

```text
controlled_distance__vivo-v21__seewo__140cm
multifactor__iphone-17-pro__aoc__angle-0__occupancy-30
```

The code and reference results use those IDs directly. `metadata/indexes/`
contains 103 ordered evaluation indexes; `scripts/prepare_data.py` validates the
public CSV manifest and produces working indexes with relative paths. A moved
dataset needs only a different `--data-root`; no filename rewriting is required.

The paper's loader reads images with OpenCV and retains BGR channel order,
normalizes with mean/std 0.5, and resizes tensors to 299x299 during collation.
Those details are retained for the released weights. The configured augmentation
resolution is not by itself the final network input size.

D1 contains 500 digital frames from 100 source videos; D2 contains 100 frames
from 20 videos and is a subset of D1. Recaptured configurations reuse source
videos across devices and conditions. For new research, group samples by source
video (and identity when relevant) and capture domain when defining
train/validation/test sets, keeping related frames in the same split.

The training index selects 204,769 images from 6,427 source videos: 555 Celeb-real,
283 YouTube-real, and 5,589 Celeb-synthesis videos. The training source videos do
not overlap with D1/D2. The paper uses this fixed video-level selection rather
than the official Celeb-DF split; identities may appear across sets.
Download all four `Celeb-DF-v2-clean_part*.zip` archives from
[Zenodo](https://zenodo.org/records/23150212) and extract them into
`data/training_data/` using the [README commands](../README.md#training-and-fine-tuning).
The optional [preprocessing workflow](PREPROCESSING.md) also supports preparing
the selection from original videos or existing preprocessed faces.
The `Celeb-DF-v2-clean/` prefix and indexed image paths are preserved. Use the
listed frames for each video, including videos with fewer than 32 images.
D1 and D2 share one physical `digital/` tree in the support
archive, avoiding duplicate copies. Their indexes select the corresponding
subsets. Source-media permissions and terms continue to apply.

## Training Archives

The four archives collectively provide the training images used for CoC, before
training-time augmentation. They are balanced by file size, with all frames from
each source video kept together in one archive. Each image appears exactly once.
They do not represent separate experiments or training/validation splits.

| Archive | Celeb-real images | YouTube-real images | Celeb-synthesis images | Total images | Source videos |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Celeb-DF-v2-clean_part1.zip` | 3,979 | 2,195 | 45,011 | 51,185 | 1,607 |
| `Celeb-DF-v2-clean_part2.zip` | 3,923 | 2,219 | 45,048 | 51,190 | 1,607 |
| `Celeb-DF-v2-clean_part3.zip` | 4,959 | 2,386 | 43,858 | 51,203 | 1,607 |
| `Celeb-DF-v2-clean_part4.zip` | 4,748 | 2,183 | 44,260 | 51,191 | 1,606 |
| Total | 17,609 | 8,983 | 178,177 | 204,769 | 6,427 |

Celeb-real and YouTube-real are labeled real; Celeb-synthesis is labeled fake.
The images are 256x256 PNGs under
`Celeb-DF-v2-clean/<source>/frames/<video>/<frame>.png`. The fixed index selects
up to 32 images per training video. The two digital validation indexes and their
images are in `ScreenDeepfakeBench_training_resources.zip`, not these four archives.
