# Data Layout and Protocol

After extracting the downloaded archives:

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
videos across devices and conditions. They are not independent identity/content
splits. For new research, partition by source video (and identity when relevant)
and capture domain before creating train/validation/test sets; do not randomly
split near-duplicate frames across those sets.

The training archive provides the 204,769 indexed frames used by the included
training recipe. D1 and D2 share one physical `digital/` tree in the support
archive, avoiding duplicate copies. Their indexes select the corresponding
subsets. Source-media permissions and terms continue to apply.
