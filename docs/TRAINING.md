# Training and Fine-Tuning

First [prepare the indexed training images](PREPROCESSING.md) from Celeb-DF-v2.
The training-support ZIP provides the fixed index and digital controls, not
the full training image collection. Evaluation of the paper weights does not
require this preprocessing step.

`scripts/train.py` starts from ImageNet Xception initialization. The
inclusive epoch loop has indices 0-10 for `--epochs 11`. It uses the paper's
optimizer, class-weighted sampling and detector-specific augmentation settings.

`scripts/fine_tune.py` starts from the named released detector checkpoint,
initializes a new optimizer, and defaults to five epochs at learning rate 2e-5.
It is not a resume command for an interrupted optimizer state. Pass
`--init-checkpoint path/to/model.pth` to start from another compatible detector.

The training loop validates on the two digital sets and retains the checkpoint
with the highest equal-weight mean AUC. D2 is a subset of D1, following the
paper's validation recipe. Use independently defined development and held-out
sets for new studies. The recaptured benchmark is reserved for evaluation.

## Checkpoints and Evaluation

The model output directory contains timestamped training logs and
`test/avg/ckpt_best.pth`; per-digital-set best checkpoints are also retained.
The terminal log prints the exact saved path. Evaluate the average-best weight
with `--weights`:

```bash
python scripts/evaluate.py --model coc --data-root data/CoC_Dataset_zenodo \
  --weights /path/to/new/test/avg/ckpt_best.pth \
  --save-predictions --output outputs/new_coc_eval
```

## Reproducible Execution

Add `--deterministic --seed 1024` to train or fine-tune. This explicitly controls
the sampler, data-loader and augmentation-worker streams and requests deterministic
GPU algorithms. Unsupported deterministic operations raise an error. Keep the
seed, library versions and hardware fixed when comparing runs; training results
can vary with these settings. Use the released checkpoints for the paper's
reported evaluation results.

## Custom Data

Use `--index-root`, `--train-dataset` and `--val-dataset` for independently prepared
splits. Each dataset has a JSON named after its ID. Entries use relative frame
paths and the labels `real` and `fake`:

```json
{
  "my_train": {
    "real": {"train": {"video_real": {"label": "real", "frames": ["real/video_real/000.png"]}}},
    "fake": {"train": {"video_fake": {"label": "fake", "frames": ["fake/video_fake/000.png"]}}}
  }
}
```

Validation indexes use `test` in place of `train`, following the underlying loader
convention. Both classes must be represented. The supplied configuration samples
up to 32 training frames or five validation frames per video; change these only
as an explicitly different experiment. Videos within each evaluation configuration
must have a consistent label and distinct parent directories.

For recaptured-data fine-tuning, define separate development and held-out sets
with source-content leakage controls. Never select the best epoch or threshold
using the final benchmark results.
