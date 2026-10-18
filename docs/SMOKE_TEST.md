# Small Workflow Check

This walkthrough uses 128 synthetic training images and 20 synthetic validation
images. It checks data loading, training, checkpoint saving/loading and evaluation
without downloading Celeb-DF or the recaptured dataset. The images are generated
patterns, not faces; their metrics are not detection results.

Install the repository dependencies and extract `ScreenDeepfakeBench_models.zip`
at the repository root. No training-support archive is needed for this check.

## Generate and Evaluate

```bash
python scripts/prepare_smoke_data.py --output data/smoke
python scripts/check_resources.py --load-models
python scripts/evaluate.py --model all --data-root data/smoke \
  --index-root data/smoke/indexes --dataset smoke_val \
  --device cpu --workers 0 --batch-size 4 --save-predictions \
  --output outputs/smoke_initial
```

Each model writes `metrics.json` and prediction files below `outputs/smoke_initial/`.
This CPU evaluation can also be run on macOS. On a CUDA machine, use `--device cuda`.

## Train, Save and Reload (CUDA)

Run one epoch through the actual training entry point:

```bash
python scripts/train.py --model coc --data-root data/smoke \
  --train-dataset smoke_train --val-dataset smoke_val \
  --epochs 1 --workers 0 --output outputs/smoke_train_coc
```

The log prints the saved checkpoint path. In a POSIX shell, find the average-best
checkpoint from this run and evaluate it:

```bash
WEIGHTS=$(find outputs/smoke_train_coc -path '*/test/avg/ckpt_best.pth' -type f)
python scripts/evaluate.py --model coc --weights "$WEIGHTS" \
  --data-root data/smoke --index-root data/smoke/indexes --dataset smoke_val \
  --workers 0 --save-predictions --output outputs/smoke_reloaded_coc
```

To test fine-tuning from the released CoC checkpoint:

```bash
python scripts/fine_tune.py --model coc --data-root data/smoke \
  --train-dataset smoke_train --val-dataset smoke_val \
  --epochs 1 --workers 0 --output outputs/smoke_finetune_coc
```

Repeat the checkpoint lookup and evaluation with `outputs/smoke_finetune_coc` and
a fresh evaluation output directory. The same commands accept `xception`, `ffd`
and `spsl` in place of `coc`; use separate output directories for every run.

Successful training should produce finite loss values and a saved checkpoint;
reloading it should produce scores for all 20 validation images. Accuracy on
these synthetic images is not an acceptance criterion. Do not pass these results
to the benchmark summary command, which accepts only the published configurations.

For the full experiment, use the [provided training data](../README.md#training-and-fine-tuning)
and evaluate on the recaptured benchmark. A successful workflow check alone does
not establish reproduction of the paper's accuracy.
