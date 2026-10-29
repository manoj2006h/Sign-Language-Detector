# Sign Language Detection with DETR

A small DETR-style object detector for three sign-language classes (`hello`,
`iloveyou`, `thankyou`), with training, held-out evaluation, and webcam
inference.

This is a DETR implementation, not a replacement architecture. The ResNet-50
backbone is initialized from ImageNet weights by default during training; the
transformer encoder/decoder and prediction heads are trained on the sign
dataset. The repository does not claim "full DETR trained from scratch".

## Architecture and preprocessing

- ResNet-50 backbone, 1x1 projection to 256 channels.
- One transformer encoder layer and one decoder layer, 8 attention heads.
- 25 learned object queries.
- Linear class head (`num_classes + 1`, including no-object) and linear box
  head.
- Images are converted to RGB, resized, normalized with ImageNet mean/std, and
  passed as `float32` tensors shaped `(B, 3, 224, 224)`.
- Training uses random crop/flip/color-jitter augmentation. Evaluation and
  webcam inference use the deterministic resize pipeline shared in
  `src/data.py`.
- Predicted boxes are normalized `(center_x, center_y, width, height)` and are
  converted to clipped pixel-space `(x1, y1, x2, y2)` using the **actual**
  camera frame dimensions.

## Setup

Requirements:

- Python 3.13 (declared in `.python-version`).
- [uv](https://docs.astral.sh/uv/) for the locked environment.
- A CUDA GPU is optional. CPU training/inference works but is slower.
- A webcam only for `realtime.py` or `collect_images.py`.

```bash
git clone https://github.com/Shreyas-a-j/Sign-Language-Detector.git
cd Sign-Language-Detector
uv sync
```

`uv sync` installs the dependencies already pinned in `uv.lock`. It does not
download the sign-language task checkpoint.

## Dataset

The loader expects this structure:

```text
data/
  train/
    images/
      sample.jpg
    labels/
      sample.txt
  test/
    images/
      sample.jpg
    labels/
      sample.txt
```

Image and label files must share the same stem. Supported image extensions are
`.jpg`, `.jpeg`, `.png`, and `.bmp` (case-insensitive).

Each non-blank annotation line is YOLO format:

```text
class_id center_x center_y width height
```

Coordinates are normalized to `[0, 1]`, and class IDs must exist in
`src/config.json`. Example:

```text
0 0.6997 0.3748 0.2242 0.4529
```

An empty `.txt` file is a valid no-object image. Every image must have a
matching `.txt` label file; an image without a matching annotation is reported
as an error instead of being silently skipped. Malformed lines, unknown
classes, missing images, non-positive sizes, and out-of-bounds boxes are
reported with the file and line number. Validate the dataset non-destructively
with:

```bash
uv run src/validate_dataset.py
```

To validate specific splits, repeat `--data-dir`:

```bash
uv run src/validate_dataset.py \
  --data-dir data/train \
  --data-dir data/test
```

The validator only reads the dataset; it never rewrites or deletes files.

`src/train.py` holds out a deterministic validation subset from `data/train`
(10% by default) and keeps `data/test` completely separate. The test split is
never used to select a checkpoint.

## Classes

Class names and colors live in `src/config.json`:

```json
{
  "classes": ["hello", "iloveyou", "thankyou"],
  "colors": [[41, 171, 202], [240, 172, 95], [131, 193, 103]]
}
```

Every checkpoint records the class names and number of classes. Loading a
checkpoint that does not match the current config fails with a clear error.

## Training

```bash
uv run src/train.py --epochs 100 --batch-size 4 --device auto
```

Useful options:

```text
--train-dir data/train          dataset split to train on
--checkpoint-dir checkpoints    output directory
--val-fraction 0.1              validation holdout from the training split
--device auto|cpu|cuda          explicit device selection
--no-pretrained-backbone        do not use/download ImageNet ResNet-50 weights
--init-checkpoint PATH          initialize from a compatible task checkpoint
--resume PATH                   resume model + optimizer + scheduler + epoch
--allow-unsafe-load             trusted legacy checkpoints only
```

Training writes:

- `checkpoints/last.pt` after every epoch, containing model, optimizer,
  scheduler, epoch, history, and validation metrics. Use it with `--resume`.
- `checkpoints/best.pt` when validation loss improves. This is the recommended
  checkpoint for evaluation and inference.

If `--val-fraction 0` is used, no validation split is created and no
`best.pt` is selected; `last.pt` is still written. This is intended only for
single-sample smoke runs.

## Evaluation

### Detection metrics (mAP, precision, recall)

```bash
uv run src/evaluate.py \
  --checkpoint checkpoints/best.pt \
  --test-dir data/test \
  --device cpu \
  --name baseline
```

This writes:

- `reports/baseline_metrics.json` — mAP@0.5, mAP@0.5:0.95, per-class AP,
  precision/recall at the documented confidence and IoU thresholds, GT/prediction
  counts, and FP/FN totals.
- `reports/error_analysis.json` — per-image matching detail, confidence
  histogram, and false positive/false negative records.
- `reports/error_analysis/*.jpg` — annotated overlays (green = GT, blue = TP,
  red = FP, orange = FN), capped by `--max-visualizations`.
- `reports/baseline_summary.md` — human-readable summary.

Metric conventions:

- Boxes are normalized xyxy. Predictions below `--score-threshold` (default
  `0.01`) are excluded from AP; AP uses 101-point COCO-style interpolation.
- mAP@0.5:0.95 averages AP over IoU 0.50, 0.55, ..., 0.95.
- Classes with no ground-truth boxes have *undefined* AP (`null`), not zero.
- Reported precision/recall use `--confidence` (default `0.5`) and `--iou`
  (default `0.5`). Metrics aggregate over the whole dataset, never per batch.
- Classification accuracy is not reported as a substitute for detection mAP.

### Loss-only smoke evaluation

```bash
uv run src/test.py --checkpoint checkpoints/best.pt --device auto
```

`test.py` runs every test batch, including incomplete final batches, and
reports held-out test loss and the number of detections above the threshold.
Use `src/evaluate.py` for detection metrics.

Checkpoint selection is explicit:

- Default for `test.py` and `realtime.py`: `checkpoints/best.pt`.
- Override with `--checkpoint PATH`.
- Add `--allow-unsafe-load` only for a trusted legacy checkpoint.
- Missing, corrupt, LFS-pointer, class-mismatched, or architecture-mismatched
  checkpoints are hard errors. There is no fallback to random or untrained
  weights.

### About `pretrained/4426_model.pt`

`pretrained/4426_model.pt` is tracked through Git LFS. In a checkout without
the LFS object, it is a 134-byte pointer, not a usable model. To fetch the
108 MB object:

```bash
git lfs install
git lfs pull
```

It is optional initialization weights, not the output of this training
pipeline. The pipeline never loads it automatically.

## Realtime webcam inference

```bash
uv run src/realtime.py \
  --checkpoint checkpoints/best.pt \
  --camera 0 \
  --device auto
```

Press `q` to quit. Use `--threshold 0.8` to change the confidence threshold,
and `--max-frames 30 --no-display` for a headless smoke test. If the camera
cannot be opened or the first frame is invalid, the script exits with an
actionable error instead of drawing or silently continuing.

The webcam path explicitly converts OpenCV BGR frames to RGB and uses the same
normalization as evaluation. Boxes are scaled with the actual frame
width/height, not a hard-coded 1920x1080.

## Image collection and labeling

```bash
uv run src/utils/collect_images.py \
  --output data/train/images \
  --camera 0 \
  --num-images 30
```

This only captures unlabeled images. Create matching `.txt` annotations in
`data/train/labels`, then run the validator. The collector defaults to
`data/train/images`; it no longer writes unlabeled files into the dataset root.

Label Studio is an optional dependency in the lockfile:

```bash
uv run label-studio
```

Export bounding boxes in YOLO format.

## Dataset audit and Amrita SLR integration

Audit the internal splits, split leakage, and the external archive:

```bash
uv run src/audit_datasets.py
```

Outputs:

- `reports/dataset_audit.json`
- `reports/dataset_compatibility.json`
- `reports/split_integrity.json`
- `reports/PHASE2_DATASET_AUDIT.md`

The audit detects byte-identical duplicates, perceptual near-duplicates, and
recording-group keys inferred from filename UUIDs. It also reports when signer
metadata is absent, which means person-independent generalization cannot be
established.

### Amrita SLR (video-level labels only)

The archive is expected at `dataset/Amrita SLR- Real Time ISL Dataset.zip` or
`dataset_zip/Amrita SLR- Real Time ISL Dataset.zip`. Inspect it without
extracting:

```bash
uv run src/prepare_amrita.py --inspect-only
```

Extract safely (the ZIP contains a nested ZIP), then build a manifest:

```bash
uv run src/prepare_amrita.py
```

Reuse an existing extraction without copying 1.5 GB again:

```bash
uv run src/prepare_amrita.py --manifest-only
```

Optional single-frame sampling for qualitative inspection:

```bash
uv run src/prepare_amrita.py --manifest-only \
  --frames-dir reports/amrita_frames \
  --frame-classes hello thankyou
```

Extraction rejects path traversal, absolute paths, drive paths, symlinks, and
non-empty destinations unless `--overwrite` is explicit. The original archive
is never modified. Extracted data and the ZIP are Git-ignored.

Important limits of the Amrita archive:

- 978 videos in 49 class folders, with video-level labels only.
- No bounding boxes, no temporal annotations, and no train/test split.
- No sign-definition documentation, license, or citation inside the archive.
- `hello` and `thankyou` match project glosses but remain
  `candidate_unverified`; `iloveyou` is absent.
- No class is mapped from English gloss alone, and no bounding boxes are
  invented. The corpus is therefore **not** used for DETR training or
  detection evaluation in Phase 2.

A mapping is only accepted when an explicit sign-definition evidence file is
provided:

```bash
uv run src/prepare_amrita.py --manifest-only --evidence path/to/evidence.json
```

See `src/utils/compat.py` for the evidence schema.

## Runtime benchmarking

```bash
uv run src/benchmark.py --device cpu --runs 20 --warmup 3 --batch-size 1
```

Writes `reports/benchmark.json` and `reports/benchmark_summary.md`. The
benchmark warms the model, separates preprocessing/model/postprocessing
latency, reports median and p95, FPS, and CPU RSS (plus CUDA peak memory when
available). Camera capture and display are excluded. CUDA stages synchronize
before timestamps. No numbers are extrapolated.

## Tests and checks

```bash
uv run python -m unittest discover -s tests -t .
uv run python -m compileall -q src tests
```

The tests use synthetic images, temporary datasets, and tiny checkpoints. They
do not require a webcam, internet access, the LFS checkpoint, or pretrained
weight downloads.

## CPU/GPU notes

- `--device auto` selects CUDA when available, otherwise CPU.
- `--device cuda` fails clearly if CUDA is unavailable; it never silently
  falls back to CPU.
- Full training on CPU is possible but slow. No FPS or benchmark numbers are
  claimed here.
- Test/realtime inference construct the model with `backbone_weights=None`
  because the full task checkpoint supplies the backbone. Training downloads
  ImageNet ResNet-50 weights unless `--no-pretrained-backbone` is used.

## Known limitations and troubleshooting

- The included dataset is small (100 training images, 15 test images). Do not
  interpret test loss or detection counts as real-world recognition accuracy.
- `checkpoints/best.pt` in this checkout is an early checkpoint (epoch 2). The
  Phase 2 baseline measured mAP@0.5 = 0.0919 and mAP@0.5:0.95 = 0.0232 on
  `data/test`; no detection reached the 0.5 confidence threshold. See
  `reports/baseline_metrics.json` for the recorded run and exact hashes.
- Split integrity is **confirmed leaked**: train and test share a recording
  group, and 110 near-duplicate pairs cross the split boundary. These numbers
  are not person-independent generalization. See `reports/split_integrity.json`.
- The Amrita SLR archive has video-level labels only. It has no boxes, no
  temporal annotations, and no verified sign definitions, so it is excluded
  from DETR training and detection evaluation.
- There is no temporal smoothing or sequence model; webcam predictions are
  per-frame.
- New classes require matching annotations, `src/config.json` changes, and a
  newly trained checkpoint. Old checkpoints with a different class list are
  rejected.
- **Git LFS pointer error:** run `git lfs install && git lfs pull`, or train a
  local model and pass `--checkpoint checkpoints/best.pt`.
- **Class count mismatch:** update `src/config.json` or use a checkpoint
  trained with the same classes.
- **Camera not opening:** try another `--camera` index, check permissions, and
  close other applications using the camera.
- **CUDA requested but unavailable:** use `--device cpu` or install a CUDA
  build of PyTorch.
- **No detections:** lower `--threshold`, verify the checkpoint/class order,
  and confirm the input images use the same sign vocabulary.
- **Image without annotation:** create a matching `.txt` file in the `labels`
  directory, even if it is empty for an image with no objects, then rerun
  `uv run src/validate_dataset.py`.
- **Offline setup:** `uv sync` needs the locked packages. Training with
  `--no-pretrained-backbone` avoids the ImageNet download; a local checkpoint
  avoids all pretrained downloads.

## Resources

- [DETR Colab walkthrough](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb)
- Phase 1 baseline report: [`docs/PHASE1_BASELINE.md`](docs/PHASE1_BASELINE.md)

## Who, When, Why?

👨🏾‍💻 Author: Manoj H

📅 Version: 1.x

📜 License: This project is licensed under the Manoj
