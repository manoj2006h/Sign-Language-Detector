# Phase 1 Baseline — Sign Language Detector

Recorded before implementation changes. Date: 2026-10-28.

## Repository state

- Branch: `main`, tracking `origin/main`.
- HEAD: `5a61320 Update author and license information in README`.
- Working tree: clean at baseline.
- No local test directory and no existing automated tests.
- Project Python requirement: `>=3.13` (`.python-version` is `3.13`).
- The baseline host had system Python 3.14.4, but no project virtual environment,
  `torch`, `torchvision`, `albumentations`, `scipy`, `torchinfo`, or OpenCV.
  Dependency installation was approved and performed after this baseline using
  the existing `uv.lock` (`uv sync`).

## Hardware

- No NVIDIA GPU detected (`nvidia-smi` unavailable).
- `torch.cuda` was not installable/available at baseline; CPU execution is the
  expected path on this machine.

## Dataset

- `data/train`: 100 images, 100 label files.
- `data/test`: 15 images, 15 label files.
- All images are 1920x1080 RGB JPEGs.
- Image stems and label stems match exactly in both splits.
- Label format is YOLO: `<class_id> <center_x> <center_y> <width> <height>`,
  with normalized coordinates.
- Classes in `src/config.json`: `hello` (0), `iloveyou` (1), `thankyou` (2).
- Observed label counts: class 0 = 39, class 1 = 37, class 2 = 39.
- All baseline annotations have five numeric fields, valid class IDs, positive
  widths/heights, and are within the normalized image bounds.
- There is no validation split. The current trainer uses the test split for
  per-epoch loss reporting, which leaks test information into model selection.

## Checkpoint

- `pretrained/4426_model.pt` is tracked through Git LFS (`.gitattributes`).
- The file in this checkout is 134 bytes and is an **LFS pointer**, not a
  PyTorch checkpoint:
  `version https://git-lfs.github.com/spec/v1`.
- The expected LFS object is 108,081,916 bytes; it is not present locally and
  Git LFS is not installed in this environment.
- Existing logs show `torch.load` failing under the PyTorch 2.6+ default
  `weights_only=True`, and `DETR.load_pretrained` logging the error but
  continuing. This means training/testing/realtime can silently run with
  random task heads and no trained weights.

## Model and preprocessing

- Architecture: torchvision ResNet-50 backbone, 1x1 projection to 256 channels,
  a 1-layer transformer encoder/decoder, 25 object queries, linear class and
  box heads (3 classes + no-object).
- `DETR.__init__` always requests ImageNet ResNet-50 weights, even for
  inference, so model construction requires a network/cache.
- Training/evaluation preprocessing: RGB PIL image, resize to 500x500, optional
  random 224x224 crop (training only), resize to 224x224, normalization with
  ImageNet statistics, tensor conversion.
- Realtime preprocessing: BGR camera frame resized directly to 224x224. It does
  not convert BGR to RGB, so channels differ from training.
- Realtime bounding boxes are rescaled using hard-coded `(1920, 1080)` rather
  than the actual camera frame dimensions.
- Test/realtime inference are not wrapped in `torch.inference_mode()`.
- There is no explicit CPU/CUDA selection.

## Reproducible issues found

1. The checked-in checkpoint is an LFS pointer and cannot be loaded.
2. Checkpoint loading failures are swallowed, leaving an untrained model that
   is still used for training/evaluation/realtime.
3. No checkpoint metadata/class/architecture validation; no best or resumable
   checkpoint format.
4. Hard-coded class count (`3`) in `train.py`, `test.py`, and `realtime.py`.
5. BGR camera frames are fed to an RGB-trained preprocessing pipeline.
6. Realtime boxes use hard-coded camera dimensions.
7. `torch.load` defaults and `map_location` are not handled explicitly.
8. Dataset transform failures are caught and the unnormalized NumPy image is
   returned, which breaks tensor shape/dtype assumptions downstream.
9. Annotation parsing assumes every line has a class and box, does not validate
   class IDs/bounds, and assumes `.jpg` images.
10. Empty annotations, empty datasets, and empty evaluation batches can cause
    confusing downstream failures or division by zero.
11. Training uses the test split as validation and saves only every 10 epochs;
    the final save may fail if `checkpoints/` does not exist, and no optimizer
    state is saved.
12. The image collector defaults to writing images directly into `data/test`
    rather than `data/test/images`, so collected images do not match the loader.
13. `get_classes()`/`get_colors()` return an error string instead of raising,
    which can be indexed or iterated as if it were a class list.
14. README has the upstream clone URL, claims training "from scratch", and does
    not document the LFS checkpoint limitation or validation split.

## Commands that were expected to work (baseline)

- `uv run src/train.py`
- `uv run src/test.py`
- `uv run src/realtime.py`

None could be considered reliable at baseline because the checkpoint is a
pointer and the failures are non-fatal. No accuracy, FPS, or benchmark numbers
are claimed in this phase.
