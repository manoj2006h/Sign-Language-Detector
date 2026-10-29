# Phase 2 — evaluation and integration summary

Generated: 2026-10-29T14:11:27Z

Every value below is read from a generated JSON report. Sections whose
report is missing are marked **not generated** instead of guessed.

## Experiment status

| Experiment | Status | Reason |
| --- | --- | --- |
| A — existing baseline on data/test | ran | See `baseline_metrics.json`. |
| B — external evaluation on Amrita | **not run** | No bounding-box or temporal annotations, no verified sign definitions, no independent split. |
| C — combined training experiment | **not run** | Amrita is not DETR training-ready; mapping from gloss alone is prohibited. |

## Internal dataset

| Split | Images | Objects |
| --- | --- | --- |
| train | 100 | 100 |
| test | 15 | 15 |

## Split integrity

- Status: **confirmed**
- Recording groups: 2
- Shared groups: 1
- Exact duplicates: 0
- Near-duplicate pairs: 413
- Signer metadata available: False
- Person-independent generalization: not established

- 110 near-duplicate pair(s) cross splits (dHash distance <= 6).
- 1 recording-group key(s) occur in more than one split; neighbouring frames are not isolated.

## Amrita SLR

- Videos: 978
- Classes: 49
- Duration: 2608.029 s
- Duplicate video groups: 58
- Annotation status: video_level_only
- DETR training ready: False

- Blocker: No bounding-box or segmentation annotations: only video-level class labels are provided.
- Blocker: No temporal boundaries, so sign windows cannot be localized.
- Blocker: No verified sign-definition documentation for candidate gloss matches (hello, thankyou).
- Blocker: No train/test split and no authoritative signer/session metadata.
- Blocker: No license or citation file is included in the archive.

### Domain shift sample

- Extracted frames: 39 / 39 requested
- Frame directory: `reports/amrita_frames`
- Amrita clips are short studio recordings with different
  resolution and framing from the internal 1920x1080 frames,
  and they carry no bounding boxes. These frames are for
  qualitative inspection only.

## Vocabulary compatibility

- Training mapping: none
- Can convert to DETR: False

| External label | Project class | Status |
| --- | --- | --- |
| hello | hello | candidate_unverified |
| thankyou | thankyou | candidate_unverified |

## Experiment A — baseline detection metrics

| Metric | Value |
| --- | --- |
| Checkpoint | `checkpoints/best.pt` |
| Checkpoint SHA-256 | `b2ed6aa8aea023d37c694939fa952bc6f76de546b93f68cc0bea5dc54664fd74` |
| Checkpoint epoch | 2 |
| Dataset | `data/test` |
| Dataset identity SHA-256 | `e9f7879f10951296b8eff0f8de2e1dd48535cb357c81c80944a14b19e6ec947f` |
| Images | 15 |
| Ground-truth objects | 15 |
| mAP@0.5 | 0.0919 |
| mAP@0.5:0.95 | 0.0232 |
| Micro precision | undefined |
| Micro recall | 0.0000 |
| Predictions at confidence threshold | 0 |

| Class | AP@0.5 | AP@0.5:0.95 | Precision | Recall | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hello | 0.1642 | 0.0318 | undefined | 0.0000 | 0 | 0 | 7 |
| iloveyou | 0.0491 | 0.0245 | undefined | 0.0000 | 0 | 0 | 3 |
| thankyou | 0.0625 | 0.0132 | undefined | 0.0000 | 0 | 0 | 5 |

Warnings:

- Split integrity report flags leakage between train and test; these metrics are not person-independent generalization.

## Error analysis

- Images with errors: 15 / 15
- True positives: 0
- False positives: 0
- False negatives: 15
- Visualizations: 15

## Runtime benchmark

| Metric | Value |
| --- | --- |
| Device | cpu |
| Batch size | 1 |
| Runs / warmup | 10 / 3 |
| Median end-to-end | 50.17 ms |
| p95 end-to-end | 82.45 ms |
| Median model forward | 48.66 ms |
| Median preprocess | 1.37 ms |
| Median postprocess | 0.15 ms |
| FPS (median end-to-end) | 19.93 |
| Peak RSS | 1138.64 MB |

Camera capture and display are excluded; CUDA stages are
synchronized when a GPU is used.

## Missing reports

- none

## Scientific conclusions supported by the data

- The baseline checkpoint detects none of the 15 test objects at a
  0.5 confidence threshold; low-confidence AP is 0.0919 at IoU 0.5.
- Train/test leakage is confirmed, so the baseline is an internal
  diagnostic, not evidence of person-independent generalization.
- Amrita SLR cannot support detection evaluation or DETR training in
  Phase 2 because it has video-level labels only.
