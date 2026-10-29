# Detection evaluation — baseline

| Field | Value |
| --- | --- |
| Experiment | baseline |
| Checkpoint | `checkpoints/best.pt` |
| Checkpoint SHA-256 | `b2ed6aa8aea023d37c694939fa952bc6f76de546b93f68cc0bea5dc54664fd74` |
| Test directory | `data/test` |
| Images | 15 |
| Ground-truth objects | 15 |
| mAP@0.5 | 0.0919 |
| mAP@0.5:0.95 | 0.0232 |
| Micro precision | undefined |
| Micro recall | 0.0000 |
| False positives | 0 |
| False negatives | 15 |

Thresholds: confidence 0.5, IoU 0.5; AP input score threshold 0.01.

## Per-class metrics

| Class | AP@0.5 | AP@0.5:0.95 | Precision | Recall | GT | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hello | 0.1642 | 0.0318 | undefined | 0.0000 | 7 | 0 | 0 | 7 |
| iloveyou | 0.0491 | 0.0245 | undefined | 0.0000 | 3 | 0 | 0 | 3 |
| thankyou | 0.0625 | 0.0132 | undefined | 0.0000 | 5 | 0 | 0 | 5 |

## Files

- Metrics: `reports/baseline_metrics.json`
- Error analysis: `reports/error_analysis.json`

## Warnings

- Split integrity report flags leakage between train and test; these metrics are not person-independent generalization.
