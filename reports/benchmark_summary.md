# Runtime benchmark

| Field | Value |
| --- | --- |
| Device | cpu |
| Checkpoint | `checkpoints/best.pt` |
| Checkpoint SHA-256 | `b2ed6aa8aea023d37c694939fa952bc6f76de546b93f68cc0bea5dc54664fd74` |
| Runs / warmup | 10 / 3 |
| Batch size | 1 |
| Input resolution | 1920x1080 |
| Median end-to-end | 50.1661 ms |
| p95 end-to-end | 82.4499 ms |
| Median model forward | 48.6632 ms |
| FPS (batch/median e2e) | 19.934 |
| Peak RSS | 1138.64 MB |

Camera capture and display are excluded. CUDA stages are synchronized. Results are machine-specific and were actually measured; they are not extrapolated.
