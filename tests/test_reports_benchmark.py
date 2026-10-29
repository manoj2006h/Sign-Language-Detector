import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.helpers import ROOT, SRC  # noqa: F401

from benchmark import benchmark, percentiles
from utils.reports import environment_info, utc_now, write_json


class FakeDETR:
    """Small deterministic stand-in that returns DETR-shaped tensors."""

    def __init__(self, num_classes=2, queries=3):
        self.num_classes = num_classes
        self.queries = queries

    def __call__(self, images):
        batch = images.shape[0]
        logits = torch.full((batch, self.queries, self.num_classes + 1), -5.0)
        logits[..., 0] = 5.0
        boxes = torch.tensor([0.5, 0.5, 0.2, 0.2], dtype=torch.float32)
        return {
            "pred_logits": logits,
            "pred_boxes": boxes.expand(batch, self.queries, 4).clone(),
        }


class PercentileTests(unittest.TestCase):
    def test_summary_values(self):
        result = percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["median_ms"], 3.0)
        self.assertEqual(result["min_ms"], 1.0)
        self.assertEqual(result["max_ms"], 5.0)
        self.assertEqual(result["p95_ms"], 5.0)

    def test_empty_list_is_rejected(self):
        with self.assertRaises(ValueError):
            percentiles([])

    def test_negative_latency_is_rejected(self):
        with self.assertRaises(ValueError):
            percentiles([1.0, -0.1])


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((48, 64, 3), dtype=np.uint8)
        self.model = FakeDETR()

    def test_cpu_benchmark_produces_valid_report_fields(self):
        result = benchmark(
            self.model,
            self.image,
            device=torch.device("cpu"),
            class_names=["hello", "thankyou"],
            runs=3,
            warmup=1,
            batch_size=1,
            threshold=0.5,
        )
        for stage in (
            "preprocess_ms",
            "forward_ms",
            "postprocess_ms",
            "end_to_end_ms",
        ):
            self.assertEqual(result["stages"][stage]["count"], 3)
            self.assertGreaterEqual(result["stages"][stage]["median_ms"], 0.0)
        self.assertEqual(result["resolution"], [64, 48])
        self.assertEqual(result["model_input_size"], [224, 224])
        self.assertGreater(result["fps"], 0.0)
        self.assertIn("peak_rss_mb", result["memory"])
        self.assertTrue(result["fps_method"])

    def test_batch_size_affects_reported_throughput(self):
        result = benchmark(
            self.model,
            self.image,
            device=torch.device("cpu"),
            class_names=["hello", "thankyou"],
            runs=2,
            warmup=0,
            batch_size=2,
            threshold=0.5,
        )
        self.assertEqual(result["batch_size"], 2)

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            benchmark(
                self.model,
                self.image,
                device=torch.device("cpu"),
                class_names=["hello"],
                runs=0,
                warmup=0,
                batch_size=1,
                threshold=0.5,
            )
        with self.assertRaises(ValueError):
            benchmark(
                self.model,
                self.image,
                device=torch.device("cpu"),
                class_names=["hello"],
                runs=1,
                warmup=0,
                batch_size=0,
                threshold=0.5,
            )


class ReportWritingTests(unittest.TestCase):
    def test_write_json_is_atomic_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "report.json"
            write_json(path, {"b": 2, "a": [1, 2]})
            self.assertEqual(json.loads(path.read_text()), {"b": 2, "a": [1, 2]})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_environment_info_has_provenance(self):
        info = environment_info("cpu")
        for key in ("timestamp", "python", "platform", "torch", "device"):
            self.assertIn(key, info)
        self.assertEqual(info["device"], "cpu")

    def test_utc_now_is_iso(self):
        value = utc_now()
        self.assertTrue(value.endswith("Z"))
        self.assertIn("T", value)


class SummaryTests(unittest.TestCase):
    def test_missing_reports_are_marked_instead_of_fabricated(self):
        from summarize_reports import REQUIRED_REPORTS, summarize

        with tempfile.TemporaryDirectory() as tmp:
            text = summarize(Path(tmp))
        self.assertIn("not generated", text)
        for name in REQUIRED_REPORTS:
            self.assertIn(f"`{name}`", text)
        self.assertIn("Amrita SLR cannot support detection evaluation", text)

    def test_summary_reads_generated_metrics(self):
        from summarize_reports import summarize

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "baseline_metrics.json",
                {
                    "checkpoint": {"path": "c.pt", "sha256": "abc", "epoch": 3},
                    "dataset": {"path": "data/test", "identity_sha256": "def"},
                    "metrics": {
                        "images": 1,
                        "ground_truth_objects": 1,
                        "mAP@0.5": 0.25,
                        "mAP@0.5:0.95": 0.1,
                        "micro": {"precision": 0.5, "recall": 1.0},
                        "predictions_at_confidence": 1,
                        "per_class": {},
                    },
                    "warnings": ["leakage"],
                },
            )
            text = summarize(root)
        self.assertIn("0.2500", text)
        self.assertIn("leakage", text)
        self.assertIn("`abc`", text)


if __name__ == "__main__":
    unittest.main()
