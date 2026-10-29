import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import ROOT, SRC, write_dataset  # noqa: F401

from data import DatasetError
from validate_dataset import main, validate_dataset


class DatasetValidatorTests(unittest.TestCase):
    def test_valid_dataset_reports_counts_without_modifying_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = write_dataset(
                Path(tmp),
                "train",
                [
                    ("one", (255, 0, 0), (16, 16), "0 0.5 0.5 0.2 0.2\n"),
                    ("two", (0, 255, 0), (16, 16), "1 0.5 0.5 0.2 0.2\n"),
                ],
            )
            before = sorted(path.name for path in split.rglob("*"))
            summary = validate_dataset(split, ["a", "b"])
            after = sorted(path.name for path in split.rglob("*"))
            self.assertEqual(before, after)
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(summary["boxes"], 2)
            self.assertEqual(summary["class_counts"], {"a": 1, "b": 1})

    def test_invalid_dataset_raises_dataset_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = write_dataset(
                Path(tmp),
                "train",
                [("bad", (0, 0, 0), (16, 16), "5 0.5 0.5 0.2 0.2\n")],
            )
            with self.assertRaises(DatasetError):
                validate_dataset(split, ["a", "b"])

    def test_unreadable_image_is_reported_by_the_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = write_dataset(
                Path(tmp),
                "train",
                [("bad", (0, 0, 0), (16, 16), "0 0.5 0.5 0.2 0.2\n")],
            )
            (split / "images" / "bad.jpg").write_bytes(b"")
            with self.assertRaises(DatasetError) as ctx:
                validate_dataset(split, ["a", "b"])
            self.assertIn("bad.jpg", str(ctx.exception))

    def test_cli_reports_a_bad_config_without_a_traceback(self):
        output = io.StringIO()
        with mock.patch.dict(
            os.environ, {"SIGN_DETR_CONFIG": "/definitely/missing/config.json"}
        ):
            with contextlib.redirect_stdout(output):
                result = main([])
        self.assertEqual(result, 1)
        self.assertIn("configuration", output.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
