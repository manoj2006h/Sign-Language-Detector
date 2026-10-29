import tempfile
import unittest
from pathlib import Path

import torch

from tests.helpers import ROOT, SRC  # noqa: F401

from utils.annotations import (
    AnnotationError,
    find_image_for_stem,
    parse_yolo_annotations,
    parse_yolo_file,
)


class AnnotationParsingTests(unittest.TestCase):
    def test_parses_multiple_boxes_and_skips_blank_lines(self):
        labels, boxes = parse_yolo_annotations(
            "0 0.5 0.5 0.2 0.4\n\n2 0.25 0.75 0.1 0.2\n", num_classes=3
        )
        self.assertEqual(labels.tolist(), [0, 2])
        self.assertTrue(torch.allclose(boxes, torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.25, 0.75, 0.1, 0.2]])))

    def test_empty_text_returns_empty_tensors(self):
        labels, boxes = parse_yolo_annotations("", num_classes=3)
        self.assertEqual(tuple(labels.shape), (0,))
        self.assertEqual(tuple(boxes.shape), (0, 4))

    def test_unknown_class_is_rejected_with_source_and_line(self):
        with self.assertRaises(AnnotationError) as ctx:
            parse_yolo_annotations("3 0.5 0.5 0.2 0.2", num_classes=3, source="x.txt")
        message = str(ctx.exception)
        self.assertIn("x.txt", message)
        self.assertIn("class", message.lower())
        self.assertIn("line 1", message)

    def test_malformed_field_count_is_rejected(self):
        with self.assertRaises(AnnotationError):
            parse_yolo_annotations("0 0.5 0.5 0.2", num_classes=3)

    def test_non_numeric_values_are_rejected(self):
        with self.assertRaises(AnnotationError):
            parse_yolo_annotations("0 x 0.5 0.2 0.2", num_classes=3)

    def test_non_finite_values_are_rejected(self):
        for bad in ("nan", "inf", "-inf"):
            with self.assertRaises(AnnotationError):
                parse_yolo_annotations(f"0 0.5 {bad} 0.2 0.2", num_classes=3)

    def test_non_positive_size_is_rejected(self):
        for box in ("0 0.5 0.5 0 0.2", "0 0.5 0.5 0.2 -0.1"):
            with self.assertRaises(AnnotationError):
                parse_yolo_annotations(box, num_classes=3)

    def test_out_of_bounds_box_is_rejected(self):
        for box in (
            "0 0.05 0.5 0.2 0.2",  # left edge < 0
            "0 0.5 0.95 0.2 0.2",  # bottom edge > 1
            "0 1.1 0.5 0.2 0.2",  # center > 1
        ):
            with self.assertRaises(AnnotationError) as ctx:
                parse_yolo_annotations(box, num_classes=3)
            self.assertIn("bound", str(ctx.exception).lower())

    def test_file_parser_reports_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.txt"
            path.write_text("9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaises(AnnotationError) as ctx:
                parse_yolo_file(path, num_classes=3)
            self.assertIn("bad.txt", str(ctx.exception))

    def test_non_utf8_annotation_is_an_annotation_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.txt"
            path.write_bytes(b"0 0.5 0.5 0.2 0.2\n\xff\xfe")
            with self.assertRaises(AnnotationError) as ctx:
                parse_yolo_file(path, num_classes=3)
            self.assertIn("invalid.txt", str(ctx.exception))

    def test_find_image_supports_case_insensitive_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.JPG").write_bytes(b"x")
            found = find_image_for_stem(root, "sample")
            self.assertIsNotNone(found)
            self.assertEqual(found.name, "sample.JPG")
            self.assertIsNone(find_image_for_stem(root, "missing"))


if __name__ == "__main__":
    unittest.main()
