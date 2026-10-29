import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tests.helpers import ROOT, SRC, write_dataset  # noqa: F401

from data import DETRData, DatasetError, build_eval_transform


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _dataset(self, split_dir, **kwargs):
        return DETRData(split_dir, classes=["a", "b"], verbose=False, **kwargs)

    def test_eval_sample_has_expected_shape_dtype_and_normalized_boxes(self):
        split = write_dataset(
            self.root,
            "train",
            [("one", (255, 0, 0), (64, 48), "0 0.5 0.5 0.25 0.25\n")],
        )
        image, target = self._dataset(split, train=False)[0]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(target["labels"].dtype, torch.long)
        self.assertTrue(torch.allclose(target["boxes"], torch.tensor([[0.5, 0.5, 0.25, 0.25]]), atol=1e-5))

    def test_rgb_channels_are_preserved_and_normalized(self):
        split = write_dataset(
            self.root,
            "train",
            [("red", (255, 0, 0), (32, 32), "0 0.5 0.5 0.5 0.5\n")],
        )
        image, _ = self._dataset(split, train=False)[0]
        # ImageNet normalization: red should be the largest channel.
        self.assertGreater(image[0].mean().item(), image[1].mean().item())
        self.assertGreater(image[0].mean().item(), image[2].mean().item())

    def test_grayscale_image_is_converted_to_three_channels(self):
        split_dir = self.root / "train"
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "labels").mkdir()
        Image.new("L", (10, 10), 128).save(split_dir / "images" / "gray.png")
        (split_dir / "labels" / "gray.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        image, _ = self._dataset(split_dir, train=False)[0]
        self.assertEqual(tuple(image.shape), (3, 224, 224))

    def test_safe_transform_falls_back_when_augmentation_drops_all_boxes(self):
        split = write_dataset(
            self.root,
            "train",
            [("boxed", (255, 0, 0), (64, 64), "0 0.5 0.5 0.25 0.25\n")],
        )
        dataset = self._dataset(split, train=True)

        class DropAllBoxes:
            def __call__(self, image, bboxes, class_labels):
                return {"image": image, "bboxes": [], "class_labels": []}

        dataset.transform = DropAllBoxes()
        image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(target["labels"].tolist(), [0])
        self.assertTrue(
            torch.allclose(
                target["boxes"], torch.tensor([[0.5, 0.5, 0.25, 0.25]]), atol=1e-5
            )
        )

    def test_duplicate_image_stems_are_rejected(self):
        split = write_dataset(
            self.root,
            "train",
            [("same", (255, 0, 0), (16, 16), "0 0.5 0.5 0.2 0.2\n")],
        )
        Image.new("RGB", (16, 16), (0, 255, 0)).save(
            split / "images" / "same.png"
        )
        with self.assertRaises(DatasetError) as ctx:
            self._dataset(split, train=False)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_empty_annotation_is_a_valid_no_object_sample(self):
        split = write_dataset(
            self.root, "train", [("empty", (0, 255, 0), (32, 32), "")]
        )
        image, target = self._dataset(split, train=True)[0]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(tuple(target["labels"].shape), (0,))
        self.assertEqual(tuple(target["boxes"].shape), (0, 4))

    def test_unknown_class_is_reported_at_dataset_construction(self):
        split = write_dataset(
            self.root, "train", [("bad", (0, 0, 255), (32, 32), "99 0.5 0.5 0.2 0.2\n")]
        )
        with self.assertRaises(DatasetError) as ctx:
            self._dataset(split, train=False)
        self.assertIn("bad", str(ctx.exception))
        self.assertIn("class", str(ctx.exception).lower())

    def test_malformed_annotation_is_reported_at_dataset_construction(self):
        split = write_dataset(
            self.root, "train", [("bad", (0, 0, 255), (32, 32), "0 0.5 0.5\n")]
        )
        with self.assertRaises(DatasetError) as ctx:
            self._dataset(split, train=False)
        self.assertIn("bad.txt", str(ctx.exception))

    def test_missing_image_is_reported(self):
        split_dir = self.root / "train"
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "labels").mkdir()
        (split_dir / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        with self.assertRaises(DatasetError) as ctx:
            self._dataset(split_dir, train=False)
        self.assertIn("image", str(ctx.exception).lower())

    def test_empty_dataset_has_length_zero(self):
        split_dir = self.root / "train"
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "labels").mkdir()
        dataset = self._dataset(split_dir, train=False)
        self.assertEqual(len(dataset), 0)

    def test_image_without_annotation_is_reported(self):
        split_dir = self.root / "train"
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "labels").mkdir()
        Image.new("RGB", (16, 16), (255, 0, 0)).save(
            split_dir / "images" / "orphan.jpg"
        )
        with self.assertRaises(DatasetError) as ctx:
            self._dataset(split_dir, train=False)
        message = str(ctx.exception).lower()
        self.assertIn("orphan", message)
        self.assertIn("annotation", message)

    def test_transform_failure_is_not_silently_replaced_by_fallback(self):
        split = write_dataset(
            self.root,
            "train",
            [("boxed", (255, 0, 0), (64, 64), "0 0.5 0.5 0.25 0.25\n")],
        )
        dataset = self._dataset(split, train=True)

        class BrokenTransform:
            def __call__(self, image, bboxes, class_labels):
                raise ValueError("synthetic transform failure")

        dataset.transform = BrokenTransform()
        with self.assertRaises(DatasetError) as ctx:
            dataset[0]
        message = str(ctx.exception)
        self.assertIn("transform", message.lower())
        self.assertIn("synthetic transform failure", message)

    def test_missing_split_directories_are_reported(self):
        with self.assertRaises(DatasetError):
            self._dataset(self.root / "missing", train=False)

    def test_indices_select_a_subset_without_modifying_underlying_dataset(self):
        split = write_dataset(
            self.root,
            "train",
            [
                ("a", (255, 0, 0), (16, 16), "0 0.5 0.5 0.2 0.2\n"),
                ("b", (0, 255, 0), (16, 16), "1 0.5 0.5 0.2 0.2\n"),
            ],
        )
        full = self._dataset(split, train=False)
        subset = self._dataset(split, train=False, indices=[1])
        self.assertEqual(len(full), 2)
        self.assertEqual(len(subset), 1)
        self.assertEqual(subset[0][1]["labels"].tolist(), [1])
        with self.assertRaises(IndexError):
            self._dataset(split, train=False, indices=[2])

    def test_inference_transform_matches_eval_geometry_without_boxes(self):
        transform = build_eval_transform(image_size=224, with_bboxes=False)
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        tensor = transform(image=image)["image"]
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
