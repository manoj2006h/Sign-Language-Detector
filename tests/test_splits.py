import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.helpers import ROOT, SRC, write_dataset  # noqa: F401

from utils.splits import (
    SplitIntegrityError,
    audit_split_integrity,
    grouped_split_indices,
    hamming_distance,
    recording_group,
)


class RecordingGroupTests(unittest.TestCase):
    def test_frames_from_one_batch_share_a_group(self):
        first = "8ae99916-thankyou-f2441f5c-6c0d-11f0-9325-cf7617c51a8e"
        second = "03dd2586-hello-31fc7bec-6c0d-11f0-9325-cf7617c51a8e"
        self.assertEqual(recording_group(first), recording_group(second))

    def test_different_batches_do_not_share_a_group(self):
        first = "aaa-hello-1111-6c0d-11f0-9325-cf7617c51a8e"
        second = "bbb-hello-2222-725a-11f0-bdf3-b791a62111e4"
        self.assertNotEqual(recording_group(first), recording_group(second))

    def test_filename_without_uuid_is_its_own_group(self):
        self.assertEqual(recording_group("plain_name"), "plain_name")


class GroupedSplitTests(unittest.TestCase):
    def test_groups_are_never_split_across_train_and_val(self):
        groups = ["a", "a", "a", "b", "b", "c", "c", "d", "d", "d"]
        train, val = grouped_split_indices(groups, val_fraction=0.4, seed=1)
        self.assertEqual(sorted(train + val), list(range(len(groups))))
        self.assertFalse(set(train) & set(val))
        train_groups = {groups[index] for index in train}
        val_groups = {groups[index] for index in val}
        self.assertFalse(train_groups & val_groups)

    def test_split_is_deterministic(self):
        groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
        first = grouped_split_indices(groups, val_fraction=0.25, seed=7)
        second = grouped_split_indices(groups, val_fraction=0.25, seed=7)
        self.assertEqual(first, second)

    def test_single_group_is_rejected(self):
        with self.assertRaises(SplitIntegrityError):
            grouped_split_indices(["a", "a"], val_fraction=0.5)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(SplitIntegrityError):
            grouped_split_indices([], val_fraction=0.5)

    def test_invalid_fraction_is_rejected(self):
        with self.assertRaises(SplitIntegrityError):
            grouped_split_indices(["a", "b"], val_fraction=0.0)


class SplitAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_shared_session_and_duplicate_across_splits_are_flagged(self):
        suffix = "6c0d-11f0-9325-cf7617c51a8e"
        train = write_dataset(
            self.root,
            "train",
            [
                (f"aaaa-hello-11111111-{suffix}", (255, 0, 0), (32, 32),
                 "0 0.5 0.5 0.5 0.5\n"),
                (f"bbbb-hello-22222222-{suffix}", (0, 255, 0), (32, 32),
                 "0 0.5 0.5 0.5 0.5\n"),
            ],
        )
        test = write_dataset(
            self.root,
            "test",
            [
                (f"cccc-hello-33333333-{suffix}", (0, 0, 255), (32, 32),
                 "0 0.5 0.5 0.5 0.5\n"),
            ],
        )
        # Byte-identical copy across splits.
        (test / "images" / "copy.jpg").write_bytes(
            (train / "images" / f"aaaa-hello-11111111-{suffix}.jpg").read_bytes()
        )
        (test / "labels" / "copy.txt").write_text("0 0.5 0.5 0.5 0.5\n")

        report = audit_split_integrity(
            self.root, class_names=["hello"], near_duplicate_distance=0
        )
        self.assertEqual(report["leakage"]["status"], "confirmed")
        self.assertEqual(len(report["exact_cross_split"]), 1)
        self.assertIn(
            "6c0d-11f0-9325-cf7617c51a8e",
            {group for group in report["grouping"]["shared_groups"]},
        )
        self.assertFalse(report["signer_metadata_available"])

    def test_clean_splits_have_no_confirmed_leakage(self):
        import numpy as np

        for split, stem, pattern in (
            (
                "train",
                "aaaa-hello-11111111-6c0d-11f0-9325-cf7617c51a8e",
                "vertical",
            ),
            (
                "test",
                "bbbb-hello-22222222-725a-11f0-bdf3-b791a62111e4",
                "horizontal",
            ),
        ):
            images = self.root / split / "images"
            labels = self.root / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            array = np.zeros((32, 32, 3), dtype=np.uint8)
            if pattern == "vertical":
                array[:, ::2] = 255
            else:
                array[::2, :] = 255
            Image.fromarray(array).save(images / f"{stem}.jpg", quality=95)
            (labels / f"{stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        report = audit_split_integrity(
            self.root, class_names=["hello"], near_duplicate_distance=0
        )
        self.assertEqual(report["leakage"]["status"], "none")
        self.assertEqual(report["exact_cross_split"], [])

    def test_near_duplicates_are_detected(self):
        import numpy as np

        rng = np.random.default_rng(0)
        pixels = (rng.random((64, 64, 3)) * 255).astype("uint8")
        for split, stem in (
            ("train", "aaaa-hello-11111111-6c0d-11f0-9325-cf7617c51a8e"),
            ("test", "bbbb-hello-22222222-6c0d-11f0-9325-cf7617c51a8e"),
        ):
            images = self.root / split / "images"
            labels = self.root / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            array = pixels.copy()
            if split == "test":
                array[0, 0] = [0, 0, 0]
            Image.fromarray(array).save(images / f"{stem}.jpg", quality=95)
            (labels / f"{stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        report = audit_split_integrity(
            self.root, class_names=["hello"], near_duplicate_distance=4
        )
        self.assertEqual(report["exact_duplicates"], [])
        self.assertTrue(report["near_duplicates"])
        self.assertEqual(report["leakage"]["status"], "confirmed")

    def test_missing_split_raises(self):
        with self.assertRaises(SplitIntegrityError):
            audit_split_integrity(self.root / "nope", class_names=["hello"])

    def test_hamming_distance(self):
        self.assertEqual(hamming_distance(0b1010, 0b1000), 1)
        self.assertEqual(hamming_distance(0, 0), 0)


if __name__ == "__main__":
    unittest.main()
