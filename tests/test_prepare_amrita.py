import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from tests.helpers import ROOT, SRC  # noqa: F401

from prepare_amrita import (
    build_manifest,
    deterministic_sample_id,
    extract_archive,
    extract_frame,
    infer_signer,
    main,
    probe_video,
)
from utils.archive import ArchiveError


def make_tree(root: Path) -> Path:
    base = root / "sign project"
    for folder, files in {
        "9.hello": ["krishnarajhello.mp4", "manideephello-1.mov"],
        "38.thankyou": ["ThankYou1.mp4"],
        "1.bird": ["Bird1.mp4"],
    }.items():
        target = base / folder
        target.mkdir(parents=True)
        for index, name in enumerate(files):
            (target / name).write_bytes(f"{folder}-{name}-{index}".encode())
    return root


def make_video(path: Path, frames: int = 6) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
    )
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV MP4 writer is unavailable")
    try:
        for index in range(frames):
            writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()
    return path


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = make_tree(Path(self.tmp.name))

    def test_manifest_records_video_level_annotation_limits(self):
        manifest = build_manifest(
            self.root,
            project_classes=["hello", "iloveyou", "thankyou"],
            probe=False,
        )
        self.assertEqual(manifest["totals"]["videos"], 4)
        self.assertEqual(manifest["totals"]["classes"], 3)
        self.assertEqual(manifest["annotation_status"], "video_level_only")
        self.assertFalse(manifest["bounding_boxes_available"])
        self.assertFalse(manifest["temporal_annotations_available"])
        self.assertFalse(manifest["detr_training_ready"])
        self.assertTrue(manifest["blockers"])
        self.assertEqual(manifest["license"], "not stated in archive")
        for sample in manifest["samples"]:
            self.assertFalse(sample["annotation"]["has_bbox"])
            self.assertEqual(sample["annotation"]["level"], "video_label_only")
            self.assertIsNone(sample["width"])
            self.assertFalse(sample["probed"])

    def test_sample_ids_are_deterministic_and_path_based(self):
        first = build_manifest(
            self.root, project_classes=["hello"], probe=False
        )
        second = build_manifest(
            self.root, project_classes=["hello"], probe=False
        )
        self.assertEqual(
            [sample["id"] for sample in first["samples"]],
            [sample["id"] for sample in second["samples"]],
        )
        self.assertEqual(
            first["samples"][0]["id"],
            deterministic_sample_id(first["samples"][0]["relative_path"]),
        )
        self.assertEqual(len({sample["id"] for sample in first["samples"]}), 4)

    def test_gloss_matches_require_verified_evidence(self):
        manifest = build_manifest(
            self.root, project_classes=["hello", "thankyou"], probe=False
        )
        matches = {
            match["external_label"]: match["status"]
            for match in manifest["compatibility"]["matches"]
        }
        self.assertEqual(matches["hello"], "candidate_unverified")
        self.assertEqual(matches["thankyou"], "candidate_unverified")
        self.assertEqual(manifest["compatibility"]["training_mapping"], {})
        self.assertIn("1.bird", manifest["compatibility"]["unmatched_external"])

        verified = build_manifest(
            self.root,
            project_classes=["hello"],
            evidence={
                "hello": {
                    "verified": True,
                    "source": "corpus sign list",
                    "definition": "documented hello sign",
                }
            },
            probe=False,
        )
        self.assertEqual(
            verified["compatibility"]["training_mapping"], {"hello": 0}
        )

    def test_duplicate_videos_are_reported(self):
        folder = self.root / "sign project" / "1.bird"
        (folder / "copy.mp4").write_bytes((folder / "Bird1.mp4").read_bytes())
        manifest = build_manifest(
            self.root, project_classes=["hello"], probe=False
        )
        self.assertEqual(len(manifest["duplicates"]), 1)
        self.assertEqual(len(manifest["duplicates"][0]["sample_ids"]), 2)

    def test_missing_or_empty_directory_is_rejected(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        with self.assertRaises(ArchiveError):
            build_manifest(empty, project_classes=["hello"], probe=False)
        with self.assertRaises(ArchiveError):
            build_manifest(empty / "missing", project_classes=["hello"], probe=False)

    def test_probe_disabled_reports_unavailable(self):
        sample = self.root / "sign project" / "9.hello" / "krishnarajhello.mp4"
        info = probe_video(sample, enabled=False)
        self.assertFalse(info["probed"])
        self.assertIn("disabled", info["error"])


class SignerInferenceTests(unittest.TestCase):
    def test_known_signer_names(self):
        self.assertEqual(infer_signer("KrishnarajHello1.mp4")[0], "krishnaraj")
        self.assertEqual(infer_signer("manideephello-2.mp4")[0], "manideep")
        self.assertEqual(infer_signer("NachikethHello3.mp4")[0], "nachiketh")

    def test_unknown_and_batch_names(self):
        self.assertEqual(infer_signer("Hello1.mp4")[0], "unknown")
        self.assertEqual(
            infer_signer("0403(21).mp4")[0], "unknown_camera_batch_0403"
        )


class FrameExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_extracts_a_frame_without_annotations(self):
        video = make_video(self.root / "clip.mp4")
        destination = self.root / "frames" / "hello.jpg"
        result = extract_frame(video, destination)
        self.assertTrue(result["ok"], result)
        self.assertTrue(destination.is_file())
        self.assertEqual((result["width"], result["height"]), (64, 48))
        self.assertFalse(destination.with_suffix(".txt").exists())

    def test_invalid_video_reports_failure(self):
        not_video = self.root / "broken.mp4"
        not_video.write_text("not a video")
        result = extract_frame(not_video, self.root / "frame.jpg")
        self.assertFalse(result["ok"])
        self.assertTrue(result["error"])

    def test_invalid_position_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_frame(self.root / "x.mp4", self.root / "f.jpg", position=1.0)


class NestedArchiveTests(unittest.TestCase):
    def test_nested_archive_is_extracted_to_output(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)

        inner_payload = root / "inner.zip"
        with zipfile.ZipFile(inner_payload, "w") as archive:
            archive.writestr("sign project/9.hello/a.mp4", b"video")

        outer = root / "outer.zip"
        with zipfile.ZipFile(outer, "w") as archive:
            archive.writestr(
                "Amrita SLR- Real Time ISL Dataset/Amrita SLR Dataset.zip",
                inner_payload.read_bytes(),
            )

        output = root / "out"
        report = extract_archive(outer, output)
        self.assertEqual(report["extracted_file_count"], 1)
        self.assertTrue((output / "sign project/9.hello/a.mp4").is_file())
        self.assertIn("inner_archive", report)


class InspectOnlyCliTests(unittest.TestCase):
    def test_inspect_only_writes_report_and_does_not_extract(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        archive = root / "data.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("a.txt", b"content")
        report = root / "report.json"
        code = main(
            ["--archive", str(archive), "--inspect-only", "--report", str(report)]
        )
        self.assertEqual(code, 0)
        payload = json.loads(report.read_text())
        self.assertEqual(payload["mode"], "inspect_only")
        self.assertFalse(payload["detr_training_ready"])
        self.assertFalse((root / "dataset").exists())

    def test_missing_archive_returns_error_code(self):
        code = main(
            [
                "--archive",
                "/nonexistent/amrita.zip",
                "--inspect-only",
                "--report",
                "/tmp/opencode/unused-report.json",
            ]
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
