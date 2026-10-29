import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from tests.helpers import ROOT, SRC  # noqa: F401

from utils.archive import (
    ArchiveError,
    inspect_archive,
    safe_extract,
    sha256_file,
    unsafe_member_reason,
)


def make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return path


class ArchiveInspectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_inspects_counts_extensions_and_sha256(self):
        archive = make_zip(
            self.root / "sample.zip",
            {
                "sign project/9.hello/a.mp4": b"video-a",
                "sign project/9.hello/b.mov": b"video-b",
                "README.txt": b"docs",
            },
        )
        info = inspect_archive(archive)
        self.assertTrue(info["valid"])
        self.assertEqual(info["entry_count"], 3)
        self.assertEqual(info["extensions"][".mp4"], 1)
        self.assertEqual(info["extensions"][".mov"], 1)
        self.assertEqual(info["documentation_files"], ["README.txt"])
        self.assertEqual(info["unsafe_entries"], [])
        self.assertEqual(info["sha256"], sha256_file(archive))

    def test_missing_archive_has_actionable_error(self):
        with self.assertRaisesRegex(ArchiveError, "Archive not found"):
            inspect_archive(self.root / "missing.zip")

    def test_empty_archive_is_rejected(self):
        path = self.root / "empty.zip"
        path.write_bytes(b"")
        with self.assertRaisesRegex(ArchiveError, "empty"):
            inspect_archive(path)

    def test_partial_download_is_rejected_with_crdownload_hint(self):
        path = self.root / "Unconfirmed 1.crdownload"
        path.write_bytes(b"PK\x03\x04 truncated")
        with self.assertRaisesRegex(ArchiveError, "not a complete ZIP"):
            inspect_archive(path)

    def test_nested_archive_is_reported(self):
        inner = make_zip(self.root / "inner.zip", {"data/x.mp4": b"x"})
        outer = make_zip(self.root / "outer.zip", {"nested.zip": inner.read_bytes()})
        info = inspect_archive(outer)
        self.assertEqual(info["nested_archives"], ["nested.zip"])
        self.assertEqual(info["extensions"][".zip"], 1)

    def test_crc_check_detects_corruption(self):
        archive = make_zip(self.root / "crc.zip", {"a.txt": b"hello"})
        payload = bytearray(archive.read_bytes())
        payload[-20] ^= 0xFF
        archive.write_bytes(payload)
        with self.assertRaises(ArchiveError):
            inspect_archive(archive, check_crc=True)


class UnsafePathTests(unittest.TestCase):
    def test_unsafe_member_names_are_detected(self):
        self.assertIn("parent", unsafe_member_reason("../evil.txt"))
        self.assertIn("absolute", unsafe_member_reason("/etc/passwd"))
        self.assertIn("drive", unsafe_member_reason("C:/Windows/system32"))
        self.assertIn("backslash", unsafe_member_reason("..\\evil.txt"))
        self.assertIsNone(unsafe_member_reason("safe/path/file.txt"))

    def test_traversal_archive_is_refused_before_writing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        archive = root / "evil.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../escaped.txt", b"bad")
        destination = root / "out"
        with self.assertRaisesRegex(ArchiveError, "unsafe"):
            safe_extract(archive, destination)
        self.assertFalse((root / "escaped.txt").exists())

    def test_symlink_members_are_refused(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        archive = root / "link.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            info = zipfile.ZipInfo("link")
            info.external_attr = (0o120777 << 16)
            handle.writestr(info, "/etc/passwd")
        with self.assertRaisesRegex(ArchiveError, "symlink"):
            safe_extract(archive, root / "out")


class SafeExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_extracts_and_preserves_relative_structure(self):
        archive = make_zip(
            self.root / "ok.zip",
            {"sign project/9.hello/a.mp4": b"payload", "README.md": b"# docs"},
        )
        before = archive.read_bytes()
        destination = self.root / "out"
        report = safe_extract(archive, destination)
        self.assertEqual(report["extracted_file_count"], 2)
        self.assertEqual(
            (destination / "sign project/9.hello/a.mp4").read_bytes(), b"payload"
        )
        self.assertEqual((destination / "README.md").read_text(), "# docs")
        self.assertEqual(archive.read_bytes(), before)

    def test_non_empty_destination_is_refused_without_overwrite(self):
        archive = make_zip(self.root / "ok.zip", {"a.txt": b"a"})
        destination = self.root / "out"
        destination.mkdir()
        (destination / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(ArchiveError, "not empty"):
            safe_extract(archive, destination)
        self.assertEqual((destination / "keep.txt").read_text(), "keep")

    def test_sha256_mismatch_is_refused(self):
        archive = make_zip(self.root / "ok.zip", {"a.txt": b"a"})
        with self.assertRaisesRegex(ArchiveError, "SHA-256 mismatch"):
            safe_extract(archive, self.root / "out", expected_sha256="0" * 64)

    def test_expected_sha256_is_accepted(self):
        archive = make_zip(self.root / "ok.zip", {"a.txt": b"a"})
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        report = safe_extract(
            archive, self.root / "out", expected_sha256=digest
        )
        self.assertEqual(report["extracted_file_count"], 1)


if __name__ == "__main__":
    unittest.main()
