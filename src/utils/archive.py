"""ZIP inspection and safe extraction.

The Amrita SLR download is a ZIP that contains another ZIP. Extraction must
never follow ``..`` components, absolute paths, or symlink members, and it
must not silently overwrite an existing dataset. ``inspect_archive`` is
read-only; ``safe_extract`` refuses unsafe archives before writing anything.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable

DOCUMENTATION_SUFFIXES = (".md", ".txt", ".pdf", ".doc", ".docx", ".html", ".rst")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".bz2", ".xz")
_DRIVE_PREFIX_LENGTH = 2


class ArchiveError(RuntimeError):
    """Raised when an archive is missing, corrupt, unsafe, or too large."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ArchiveError(f"Could not read {path}: {exc}") from exc
    return digest.hexdigest()


def unsafe_member_reason(name: str) -> str | None:
    """Return why a ZIP member name is unsafe, or ``None`` if it is safe.

    Rejects absolute paths, Windows drive/UNC paths, backslash separators, NUL
    bytes, empty names, and any ``..`` path component.
    """
    if not name or not name.strip():
        return "empty member name"
    if "\x00" in name:
        return "NUL byte in member name"
    if "\\" in name:
        return "backslash path separator"
    if name.startswith("/"):
        return "absolute path"
    if len(name) >= _DRIVE_PREFIX_LENGTH and name[1] == ":" and name[0].isalpha():
        return "Windows drive path"
    parts = name.split("/")
    if any(part == ".." for part in parts):
        return "parent-directory traversal"
    return None


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _extension(name: str) -> str:
    return PurePosixPath(name).suffix.lower()


def _check_zip(path: Path) -> zipfile.ZipFile:
    if not path.exists():
        raise ArchiveError(
            f"Archive not found: {path}. Download it before running extraction."
        )
    if not path.is_file():
        raise ArchiveError(f"Archive path is not a file: {path}")
    if path.stat().st_size == 0:
        raise ArchiveError(f"Archive is empty: {path}")
    try:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ArchiveError(
            f"{path} is not a complete ZIP archive ({exc}). A ``.crdownload`` or "
            "``Unconfirmed`` file usually means the download was interrupted; "
            "the central directory is missing and the archive cannot be used."
        ) from exc


def inspect_archive(
    path: str | Path,
    *,
    include_sha256: bool = True,
    max_sample_entries: int = 30,
    check_crc: bool = False,
) -> dict:
    """Inspect a ZIP without extracting it.

    Returns a JSON-serializable dictionary with counts, extensions, top-level
    directories, nested archives, documentation, and unsafe member names.
    """
    resolved = Path(path)
    with _check_zip(resolved) as archive:
        infos = archive.infolist()
        files = [info for info in infos if not info.filename.endswith("/")]
        unsafe = [
            {"name": info.filename, "reason": reason}
            for info in infos
            if (reason := unsafe_member_reason(info.filename))
        ]
        symlinks = [info.filename for info in infos if _is_symlink(info)]
        extensions = Counter(
            _extension(info.filename) or "<none>" for info in files
        )
        top_level = Counter(
            info.filename.split("/", 1)[0] for info in infos if info.filename
        )
        nested = [
            info.filename
            for info in files
            if _extension(info.filename) in ARCHIVE_SUFFIXES
        ]
        documentation = [
            info.filename
            for info in files
            if _extension(info.filename) in DOCUMENTATION_SUFFIXES
        ]

        result = {
            "path": str(resolved),
            "size_bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved) if include_sha256 else None,
            "valid": True,
            "entry_count": len(infos),
            "file_count": len(files),
            "directory_count": len(infos) - len(files),
            "total_uncompressed_bytes": sum(info.file_size for info in infos),
            "total_compressed_bytes": sum(info.compress_size for info in infos),
            "extensions": dict(sorted(extensions.items())),
            "top_level": dict(top_level.most_common()),
            "nested_archives": nested,
            "documentation_files": documentation,
            "unsafe_entries": unsafe,
            "symlink_entries": symlinks,
            "sample_entries": [
                {"name": info.filename, "size_bytes": info.file_size}
                for info in infos[:max_sample_entries]
            ],
        }
        if check_crc:
            bad_member = archive.testzip()
            result["crc_valid"] = bad_member is None
            result["crc_bad_member"] = bad_member
            if bad_member is not None:
                raise ArchiveError(
                    f"CRC check failed for member {bad_member!r} in {resolved}"
                )
        return result


def _safe_target(destination: Path, name: str) -> Path:
    target = (destination / name).resolve()
    try:
        target.relative_to(destination)
    except ValueError as exc:
        raise ArchiveError(
            f"Refusing to extract unsafe member {name!r} outside {destination}"
        ) from exc
    return target


def safe_extract(
    path: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str | None = None,
    check_crc: bool = False,
    overwrite: bool = False,
    free_space_margin: float = 1.05,
) -> dict:
    """Extract a ZIP after validating every member path.

    The destination must be empty or absent unless ``overwrite=True``. Symlink
    members and path traversal are rejected before any bytes are written. The
    original archive is never modified.
    """
    source = Path(path)
    dest = Path(destination)
    report = inspect_archive(source, check_crc=check_crc)

    if report["unsafe_entries"]:
        details = ", ".join(
            f"{item['name']!r} ({item['reason']})"
            for item in report["unsafe_entries"][:5]
        )
        raise ArchiveError(f"Archive contains unsafe paths: {details}")
    if report["symlink_entries"]:
        raise ArchiveError(
            "Archive contains symlink members, which are not extracted: "
            f"{report['symlink_entries'][:5]}"
        )
    if expected_sha256 and report["sha256"] != expected_sha256:
        raise ArchiveError(
            f"Archive SHA-256 mismatch: expected {expected_sha256}, "
            f"got {report['sha256']}"
        )

    if dest.exists():
        if not dest.is_dir():
            raise ArchiveError(f"Extraction destination is not a directory: {dest}")
        existing = list(dest.iterdir())
        if existing and not overwrite:
            raise ArchiveError(
                f"Extraction destination is not empty: {dest}. "
                "Use a new directory or pass overwrite=True explicitly."
            )
    else:
        dest.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(dest).free
    required = int(report["total_uncompressed_bytes"] * free_space_margin)
    if required > free:
        raise ArchiveError(
            f"Not enough disk space for extraction: need {required} bytes, "
            f"{free} bytes available at {dest}"
        )

    resolved_dest = dest.resolve()
    extracted: list[str] = []
    with _check_zip(source) as archive:
        for info in archive.infolist():
            _safe_target(resolved_dest, info.filename)
            target = resolved_dest / info.filename
            if info.filename.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not overwrite:
                raise ArchiveError(
                    f"Refusing to overwrite existing file during extraction: {target}"
                )
            try:
                with archive.open(info) as reader, target.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
            except (OSError, zipfile.BadZipFile) as exc:
                raise ArchiveError(
                    f"Failed to extract {info.filename!r} from {source}: {exc}"
                ) from exc
            written = target.stat().st_size
            if written != info.file_size:
                raise ArchiveError(
                    f"Size mismatch extracting {info.filename!r}: expected "
                    f"{info.file_size} bytes, wrote {written}"
                )
            extracted.append(info.filename)

    return {
        "archive": report,
        "destination": str(dest),
        "extracted_files": extracted,
        "extracted_file_count": len(extracted),
        "extracted_bytes": sum(
            (dest / name).stat().st_size for name in extracted
        ),
    }


def find_documentation(root: str | Path) -> list[str]:
    """List documentation/license/citation files under an extracted dataset."""
    base = Path(root)
    if not base.is_dir():
        return []
    found: list[str] = []
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in DOCUMENTATION_SUFFIXES:
            found.append(str(path.relative_to(base)))
    return sorted(found)


def iter_media_files(root: str | Path, extensions: Iterable[str]) -> list[Path]:
    """Return sorted media files under ``root`` with the given extensions."""
    base = Path(root)
    suffixes = {extension.lower() for extension in extensions}
    if not base.is_dir():
        return []
    return sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
