"""Dataset label compatibility for external sign-language corpora.

External datasets and the detector do not necessarily share vocabulary,
spelling, or sign definitions. Two labels are only treated as the same class
when an explicit, evidence-backed sign definition is supplied. English-gloss
similarity alone is recorded as a *candidate* and never used for training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

_ORDER_PREFIX = re.compile(r"^\s*(\d+)\s*[._\-)]\s*")
_SEPARATORS = re.compile(r"[\s_\-.]+")


class CompatibilityError(ValueError):
    """Raised when a compatibility request is malformed."""


@dataclass(frozen=True)
class ExternalClass:
    """One class folder discovered in an external corpus."""

    folder: str
    label: str
    order: int | None
    sample_count: int = 0


@dataclass
class CompatibilityReport:
    """Mapping decisions plus everything that could not be verified."""

    project_classes: list[str]
    external_classes: list[ExternalClass]
    matches: list[dict] = field(default_factory=list)
    unmatched_external: list[str] = field(default_factory=list)
    missing_project_classes: list[str] = field(default_factory=list)
    duplicate_external_labels: list[str] = field(default_factory=list)

    def training_mapping(self) -> dict[str, int]:
        """Return only *verified* external label -> project class index pairs."""
        return {
            match["external_label"]: match["project_index"]
            for match in self.matches
            if match["status"] == "verified"
        }

    def to_dict(self) -> dict:
        return {
            "project_classes": list(self.project_classes),
            "external_classes": [
                {
                    "folder": item.folder,
                    "label": item.label,
                    "order": item.order,
                    "sample_count": item.sample_count,
                }
                for item in self.external_classes
            ],
            "matches": self.matches,
            "unmatched_external": list(self.unmatched_external),
            "missing_project_classes": list(self.missing_project_classes),
            "duplicate_external_labels": list(self.duplicate_external_labels),
            "training_mapping": self.training_mapping(),
            "mapping_policy": (
                "Only classes with verified sign-definition evidence are mapped. "
                "Gloss similarity alone is insufficient."
            ),
        }


def normalize_label(label: str) -> str:
    """Normalize spelling for comparison only (never for training ids)."""
    if not isinstance(label, str):
        raise CompatibilityError(f"Label must be a string, got {type(label).__name__}")
    normalized = _SEPARATORS.sub("", label.strip().lower())
    return normalized


def parse_class_folder(folder: str) -> tuple[int | None, str]:
    """Parse ``9.hello`` into ``(9, "hello")``; plain names get ``None`` order."""
    if not isinstance(folder, str) or not folder.strip():
        raise CompatibilityError("Class folder name must be a non-empty string")
    name = folder.strip()
    match = _ORDER_PREFIX.match(name)
    if match:
        order = int(match.group(1))
        label = name[match.end() :].strip()
    else:
        order = None
        label = name
    if not label:
        raise CompatibilityError(f"Class folder {folder!r} has no label")
    return order, label


def discover_external_classes(
    folders: Iterable[str], sample_counts: Mapping[str, int] | None = None
) -> list[ExternalClass]:
    """Build validated :class:`ExternalClass` records from folder names."""
    counts = sample_counts or {}
    discovered: list[ExternalClass] = []
    seen: set[str] = set()
    for folder in folders:
        order, label = parse_class_folder(folder)
        if folder in seen:
            raise CompatibilityError(f"Duplicate class folder {folder!r}")
        seen.add(folder)
        discovered.append(
            ExternalClass(
                folder=folder,
                label=label,
                order=order,
                sample_count=int(counts.get(folder, 0)),
            )
        )
    discovered.sort(
        key=lambda item: (
            item.order is None,
            item.order if item.order is not None else 0,
            item.label,
        )
    )
    return discovered


def compare_vocabularies(
    project_classes: Iterable[str],
    external_classes: Iterable[ExternalClass],
    *,
    evidence: Mapping[str, Mapping] | None = None,
) -> CompatibilityReport:
    """Compare vocabularies and classify every potential match.

    ``evidence`` maps an *external label* (original spelling) to a record such
    as::

        {"verified": True,
         "source": "dataset README section 2",
         "definition": "flat hand touches the forehead and moves outward"}

    A match is ``verified`` only when ``verified`` is true, a source and a
    non-empty definition are present, and the labels normalize equal. Otherwise
    a gloss-equal pair is recorded as ``candidate_unverified``.
    """
    project_list = [name for name in project_classes]
    if not project_list:
        raise CompatibilityError("Project class list must not be empty")
    external_list = list(external_classes)
    evidence = evidence or {}

    by_normalized: dict[str, list[ExternalClass]] = {}
    for external in external_list:
        by_normalized.setdefault(normalize_label(external.label), []).append(external)

    duplicate_labels = sorted(
        normalized
        for normalized, items in by_normalized.items()
        if len(items) > 1
    )

    report = CompatibilityReport(
        project_classes=project_list,
        external_classes=external_list,
        duplicate_external_labels=duplicate_labels,
    )
    matched_external: set[str] = set()

    for project_index, project_name in enumerate(project_list):
        project_key = normalize_label(project_name)
        for external in by_normalized.get(project_key, []):
            matched_external.add(external.folder)
            record = evidence.get(external.label, {})
            verified = bool(
                record.get("verified") is True
                and str(record.get("source", "")).strip()
                and str(record.get("definition", "")).strip()
            )
            if normalize_label(external.label) in duplicate_labels:
                status = "ambiguous"
                reason = (
                    "Multiple external labels normalize to the same gloss; "
                    "cannot choose one without sign-definition evidence."
                )
            elif verified:
                status = "verified"
                reason = "Explicit sign-definition evidence supplied."
            else:
                status = "candidate_unverified"
                reason = (
                    "English gloss matches, but the archive provides no sign "
                    "definition. Not usable for training or external evaluation."
                )
            report.matches.append(
                {
                    "external_label": external.label,
                    "external_folder": external.folder,
                    "external_order": external.order,
                    "project_class": project_name,
                    "project_index": project_index,
                    "status": status,
                    "reason": reason,
                    "evidence": {
                        "verified": bool(record.get("verified", False)),
                        "source": record.get("source"),
                        "definition": record.get("definition"),
                    },
                }
            )

    report.unmatched_external = sorted(
        external.folder
        for external in external_list
        if external.folder not in matched_external
    )
    report.missing_project_classes = [
        name
        for name in project_list
        if not any(
            match["project_class"] == name and match["status"] == "verified"
            for match in report.matches
        )
    ]
    return report


def assert_mapping_is_safe(report: CompatibilityReport) -> dict[str, int]:
    """Return the training mapping, refusing unresolved/ambiguous classes."""
    unresolved = [
        match
        for match in report.matches
        if match["status"] in {"candidate_unverified", "ambiguous"}
    ]
    mapping = report.training_mapping()
    if not mapping:
        raise CompatibilityError(
            "No external classes have verified sign-definition evidence; "
            "the dataset cannot be mapped for DETR training. Unresolved: "
            + ", ".join(sorted({match["external_folder"] for match in unresolved}))
        )
    if unresolved:
        raise CompatibilityError(
            "Refusing a partial mapping while unresolved candidates exist: "
            + ", ".join(sorted({match["external_folder"] for match in unresolved}))
        )
    return mapping
