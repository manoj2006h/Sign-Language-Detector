import unittest

from tests.helpers import ROOT, SRC  # noqa: F401

from utils.compat import (
    CompatibilityError,
    assert_mapping_is_safe,
    compare_vocabularies,
    discover_external_classes,
    normalize_label,
    parse_class_folder,
)


def external(folders, counts=None):
    return discover_external_classes(folders, counts or {})


class FolderParsingTests(unittest.TestCase):
    def test_ordered_folder_is_parsed(self):
        self.assertEqual(parse_class_folder("9.hello"), (9, "hello"))
        self.assertEqual(parse_class_folder("38.thankyou"), (38, "thankyou"))

    def test_plain_folder_has_no_order(self):
        self.assertEqual(parse_class_folder("hello"), (None, "hello"))

    def test_empty_folder_is_rejected(self):
        with self.assertRaises(CompatibilityError):
            parse_class_folder(" ")
        with self.assertRaises(CompatibilityError):
            parse_class_folder("9.")


class VocabularyTests(unittest.TestCase):
    def test_gloss_match_without_evidence_is_only_a_candidate(self):
        report = compare_vocabularies(
            ["hello", "iloveyou", "thankyou"],
            external(["9.hello", "38.thankyou", "1.bird"]),
        )
        self.assertEqual(report.training_mapping(), {})
        statuses = {
            match["external_label"]: match["status"] for match in report.matches
        }
        self.assertEqual(statuses["hello"], "candidate_unverified")
        self.assertEqual(statuses["thankyou"], "candidate_unverified")
        self.assertIn("1.bird", report.unmatched_external)
        self.assertIn("iloveyou", report.missing_project_classes)
        with self.assertRaisesRegex(CompatibilityError, "No external classes"):
            assert_mapping_is_safe(report)

    def test_verified_evidence_enables_mapping(self):
        evidence = {
            "hello": {
                "verified": True,
                "source": "corpus documentation v1.2, sign list",
                "definition": "open flat hand moves away from the forehead",
            }
        }
        report = compare_vocabularies(
            ["hello"], external(["9.hello"]), evidence=evidence
        )
        self.assertEqual(report.training_mapping(), {"hello": 0})
        self.assertEqual(report.matches[0]["status"], "verified")
        self.assertEqual(assert_mapping_is_safe(report), {"hello": 0})

    def test_incomplete_evidence_is_rejected(self):
        evidence = {"hello": {"verified": True, "source": "", "definition": ""}}
        report = compare_vocabularies(
            ["hello"], external(["9.hello"]), evidence=evidence
        )
        self.assertEqual(report.matches[0]["status"], "candidate_unverified")

    def test_gloss_lookalike_is_not_mapped(self):
        # "thankyou" and "thanks" are not assumed equivalent without evidence.
        report = compare_vocabularies(["thankyou"], external(["3.thanks"]))
        self.assertEqual(report.matches, [])
        self.assertIn("3.thanks", report.unmatched_external)

    def test_duplicate_normalized_labels_are_ambiguous(self):
        report = compare_vocabularies(
            ["hello"], external(["hello", "9.hello"])
        )
        self.assertEqual(len(report.matches), 2)
        self.assertTrue(all(m["status"] == "ambiguous" for m in report.matches))
        self.assertIn("hello", report.duplicate_external_labels)
        with self.assertRaises(CompatibilityError):
            assert_mapping_is_safe(report)

    def test_unknown_label_stays_unmatched(self):
        report = compare_vocabularies(
            ["hello"], external(["50.power", "49.missing"])
        )
        self.assertEqual(report.matches, [])
        self.assertEqual(
            report.unmatched_external, ["49.missing", "50.power"]
        )

    def test_normalization_handles_case_and_separators(self):
        self.assertEqual(normalize_label("Thank You "), normalize_label("thank_you"))
        report = compare_vocabularies(["thankyou"], external(["38.thank you"]))
        self.assertEqual(report.matches[0]["status"], "candidate_unverified")

    def test_counts_are_preserved(self):
        classes = discover_external_classes(
            ["9.hello"], {"9.hello": 18}
        )
        self.assertEqual(classes[0].sample_count, 18)

    def test_empty_project_vocabulary_is_rejected(self):
        with self.assertRaises(CompatibilityError):
            compare_vocabularies([], external(["9.hello"]))


if __name__ == "__main__":
    unittest.main()
