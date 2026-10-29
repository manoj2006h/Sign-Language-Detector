import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import ROOT, SRC  # noqa: F401  (adds src to sys.path)

from utils.setup import ConfigError, get_classes, get_colors, get_num_classes, load_config


class ConfigTests(unittest.TestCase):
    def _config_file(self, payload):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        with tmp:
            json.dump(payload, tmp)
        return tmp.name

    def test_loads_classes_colors_and_count(self):
        path = self._config_file(
            {"classes": ["hello", "thanks"], "colors": [[1, 2, 3], [4, 5, 6]]}
        )
        self.assertEqual(get_classes(path), ["hello", "thanks"])
        self.assertEqual(get_colors(path), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(get_num_classes(path), 2)
        self.assertEqual(load_config(path)["classes"], ["hello", "thanks"])

    def test_classes_do_not_require_colors(self):
        path = self._config_file({"classes": ["hello"]})
        self.assertEqual(get_classes(path), ["hello"])
        self.assertEqual(get_num_classes(path), 1)

    def test_missing_file_is_a_clear_error(self):
        with self.assertRaises(ConfigError) as ctx:
            load_config("/definitely/not/here/config.json")
        self.assertIn("not found", str(ctx.exception).lower())

    def test_empty_classes_rejected(self):
        path = self._config_file({"classes": [], "colors": []})
        with self.assertRaises(ConfigError):
            get_classes(path)

    def test_duplicate_classes_rejected(self):
        path = self._config_file({"classes": ["a", "a"]})
        with self.assertRaises(ConfigError):
            get_classes(path)

    def test_non_string_class_rejected(self):
        path = self._config_file({"classes": ["a", 2]})
        with self.assertRaises(ConfigError):
            get_classes(path)

    def test_class_names_cannot_have_surrounding_whitespace(self):
        path = self._config_file({"classes": [" hello"]})
        with self.assertRaises(ConfigError):
            get_classes(path)

    def test_color_count_must_match_classes(self):
        path = self._config_file({"classes": ["a", "b"], "colors": [[1, 2, 3]]})
        with self.assertRaises(ConfigError) as ctx:
            get_colors(path)
        self.assertIn("one color per class", str(ctx.exception).lower())

    def test_color_channels_must_be_rgb_bytes(self):
        for bad_color in ([1, 2], [1, 2, 300], [-1, 2, 3], [1, 2, "x"]):
            path = self._config_file({"classes": ["a"], "colors": [bad_color]})
            with self.assertRaises(ConfigError):
                get_colors(path)

    def test_malformed_json_is_a_clear_error(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        with tmp:
            tmp.write("{not json")
        with self.assertRaises(ConfigError):
            load_config(tmp.name)


if __name__ == "__main__":
    unittest.main()
