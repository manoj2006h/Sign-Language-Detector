"""Configuration loading and validation.

The class list is the single source of truth for class names and the number of
classes. Keeping it validated here prevents callers from silently treating an
error string as a class list.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
CONFIG_ENV_VAR = "SIGN_DETR_CONFIG"


class ConfigError(ValueError):
    """Raised when the project configuration is missing or invalid."""


def _config_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and parse the JSON config without interpreting its contents."""
    resolved = _config_path(path)
    if not resolved.is_file():
        raise ConfigError(
            f"Config file not found: {resolved}. "
            f"Create src/config.json or set {CONFIG_ENV_VAR}."
        )
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file is not valid JSON ({resolved}): {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read config file ({resolved}): {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError(f"Config root must be a JSON object in {resolved}")
    return config


def get_classes(path: str | Path | None = None) -> list[str]:
    """Return the validated, ordered class names from the config."""
    resolved = _config_path(path)
    config = load_config(resolved)
    classes = config.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ConfigError(
            f"'classes' must be a non-empty list in {resolved}, "
            'e.g. {"classes": ["hello", "iloveyou", "thankyou"]}'
        )
    if any(not isinstance(name, str) or not name.strip() for name in classes):
        raise ConfigError(f"Every class name must be a non-empty string in {resolved}")
    if any(name != name.strip() for name in classes):
        raise ConfigError(
            f"Class names must not have leading or trailing whitespace in {resolved}"
        )
    if len(set(classes)) != len(classes):
        raise ConfigError(f"Class names must be unique in {resolved}")
    return list(classes)


def get_colors(path: str | Path | None = None) -> list[list[int]]:
    """Return one validated RGB color per class."""
    resolved = _config_path(path)
    config = load_config(resolved)
    classes = get_classes(resolved)
    colors = config.get("colors")
    if not isinstance(colors, list) or not colors:
        raise ConfigError(
            f"'colors' must be a non-empty list in {resolved}, "
            "one RGB color per class"
        )
    if len(colors) != len(classes):
        raise ConfigError(
            f"Please specify one color per class in {resolved}. "
            f"You have {len(colors)} colors and {len(classes)} classes."
        )

    validated: list[list[int]] = []
    for index, color in enumerate(colors):
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            raise ConfigError(
                f"Color {index} in {resolved} must be [R, G, B], got {color!r}"
            )
        if any(
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 0 <= channel <= 255
            for channel in color
        ):
            raise ConfigError(
                f"Color {index} in {resolved} must contain integer values 0-255, "
                f"got {color!r}"
            )
        validated.append([int(channel) for channel in color])
    return validated


def get_num_classes(path: str | Path | None = None) -> int:
    """Return the number of configured classes."""
    return len(get_classes(path))


if __name__ == "__main__":
    print(get_classes())
    print(get_colors())
