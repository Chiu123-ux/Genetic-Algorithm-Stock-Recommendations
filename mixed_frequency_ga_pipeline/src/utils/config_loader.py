"""Utilities for loading and accessing project configuration."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any


_CONFIG_FILENAME = "config.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / _CONFIG_FILENAME


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or accessed."""


@lru_cache(maxsize=8)
def load_config(override_path: str | None = None) -> dict[str, Any]:
    """Return the project configuration parsed from ``config.json``.

    Raises
    ------
    ConfigError
        If the configuration file is missing or contains invalid JSON.
    """

    if override_path is None:
        override_path = os.getenv("GP_CONFIG")

    config_path = Path(override_path) if override_path else _CONFIG_PATH

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Invalid JSON in configuration file: {exc}") from exc


def get_config_value(*keys: str, default: Any | None = None, config_path: str | None = None) -> Any:
    """Retrieve a nested configuration value using a sequence of keys.

    Parameters
    ----------
    *keys:
        Hierarchical keys used to walk the configuration dictionary.
    default:
        Optional fallback if the key chain does not exist.

    Returns
    -------
    Any
        The configuration value or ``default`` when provided.

    Raises
    ------
    ConfigError
        If any key in ``keys`` is missing and ``default`` is not supplied.
    """

    config = load_config(config_path)
    current: Any = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            if default is not None:
                return default
            raise ConfigError(f"Missing configuration value for keys: {keys}")

    return current


def project_root() -> Path:
    """Return the absolute path to the project root directory."""

    return _PROJECT_ROOT


def config_path() -> Path:
    override = os.getenv("GP_CONFIG")
    return Path(override) if override else _CONFIG_PATH


def set_config_override(path: str | None) -> None:
    """Override the active configuration file path for subsequent lookups."""

    load_config.cache_clear()
    if path:
        os.environ["GP_CONFIG"] = path
    else:
        os.environ.pop("GP_CONFIG", None)
