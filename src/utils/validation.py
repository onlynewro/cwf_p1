"""Validation helpers for configuration-driven resource loading."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class ConfigValidationError(RuntimeError):
    """Raised when configuration-provided resources are invalid."""


def resolve_path(path: str | Path, base_dir: Optional[str | Path] = None) -> Path:
    """Return the absolute :class:`~pathlib.Path` for ``path``.

    Parameters
    ----------
    path:
        Path (absolute or relative) to resolve.
    base_dir:
        Optional base directory that relative paths should be resolved against.
    """
    if path is None:
        raise ConfigValidationError("No path provided for resolution")
    raw = Path(path)
    if raw.expanduser().is_absolute():
        return raw.expanduser().resolve()
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    return (base / raw).expanduser().resolve()


def require_existing_file(path: str | Path,
                          base_dir: Optional[str | Path] = None,
                          description: str | None = None) -> str:
    """Ensure that ``path`` points to an existing file.

    Parameters
    ----------
    path:
        The file path to validate.
    base_dir:
        Optional directory used to resolve relative ``path`` values.
    description:
        Human friendly label to include in error messages.

    Returns
    -------
    str
        The resolved absolute path string.

    Raises
    ------
    ConfigValidationError
        If ``path`` does not exist or is not a file.
    """
    description = description or "file"
    resolved = resolve_path(path, base_dir=base_dir)
    if not resolved.exists():
        raise ConfigValidationError(f"Configured {description} not found: {resolved}")
    if not resolved.is_file():
        raise ConfigValidationError(f"Configured {description} is not a file: {resolved}")
    return str(resolved)


def ensure_paths_exist(entries, base_dir: Optional[str | Path] = None, kind: str = "file"):
    """Validate that a sequence of paths exists.

    Parameters
    ----------
    entries:
        Iterable of ``(path, description)`` pairs.
    base_dir:
        Base directory used when resolving relative paths.
    kind:
        Descriptor included in error messages (defaults to ``"file"``).

    Raises
    ------
    ConfigValidationError
        If any path does not exist.
    """
    for path, description in entries:
        if path is None:
            raise ConfigValidationError(f"Missing {description or kind} path in configuration")
        require_existing_file(path, base_dir=base_dir, description=description or kind)
