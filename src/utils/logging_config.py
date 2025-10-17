"""Structured logging utilities for run tracking."""
from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Mapping, Optional

try:  # Optional heavy dependencies
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    _np = None  # type: ignore

try:
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    _pd = None  # type: ignore


def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _generate_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _coerce_json_serializable(value: Any) -> Any:
    """Best-effort conversion of complex objects into JSON-serializable forms."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _coerce_json_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_json_serializable(item) for item in value]
    if _np is not None and isinstance(value, _np.ndarray):
        return value.tolist()
    if hasattr(value, "_asdict"):
        return _coerce_json_serializable(value._asdict())
    if hasattr(value, "__dict__"):
        return _coerce_json_serializable(vars(value))
    return str(value)


def compute_sha256(path: Path) -> Optional[str]:
    """Compute the SHA-256 checksum for *path* if it exists."""
    try:
        resolved = path.expanduser().resolve()
    except FileNotFoundError:
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    digest = sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class StructuredLogger:
    """Simple structured logger writing JSONL events per run."""

    run_id: str = None  # type: ignore[assignment]
    base_dir: Path = Path("results") / "runs"
    console_level: int = logging.INFO

    def __post_init__(self) -> None:
        if self.run_id is None:
            self.run_id = _generate_run_id()
        self.base_dir = Path(self.base_dir)
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.run_dir / "events.jsonl"
        self._lock = Lock()

        logger_name = f"cwf.run.{self.run_id}"
        self._logger = logging.getLogger(logger_name)
        self._logger.handlers = []
        self._logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(console_handler)
        self._logger.propagate = False

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def events_path(self) -> Path:
        return self._events_path

    def log_event(
        self,
        event_type: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        level: int = logging.INFO,
        message: Optional[str] = None,
    ) -> None:
        """Append an event to the JSONL log and optionally emit to the console."""
        record: Dict[str, Any] = {
            "timestamp": _timestamp(),
            "event": event_type,
            "level": logging.getLevelName(level),
        }
        if payload:
            record["payload"] = _coerce_json_serializable(payload)
        with self._lock:
            with self._events_path.open("a", encoding="utf-8") as handle:
                json.dump(record, handle, sort_keys=True)
                handle.write("\n")
        if message:
            self._logger.log(level, message)

    # Convenience helpers -------------------------------------------------
    def artifact_path(self, relative: str | Path) -> Path:
        path = self.run_dir / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, relative: str | Path, data: Mapping[str, Any] | Iterable[Any]) -> Path:
        path = self.artifact_path(relative)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_coerce_json_serializable(data), handle, indent=2, sort_keys=True)
        return path

    def save_text(self, relative: str | Path, text: str) -> Path:
        path = self.artifact_path(relative)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def save_dataframe(self, relative: str | Path, dataframe: Any, *, index: bool = False) -> Optional[Path]:
        if _pd is None:
            return None
        if not hasattr(dataframe, "to_csv"):
            return None
        path = self.artifact_path(relative)
        dataframe.to_csv(path, index=index)
        return path


def collect_environment_metadata() -> Dict[str, Any]:
    """Gather a lightweight snapshot of the execution environment."""
    metadata: Dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    for module_name in ["numpy", "scipy", "yaml"]:
        try:
            module = __import__(module_name)
        except Exception:  # pragma: no cover - optional dependency may be missing
            continue
        metadata[f"{module_name}_version"] = getattr(module, "__version__", "unknown")
    return metadata


def build_run_metadata(
    logger: StructuredLogger,
    *,
    arguments: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    dataset_summary: Mapping[str, Any],
    results_path: Path,
    checksums: Mapping[str, Optional[str]],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a JSON-serializable metadata payload for the run."""
    metadata: Dict[str, Any] = {
        "run_id": logger.run_id,
        "timestamp": _timestamp(),
        "arguments": _coerce_json_serializable(arguments),
        "config_snapshot": _coerce_json_serializable(config_snapshot),
        "dataset_summary": _coerce_json_serializable(dataset_summary),
        "results_path": str(results_path),
        "environment": collect_environment_metadata(),
        "checksums": {
            key: value for key, value in checksums.items() if value is not None
        },
    }
    if extra:
        metadata.update(_coerce_json_serializable(extra))
    return metadata


__all__ = [
    "StructuredLogger",
    "build_run_metadata",
    "collect_environment_metadata",
    "compute_sha256",
]
