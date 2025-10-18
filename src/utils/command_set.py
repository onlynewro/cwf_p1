"""Utility helpers for executing ordered command sequences sequentially."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .logging_config import StructuredLogger


@dataclass
class Command:
    """Representation of an executable command within a command set."""

    name: str
    callback: Callable[[], Any]
    description: Optional[str] = None


class CommandSet:
    """Execute a sequence of commands in a deterministic, single-threaded order."""

    def __init__(self, name: str = "command_set", logger: Optional[StructuredLogger] = None) -> None:
        self.name = name
        self._logger = logger
        self._commands: List[Command] = []

    def register(self, name: str, callback: Callable[[], Any], description: Optional[str] = None) -> None:
        """Register a new command to be executed sequentially."""

        self._commands.append(Command(name, callback, description))

    def execute(self) -> Dict[str, Any]:
        """Run all registered commands sequentially and return their outputs."""

        results: Dict[str, Any] = {}
        for command in self._commands:
            if self._logger is not None:
                self._logger.log_event(
                    f"{self.name}.{command.name}.start",
                    {
                        "command": command.name,
                        "description": command.description,
                    },
                    message=f"Command '{command.name}' starting (single-thread mode).",
                )
            try:
                output = command.callback()
            except Exception as exc:  # pylint: disable=broad-except
                if self._logger is not None:
                    self._logger.log_event(
                        f"{self.name}.{command.name}.error",
                        {
                            "command": command.name,
                            "error": str(exc),
                        },
                        level=logging.ERROR,
                        message=f"Command '{command.name}' failed: {exc}",
                    )
                raise
            else:
                results[command.name] = output
                if self._logger is not None:
                    self._logger.log_event(
                        f"{self.name}.{command.name}.complete",
                        {"command": command.name},
                        message=f"Command '{command.name}' completed successfully.",
                    )
        return results
