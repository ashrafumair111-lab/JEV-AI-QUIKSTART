"""The user-facing exception hierarchy used across the JEV AI Quickstart.

Every failure a user can act on is raised as a :class:`QuickstartError`
subclass: the message reads like a sentence and ``hint`` says what to do next.
``src/main.py`` renders these and nothing else, so a traceback always means a
bug in this repository rather than something the reader did wrong.
"""

from __future__ import annotations

__all__ = [
    "ConfigurationError",
    "MissingDependencyError",
    "QuickstartError",
    "UpstreamError",
]


class QuickstartError(RuntimeError):
    """Base class for errors that are safe and useful to show to a user.

    Args:
        message: One sentence describing what went wrong.
        hint: Optional multi-line instructions to resolve the problem.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def render(self) -> str:
        """Return the message plus its hint, formatted for a terminal."""
        if self.hint:
            return f"{self.message}\n\nHow to fix it:\n{self.hint}"
        return self.message


class ConfigurationError(QuickstartError):
    """The environment is incomplete or invalid.

    Raised for a missing API key, an unparsable number, or thresholds that
    contradict each other.
    """


class MissingDependencyError(QuickstartError):
    """A third-party package required for the requested path is not installed."""


class UpstreamError(QuickstartError):
    """Jev, the LLM provider or the network returned an error.

    Raised when the condition is not something the user can fix by editing
    ``.env`` alone (rate limits, outages, invalid credentials at the provider).
    """
