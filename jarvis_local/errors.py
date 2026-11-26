"""Exception types shared across Jarvis Local."""

from __future__ import annotations


class JarvisError(RuntimeError):
    """Base class for expected, user-facing errors."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class ConfigError(JarvisError):
    """Invalid or missing configuration."""


class BackendError(JarvisError):
    """The local model backend is unavailable or returned an error."""


class PolicyError(JarvisError):
    """A filesystem, command-bravery, or approval policy was violated."""


class SkillError(JarvisError):
    """A skill could not be installed, loaded, or validated."""


class IntegrationError(JarvisError):
    """A third-party integration (for example Google) is not set up."""
