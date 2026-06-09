"""Typed exception hierarchy for the Subvocal SDK.

Every error the SDK raises derives from :class:`SubvocalError`, so integrators
can catch one base class at the application boundary::

    try:
        pipeline.step()
    except SubvocalError as e:
        logger.error("subvocal failure: %s", e)

Subclasses also inherit the builtin exception type the SDK historically raised
in each situation (``RuntimeError``, ``ImportError``, ...), so pre-existing
``except RuntimeError`` handlers keep working across upgrades.
"""


class SubvocalError(Exception):
    """Base class for all errors raised by the Subvocal SDK."""


class ConfigurationError(SubvocalError, ValueError):
    """Invalid or missing configuration (API keys, parameters, paths)."""


class HardwareError(SubvocalError, RuntimeError):
    """A hardware source failed: not connected, stream fault, or parse error."""


class MissingDependencyError(SubvocalError, ImportError):
    """An optional dependency is required for the requested feature.

    The message always names the pip extra that provides it.
    """


class ProviderError(SubvocalError, RuntimeError):
    """An LLM provider call failed after exhausting retries."""


class DecodingError(SubvocalError, RuntimeError):
    """Shorthand-to-intent decoding produced no usable result."""


class PolicyViolationError(SubvocalError, PermissionError):
    """An action was rejected by the authorization policy engine.

    Raised only when the pipeline is configured with
    ``raise_on_policy_violation=True``; by default rejections are traced and
    reported through callbacks without raising.
    """


class CalibrationError(SubvocalError, RuntimeError):
    """Per-user classifier calibration or model loading failed."""


__all__ = [
    "SubvocalError",
    "ConfigurationError",
    "HardwareError",
    "MissingDependencyError",
    "ProviderError",
    "DecodingError",
    "PolicyViolationError",
    "CalibrationError",
]
