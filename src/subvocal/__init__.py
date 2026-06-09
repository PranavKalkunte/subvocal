"""Subvocal SDK: hardware-agnostic middleware connecting sEMG silent-speech
interfaces to LLM-driven agents.

The top-level package re-exports the core public API::

    from subvocal import SubvocalPipeline, Sample, Frame, CommandToken

Optional subsystems live in subpackages and may require extras:

- ``subvocal.emg_core`` — DSP and classifiers (``pip install "subvocal[ml]"``)
- ``subvocal.hardware`` — drivers and public-dataset loaders (``[hardware]`` for datasets)
- ``subvocal.tts`` — audio feedback (``[tts]`` outside macOS)
- ``subvocal.mcp`` — Model Context Protocol server (``subvocal-mcp`` console command)
"""

import logging

from .core.interfaces import ActionExecutor, ContextProvider, HardwareSource, LLMProvider
from .core.llm_providers import HeuristicProvider, resolve_provider
from .core.models import Action, CommandToken, Frame, Intent, Sample
from .core.pipeline import PipelineStats, SubvocalPipeline
from .exceptions import (
    CalibrationError,
    ConfigurationError,
    DecodingError,
    HardwareError,
    MissingDependencyError,
    PolicyViolationError,
    ProviderError,
    SubvocalError,
)
from .paths import get_data_dir, get_models_dir

__version__ = "1.0.0rc1"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Sample",
    "Frame",
    "CommandToken",
    "Intent",
    "Action",
    "HardwareSource",
    "LLMProvider",
    "ActionExecutor",
    "ContextProvider",
    "SubvocalPipeline",
    "PipelineStats",
    "HeuristicProvider",
    "resolve_provider",
    "get_data_dir",
    "get_models_dir",
    "SubvocalError",
    "ConfigurationError",
    "HardwareError",
    "MissingDependencyError",
    "ProviderError",
    "DecodingError",
    "PolicyViolationError",
    "CalibrationError",
    "__version__",
]
