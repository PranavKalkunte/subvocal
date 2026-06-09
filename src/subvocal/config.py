import copy
import os
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from subvocal.exceptions import ConfigurationError


class HardwareConfig(BaseModel):
    model_config = {"extra": "forbid"}

    sample_rate: int = Field(default=250, description="Sample rate of the hardware in Hz")
    num_channels: int = Field(default=4, description="Number of EMG channels")
    port: str | None = Field(default=None, description="Serial port or interface address")
    simulated: bool = Field(default=True, description="Whether to run in simulated/replay mode")


class DSPConfig(BaseModel):
    model_config = {"extra": "forbid"}

    bandpass_low: float = Field(default=1.3, description="Low cutoff frequency for bandpass in Hz")
    bandpass_high: float = Field(default=50.0, description="High cutoff frequency for bandpass in Hz")
    bandpass_order: int = Field(default=4, description="Butterworth bandpass filter order")
    notch_freq: float = Field(default=60.0, description="Notch filter frequency (60.0 for US, 50.0 for EU)")
    notch_q: float = Field(default=30.0, description="Notch filter Q factor")
    smooth_window: int = Field(default=3, description="Samples window for moving average smoothing")


class ClassifierConfig(BaseModel):
    model_config = {"extra": "forbid"}

    type: str = Field(default="rf", description="Classifier model type ('rf', 'svm', 'cnn', 'gru')")
    model_path: str | None = Field(default=None, description="Path to trained model weights file")
    confidence_threshold: float = Field(default=0.75, description="Min classification confidence floor")
    cooldown_ms: int = Field(default=900, description="Debounce cooldown in milliseconds after positive detection")
    rf_n_estimators: int = Field(default=200, description="Number of estimators for Random Forest")


class ProviderConfig(BaseModel):
    model_config = {"extra": "forbid"}

    llm_provider: str = Field(default="mock", description="LLM provider name ('gemini', 'mock')")
    api_key: str | None = Field(default=None, description="API key for the LLM provider")
    model_name: str | None = Field(default=None, description="LLM model name to resolve intents")
    temperature: float = Field(default=0.0, description="LLM temperature parameter")


class PolicyConfig(BaseModel):
    model_config = {"extra": "forbid"}

    dry_run: bool = Field(default=False, description="Dry run mode (resolves intents but does not execute actions)")
    raise_on_policy_violation: bool = Field(default=False, description="Raise PolicyViolationError if unauthorized")


class TelemetryConfig(BaseModel):
    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=False, description="Enable prometheus exporter and event logs")
    prometheus_port: int = Field(default=8000, description="Port for prometheus exporter")
    flush_interval_seconds: float = Field(default=30.0, description="Stats flush interval in seconds")


class RuntimeConfig(BaseModel):
    model_config = {"extra": "forbid"}

    phrase_timeout_seconds: float = Field(default=1.5, description="Silence duration in seconds before triggering intent")
    session_liveness_timeout: float = Field(default=30.0, description="Timeout in seconds for watchdog expectation of frames")


class AuthConfig(BaseModel):
    model_config = {"extra": "forbid"}

    api_key: str | None = Field(default=None, description="API key required for external calls to MCP/HTTP server")


class SubvocalConfig(BaseModel):
    model_config = {"extra": "forbid"}

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    dsp: DSPConfig = Field(default_factory=DSPConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)


def merge_env_overrides(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Inspects environment variables for SUBVOCAL_ keys and merges them into the config dict."""
    merged = copy.deepcopy(config_dict)
    prefix = "SUBVOCAL_"

    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        # Convert e.g. SUBVOCAL_HARDWARE__SAMPLE_RATE to ['hardware', 'sample_rate']
        key_path = env_key[len(prefix):].lower()
        parts = key_path.split("__")

        curr = merged
        for part in parts[:-1]:
            if part not in curr or not isinstance(curr[part], dict):
                curr[part] = {}
            curr = curr[part]

        last_part = parts[-1]

        # Parse string values to Python types
        val: Any = env_val
        val_lower = env_val.lower()
        if val_lower in ("true", "yes", "1"):
            val = True
        elif val_lower in ("false", "no", "0"):
            val = False
        elif val_lower in ("none", "null"):
            val = None
        else:
            try:
                if "." in env_val:
                    val = float(env_val)
                else:
                    val = int(env_val)
            except ValueError:
                pass

        curr[last_part] = val

    return merged


def load_config(path: str | None = None) -> SubvocalConfig:
    """Loads a SubvocalConfig from a YAML file (if provided) and merges environment overrides."""
    config_dict: dict[str, Any] = {}

    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if content is not None:
                    if not isinstance(content, dict):
                        raise ConfigurationError("Configuration file root must be a YAML dictionary")
                    config_dict = content
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to read configuration file at '{path}': {e}") from e

    merged_dict = merge_env_overrides(config_dict)

    try:
        return SubvocalConfig.model_validate(merged_dict)
    except ValidationError as e:
        raise ConfigurationError(f"Invalid configuration parameters: {e}") from e
