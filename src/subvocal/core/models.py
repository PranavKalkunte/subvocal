"""Core data models representing the flow from raw physiological signals to actions.

Defines:
- Sample: A single point-in-time multi-channel sensor reading.
- Frame: A window of Samples buffered for classifier consumption.
- CommandToken: A token/shorthand output by a classifier.
- Intent: Reconstructed semantic command and arguments.
- Action: Executable instruction dispatched to an executor.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class Sample(BaseModel):
    """A single multichannel biometric time-series sample."""
    timestamp: float = Field(description="Epoch timestamp of the sample in seconds")
    channels: list[float] = Field(description="Electrophysiological readings from each electrode channel")
    sample_index: int | None = Field(default=None, description="Optional continuous sample index")

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("channels must be non-empty (at least one electrode channel)")
        # Optional: enforce reasonable channel count to catch misconfiguration early
        if len(v) > 128:
            raise ValueError("channels length exceeds plausible electrode count (>128)")
        return v


class Frame(BaseModel):
    """A window/segment of Samples buffered for signal processing or ML inference."""
    samples: list[Sample] = Field(description="Ordered list of samples in this frame")
    start_time: float = Field(description="Epoch timestamp representing the frame start")
    end_time: float = Field(description="Epoch timestamp representing the frame end")
    fs: float = Field(description="Sampling frequency of the hardware in Hz")

    @model_validator(mode="after")
    def validate_frame_ordering(self) -> "Frame":
        if self.end_time <= self.start_time:
            raise ValueError(f"end_time ({self.end_time}) must be greater than start_time ({self.start_time})")
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")
        return self

    def to_numpy(self) -> np.ndarray:
        """Converts the frame samples into a 2D NumPy array.

        Returns:
            np.ndarray of shape (num_samples, num_channels)
        """
        if not self.samples:
            return np.empty((0, 0))
        # Extract channels list from each sample
        data = [sample.channels for sample in self.samples]
        return np.array(data, dtype=np.float32)


class CommandToken(BaseModel):
    """A token predicted by the raw classifier with associated confidence."""
    text: str = Field(description="Raw phonetic shorthand command or token (e.g., 'clk', 'typ', 'gt')")
    confidence: float = Field(description="Softmax or classifier confidence score between 0.0 and 1.0")
    timestamp: float = Field(description="Epoch timestamp when the token was classified")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional classifier specific metadata")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v


class Intent(BaseModel):
    """Semantic intent reconstructed from a token stream and active context."""
    command: str = Field(description="Reconstructed base command (e.g., 'GOTO', 'SEARCH', 'CLICK')")
    arguments: list[str] = Field(default_factory=list, description="Reconstructed command arguments")
    confidence: float = Field(description="Intent reconstruction confidence score between 0.0 and 1.0")
    resolved_text: str = Field(description="Full resolved command string (e.g., 'GOTO google.com')")
    raw_shorthand: str = Field(description="Raw input shorthand string that was decoded")
    timestamp: float = Field(description="Epoch timestamp when the intent was resolved")
    context_snapshot_id: str | None = Field(default=None, description="Identifier linking to the UserContext snapshot used")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v


class Action(BaseModel):
    """The executable instruction dispatched to the system or agent executor."""
    action_type: str = Field(description="Type of action (e.g., 'click', 'type', 'goto', 'tts')")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters mapping to device tools or APIs")
    intent_id: str = Field(description="Unique identifier of the Intent that triggered this Action")
    timestamp: float = Field(description="Epoch timestamp when the action was generated")
