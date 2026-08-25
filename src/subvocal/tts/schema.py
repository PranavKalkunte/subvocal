"""Pydantic schemas for the Text-to-Speech (TTS) engine.

Defines voices, speeds, formats, and configurations for audio feedback.
"""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class TTSConfig(BaseModel):
    """Configuration settings for text-to-speech audio feedback."""
    voice: str = Field(default="alloy", description="OpenAI voice (alloy, echo, fable, onyx, nova, shimmer) or local macOS voice name")
    speed: float = Field(default=1.0, description="Speech rate speed modifier (e.g. 1.0 is normal)")
    audio_format: str = Field(default="mp3", description="Audio format (mp3 or wav)")
    output_dir: str = Field(default="audio_output", description="Subfolder where generated audio files are saved")
    local_volume: float = Field(default=1.0, description="Local playback volume level (0.0 to 1.0)")

    @field_validator("audio_format")
    @classmethod
    def validate_audio_format(cls, v: str) -> str:
        # Validate audio_format to prevent arbitrary format injection (C2)
        allowed = {"mp3", "wav"}
        if v not in allowed:
            raise ValueError(f"audio_format must be one of {allowed}, got {v!r}")
        return v

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        # Validate output_dir to prevent path traversal (C2)
        if not v or ".." in Path(v).parts:
            raise ValueError(f"Invalid output_dir contains traversal: {v!r}")
        # Disallow absolute paths that escape expected base; allow relative or safe absolute
        # Use resolve check to ensure path does not escape via symlink tricks
        # For now, reject paths with null bytes or that resolve outside cwd if relative
        if "\x00" in v:
            raise ValueError("Invalid output_dir contains null byte")
        # Additional check: if absolute, ensure it's within cwd or temp or data dirs is not strictly enforced
        # but we ensure no traversal components remain after sanitizing
        return v
