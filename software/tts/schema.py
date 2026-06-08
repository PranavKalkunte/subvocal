"""Pydantic schemas for the Text-to-Speech (TTS) engine.

Defines voices, speeds, formats, and configurations for audio feedback.
"""

from typing import Optional
from pydantic import BaseModel, Field


class TTSConfig(BaseModel):
    """Configuration settings for text-to-speech audio feedback."""
    voice: str = Field(default="alloy", description="OpenAI voice (alloy, echo, fable, onyx, nova, shimmer) or local macOS voice name")
    speed: float = Field(default=1.0, description="Speech rate speed modifier (e.g. 1.0 is normal)")
    audio_format: str = Field(default="mp3", description="Audio format (mp3 or wav)")
    output_dir: str = Field(default="audio_output", description="Subfolder where generated audio files are saved")
    local_volume: float = Field(default=1.0, description="Local playback volume level (0.0 to 1.0)")
