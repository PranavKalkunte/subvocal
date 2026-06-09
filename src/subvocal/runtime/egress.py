import json
import logging
from typing import Any

logger = logging.getLogger("subvocal.runtime.egress")


class EgressManager:
    """Coordinates downstream data exports, JSONL tracing, and audio speech feedback."""

    def __init__(self, trace_path: str | None = None, tts_engine: Any | None = None):
        self.trace_path = trace_path
        self.tts_engine = tts_engine

    def speak(self, text: str) -> None:
        """Plays speech feedback to the user via TTS."""
        if self.tts_engine:
            try:
                self.tts_engine.speak(text)
            except Exception:
                logger.exception("EgressManager speak failed")
        else:
            logger.info("[Egress Mock TTS]: %s", text)

    def write_trace(self, trace_entry: dict) -> None:
        """Appends a trace entry to the pipeline trace log file."""
        if not self.trace_path:
            return
        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_entry) + "\n")
        except Exception:
            logger.exception("EgressManager failed to write pipeline trace")

    def record_signal(self, output_path: str, frames: list[Any]) -> None:
        """Saves raw biometric frames to a JSON document for training/fine-tuning models."""
        try:
            serialized = []
            for frame in frames:
                serialized.append({
                    "start_time": getattr(frame, "start_time", 0.0),
                    "end_time": getattr(frame, "end_time", 0.0),
                    "samples_count": len(frame.to_numpy()) if hasattr(frame, "to_numpy") else 0,
                })
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2)
            logger.info("Successfully recorded %d biometric frames to %s", len(frames), output_path)
        except Exception:
            logger.exception("EgressManager failed to record raw signal dataset")
