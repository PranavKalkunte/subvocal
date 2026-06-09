"""Correction-capture loop and fine-tuning hook implementation for Subvocal SDK.
"""

import json
import logging
import os
import time
from typing import Any

from pydantic import BaseModel, Field

from subvocal.context.schema import UserContext

logger = logging.getLogger(__name__)


class CorrectionLogEntry(BaseModel):
    """Represents a single correction entry logged when a user fixes an incorrect reconstruction."""
    timestamp: float = Field(default_factory=time.time, description="Epoch timestamp of the correction")
    raw_shorthand: str = Field(description="Raw shorthand query input by user")
    decoded_intent: str = Field(description="The incorrect decoded intent text returned by the system")
    corrected_intent: str = Field(description="The correct intent text provided by the user as correction")
    context_snapshot: UserContext = Field(description="Snapshot of the UserContext when the prediction was made")


class CorrectionManager:
    """Manages local storage and retrieval of user correction log entries."""

    def __init__(self, log_path: str | None = None):
        """Initializes the manager.

        Args:
            log_path: Path to the JSONL log file. Defaults to 'sdk/data/corrections_log.jsonl'.
        """
        if log_path is None:
            self.log_path = os.path.join(
                os.path.expanduser("~"), ".subvocal", "data", "corrections_log.jsonl"
            )
        else:
            self.log_path = os.path.abspath(log_path)

    def log_correction(
        self,
        raw_shorthand: str,
        decoded_intent: str,
        corrected_intent: str,
        context: UserContext,
    ) -> CorrectionLogEntry:
        """Logs a new correction entry to the local JSONL file."""
        entry = CorrectionLogEntry(
            timestamp=time.time(),
            raw_shorthand=raw_shorthand,
            decoded_intent=decoded_intent,
            corrected_intent=corrected_intent,
            context_snapshot=context,
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Write to JSONL
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

        return entry

    def get_corrections(self) -> list[CorrectionLogEntry]:
        """Retrieves all logged corrections from the local file."""
        if not os.path.exists(self.log_path):
            return []

        entries = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(CorrectionLogEntry.model_validate_json(line))
                    except Exception as e:
                        # Skip corrupted lines gracefully
                        logger.warning("Failed to parse correction log line: %s", e)
        return entries

    def clear_logs(self) -> None:
        """Clears all logged corrections in the file."""
        if os.path.exists(self.log_path):
            os.remove(self.log_path)


class FinetuningHook:
    """Converts correction logs into training datasets for fine-tuning LLMs."""

    @staticmethod
    def _build_user_prompt(entry: "CorrectionLogEntry") -> str:
        """Builds the canonical user prompt for a correction entry, matching live inference format."""
        from subvocal.shorthand.decoder import heuristic_decode_phrase

        from .prompts import PromptManager

        contacts_str = ", ".join([f"{c.name} ({c.shorthand_name})" for c in entry.context_snapshot.contacts])
        calendar_str = ", ".join(
            [f"{e.title} at {e.start_time} ({e.shorthand_title})" for e in entry.context_snapshot.calendar]
        )
        web_elements_str = ", ".join(
            [f"{el.label} [{el.element_type}] ({el.shorthand_label})" for el in entry.context_snapshot.app_state.visible_elements]
        )
        history_str = "\n".join([f"{m.role}: {m.text}" for m in entry.context_snapshot.conversation_history])

        heur_phrase, _ = heuristic_decode_phrase(
            entry.raw_shorthand,
            ui_elements=[el.label for el in entry.context_snapshot.app_state.visible_elements],
            contacts=[c.name for c in entry.context_snapshot.contacts],
            calendar_events=[e.title for e in entry.context_snapshot.calendar],
        )

        return PromptManager().format_prompt(
            noisy_input=entry.raw_shorthand,
            heuristic_recommendation=heur_phrase,
            web_context=web_elements_str or "(none)",
            calendar=calendar_str or "(none)",
            contacts=contacts_str or "(none)",
            history=history_str or "(none)",
            version="v1",
        )

    @staticmethod
    def export_to_openai(
        entries: list[CorrectionLogEntry],
        system_instruction: str = "Translate silent speech shorthand and context into correct system actions.",
    ) -> list[dict[str, Any]]:
        """Converts entries to OpenAI chat fine-tuning format.

        Format:
            {"messages": [{"role": "system", "content": ...}, {"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
        """
        return [
            {
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": FinetuningHook._build_user_prompt(entry)},
                    {"role": "assistant", "content": entry.corrected_intent},
                ]
            }
            for entry in entries
        ]

    @staticmethod
    def export_to_gemini(
        entries: list[CorrectionLogEntry],
        system_instruction: str = "Translate silent speech shorthand and context into correct system actions.",
    ) -> list[dict[str, Any]]:
        """Converts entries to Google Gemini fine-tuning format.

        Format:
            {"contents": [...], "systemInstruction": {...}}
        """
        return [
            {
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "contents": [
                    {"role": "user", "parts": [{"text": FinetuningHook._build_user_prompt(entry)}]},
                    {"role": "model", "parts": [{"text": entry.corrected_intent}]},
                ],
            }
            for entry in entries
        ]

    @staticmethod
    def export_to_jsonl(data: list[dict[str, Any]], output_path: str) -> int:
        """Writes the exported dataset list to a JSONL file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return len(data)
