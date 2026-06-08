"""Correction-capture loop and fine-tuning hook implementation for Subvocal SDK.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from context.schema import UserContext


class CorrectionLogEntry(BaseModel):
    """Represents a single correction entry logged when a user fixes an incorrect reconstruction."""
    timestamp: float = Field(default_factory=time.time, description="Epoch timestamp of the correction")
    raw_shorthand: str = Field(description="Raw shorthand query input by user")
    decoded_intent: str = Field(description="The incorrect decoded intent text returned by the system")
    corrected_intent: str = Field(description="The correct intent text provided by the user as correction")
    context_snapshot: UserContext = Field(description="Snapshot of the UserContext when the prediction was made")


class CorrectionManager:
    """Manages local storage and retrieval of user correction log entries."""

    def __init__(self, log_path: Optional[str] = None):
        """Initializes the manager.

        Args:
            log_path: Path to the JSONL log file. Defaults to 'sdk/data/corrections_log.jsonl'.
        """
        if log_path is None:
            # Resolve relative to the current workspace sdk/data directory
            self.log_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "data", "corrections_log.jsonl")
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

    def get_corrections(self) -> List[CorrectionLogEntry]:
        """Retrieves all logged corrections from the local file."""
        if not os.path.exists(self.log_path):
            return []

        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(CorrectionLogEntry.model_validate_json(line))
                    except Exception as e:
                        # Skip corrupted lines gracefully
                        print(f"[Warning] Failed to parse correction log line: {e}")
        return entries

    def clear_logs(self) -> None:
        """Clears all logged corrections in the file."""
        if os.path.exists(self.log_path):
            os.remove(self.log_path)


class FinetuningHook:
    """Converts correction logs into training datasets for fine-tuning LLMs."""

    @staticmethod
    def export_to_openai(
        entries: List[CorrectionLogEntry],
        system_instruction: str = "Translate silent speech shorthand and context into correct system actions.",
    ) -> List[Dict[str, Any]]:
        """Converts entries to OpenAI chat fine-tuning format.

        Format:
            {"messages": [{"role": "system", "content": ...}, {"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
        """
        from .prompts import PromptManager
        prompt_mgr = PromptManager()

        dataset = []
        for entry in entries:
            # Reconstruct the user prompt context
            contacts_str = ", ".join([f"{c.name} ({c.shorthand_name})" for c in entry.context_snapshot.contacts])
            calendar_str = ", ".join([f"{e.title} ({e.shorthand_title})" for e in entry.context_snapshot.calendar])
            web_elements_str = ", ".join([f"{el.label} ({el.shorthand_label})" for el in entry.context_snapshot.app_state.visible_elements])
            history_str = "\n".join([f"{m.role}: {m.text}" for m in entry.context_snapshot.conversation_history])

            user_prompt = prompt_mgr.format_prompt(
                noisy_input=entry.raw_shorthand,
                heuristic_recommendation=entry.decoded_intent,  # Use predicted intent as heuristic recommendation
                web_context=web_elements_str or "(none)",
                calendar=calendar_str or "(none)",
                contacts=contacts_str or "(none)",
                history=history_str or "(none)",
                version="v1"
            )

            dataset.append({
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": entry.corrected_intent}
                ]
            })
        return dataset

    @staticmethod
    def export_to_gemini(
        entries: List[CorrectionLogEntry],
        system_instruction: str = "Translate silent speech shorthand and context into correct system actions.",
    ) -> List[Dict[str, Any]]:
        """Converts entries to Google Gemini fine-tuning format.

        Format:
            {"contents": [{"role": "user", "parts": [{"text": ...}]}, {"role": "model", "parts": [{"text": ...}]}], "systemInstruction": {"parts": [{"text": ...}]}}
        """
        from .prompts import PromptManager
        prompt_mgr = PromptManager()

        dataset = []
        for entry in entries:
            contacts_str = ", ".join([f"{c.name} ({c.shorthand_name})" for c in entry.context_snapshot.contacts])
            calendar_str = ", ".join([f"{e.title} ({e.shorthand_title})" for e in entry.context_snapshot.calendar])
            web_elements_str = ", ".join([f"{el.label} ({el.shorthand_label})" for el in entry.context_snapshot.app_state.visible_elements])
            history_str = "\n".join([f"{m.role}: {m.text}" for m in entry.context_snapshot.conversation_history])

            user_prompt = prompt_mgr.format_prompt(
                noisy_input=entry.raw_shorthand,
                heuristic_recommendation=entry.decoded_intent,
                web_context=web_elements_str or "(none)",
                calendar=calendar_str or "(none)",
                contacts=contacts_str or "(none)",
                history=history_str or "(none)",
                version="v1"
            )

            dataset.append({
                "systemInstruction": {
                    "parts": [{"text": system_instruction}]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt}]
                    },
                    {
                        "role": "model",
                        "parts": [{"text": entry.corrected_intent}]
                    }
                ]
            })
        return dataset

    @staticmethod
    def export_to_jsonl(data: List[Dict[str, Any]], output_path: str) -> int:
        """Writes the exported dataset list to a JSONL file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return len(data)
