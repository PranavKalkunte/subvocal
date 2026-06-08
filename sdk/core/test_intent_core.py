"""Unit tests for the intent reconstruction core (prompts, providers, context aggregation, corrections).
"""

import unittest
import os
import tempfile
import time
import json
from typing import List

from context.schema import (
    UserContext,
    Contact,
    CalendarEvent,
    LocationInfo,
    AppState,
    UIElement,
)
from core.models import CommandToken
from core.prompts import PromptManager
from core.llm_providers import (
    ClaudeProvider,
    OpenAIProvider,
    GeminiProvider,
    LlamaProvider,
)
from context.providers import (
    CalendarContextProvider,
    ContactsContextProvider,
    LocationContextProvider,
    AppStateContextProvider,
    CompositeContextProvider,
)
from core.corrections import CorrectionManager, FinetuningHook


class TestIntentCore(unittest.TestCase):
    """Test suite for the intent reconstruction module components."""

    def test_prompt_manager(self):
        """Test that PromptManager retrieves and formats templates correctly."""
        pm = PromptManager(default_version="v1")
        
        # Verify v1 and v2 template presence
        self.assertIn("v1", pm.templates)
        self.assertIn("v2", pm.templates)
        
        # Format prompt and check variables substitution
        prompt_str = pm.format_prompt(
            noisy_input="gt ggl",
            heuristic_recommendation="GOTO google.com",
            web_context="Submit Button",
            calendar="Team Sync",
            contacts="Alice Smith",
            history="user: hello",
            version="v1"
        )
        self.assertIn("gt ggl", prompt_str)
        self.assertIn("GOTO google.com", prompt_str)
        self.assertIn("Team Sync", prompt_str)
        self.assertIn("Alice Smith", prompt_str)

    def test_context_providers(self):
        """Test modular context providers and composite context aggregation."""
        contact = Contact(id="1", name="Bob", shorthand_name="bb", email="bob@dev.com")
        event = CalendarEvent(id="1", title="Code Review", start_time="2026-06-08T09:00:00Z", end_time="2026-06-08T10:00:00Z", shorthand_title="cd rvw")
        loc = LocationInfo(latitude=37.7749, longitude=-122.4194, place_name="San Francisco")
        ui = UIElement(element_id="btn-1", element_type="button", label="Confirm", shorthand_label="cnfm")
        app = AppState(current_app="Terminal", visible_elements=[ui])

        # Instantiate modular providers
        p_contacts = ContactsContextProvider([contact])
        p_calendar = CalendarContextProvider([event])
        p_location = LocationContextProvider(loc)
        p_app_state = AppStateContextProvider(app)

        # Composite provider
        composite = CompositeContextProvider([p_contacts, p_calendar, p_location, p_app_state])
        merged_ctx = composite.get_context()

        # Assert correct merges
        self.assertEqual(len(merged_ctx.contacts), 1)
        self.assertEqual(merged_ctx.contacts[0].name, "Bob")
        self.assertEqual(len(merged_ctx.calendar), 1)
        self.assertEqual(merged_ctx.calendar[0].title, "Code Review")
        self.assertEqual(merged_ctx.location.place_name, "San Francisco")
        self.assertEqual(merged_ctx.app_state.current_app, "Terminal")
        self.assertEqual(len(merged_ctx.app_state.visible_elements), 1)

    def test_llm_providers_mock(self):
        """Test all concrete providers using local mock HTTP responses."""
        tokens = [
            CommandToken(text="gt", confidence=0.9, timestamp=time.time()),
            CommandToken(text="ggl", confidence=0.8, timestamp=time.time())
        ]
        context = UserContext(
            app_state=AppState(current_app="Browser", visible_elements=[])
        )

        # 1. Claude Provider Mock
        claude = ClaudeProvider(api_key="mock", mock_response="GOTO google.com")
        intent_claude = claude.reconstruct_intent(tokens, context)
        self.assertEqual(intent_claude.command, "GOTO")
        self.assertEqual(intent_claude.arguments, ["google.com"])
        self.assertEqual(intent_claude.resolved_text, "GOTO google.com")

        # 2. OpenAI Provider Mock
        openai = OpenAIProvider(api_key="mock", mock_response="CLICK Search")
        intent_openai = openai.reconstruct_intent(tokens, context)
        self.assertEqual(intent_openai.command, "CLICK")
        self.assertEqual(intent_openai.arguments, ["Search"])

        # 3. Gemini Provider Mock
        gemini = GeminiProvider(api_key="mock", mock_response="TYPE Hello")
        intent_gemini = gemini.reconstruct_intent(tokens, context)
        self.assertEqual(intent_gemini.command, "TYPE")
        self.assertEqual(intent_gemini.arguments, ["Hello"])

        # 4. Llama Provider Mock
        llama = LlamaProvider(ollama_host="http://localhost:11434", mock_response="SEARCH weather")
        intent_llama = llama.reconstruct_intent(tokens, context)
        self.assertEqual(intent_llama.command, "SEARCH")
        self.assertEqual(intent_llama.arguments, ["weather"])

    def test_corrections_and_finetuning(self):
        """Test logging corrections and exporting them to LLM fine-tuning files."""
        context = UserContext(
            contacts=[Contact(id="1", name="Alice", shorthand_name="alc")],
            calendar=[],
            app_state=AppState(current_app="Home")
        )

        # Create temporary file for logging corrections
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            manager = CorrectionManager(log_path=tmp_path)
            
            # 1. Log a correction
            entry = manager.log_correction(
                raw_shorthand="typ alc",
                decoded_intent="TYPE Alex",
                corrected_intent="TYPE Alice",
                context=context
            )
            self.assertEqual(entry.raw_shorthand, "typ alc")
            self.assertEqual(entry.corrected_intent, "TYPE Alice")

            # 2. Retrieve correction and verify
            entries = manager.get_corrections()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].decoded_intent, "TYPE Alex")
            self.assertEqual(entries[0].context_snapshot.contacts[0].name, "Alice")

            # 3. Test OpenAI export
            openai_data = FinetuningHook.export_to_openai(entries)
            self.assertEqual(len(openai_data), 1)
            self.assertEqual(openai_data[0]["messages"][2]["content"], "TYPE Alice")
            self.assertEqual(openai_data[0]["messages"][0]["role"], "system")

            # 4. Test Gemini export
            gemini_data = FinetuningHook.export_to_gemini(entries)
            self.assertEqual(len(gemini_data), 1)
            self.assertEqual(gemini_data[0]["contents"][1]["parts"][0]["text"], "TYPE Alice")

            # 5. Test JSONL export writing
            export_path = tmp_path + ".export.jsonl"
            count = FinetuningHook.export_to_jsonl(openai_data, export_path)
            self.assertEqual(count, 1)
            self.assertTrue(os.path.exists(export_path))
            
            # Read and verify exported file content
            with open(export_path, "r", encoding="utf-8") as f:
                content = json.loads(f.read().strip())
                self.assertEqual(content["messages"][2]["content"], "TYPE Alice")

            # Clean up export file
            if os.path.exists(export_path):
                os.remove(export_path)

        finally:
            # Clean up logs
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
