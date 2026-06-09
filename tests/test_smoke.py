"""End-to-end smoke test for the Subvocal Middleware pipeline.

Simulates raw physiological signal ingestion, filters and classifies the frame,
buffers the classified token, triggers intent reconstruction, and dispatches the action.
"""

import os
import sys
import time
import unittest
from typing import List, Optional


from subvocal.core.models import Frame, CommandToken, Intent, Action
from subvocal.core.interfaces import LLMProvider, ActionExecutor, ContextProvider
from subvocal.core.pipeline import SubvocalPipeline
from subvocal.context.schema import UserContext, AppState
from subvocal.hardware.drivers import SyntheticSignalGenerator


class SmokeExecutor(ActionExecutor):
    """Action executor recording dispatched commands for verification."""
    def __init__(self):
        self.executed_actions: List[Action] = []

    def execute(self, action: Action):
        self.executed_actions.append(action)

    def can_execute(self, action: Action) -> bool:
        return action.action_type in ["click", "scroll"]


class SmokeLLM(LLMProvider):
    """Intent decoder translating shorthand tokens."""
    def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent:  # noqa: ARG002
        phrase = " ".join([t.text for t in tokens])
        if "clk" in phrase:
            return Intent(
                command="CLICK",
                arguments=["target: document"],
                resolved_text="Click on the document",
                raw_shorthand=phrase,
                confidence=0.95,
                timestamp=time.time()
            )
        return Intent(
            command="SCROLL",
            arguments=["down"],
            resolved_text="Scroll down",
            raw_shorthand=phrase,
            confidence=0.90,
            timestamp=time.time()
        )

    def get_provider_name(self) -> str:
        return "smoke_llm"


class SmokeContext(ContextProvider):
    """Context provider returning static app state."""
    def get_context(self) -> UserContext:
        return UserContext(
            app_state=AppState(current_app="Browser", page_title="Walkthrough")
        )

    def get_provider_name(self) -> str:
        return "smoke_context"


class TestSubvocalEndToEndSmoke(unittest.TestCase):
    """End-to-end validation of the subvocal middleware stack."""

    def test_e2e_pipeline_execution(self):
        # 1. Initialize components
        hardware = SyntheticSignalGenerator(fs=250.0, num_channels=8)
        executor = SmokeExecutor()
        llm = SmokeLLM()
        context = SmokeContext()

        # Simple RMS classifier function: if amplitude on channel 2 bursts, emit "clk"
        def classify_fn(frame: Frame) -> Optional[CommandToken]:
            import numpy as np
            arr = frame.to_numpy()
            if len(arr) == 0:
                return None
            rms = np.sqrt(np.mean(arr[:, 2] ** 2))
            if rms > 0.8:  # Threshold for synthetic burst
                return CommandToken(text="clk", confidence=0.98, timestamp=time.time())
            return None

        # 2. Build Pipeline
        pipeline = SubvocalPipeline(
            hardware=hardware,
            classify_fn=classify_fn,
            llm_provider=llm,
            context_provider=context,
            executor=executor,
            phrase_timeout_seconds=0.4  # Short timeout for fast test
        )

        hardware.start()
        self.assertTrue(hardware.is_connected())

        try:
            # Step A: Feed resting baseline signals (no gestures)
            for _ in range(5):
                pipeline.step(window_ms=50)
                time.sleep(0.01)

            # Assert buffer is empty
            self.assertEqual(len(pipeline.token_buffer), 0)
            self.assertEqual(len(executor.executed_actions), 0)

            # Step B: Trigger a synthetic contraction burst (clk gesture)
            hardware.trigger_command(command="clk", duration_ms=150)
            time.sleep(0.05)  # Let it propagate

            # Read frame during the burst to register classification token
            action_step = pipeline.step(window_ms=50)
            self.assertIsNone(action_step)  # Action not processed yet (waiting for timeout)
            self.assertEqual(len(pipeline.token_buffer), 1)
            self.assertEqual(pipeline.token_buffer[0].text, "clk")

            # Step C: Let the phrase timeout elapse (0.4s) to trigger reconstruction (2× margin for CI)
            time.sleep(0.8)
            
            # Step the pipeline again, which triggers phrase processing due to timeout
            action_result = pipeline.step(window_ms=50)

            # 3. Assert execution success
            self.assertIsNotNone(action_result)
            self.assertEqual(action_result.action_type, "click")
            self.assertEqual(len(pipeline.token_buffer), 0)  # Buffer cleared
            
            # Verify Executor received the action
            self.assertEqual(len(executor.executed_actions), 1)
            self.assertEqual(executor.executed_actions[0].action_type, "click")
            self.assertEqual(executor.executed_actions[0].params.get("resolved_text"), "Click on the document")

        finally:
            hardware.stop()


if __name__ == "__main__":
    unittest.main()
