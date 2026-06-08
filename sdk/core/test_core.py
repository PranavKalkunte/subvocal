"""Unit tests for the core models, interfaces, and pipeline orchestrator.
"""

import unittest
import time
import numpy as np
from typing import List, Optional, Any
from context.schema import UserContext, AppState

from core.models import Sample, Frame, CommandToken, Intent, Action
from core.interfaces import HardwareSource, LLMProvider, ActionExecutor, ContextProvider
from core.pipeline import SubvocalPipeline


# --- Mock Implementations for Testing ---

class MockHardwareSource(HardwareSource):
    """Generates continuous fake samples representing a 4-channel sEMG setup."""

    def __init__(self, fs: float = 1000.0, num_channels: int = 4):
        self._fs = fs
        self._num_channels = num_channels
        self._connected = False
        self._sample_counter = 0

    def start(self) -> None:
        self._connected = True

    def stop(self) -> None:
        self._connected = False

    def read_frame(self, window_ms: int) -> Frame:
        now = time.time()
        num_samples = int((window_ms / 1000.0) * self._fs)
        samples = []
        for _ in range(num_samples):
            self._sample_counter += 1
            # Generate dummy signal (sine waves + noise)
            t = self._sample_counter / self._fs
            channels = [float(np.sin(2 * np.pi * 10.0 * t + i) + np.random.normal(0, 0.1)) for i in range(self._num_channels)]
            samples.append(Sample(timestamp=now, channels=channels, sample_index=self._sample_counter))

        return Frame(
            samples=samples,
            start_time=now - (window_ms / 1000.0),
            end_time=now,
            fs=self._fs
        )

    def is_connected(self) -> bool:
        return self._connected


class MockLLMProvider(LLMProvider):
    """Simulates shorthand intent reconstruction."""

    def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent:
        # Join tokens into a shorthand representation
        shorthand = " ".join([t.text for t in tokens])

        # Simple rule-based mapping to mimic reconstruction
        if "gt" in shorthand or "goto" in shorthand:
            cmd = "GOTO"
            args = ["google.com"]
        elif "clk" in shorthand or "click" in shorthand:
            cmd = "CLICK"
            args = ["Search Button"]
        else:
            cmd = "UNKNOWN"
            args = []

        resolved = f"{cmd} {' '.join(args)}".strip()
        return Intent(
            command=cmd,
            arguments=args,
            confidence=0.95,
            resolved_text=resolved,
            raw_shorthand=shorthand,
            timestamp=time.time()
        )

    def get_provider_name(self) -> str:
        return "mock_llm"


class MockActionExecutor(ActionExecutor):
    """Tracks executed actions for verification."""

    def __init__(self):
        self.executed_actions: List[Action] = []

    def execute(self, action: Action) -> Any:
        self.executed_actions.append(action)
        return {"status": "success", "action_id": action.intent_id}

    def can_execute(self, action: Action) -> bool:
        return action.action_type in ("goto", "click", "unknown")


class MockContextProvider(ContextProvider):
    """Provides a dummy UserContext snapshot."""

    def get_context(self) -> UserContext:
        return UserContext(
            app_state=AppState(
                current_app="Browser",
                page_title="Google Search",
                visible_elements=[]
            )
        )

    def get_provider_name(self) -> str:
        return "mock_context"


# --- Unit Tests Suite ---

class TestCoreAPI(unittest.TestCase):
    """Test cases for models, interfaces, and pipeline coordination."""

    def test_sample_and_frame_serialization(self):
        """Verify that Sample and Frame models serialize and deserialize correctly."""
        s = Sample(timestamp=123456789.0, channels=[0.1, -0.2, 0.35], sample_index=42)
        self.assertEqual(s.sample_index, 42)
        self.assertEqual(s.channels[1], -0.2)

        # Build Frame
        frame = Frame(
            samples=[s, s],
            start_time=123456789.0,
            end_time=123456789.1,
            fs=1000.0
        )
        self.assertEqual(len(frame.samples), 2)
        
        # Test numpy conversion
        arr = frame.to_numpy()
        self.assertEqual(arr.shape, (2, 3))
        self.assertEqual(arr[0, 1], -0.2)

    def test_pipeline_streaming_and_timeout(self):
        """Test that SubvocalPipeline correctly debounces, aggregates tokens, and triggers actions on timeout."""
        hw = MockHardwareSource()
        hw.start()

        # Mock classifier callback: returns a token every 3rd call, otherwise None
        call_count = 0
        def dummy_classify(frame: Frame) -> Optional[CommandToken]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return CommandToken(text="gt", confidence=0.85, timestamp=time.time())
            elif call_count == 3:
                return CommandToken(text="ggl", confidence=0.90, timestamp=time.time())
            return None

        llm = MockLLMProvider()
        ctx = MockContextProvider()
        exec_tracker = MockActionExecutor()

        # Instantiate pipeline with short timeout (e.g. 0.05 seconds) for faster testing
        pipeline = SubvocalPipeline(
            hardware=hw,
            classify_fn=dummy_classify,
            llm_provider=llm,
            context_provider=ctx,
            executor=exec_tracker,
            phrase_timeout_seconds=0.05
        )

        # 1. Step 1: produces 'gt' token
        action = pipeline.step(window_ms=10)
        self.assertIsNone(action)
        self.assertEqual(len(pipeline.token_buffer), 1)
        self.assertEqual(pipeline.token_buffer[0].text, "gt")

        # 2. Step 2: produces None, no timeout
        action = pipeline.step(window_ms=10)
        self.assertIsNone(action)

        # 3. Step 3: produces 'ggl' token
        action = pipeline.step(window_ms=10)
        self.assertIsNone(action)
        self.assertEqual(len(pipeline.token_buffer), 2)

        # 4. Wait for timeout and step again
        time.sleep(0.06)
        action = pipeline.step(window_ms=10)

        # Verify action was executed
        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, "goto")
        self.assertEqual(action.params["arguments"], ["google.com"])
        self.assertEqual(len(pipeline.token_buffer), 0)  # Buffer should be cleared
        self.assertEqual(len(exec_tracker.executed_actions), 1)

    def test_pipeline_force_process(self):
        """Verify that process_phrase can be called manually to bypass the timeout."""
        hw = MockHardwareSource()
        hw.start()

        def dummy_classify(frame: Frame) -> Optional[CommandToken]:
            return CommandToken(text="clk", confidence=0.99, timestamp=time.time())

        llm = MockLLMProvider()
        ctx = MockContextProvider()
        exec_tracker = MockActionExecutor()

        pipeline = SubvocalPipeline(
            hardware=hw,
            classify_fn=dummy_classify,
            llm_provider=llm,
            context_provider=ctx,
            executor=exec_tracker,
            phrase_timeout_seconds=10.0  # long timeout
        )

        # Run one step
        pipeline.step(window_ms=10)
        self.assertEqual(len(pipeline.token_buffer), 1)

        # Force process
        action = pipeline.process_phrase()
        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, "click")
        self.assertEqual(len(pipeline.token_buffer), 0)
        self.assertEqual(len(exec_tracker.executed_actions), 1)
