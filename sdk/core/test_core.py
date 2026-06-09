"""Unit tests for the core models, interfaces, and pipeline orchestrator.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import time
from typing import Optional

from core.models import Sample, Frame, CommandToken
from core.pipeline import SubvocalPipeline
from core.testing import MockHardwareSource, MockLLMProvider, MockActionExecutor, MockContextProvider


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
        def dummy_classify(_frame: Frame) -> Optional[CommandToken]:
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

        # 4. Wait for timeout and step again (3× the timeout to avoid timer jitter on CI)
        time.sleep(0.15)
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

        def dummy_classify(_frame: Frame) -> Optional[CommandToken]:
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
