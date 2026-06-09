import time
import unittest

from subvocal.config import load_config
from subvocal.core.testing import MockActionExecutor, MockContextProvider, MockHardwareSource, MockLLMProvider
from subvocal.exceptions import HardwareError
from subvocal.runtime import Session, SessionWorker


class TestRuntimeSubsystems(unittest.TestCase):
    def test_session_lifecycle_and_state(self):
        """Verify session starts in STARTING, transitions to ACTIVE on data, and CLOSES."""
        config = load_config()
        config.runtime.phrase_timeout_seconds = 0.05
        hw = MockHardwareSource()
        llm = MockLLMProvider()
        ctx = MockContextProvider()
        executor = MockActionExecutor()

        states = []
        def on_state(s):
            states.append(s)

        session = Session(
            id="test-session",
            config=config,
            hardware=hw,
            classify_fn=lambda f: None,
            llm_provider=llm,
            context_provider=ctx,
            executor=executor,
            on_state_changed=on_state,
        )

        self.assertEqual(session.state, Session.STATE_CLOSED)

        session.start()
        self.assertEqual(session.state, Session.STATE_STARTING)
        self.assertIn(Session.STATE_STARTING, states)

        # Ingest 3 frames to trigger ACTIVE state (satisfying tracker.samples_required=3)
        for _ in range(3):
            f = hw.read_frame(window_ms=10)
            session.push_frame(f)
            time.sleep(0.005)  # yield slightly
        
        # Wait for the OpsQueue to process it
        time.sleep(0.05)
        self.assertEqual(session.state, Session.STATE_ACTIVE)

        session.stop()
        self.assertEqual(session.state, Session.STATE_CLOSED)

    def test_session_watchdog_timeout(self):
        """Verify session watchdog triggers an error when frame count stalls."""
        config = load_config()
        # Set a very low watchdog timeout (e.g. 0.05s) for testing
        config.runtime.session_liveness_timeout = 0.05
        hw = MockHardwareSource()
        llm = MockLLMProvider()
        ctx = MockContextProvider()
        executor = MockActionExecutor()

        errors = []
        session = Session(
            id="watchdog-session",
            config=config,
            hardware=hw,
            classify_fn=lambda f: None,
            llm_provider=llm,
            context_provider=ctx,
            executor=executor,
            on_error=errors.append,
        )

        session.start()
        # Wait long enough for the watchdog timer to check twice
        time.sleep(0.12)

        session.stop()

        self.assertTrue(len(errors) > 0)
        self.assertTrue(any(isinstance(e, HardwareError) for e in errors))

    def test_session_worker_capacity(self):
        """Verify SessionWorker tracks capacity, registers sessions, and closes them."""
        config = load_config()
        worker = SessionWorker(config, max_sessions=2)

        hw = MockHardwareSource()
        llm = MockLLMProvider()
        ctx = MockContextProvider()
        executor = MockActionExecutor()

        worker.create_session("s1", hw, lambda f: None, llm, ctx, executor)
        worker.create_session("s2", hw, lambda f: None, llm, ctx, executor)

        self.assertEqual(worker.load, 1.0)
        self.assertEqual(worker.status, "full")

        # Exceed capacity
        with self.assertRaises(ValueError):
            worker.create_session("s3", hw, lambda f: None, llm, ctx, executor)

        worker.close()
        self.assertEqual(worker.load, 0.0)
        self.assertEqual(worker.status, "idle")
