"""Unit tests for safety, policies, dry-run, and tracing APIs."""

import os
import sys
import json
import time
import unittest
from typing import List, Dict, Any


from subvocal.core.models import Frame, CommandToken, Intent, Action
from subvocal.core.interfaces import HardwareSource, LLMProvider, ActionExecutor, ContextProvider
from subvocal.core.pipeline import SubvocalPipeline
from subvocal.context.schema import UserContext, AppState
from subvocal.emg_core import config

from subvocal.core.security import (
    ConfidenceThresholdPolicy,
    CommandWhitelistPolicy,
    ContextBoundPolicy,
    PolicyEngine
)


# ══════════════════════════════════════════════════════════════════════════════
# Mocks
# ══════════════════════════════════════════════════════════════════════════════

class MockHardware(HardwareSource):
    def start(self): pass
    def stop(self): pass
    def read_frame(self, window_ms: int) -> Frame:
        return Frame(samples=[], start_time=time.time(), end_time=time.time(), fs=250.0)
    def is_connected(self) -> bool: return True


class MockLLM(LLMProvider):
    def __init__(self, cmd="CLICK", confidence=0.9):
        self.cmd = cmd
        self.confidence = confidence

    def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent:
        return Intent(
            command=self.cmd,
            arguments=[],
            resolved_text=self.cmd,
            raw_shorthand="clk",
            confidence=self.confidence,
            timestamp=time.time()
        )
    def get_provider_name(self) -> str: return "mock"


class MockContext(ContextProvider):
    def __init__(self, app="Chrome"):
        self.app = app
    def get_context(self) -> UserContext:
        return UserContext(
            app_state=AppState(current_app=self.app)
        )
    def get_provider_name(self) -> str: return "mock"


class MockExecutor(ActionExecutor):
    def __init__(self):
        self.executed = []
    def execute(self, action: Action):
        self.executed.append(action)
    def can_execute(self, action: Action) -> bool: return True


# Mock HTTP request wrapper for handler testing
class MockWfile:
    def __init__(self):
        self.data = b""
    def write(self, b):
        self.data += b


# ══════════════════════════════════════════════════════════════════════════════
# Test Cases
# ══════════════════════════════════════════════════════════════════════════════

class TestSubvocalSecurity(unittest.TestCase):

    def setUp(self):
        self.trace_file = os.path.join(config.DATA_DIR, "pipeline_traces.jsonl")
        if os.path.exists(self.trace_file):
            os.remove(self.trace_file)

        self.context = UserContext(
            app_state=AppState(current_app="Terminal")
        )

    def tearDown(self):
        if os.path.exists(self.trace_file):
            os.remove(self.trace_file)

    def test_01_confidence_policy(self):
        """Test confidence threshold checks."""
        policy = ConfidenceThresholdPolicy(threshold=0.85)
        
        # Approved: confidence = 0.90
        act_ok = Action(action_type="click", params={"confidence": 0.90}, intent_id="1", timestamp=time.time())
        self.assertTrue(policy.is_authorized(act_ok, self.context))

        # Rejected: confidence = 0.75
        act_low = Action(action_type="click", params={"confidence": 0.75}, intent_id="2", timestamp=time.time())
        self.assertFalse(policy.is_authorized(act_low, self.context))

    def test_02_whitelist_policy(self):
        """Test command whitelist restriction."""
        policy = CommandWhitelistPolicy(allowed_commands=["GOTO", "CLICK"])

        # Authorized: click
        act_ok = Action(action_type="click", params={}, intent_id="1", timestamp=time.time())
        self.assertTrue(policy.is_authorized(act_ok, self.context))

        # Unauthorized: execute
        act_bad = Action(action_type="execute", params={}, intent_id="2", timestamp=time.time())
        self.assertFalse(policy.is_authorized(act_bad, self.context))

    def test_03_context_bound_policy(self):
        """Test context application boundaries for sensitive actions."""
        policy = ContextBoundPolicy(
            sensitive_commands=["EXECUTE", "DELETE"],
            safe_applications=["Terminal", "VSCode"]
        )

        act_sensitive = Action(action_type="delete", params={}, intent_id="1", timestamp=time.time())

        # Safe Context: Terminal
        ctx_safe = UserContext(app_state=AppState(current_app="Terminal"))
        self.assertTrue(policy.is_authorized(act_sensitive, ctx_safe))

        # Unsafe Context: Chrome
        ctx_unsafe = UserContext(app_state=AppState(current_app="Chrome"))
        self.assertFalse(policy.is_authorized(act_sensitive, ctx_unsafe))

    def test_04_pipeline_unauthorized_trace(self):
        """Test pipeline policy rejection and trace log entry."""
        policy_engine = PolicyEngine([
            ConfidenceThresholdPolicy(threshold=0.9)
        ])
        
        # LLM returns intent with confidence = 0.75 (violates policy)
        pipeline = SubvocalPipeline(
            hardware=MockHardware(),
            classify_fn=lambda f: CommandToken(text="clk", confidence=0.9, timestamp=time.time()),
            llm_provider=MockLLM(cmd="CLICK", confidence=0.75),
            context_provider=MockContext(),
            executor=MockExecutor(),
            policy_engine=policy_engine
        )

        pipeline._token_buffer.append(CommandToken(text="clk", confidence=0.9, timestamp=time.time()))
        res = pipeline.process_phrase()
        
        # Verify action execution is blocked
        self.assertIsNone(res)
        self.assertEqual(len(pipeline.executor.executed), 0)

        # Verify trace entry written to JSONL
        self.assertTrue(os.path.exists(self.trace_file))
        with open(self.trace_file, "r") as f:
            trace = json.loads(f.read().strip())
        self.assertEqual(trace["status"], "REJECTED_UNAUTHORIZED")
        self.assertFalse(trace["authorized"])

    def test_05_pipeline_dry_run(self):
        """Test pipeline dry-run compilation without side-effects."""
        executor = MockExecutor()
        pipeline = SubvocalPipeline(
            hardware=MockHardware(),
            classify_fn=lambda f: CommandToken(text="clk", confidence=0.9, timestamp=time.time()),
            llm_provider=MockLLM(cmd="CLICK", confidence=0.95),
            context_provider=MockContext(),
            executor=executor,
            dry_run=True
        )

        pipeline._token_buffer.append(CommandToken(text="clk", confidence=0.9, timestamp=time.time()))
        res = pipeline.process_phrase()

        # Action is compiled and returned, but not executed
        self.assertIsNotNone(res)
        self.assertEqual(res.action_type, "click")
        self.assertEqual(len(executor.executed), 0)

        # Verify trace entry
        with open(self.trace_file, "r") as f:
            trace = json.loads(f.read().strip())
        self.assertEqual(trace["status"], "DRY_RUN")
        self.assertTrue(trace["dry_run"])


if __name__ == "__main__":
    unittest.main()
