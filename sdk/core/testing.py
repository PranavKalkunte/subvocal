"""Shared test fixtures for Subvocal SDK unit tests."""

import time
import numpy as np
from typing import List, Any

from context.schema import UserContext, AppState
from core.models import Sample, Frame, CommandToken, Intent, Action
from core.interfaces import HardwareSource, LLMProvider, ActionExecutor, ContextProvider


class MockHardwareSource(HardwareSource):
    """Generates continuous synthetic sEMG samples (sine + noise)."""

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
            t = self._sample_counter / self._fs
            channels = [
                float(np.sin(2 * np.pi * 10.0 * t + i) + np.random.normal(0, 0.1))
                for i in range(self._num_channels)
            ]
            samples.append(Sample(timestamp=now, channels=channels, sample_index=self._sample_counter))
        return Frame(samples=samples, start_time=now - (window_ms / 1000.0), end_time=now, fs=self._fs)

    def is_connected(self) -> bool:
        return self._connected


class MockLLMProvider(LLMProvider):
    """Maps shorthand token text patterns to GOTO or CLICK intents."""

    def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent:
        shorthand = " ".join([t.text for t in tokens])
        if "gt" in shorthand or "goto" in shorthand:
            cmd, args = "GOTO", ["google.com"]
        elif "clk" in shorthand or "click" in shorthand:
            cmd, args = "CLICK", ["Search Button"]
        else:
            cmd, args = "UNKNOWN", []
        return Intent(
            command=cmd,
            arguments=args,
            confidence=0.95,
            resolved_text=f"{cmd} {' '.join(args)}".strip(),
            raw_shorthand=shorthand,
            timestamp=time.time()
        )

    def get_provider_name(self) -> str:
        return "mock_llm"


class MockActionExecutor(ActionExecutor):
    """Records dispatched actions for assertion in tests."""

    def __init__(self):
        self.executed_actions: List[Action] = []

    def execute(self, action: Action) -> Any:
        self.executed_actions.append(action)
        return {"status": "success", "action_id": action.intent_id}

    def can_execute(self, action: Action) -> bool:
        return action.action_type in ("goto", "click")


class MockContextProvider(ContextProvider):
    """Returns a static browser UserContext snapshot."""

    def get_context(self) -> UserContext:
        return UserContext(
            app_state=AppState(current_app="Browser", page_title="Google Search", visible_elements=[])
        )

    def get_provider_name(self) -> str:
        return "mock_context"
