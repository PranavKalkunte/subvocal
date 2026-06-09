"""SubvocalPipeline class coordinating the hardware, classification, and execution layers.
"""

import time
from typing import List, Optional, Callable, Dict, Any
import uuid

from subvocal.context.schema import UserContext
from .models import Frame, CommandToken, Intent, Action
from .interfaces import HardwareSource, LLMProvider, ActionExecutor, ContextProvider


class SubvocalPipeline:
    """Orchestrates hardware ingestion, classifier decoding, intent reconstruction, and action execution."""

    def __init__(
        self,
        hardware: HardwareSource,
        classify_fn: Callable[[Frame], Optional[CommandToken]],
        llm_provider: LLMProvider,
        context_provider: ContextProvider,
        executor: ActionExecutor,
        phrase_timeout_seconds: float = 1.5,
        policy_engine: Optional[Any] = None,
        dry_run: bool = False,
        trace_path: Optional[str] = None,
    ):
        """Initializes the subvocal pipeline.

        Args:
            hardware: The hardware source to read sEMG frames from.
            classify_fn: A function that takes a Frame and returns an optional CommandToken.
            llm_provider: The LLM intent decoder.
            context_provider: The active user context source.
            executor: The device or tool action dispatcher.
            phrase_timeout_seconds: Duration of silence (no tokens) to wait before triggering intent reconstruction.
            policy_engine: Optional authorization and security policy checker.
            dry_run: If True, resolves intents and compiles actions but does not run them.
            trace_path: Destination JSONL file for execution traces. Defaults to
                ``pipeline_traces.jsonl`` inside the user data directory
                (see :mod:`subvocal.paths`).
        """
        self.hardware = hardware
        self.classify_fn = classify_fn
        self.llm_provider = llm_provider
        self.context_provider = context_provider
        self.executor = executor
        self.phrase_timeout_seconds = phrase_timeout_seconds
        self.policy_engine = policy_engine
        self.dry_run = dry_run
        self.trace_path = trace_path

        self._token_buffer: List[CommandToken] = []
        self._last_token_time: float = 0.0

    @property
    def token_buffer(self) -> List[CommandToken]:
        """Returns the current accumulated tokens in the buffer."""
        return self._token_buffer

    def clear_buffer(self) -> None:
        """Clears the accumulated command token buffer."""
        self._token_buffer.clear()
        self._last_token_time = 0.0

    def step(self, window_ms: int = 100) -> Optional[Action]:
        """Performs a single step of the pipeline.

        1. Ingests a new frame of raw sEMG data from the hardware source.
        2. Classifies the frame into a potential command token.
        3. Accumulates the token and checks for phrase completion timeout.
        4. If timed out, reconstructs the intent and executes the action.

        Args:
            window_ms: Buffer window duration in milliseconds to acquire.

        Returns:
            An executed Action if an intent was resolved and dispatched, otherwise None.
        """
        if not self.hardware.is_connected():
            return None

        # 1. Read raw frame
        frame = self.hardware.read_frame(window_ms)

        # 2. Run classification
        token = self.classify_fn(frame)
        now = time.time()

        if token is not None:
            self._token_buffer.append(token)
            self._last_token_time = now

        # 3. Check for phrase timeout trigger
        if self._token_buffer and (now - self._last_token_time) >= self.phrase_timeout_seconds:
            return self.process_phrase()

        return None

    def _write_trace(self, trace_entry: Dict[str, Any]) -> None:
        """Appends a structured trace record to the local JSONL log file."""
        import json
        import os

        from subvocal.paths import get_data_dir

        trace_path = self.trace_path or os.path.join(get_data_dir(), "pipeline_traces.jsonl")
        with open(trace_path, "a") as f:
            f.write(json.dumps(trace_entry) + "\n")

    def process_phrase(self) -> Optional[Action]:
        """Forces immediate decoding and execution of the accumulated tokens.

        Returns:
            The executed Action if successful, otherwise None.
        """
        if not self._token_buffer:
            return None

        tokens_to_process = list(self._token_buffer)
        self.clear_buffer()

        # 1. Retrieve context snapshot
        context = self.context_provider.get_context()

        # Initialize trace variables
        intent = None
        action = None
        authorized = True
        status = "PENDING"
        execution_error = None

        # 2. Reconstruct intent via LLM
        try:
            intent = self.llm_provider.reconstruct_intent(tokens_to_process, context)

            # 3. Create Action from Intent
            action = Action(
                action_type=intent.command.lower(),
                params={
                    "arguments": intent.arguments,
                    "resolved_text": intent.resolved_text,
                    "confidence": intent.confidence,
                },
                intent_id=str(uuid.uuid4()),
                timestamp=time.time(),
            )

            # 4. Check Policy Engine authorization
            if self.policy_engine:
                authorized = self.policy_engine.is_authorized(action, context)

            if not authorized:
                status = "REJECTED_UNAUTHORIZED"
            elif self.dry_run:
                status = "DRY_RUN"
            else:
                # 5. Execute action
                if self.executor.can_execute(action):
                    self.executor.execute(action)
                    status = "SUCCESS"
                else:
                    status = "FAILED_CANNOT_EXECUTE"

        except Exception as e:
            status = "ERROR"
            execution_error = str(e)

        # Log structured trace
        trace_entry = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "tokens": [t.model_dump() for t in tokens_to_process],
            "context": context.model_dump() if context else None,
            "intent": intent.model_dump() if intent else None,
            "action": action.model_dump() if action else None,
            "authorized": authorized,
            "dry_run": self.dry_run,
            "status": status,
            "error": execution_error
        }
        self._write_trace(trace_entry)

        # Return action if success or dry run
        if status in ["SUCCESS", "DRY_RUN"] and action:
            return action

        return None

