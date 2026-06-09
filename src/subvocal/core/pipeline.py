"""SubvocalPipeline class coordinating the hardware, classification, and execution layers.
"""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from subvocal.exceptions import PolicyViolationError

from .interfaces import ActionExecutor, ContextProvider, HardwareSource, LLMProvider
from .models import Action, CommandToken, Frame, Intent

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Running counters for a pipeline instance, exposed as ``pipeline.stats``."""

    frames_processed: int = 0
    tokens_classified: int = 0
    phrases_processed: int = 0
    intents_resolved: int = 0
    actions_executed: int = 0
    actions_blocked: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        """Returns the counters as a plain dictionary (for logging/export)."""
        return {
            "frames_processed": self.frames_processed,
            "tokens_classified": self.tokens_classified,
            "phrases_processed": self.phrases_processed,
            "intents_resolved": self.intents_resolved,
            "actions_executed": self.actions_executed,
            "actions_blocked": self.actions_blocked,
            "errors": self.errors,
            "uptime_seconds": time.time() - self.started_at,
        }


class SubvocalPipeline:
    """Orchestrates hardware ingestion, classifier decoding, intent reconstruction, and action execution."""

    def __init__(
        self,
        hardware: HardwareSource,
        classify_fn: Callable[[Frame], CommandToken | None],
        llm_provider: LLMProvider,
        context_provider: ContextProvider,
        executor: ActionExecutor,
        phrase_timeout_seconds: float = 1.5,
        policy_engine: Any | None = None,
        dry_run: bool = False,
        trace_path: str | None = None,
        raise_on_policy_violation: bool = False,
        on_token: Callable[[CommandToken], None] | None = None,
        on_intent: Callable[[Intent], None] | None = None,
        on_action: Callable[[Action, str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
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
            raise_on_policy_violation: If True, a policy rejection raises
                :class:`subvocal.exceptions.PolicyViolationError` instead of
                being silently traced.
            on_token: Observer invoked with each classified CommandToken.
            on_intent: Observer invoked with each reconstructed Intent.
            on_action: Observer invoked with each terminal (action, status)
                pair; status is one of SUCCESS, DRY_RUN, REJECTED_UNAUTHORIZED,
                FAILED_CANNOT_EXECUTE.
            on_error: Observer invoked with exceptions raised during phrase
                processing. Observer failures are logged, never propagated.
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
        self.raise_on_policy_violation = raise_on_policy_violation
        self.on_token = on_token
        self.on_intent = on_intent
        self.on_action = on_action
        self.on_error = on_error
        self.stats = PipelineStats()

        self._token_buffer: list[CommandToken] = []
        self._last_token_time: float = 0.0

    def _notify(self, callback: Callable | None, *args: Any) -> None:
        """Invokes an observer callback without letting it break the pipeline."""
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            logger.exception("Pipeline observer %r raised; continuing", callback)

    @property
    def token_buffer(self) -> list[CommandToken]:
        """Returns the current accumulated tokens in the buffer."""
        return self._token_buffer

    def clear_buffer(self) -> None:
        """Clears the accumulated command token buffer."""
        self._token_buffer.clear()
        self._last_token_time = 0.0

    def step(self, window_ms: int = 100) -> Action | None:
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
        self.stats.frames_processed += 1

        # 2. Run classification
        token = self.classify_fn(frame)
        now = time.time()

        if token is not None:
            self._token_buffer.append(token)
            self._last_token_time = now
            self.stats.tokens_classified += 1
            self._notify(self.on_token, token)

        # 3. Check for phrase timeout trigger
        if self._token_buffer and (now - self._last_token_time) >= self.phrase_timeout_seconds:
            return self.process_phrase()

        return None

    def _write_trace(self, trace_entry: dict[str, Any]) -> None:
        """Appends a structured trace record to the local JSONL log file."""
        import json
        import os

        from subvocal.paths import get_data_dir

        trace_path = self.trace_path or os.path.join(get_data_dir(), "pipeline_traces.jsonl")
        with open(trace_path, "a") as f:
            f.write(json.dumps(trace_entry) + "\n")

    def process_phrase(self) -> Action | None:
        """Forces immediate decoding and execution of the accumulated tokens.

        Returns:
            The executed Action if successful, otherwise None.
        """
        if not self._token_buffer:
            return None

        tokens_to_process = list(self._token_buffer)
        self.clear_buffer()
        self.stats.phrases_processed += 1

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
            self.stats.intents_resolved += 1
            self._notify(self.on_intent, intent)

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
                self.stats.actions_blocked += 1
                if self.raise_on_policy_violation:
                    raise PolicyViolationError(
                        f"Action '{action.action_type}' rejected by policy engine."
                    )
            elif self.dry_run:
                status = "DRY_RUN"
            else:
                # 5. Execute action
                if self.executor.can_execute(action):
                    self.executor.execute(action)
                    status = "SUCCESS"
                    self.stats.actions_executed += 1
                else:
                    status = "FAILED_CANNOT_EXECUTE"

        except PolicyViolationError as e:
            self.stats.errors += 1
            self._notify(self.on_error, e)
            self._write_trace({
                "trace_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "tokens": [t.model_dump() for t in tokens_to_process],
                "context": context.model_dump() if context else None,
                "intent": intent.model_dump() if intent else None,
                "action": action.model_dump() if action else None,
                "authorized": False,
                "dry_run": self.dry_run,
                "status": "REJECTED_UNAUTHORIZED",
                "error": str(e),
            })
            raise
        except Exception as e:
            status = "ERROR"
            execution_error = str(e)
            self.stats.errors += 1
            self._notify(self.on_error, e)

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

        if action and status in ("SUCCESS", "DRY_RUN", "REJECTED_UNAUTHORIZED", "FAILED_CANNOT_EXECUTE"):
            self._notify(self.on_action, action, status)

        # Return action if success or dry run
        if status in ["SUCCESS", "DRY_RUN"] and action:
            return action

        return None

