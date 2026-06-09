"""SubvocalPipeline class coordinating the hardware, classification, and execution layers.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from subvocal.config import load_config
from subvocal.runtime.session import Session

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
        telemetry_service: Any | None = None,
    ):
        self.stats = PipelineStats()
        self._trace_path = trace_path

        # Save callbacks
        self.on_token = on_token
        self.on_intent = on_intent
        self.on_action = on_action
        self.on_error = on_error

        # Build config dynamically from constructor parameters
        config = load_config()
        config.runtime.phrase_timeout_seconds = phrase_timeout_seconds
        config.policy.dry_run = dry_run
        config.policy.raise_on_policy_violation = raise_on_policy_violation

        # Dynamically initialize telemetry service if not provided
        if telemetry_service is None:
            if config.telemetry.enabled:
                from subvocal.telemetry.service import PrometheusTelemetry
                telemetry_service = PrometheusTelemetry(config)
            else:
                from subvocal.telemetry.service import NullTelemetry
                telemetry_service = NullTelemetry()

        self.telemetry = telemetry_service

        # Define internal callbacks to update stats & forward to user callbacks
        def internal_on_token(token):
            self.stats.tokens_classified += 1
            if self.on_token:
                self.on_token(token)

        def internal_on_intent(intent):
            self.stats.intents_resolved += 1
            if self.on_intent:
                self.on_intent(intent)

        def internal_on_action(action, status):
            self.stats.phrases_processed += 1
            if status == "SUCCESS":
                self.stats.actions_executed += 1
            elif status == "REJECTED_UNAUTHORIZED":
                self.stats.actions_blocked += 1
            if self.on_action:
                self.on_action(action, status)

        def internal_on_error(err):
            self.stats.errors += 1
            if self.on_error:
                self.on_error(err)

        # Initialize Session
        self._session = Session(
            id="default-pipeline",
            config=config,
            hardware=hardware,
            classify_fn=classify_fn,
            llm_provider=llm_provider,
            context_provider=context_provider,
            executor=executor,
            policy_engine=policy_engine,
            trace_path=trace_path,
            on_token=internal_on_token,
            on_intent=internal_on_intent,
            on_action=internal_on_action,
            on_error=internal_on_error,
            telemetry=self.telemetry,
        )
        self._session.start()

    # Property mappings to underlying Session delegates
    @property
    def trace_path(self) -> str | None:
        if hasattr(self, "_session"):
            return self._session.trace_path
        return getattr(self, "_trace_path", None)

    @trace_path.setter
    def trace_path(self, val: str | None) -> None:
        self._trace_path = val
        if hasattr(self, "_session"):
            self._session.trace_path = val
    @property
    def hardware(self) -> HardwareSource:
        return self._session.hardware

    @hardware.setter
    def hardware(self, val: HardwareSource) -> None:
        self._session.hardware = val

    @property
    def classify_fn(self) -> Callable[[Frame], CommandToken | None]:
        return self._session.classify_fn

    @classify_fn.setter
    def classify_fn(self, val: Callable[[Frame], CommandToken | None]) -> None:
        self._session.classify_fn = val

    @property
    def llm_provider(self) -> LLMProvider:
        return self._session.llm_provider

    @llm_provider.setter
    def llm_provider(self, val: LLMProvider) -> None:
        self._session.llm_provider = val

    @property
    def context_provider(self) -> ContextProvider:
        return self._session.context_provider

    @context_provider.setter
    def context_provider(self, val: ContextProvider) -> None:
        self._session.context_provider = val

    @property
    def executor(self) -> ActionExecutor:
        return self._session.executor

    @executor.setter
    def executor(self, val: ActionExecutor) -> None:
        self._session.executor = val

    @property
    def policy_engine(self) -> Any:
        return self._session.policy_engine

    @policy_engine.setter
    def policy_engine(self, val: Any) -> None:
        self._session.policy_engine = val

    @property
    def phrase_timeout_seconds(self) -> float:
        return self._session.config.runtime.phrase_timeout_seconds

    @phrase_timeout_seconds.setter
    def phrase_timeout_seconds(self, val: float) -> None:
        self._session.config.runtime.phrase_timeout_seconds = val

    @property
    def dry_run(self) -> bool:
        return self._session.config.policy.dry_run

    @dry_run.setter
    def dry_run(self, val: bool) -> None:
        self._session.config.policy.dry_run = val

    @property
    def raise_on_policy_violation(self) -> bool:
        return self._session.config.policy.raise_on_policy_violation

    @raise_on_policy_violation.setter
    def raise_on_policy_violation(self, val: bool) -> None:
        self._session.config.policy.raise_on_policy_violation = val

    @property
    def token_buffer(self) -> list[CommandToken]:
        """Returns the current accumulated tokens in the buffer."""
        return self._session.token_buffer

    @property
    def _token_buffer(self) -> list[CommandToken]:
        """Internal token buffer reference for backward compatibility."""
        return self._session._token_buffer

    @property
    def _last_token_time(self) -> float:
        return self._session._last_token_time

    @_last_token_time.setter
    def _last_token_time(self, val: float) -> None:
        self._session._last_token_time = val

    def clear_buffer(self) -> None:
        """Clears the accumulated command token buffer."""
        event = threading.Event()
        def clear_op():
            try:
                self._session._token_buffer.clear()
                self._session._last_token_time = 0.0
            finally:
                event.set()
        self._session._ops_queue.enqueue(clear_op)
        event.wait()

    def step(self, window_ms: int = 100) -> Action | None:
        """Performs a single step of the pipeline.

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

        # Synchronously block until OpsQueue processes the frame to maintain step API contract
        res: dict[str, Any] = {"action": None, "exception": None}
        event = threading.Event()

        def process_and_signal():
            try:
                original_on_action = self._session._on_action
                def capture_action(action, status):
                    if status in ("SUCCESS", "DRY_RUN"):
                        res["action"] = action
                    self._session._on_action = original_on_action
                    if original_on_action is not None:
                        original_on_action(action, status)

                self._session._on_action = capture_action
                self._session._process_frame(frame)
            except Exception as e:
                res["exception"] = e
            finally:
                event.set()

        self._session._ops_queue.enqueue(process_and_signal)
        event.wait()

        if res["exception"]:
            raise res["exception"]
        return res["action"]

    def process_phrase(self) -> Action | None:
        """Forces immediate decoding and execution of the accumulated tokens.

        Returns:
            The executed Action if successful, otherwise None.
        """
        res: dict[str, Any] = {"action": None, "exception": None}
        event = threading.Event()

        def process_and_signal():
            try:
                original_on_action = self._session._on_action
                def capture_action(action, status):
                    if status in ("SUCCESS", "DRY_RUN"):
                        res["action"] = action
                    self._session._on_action = original_on_action
                    if original_on_action is not None:
                        original_on_action(action, status)

                self._session._on_action = capture_action
                self._session.process_phrase()
            except Exception as e:
                res["exception"] = e
            finally:
                event.set()

        self._session._ops_queue.enqueue(process_and_signal)
        event.wait()

        if res["exception"]:
            raise res["exception"]
        return res["action"]
