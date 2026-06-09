import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from subvocal.config import SubvocalConfig
from subvocal.core.interfaces import ActionExecutor, ContextProvider, HardwareSource, LLMProvider
from subvocal.core.models import Action, CommandToken, Frame, Intent
from subvocal.exceptions import HardwareError, PolicyViolationError
from subvocal.stream import FrameRing, SignalLevel, SignalQualityScorer, StreamStats, StreamTracker
from subvocal.utils import OpsQueue

logger = logging.getLogger(__name__)


class Session:
    """Represents an active silent speech processing session.

    Co-ordinates real-time signal level monitoring, stream tracking, quality scoring,
    and intent decoding on a background OpsQueue thread.
    """

    STATE_STARTING = "starting"
    STATE_ACTIVE = "active"
    STATE_DEGRADED = "degraded"
    STATE_CLOSED = "closed"

    def __init__(
        self,
        id: str,
        config: SubvocalConfig,
        hardware: HardwareSource,
        classify_fn: Callable[[Frame], CommandToken | None],
        llm_provider: LLMProvider,
        context_provider: ContextProvider,
        executor: ActionExecutor,
        policy_engine: Any | None = None,
        trace_path: str | None = None,
        on_state_changed: Callable[[str], None] | None = None,
        on_token: Callable[[CommandToken], None] | None = None,
        on_intent: Callable[[Intent], None] | None = None,
        on_action: Callable[[Action, str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        telemetry: Any | None = None,
        data_channel: Any | None = None,
    ):
        self.id = id
        self.config = config
        self.hardware = hardware
        self.classify_fn = classify_fn
        self.llm_provider = llm_provider
        self.context_provider = context_provider
        self.executor = executor
        self.policy_engine = policy_engine
        self.trace_path = trace_path
        self.data_channel = data_channel

        # Observer Callbacks
        self._on_state_changed = on_state_changed
        self._on_token = on_token
        self._on_intent = on_intent
        self._on_action = on_action
        self._on_error = on_error

        # Telemetry
        if telemetry is None:
            from subvocal.telemetry.service import NullTelemetry
            self.telemetry = NullTelemetry()
        else:
            self.telemetry = telemetry

        # Stream Monitoring Subsystems
        self.buffer = FrameRing(max_size=200)
        self.stats = StreamStats()
        self.level = SignalLevel(
            active_level=config.classifier.confidence_threshold * 0.1,  # adapted
            min_percentile=40.0,
            update_interval_ms=400,
            smooth_intervals=2,
        )
        self.tracker = StreamTracker(samples_required=3, cycles_required=5)
        self.scorer = SignalQualityScorer(clip_max=5.0)

        # State management
        self._lock = threading.Lock()
        self._state = self.STATE_CLOSED
        self._ops_queue = OpsQueue(f"session-{id}", flush_on_stop=True)

        self._token_buffer: list[CommandToken] = []
        self._last_token_time = 0.0

        # Liveness watchdog
        self._watchdog_timer = None
        self._last_frame_count = 0
        self._watchdog_interval = config.runtime.session_liveness_timeout

        # Connect callbacks
        self.tracker.on_status_changed(self._handle_tracker_status_change)
        self.scorer.on_status_changed(self._handle_quality_status_change)

    @property
    def state(self) -> str:
        """Returns the current state of the session state machine."""
        with self._lock:
            return self._state

    @property
    def token_buffer(self) -> list[CommandToken]:
        """Returns the raw reference to the accumulated token buffer."""
        return self._token_buffer

    def start(self) -> None:
        """Initializes and starts the session."""
        with self._lock:
            if self._state != self.STATE_CLOSED:
                return
            self._state = self.STATE_STARTING
            self._token_buffer.clear()
            self._last_token_time = 0.0
            self._last_frame_count = 0

            # Start hardware and background queue
            self.hardware.start()
            self._ops_queue.start()

            # Trigger liveness watchdog
            self._schedule_watchdog()

            # Record telemetry
            self.telemetry.session_started(self.id, self.config)

            if self._on_state_changed:
                self._on_state_changed(self._state)

    def stop(self) -> None:
        """Gracefully stops and closes the session."""
        with self._lock:
            if self._state == self.STATE_CLOSED:
                return
            self._state = self.STATE_CLOSED

            # Cancel watchdog
            if self._watchdog_timer:
                self._watchdog_timer.cancel()
                self._watchdog_timer = None

            # Stop hardware and wait for OpsQueue to drain
            self.hardware.stop()
            self._ops_queue.stop()

            # Record telemetry
            self.telemetry.session_ended(self.id, self.stats)

            if self._on_state_changed:
                self._on_state_changed(self._state)

    def push_frame(self, frame: Frame) -> None:
        """Ingests a new Frame, running stream checks and classification asynchronously on OpsQueue."""
        with self._lock:
            if self._state == self.STATE_CLOSED:
                return

        # Enqueue frame processing
        self._ops_queue.enqueue(self._process_frame, frame)

    def _process_frame(self, frame: Frame) -> None:
        try:
            # 1. Update stats & buffer
            self.stats.observe(frame)
            self.buffer.push(frame)

            # 2. Run signal level & quality evaluations
            self.level.observe(frame)
            self.scorer.update(frame)

            # 3. Observe active stream status
            _, is_active = self.level.get_level(frame.end_time)
            self.tracker.observe(is_active)

            # If connection quality is lost, don't run classifier
            if self.scorer.quality == SignalQualityScorer.QUALITY_LOST:
                return

            # 4. Run classification
            token = self.classify_fn(frame)
            now = time.time()

            # Broadcast frame metrics to data channel
            if self.data_channel:
                self.data_channel.broadcast({
                    "session_id": self.id,
                    "event": "frame_processed",
                    "timestamp": frame.end_time,
                    "quality_score": self.scorer.score,
                    "quality_state": self.scorer.quality,
                    "token": token.text if token else None,
                })

            if token is not None:
                with self._lock:
                    self._token_buffer.append(token)
                    self._last_token_time = now
                self._notify(self._on_token, token)

            # 5. Check phrase timeout based on silence duration
            with self._lock:
                buffer_len = len(self._token_buffer)
                last_time = self._last_token_time
            
            if buffer_len > 0 and (now - last_time) >= self.config.runtime.phrase_timeout_seconds:
                self.process_phrase()
        except Exception as e:
            self._notify(self._on_error, e)
            raise

    def process_phrase(self) -> Action | None:
        """Forces immediate decoding of the current token buffer and runs the resolved action."""
        with self._lock:
            if not self._token_buffer:
                return None
            tokens = list(self._token_buffer)
            self._token_buffer.clear()
            self._last_token_time = 0.0

        # Retrieve context snapshot
        context = self.context_provider.get_context()
        intent = None
        action = None
        authorized = True
        status = "PENDING"
        error_msg = None

        phrase_id = str(uuid.uuid4())
        duration = 1.0
        if tokens:
            duration = max(0.1, tokens[-1].timestamp - tokens[0].timestamp)
        self.telemetry.phrase_detected(self.id, phrase_id, duration)

        try:
            # Reconstruct intent via LLM
            intent = self.llm_provider.reconstruct_intent(tokens, context)
            self._notify(self._on_intent, intent)
            self.telemetry.intent_resolved(self.id, phrase_id, intent.command, intent.confidence)

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

            # Check Policy Engine authorization
            if self.policy_engine:
                authorized = self.policy_engine.is_authorized(action, context)

            if not authorized:
                status = "REJECTED_UNAUTHORIZED"
                self.telemetry.action_blocked(self.id, action.action_type, "policy_violation")
                if self.config.policy.raise_on_policy_violation:
                    raise PolicyViolationError(f"Action '{action.action_type}' rejected by policy engine.")
            elif self.config.policy.dry_run or (
                self._get_grants() and self._get_grants().enforced_dry_run
            ):
                status = "DRY_RUN"
                self.telemetry.action_executed(self.id, action.action_type, status)
            else:
                # Execute action
                if self.executor.can_execute(action):
                    self.executor.execute(action)
                    status = "SUCCESS"
                else:
                    status = "FAILED_CANNOT_EXECUTE"
                self.telemetry.action_executed(self.id, action.action_type, status)

        except PolicyViolationError as e:
            self._notify(self._on_error, e)
            status = "REJECTED_UNAUTHORIZED"
            error_msg = str(e)
            self._write_trace(tokens, context, intent, action, authorized, status, error_msg)
            raise
        except Exception as e:
            self._notify(self._on_error, e)
            status = "ERROR"
            error_msg = str(e)
            act_type = action.action_type if action else "unknown"
            self.telemetry.action_executed(self.id, act_type, "ERROR")

        self._write_trace(tokens, context, intent, action, authorized, status, error_msg)

        if action:
            self._notify(self._on_action, action, status)
            if status in ("SUCCESS", "DRY_RUN"):
                return action
        return None

    def _write_trace(self, tokens, context, intent, action, authorized, status, error_msg) -> None:
        import json
        import os

        from subvocal.paths import get_data_dir

        trace_entry = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "tokens": [t.model_dump() for t in tokens],
            "context": context.model_dump() if context else None,
            "intent": intent.model_dump() if intent else None,
            "action": action.model_dump() if action else None,
            "authorized": authorized,
            "dry_run": self.config.policy.dry_run,
            "status": status,
            "error": error_msg,
        }

        path = self.trace_path or os.path.join(get_data_dir(), "pipeline_traces.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_entry) + "\n")
        except Exception:
            logger.exception("Failed to write pipeline trace")

    def _notify(self, callback: Callable | None, *args: Any) -> None:
        if callback is None:
            return
        if callback is self._on_error:
            err_type = type(args[0]).__name__ if args else "Exception"
            self.telemetry.error_occurred(self.id, err_type)
        try:
            callback(*args)
        except Exception:
            logger.exception("Session callback execution error")

    def _handle_tracker_status_change(self, status: str) -> None:
        with self._lock:
            if self._state == self.STATE_CLOSED:
                return
            if status == StreamTracker.STATUS_ACTIVE:
                self._state = self.STATE_ACTIVE
            elif status == StreamTracker.STATUS_STOPPED:
                # If stream stops, force process any remaining tokens in buffer
                self._ops_queue.enqueue(self.process_phrase)
                
            if self._on_state_changed:
                self._on_state_changed(self._state)

    def _handle_quality_status_change(self, quality: str) -> None:
        with self._lock:
            if self._state == self.STATE_CLOSED:
                return
            if quality == SignalQualityScorer.QUALITY_LOST:
                self._state = self.STATE_DEGRADED
                # Force process on degradation
                self._ops_queue.enqueue(self.process_phrase)
            elif quality == SignalQualityScorer.QUALITY_POOR:
                self._state = self.STATE_DEGRADED
            elif quality in (SignalQualityScorer.QUALITY_GOOD, SignalQualityScorer.QUALITY_EXCELLENT):
                self._state = self.STATE_ACTIVE

            self.telemetry.quality_changed(self.id, self.scorer.score, quality)

            if self._on_state_changed:
                self._on_state_changed(self._state)

    def _schedule_watchdog(self) -> None:
        def check_liveness():
            with self._lock:
                if self._state == self.STATE_CLOSED:
                    return

            current_frames = self.stats.total_frames
            if current_frames == self._last_frame_count:
                # Liveness heartbeat timeout! Transition state to degraded/closed
                logger.warning("Session %s liveness watchdog triggered: no incoming frames.", self.id)
                self._handle_quality_status_change(SignalQualityScorer.QUALITY_LOST)
                self._notify(self._on_error, HardwareError("Session stream liveness heartbeat timed out."))

            self._last_frame_count = current_frames
            
            with self._lock:
                if self._state != self.STATE_CLOSED:
                    self._watchdog_timer = threading.Timer(self._watchdog_interval, check_liveness)
                    self._watchdog_timer.name = f"Session-Watchdog-{self.id}"
                    self._watchdog_timer.daemon = True
                    self._watchdog_timer.start()

        self._watchdog_timer = threading.Timer(self._watchdog_interval, check_liveness)
        self._watchdog_timer.name = f"Session-Watchdog-{self.id}"
        self._watchdog_timer.daemon = True
        self._watchdog_timer.start()

    def _get_grants(self) -> Any:
        from subvocal.auth.grants import get_context_grants
        return get_context_grants()
