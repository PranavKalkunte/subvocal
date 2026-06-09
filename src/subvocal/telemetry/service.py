import logging
import threading
from typing import Any

from subvocal.config import SubvocalConfig

logger = logging.getLogger("subvocal.telemetry")

class TelemetryService:
    """Interface for telemetry exporting, based on LiveKit telemetry service design."""

    def session_started(self, session_id: str, config: SubvocalConfig) -> None:
        pass

    def session_ended(self, session_id: str, stats: Any) -> None:
        pass

    def phrase_detected(self, session_id: str, phrase_id: str, duration_seconds: float) -> None:
        pass

    def intent_resolved(self, session_id: str, phrase_id: str, intent_name: str, confidence: float) -> None:
        pass

    def action_executed(self, session_id: str, action_type: str, status: str) -> None:
        pass

    def action_blocked(self, session_id: str, action_type: str, reason: str) -> None:
        pass

    def provider_retry(self, session_id: str, provider_name: str, attempt: int, error: str) -> None:
        pass

    def quality_changed(self, session_id: str, quality_score: float, quality_state: str) -> None:
        pass

    def error_occurred(self, session_id: str, error_type: str) -> None:
        pass


class NullTelemetry(TelemetryService):
    """Default telemetry implementation that does nothing."""
    pass


# Registry to track active Prometheus exporter ports globally across sessions
_started_ports = set()
_metrics_lock = threading.Lock()

# Global metrics references to prevent duplicate registrations
_sessions_active: Any = None
_sessions_total: Any = None
_phrases_total: Any = None
_intents_total: Any = None
_actions_executed_total: Any = None
_actions_blocked_total: Any = None
_provider_retries_total: Any = None
_signal_quality_score: Any = None
_errors_total: Any = None


class PrometheusTelemetry(TelemetryService):
    """Telemetry service that exports metrics to Prometheus."""

    def __init__(self, config: SubvocalConfig):
        self.config = config
        self._enabled = config.telemetry.enabled
        self._port = config.telemetry.prometheus_port

        if not self._enabled:
            return

        try:
            import prometheus_client
        except ImportError:
            logger.warning("prometheus-client is not installed. Prometheus telemetry is disabled.")
            self._enabled = False
            return

        with _metrics_lock:
            global _sessions_active, _sessions_total, _phrases_total, _intents_total, _actions_executed_total
            global _actions_blocked_total, _provider_retries_total, _signal_quality_score, _errors_total

            if _sessions_active is None:
                _sessions_active = prometheus_client.Gauge(
                    "subvocal_sessions_active",
                    "Number of currently active subvocal sessions"
                )
                _sessions_total = prometheus_client.Counter(
                    "subvocal_sessions_total",
                    "Total number of subvocal sessions started"
                )
                _phrases_total = prometheus_client.Counter(
                    "subvocal_phrases_total",
                    "Total number of phrases detected",
                    ["session_id"]
                )
                _intents_total = prometheus_client.Counter(
                    "subvocal_intents_total",
                    "Total number of intents resolved",
                    ["session_id", "intent"]
                )
                _actions_executed_total = prometheus_client.Counter(
                    "subvocal_actions_executed_total",
                    "Total number of actions executed",
                    ["session_id", "action_type", "status"]
                )
                _actions_blocked_total = prometheus_client.Counter(
                    "subvocal_actions_blocked_total",
                    "Total number of actions blocked by policy",
                    ["session_id", "action_type", "reason"]
                )
                _provider_retries_total = prometheus_client.Counter(
                    "subvocal_provider_retries_total",
                    "Total number of provider retries",
                    ["session_id", "provider_name"]
                )
                _signal_quality_score = prometheus_client.Gauge(
                    "subvocal_signal_quality_score",
                    "Current physiological signal quality score (MOS score 0-5.0)",
                    ["session_id", "quality_state"]
                )
                _errors_total = prometheus_client.Counter(
                    "subvocal_errors_total",
                    "Total number of pipeline/session errors",
                    ["session_id", "error_type"]
                )

            if self._port not in _started_ports:
                try:
                    prometheus_client.start_http_server(self._port)
                    _started_ports.add(self._port)
                    logger.info("Started Prometheus metrics server on port %d", self._port)
                except Exception as e:
                    logger.error("Failed to start Prometheus server on port %d: %s", self._port, e)

    def session_started(self, session_id: str, config: SubvocalConfig) -> None:
        if not self._enabled:
            return
        _sessions_active.inc()
        _sessions_total.inc()

    def session_ended(self, session_id: str, stats: Any) -> None:
        if not self._enabled:
            return
        _sessions_active.dec()

    def phrase_detected(self, session_id: str, phrase_id: str, duration_seconds: float) -> None:
        if not self._enabled:
            return
        _phrases_total.labels(session_id=session_id).inc()

    def intent_resolved(self, session_id: str, phrase_id: str, intent_name: str, confidence: float) -> None:
        if not self._enabled:
            return
        _intents_total.labels(session_id=session_id, intent=intent_name).inc()

    def action_executed(self, session_id: str, action_type: str, status: str) -> None:
        if not self._enabled:
            return
        _actions_executed_total.labels(
            session_id=session_id, action_type=action_type, status=status
        ).inc()

    def action_blocked(self, session_id: str, action_type: str, reason: str) -> None:
        if not self._enabled:
            return
        _actions_blocked_total.labels(
            session_id=session_id, action_type=action_type, reason=reason
        ).inc()

    def provider_retry(self, session_id: str, provider_name: str, attempt: int, error: str) -> None:
        if not self._enabled:
            return
        _provider_retries_total.labels(
            session_id=session_id, provider_name=provider_name
        ).inc()

    def quality_changed(self, session_id: str, quality_score: float, quality_state: str) -> None:
        if not self._enabled:
            return
        _signal_quality_score.labels(
            session_id=session_id, quality_state=quality_state
        ).set(quality_score)

    def error_occurred(self, session_id: str, error_type: str) -> None:
        if not self._enabled:
            return
        _errors_total.labels(session_id=session_id, error_type=error_type).inc()
