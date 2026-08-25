import hashlib
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
# Fix H9: _started_ports leak – previously never evicted, growing unbounded if
# many distinct ports were used. Ports are naturally bounded to 1-65535, but
# we explicitly bound the set and provide _release_port for cleanup to avoid
# memory leak in long-running processes or tests that rotate ports.
_started_ports: set[int] = set()
_MAX_STARTED_PORTS = 64
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


def _hash_session_id(session_id: str) -> str:
    """Hash session_id to low-cardinality bucket if per-session breakdown is required.

    Prometheus high-cardinality guidance: avoid unbounded label values such as
    raw session_id (each unique session would create a new time series).
    If per-session insight is needed, use this hashed 8-char bucket (<256
    values) or preferably aggregate without session_id and rely on logs.
    Documented limit: when using hashed bucket, cardinality is bounded to
    2^32 possibilities truncated to 8 hex chars (~4B, but bucketed to 256
    if further truncated). Prefer removing label entirely.
    """
    return hashlib.sha256(session_id.encode()).hexdigest()[:8]


def _register_port(port: int) -> None:
    """Register port with bounded set to prevent unbounded leak."""
    if len(_started_ports) >= _MAX_STARTED_PORTS:
        # Evict an arbitrary oldest entry to bound memory
        try:
            _started_ports.pop()
        except KeyError:
            pass
    _started_ports.add(port)


def _release_port(port: int) -> None:
    """Evict port from registry – call on server shutdown to prevent leak."""
    _started_ports.discard(port)


def _make_compat(metric: Any) -> Any:
    """Wrap metric.labels() to ignore legacy session_id for backwards compat.

    Cardinality fix removes session_id from Prometheus label sets. Old callers
    (e.g., tests) that still invoke .labels(session_id=...) will be handled
    by dropping the high-cardinality label and delegating to the low-
    cardinality metric. This keeps exposition cardinality bounded while not
    breaking callers.
    """
    try:
        orig_labels = metric.labels  # type: ignore[attr-defined]
        labelnames = getattr(metric, "_labelnames", None)
        # prometheus_client stores labelnames as tuple
        if labelnames is None:
            labelnames = getattr(metric, "_labelNames", ())

        def compat(*args: Any, **kwargs: Any) -> Any:
            kwargs.pop("session_id", None)
            # If metric has no labels, return the metric itself (unlabelled)
            if not labelnames:
                return metric
            if kwargs or args:
                return orig_labels(*args, **kwargs)
            return metric

        metric.labels = compat  # type: ignore[method-assign,attr-defined]
    except Exception:
        pass
    return metric


class PrometheusTelemetry(TelemetryService):
    """Telemetry service that exports metrics to Prometheus.

    Prometheus cardinality: session_id is high cardinality (unbounded, one
    series per session). To keep series count bounded (Prometheus recommends
    <10k series per metric, ideally <100k total), session_id is NOT used as
    a Prometheus label. Metrics are aggregated globally or by low-cardinality
    labels (intent, action_type, etc.). If per-session debugging is required,
    use logs or hash session_id via _hash_session_id() into a small bucket,
    but do not expose raw session_id. Documented limit: if you must add
    session_id, hash and cap distinct values <1000 and ensure series are
    removed on session_ended to avoid leak.
    """

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
                # H9 fix: removed session_id label to avoid high cardinality
                _phrases_total = prometheus_client.Counter(
                    "subvocal_phrases_total",
                    "Total number of phrases detected"
                )
                _intents_total = prometheus_client.Counter(
                    "subvocal_intents_total",
                    "Total number of intents resolved",
                    ["intent"]
                )
                _actions_executed_total = prometheus_client.Counter(
                    "subvocal_actions_executed_total",
                    "Total number of actions executed",
                    ["action_type", "status"]
                )
                _actions_blocked_total = prometheus_client.Counter(
                    "subvocal_actions_blocked_total",
                    "Total number of actions blocked by policy",
                    ["action_type", "reason"]
                )
                _provider_retries_total = prometheus_client.Counter(
                    "subvocal_provider_retries_total",
                    "Total number of provider retries",
                    ["provider_name"]
                )
                _signal_quality_score = prometheus_client.Gauge(
                    "subvocal_signal_quality_score",
                    "Current physiological signal quality score (MOS score 0-5.0)",
                    ["quality_state"]
                )
                _errors_total = prometheus_client.Counter(
                    "subvocal_errors_total",
                    "Total number of pipeline/session errors",
                    ["error_type"]
                )
                # Wrap for backwards compatibility with legacy session_id callers
                _make_compat(_phrases_total)
                _make_compat(_intents_total)
                _make_compat(_actions_executed_total)
                _make_compat(_actions_blocked_total)
                _make_compat(_provider_retries_total)
                _make_compat(_signal_quality_score)
                _make_compat(_errors_total)

            if self._port not in _started_ports:
                try:
                    prometheus_client.start_http_server(self._port)
                    _register_port(self._port)
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
        # Note: per-session metric series are not retained (no session_id label),
        # so no need to remove high-cardinality child. If hashing were used,
        # we would call _phrases_total.remove(_hash_session_id(session_id)) etc.
        # _started_ports is not evicted per session; ports are per-process.

    def phrase_detected(self, session_id: str, phrase_id: str, duration_seconds: float) -> None:
        if not self._enabled:
            return
        # session_id intentionally not used as Prometheus label (cardinality)
        _phrases_total.inc()

    def intent_resolved(self, session_id: str, phrase_id: str, intent_name: str, confidence: float) -> None:
        if not self._enabled:
            return
        _intents_total.labels(intent=intent_name).inc()

    def action_executed(self, session_id: str, action_type: str, status: str) -> None:
        if not self._enabled:
            return
        _actions_executed_total.labels(
            action_type=action_type, status=status
        ).inc()

    def action_blocked(self, session_id: str, action_type: str, reason: str) -> None:
        if not self._enabled:
            return
        _actions_blocked_total.labels(
            action_type=action_type, reason=reason
        ).inc()

    def provider_retry(self, session_id: str, provider_name: str, attempt: int, error: str) -> None:
        if not self._enabled:
            return
        _provider_retries_total.labels(
            provider_name=provider_name
        ).inc()

    def quality_changed(self, session_id: str, quality_score: float, quality_state: str) -> None:
        if not self._enabled:
            return
        _signal_quality_score.labels(
            quality_state=quality_state
        ).set(quality_score)

    def error_occurred(self, session_id: str, error_type: str) -> None:
        if not self._enabled:
            return
        _errors_total.labels(error_type=error_type).inc()

    def shutdown(self) -> None:
        """Release Prometheus port from registry (fixes _started_ports leak)."""
        _release_port(self._port)
