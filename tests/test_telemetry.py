import unittest
from unittest.mock import MagicMock

from subvocal.config import load_config
from subvocal.telemetry.service import NullTelemetry, PrometheusTelemetry


class TestTelemetry(unittest.TestCase):

    def test_null_telemetry_methods_no_op(self):
        """Verify NullTelemetry methods can be invoked without error."""
        telemetry = NullTelemetry()
        config = load_config()

        # Should not raise any exceptions
        telemetry.session_started("test-session", config)
        telemetry.session_ended("test-session", MagicMock())
        telemetry.phrase_detected("test-session", "p-1", 1.2)
        telemetry.intent_resolved("test-session", "p-1", "play", 0.95)
        telemetry.action_executed("test-session", "play", "SUCCESS")
        telemetry.action_blocked("test-session", "play", "unauthorized")
        telemetry.quality_changed("test-session", 4.2, "EXCELLENT")
        telemetry.error_occurred("test-session", "ValueError")

    def test_prometheus_telemetry_disabled(self):
        """Verify PrometheusTelemetry does not export if configured as disabled."""
        config = load_config()
        config.telemetry.enabled = False

        telemetry = PrometheusTelemetry(config)
        self.assertFalse(telemetry._enabled)

    def test_prometheus_telemetry_enabled_and_events(self):
        """Verify PrometheusTelemetry exports metrics correctly when enabled."""
        config = load_config()
        config.telemetry.enabled = True
        # Choose a high port to avoid collision
        config.telemetry.prometheus_port = 28099

        telemetry = PrometheusTelemetry(config)
        self.assertTrue(telemetry._enabled)

        # Get initial values of metrics if they exist
        from subvocal.telemetry.service import (
            _actions_blocked_total,
            _actions_executed_total,
            _errors_total,
            _intents_total,
            _phrases_total,
            _sessions_active,
            _sessions_total,
            _signal_quality_score,
        )

        initial_active = _sessions_active.value() if hasattr(_sessions_active, "value") else 0
        initial_total = _sessions_total.value() if hasattr(_sessions_total, "value") else 0

        # Trigger events
        session_id = "test-prometheus-session"
        telemetry.session_started(session_id, config)
        telemetry.phrase_detected(session_id, "phrase-123", 1.5)
        telemetry.intent_resolved(session_id, "phrase-123", "volume_up", 0.88)
        telemetry.action_executed(session_id, "volume_up", "SUCCESS")
        telemetry.action_blocked(session_id, "shutdown", "policy_denied")
        telemetry.quality_changed(session_id, 3.8, "GOOD")
        telemetry.error_occurred(session_id, "HardwareError")

        # Verify active session incremented
        self.assertEqual(_sessions_active._value.get(), initial_active + 1)
        self.assertEqual(_sessions_total._value.get(), initial_total + 1)

        # Verify labels and metric increments
        self.assertEqual(
            _phrases_total.labels(session_id=session_id)._value.get(), 1
        )
        self.assertEqual(
            _intents_total.labels(session_id=session_id, intent="volume_up")._value.get(), 1
        )
        self.assertEqual(
            _actions_executed_total.labels(
                session_id=session_id, action_type="volume_up", status="SUCCESS"
            )._value.get(), 1
        )
        self.assertEqual(
            _actions_blocked_total.labels(
                session_id=session_id, action_type="shutdown", reason="policy_denied"
            )._value.get(), 1
        )
        self.assertEqual(
            _signal_quality_score.labels(session_id=session_id, quality_state="GOOD")._value.get(), 3.8
        )
        self.assertEqual(
            _errors_total.labels(session_id=session_id, error_type="HardwareError")._value.get(), 1
        )

        # End session and verify active session decremented
        telemetry.session_ended(session_id, MagicMock())
        self.assertEqual(_sessions_active._value.get(), initial_active)


if __name__ == "__main__":
    unittest.main()
