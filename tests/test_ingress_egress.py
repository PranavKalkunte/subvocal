import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from subvocal.core.testing import MockHardwareSource
from subvocal.runtime.egress import EgressManager
from subvocal.runtime.ingress import IngressManager


class TestIngressEgressManagers(unittest.TestCase):

    def setUp(self):
        self.temp_fd, self.temp_path = tempfile.mkstemp()
        os.close(self.temp_fd)

    def tearDown(self):
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)

    def test_ingress_manager_failover_lifecycle(self):
        """Verify IngressManager starts the primary and fails over to the fallback source."""
        manager = IngressManager()
        
        primary = MockHardwareSource()
        fallback = MockHardwareSource()

        manager.register_source("primary", primary, is_fallback=False)
        manager.register_source("fallback", fallback, is_fallback=True)

        self.assertEqual(manager.active_name, "primary")
        self.assertFalse(primary.is_connected())
        self.assertFalse(fallback.is_connected())

        # Start primary
        manager.start()
        self.assertTrue(primary.is_connected())
        self.assertFalse(fallback.is_connected())

        # Trigger failover
        manager.trigger_failover()
        self.assertEqual(manager.active_name, "fallback")
        # Primary should be stopped
        self.assertFalse(primary.is_connected())
        # Fallback should be started
        self.assertTrue(fallback.is_connected())

        # Stop
        manager.stop()
        self.assertFalse(fallback.is_connected())

    def test_egress_manager_write_trace(self):
        """Verify EgressManager writes tracing to JSONL correctly."""
        egress = EgressManager(trace_path=self.temp_path)
        trace_data = {"key": "val", "step": 1}
        
        egress.write_trace(trace_data)

        # Read back
        with open(self.temp_path, encoding="utf-8") as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 1)
        decoded = json.loads(lines[0])
        self.assertEqual(decoded["key"], "val")
        self.assertEqual(decoded["step"], 1)

    def test_egress_manager_mock_speak(self):
        """Verify speak delegates to the TTS engine."""
        mock_tts = MagicMock()
        egress = EgressManager(tts_engine=mock_tts)

        egress.speak("hello silent speech")
        mock_tts.speak.assert_called_once_with("hello silent speech")

    def test_egress_manager_record_signal(self):
        """Verify record_signal serializes and saves sEMG frame metadata."""
        egress = EgressManager()
        
        # Mock frame with start_time and end_time
        class MockFrame:
            def __init__(self, start, end):
                self.start_time = start
                self.end_time = end
            def to_numpy(self):
                return [1.0, 2.0]

        frames = [MockFrame(0.0, 0.5), MockFrame(0.5, 1.0)]
        egress.record_signal(self.temp_path, frames)

        # Load back
        with open(self.temp_path, encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["start_time"], 0.0)
        self.assertEqual(data[0]["end_time"], 0.5)
        self.assertEqual(data[0]["samples_count"], 2)


if __name__ == "__main__":
    unittest.main()
