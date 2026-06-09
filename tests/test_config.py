import os
import tempfile
import unittest

from subvocal.config import load_config
from subvocal.exceptions import ConfigurationError


class TestConfigSystem(unittest.TestCase):
    def setUp(self):
        # Clear any existing subvocal env vars
        self._old_env = {}
        for k in list(os.environ.keys()):
            if k.startswith("SUBVOCAL_"):
                self._old_env[k] = os.environ[k]
                del os.environ[k]

    def tearDown(self):
        # Restore environment
        for k in list(os.environ.keys()):
            if k.startswith("SUBVOCAL_"):
                del os.environ[k]
        for k, v in self._old_env.items():
            os.environ[k] = v

    def test_load_default_config(self):
        """Verify load_config() returns default values when no file is provided."""
        cfg = load_config()
        self.assertEqual(cfg.hardware.sample_rate, 250)
        self.assertEqual(cfg.dsp.bandpass_high, 50.0)
        self.assertTrue(cfg.hardware.simulated)
        self.assertFalse(cfg.policy.dry_run)

    def test_load_from_yaml(self):
        """Verify values are loaded correctly from a YAML config file."""
        content = """
hardware:
  sample_rate: 1000
  num_channels: 8
  simulated: false
policy:
  dry_run: true
"""
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            cfg = load_config(temp_path)
            self.assertEqual(cfg.hardware.sample_rate, 1000)
            self.assertEqual(cfg.hardware.num_channels, 8)
            self.assertFalse(cfg.hardware.simulated)
            self.assertTrue(cfg.policy.dry_run)
        finally:
            os.unlink(temp_path)

    def test_strict_mode_rejects_unknown_keys(self):
        """Verify ConfigurationError is raised when unknown keys are present in YAML."""
        content = """
hardware:
  sample_rate: 250
  unknown_parameter: 1234
"""
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            with self.assertRaises(ConfigurationError):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_env_overrides(self):
        """Verify environment variables override loaded config values."""
        # Simple flat key override
        os.environ["SUBVOCAL_POLICY__DRY_RUN"] = "true"
        # Nested key override
        os.environ["SUBVOCAL_HARDWARE__SAMPLE_RATE"] = "500"
        # Overriding a string
        os.environ["SUBVOCAL_CLASSIFIER__TYPE"] = "cnn"
        # Override with floats
        os.environ["SUBVOCAL_DSP__BANDPASS_LOW"] = "2.5"

        cfg = load_config()
        self.assertTrue(cfg.policy.dry_run)
        self.assertEqual(cfg.hardware.sample_rate, 500)
        self.assertEqual(cfg.classifier.type, "cnn")
        self.assertEqual(cfg.dsp.bandpass_low, 2.5)

    def test_invalid_env_types_raise_validation_error(self):
        """Verify validation errors are properly converted to ConfigurationError on invalid env types."""
        os.environ["SUBVOCAL_HARDWARE__SAMPLE_RATE"] = "not-an-integer"
        with self.assertRaises(ConfigurationError):
            load_config()
