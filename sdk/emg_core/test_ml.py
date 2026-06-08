"""End-to-end integration tests for sEMG DSP and ML components."""

import os
import shutil
import numpy as np
import unittest

# Add package path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emg_core import config
from emg_core.dsp.filters import preprocess_multichannel
from emg_core.dsp.features import extract_features
from emg_core.ml.train import train_model
from emg_core.ml.infer import InferenceEngine


class TestEMGCoreML(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup synthetic environment."""
        cls.user_id = "test_user_integration"
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.MODELS_DIR, exist_ok=True)

        # Generate differentiable synthetic calibration segments
        # 150 samples per segment, 4 channels
        cls.commands = ["OPEN", "CLOSE", "CLICK", "TYPE"]
        segments = []
        labels = []

        np.random.seed(42)
        # Create 15 samples per class = 60 segments total
        for i, cmd in enumerate(cls.commands):
            for _ in range(15):
                # Generate random base noise
                seg = np.random.normal(0.0, 1.0, (150, 4))
                # Add command-specific pattern (frequency) to make it highly classifiable
                t = np.linspace(0, 2*np.pi, 150)
                freq = (i + 1) * 5.0
                pattern = np.sin(t * freq)[:, np.newaxis]
                seg += pattern * 2.0  # boost signal-to-noise ratio

                segments.append(seg)
                labels.append(cmd)

        # Save to config.DATA_DIR
        cls.data_path = os.path.join(config.DATA_DIR, f"{cls.user_id}_calib.npz")
        np.savez(cls.data_path, segments=segments, labels=labels)

    @classmethod
    def tearDownClass(cls):
        """Clean up generated files."""
        if os.path.exists(cls.data_path):
            os.remove(cls.data_path)

        # Remove models
        for m_type in ["rf", "cnn", "gru"]:
            model_path = os.path.join(config.MODELS_DIR, f"{cls.user_id}_model_{m_type}.joblib")
            if os.path.exists(model_path):
                os.remove(model_path)
            model_path_pth = os.path.join(config.MODELS_DIR, f"{cls.user_id}_model_{m_type}.pth")
            if os.path.exists(model_path_pth):
                os.remove(model_path_pth)

    def test_01_dsp_pipeline(self):
        """Test preprocessing and feature extraction."""
        test_seg = np.random.normal(0.0, 1.0, (150, 4))
        p_seg = preprocess_multichannel(
            test_seg,
            fs=config.SAMPLE_RATE,
            low=config.BANDPASS_LOW,
            high=config.BANDPASS_HIGH
        )
        self.assertEqual(p_seg.shape, (150, 4))

        features = extract_features(p_seg, sample_rate=config.SAMPLE_RATE)
        # 4 channels * 5 TD0 features * 21 context stacked * 2 aggregations (mean/std) = 840 features
        self.assertEqual(len(features), 840)

    def test_02_random_forest_pipeline(self):
        """Train and evaluate scikit-learn Random Forest model."""
        res = train_model(self.user_id, model_type="rf", test_size=0.25)
        self.assertIn("accuracy", res)
        self.assertIn("confusion_matrix", res)
        self.assertEqual(res["labels"], sorted(self.commands))

        # Verify inference
        engine = InferenceEngine(self.user_id, model_type="rf")
        test_seg = np.random.normal(0.0, 1.0, (150, 4))
        cmd, prob, proba = engine.predict_raw(test_seg)
        self.assertIn(cmd, self.commands)
        self.assertTrue(0.0 <= prob <= 1.0)
        self.assertEqual(len(proba), len(self.commands))

    def test_03_cnn_pipeline(self):
        """Train and evaluate PyTorch 1D CNN model."""
        res = train_model(self.user_id, model_type="cnn", test_size=0.25)
        self.assertIn("accuracy", res)
        self.assertIn("confusion_matrix", res)

        # Verify inference
        engine = InferenceEngine(self.user_id, model_type="cnn")
        test_seg = np.random.normal(0.0, 1.0, (150, 4))
        cmd, prob, proba = engine.predict_raw(test_seg)
        self.assertIn(cmd, self.commands)
        self.assertTrue(0.0 <= prob <= 1.0)
        self.assertEqual(len(proba), len(self.commands))

    def test_04_gru_pipeline(self):
        """Train and evaluate PyTorch GRU model."""
        res = train_model(self.user_id, model_type="gru", test_size=0.25)
        self.assertIn("accuracy", res)
        self.assertIn("confusion_matrix", res)

        # Verify inference
        engine = InferenceEngine(self.user_id, model_type="gru")
        test_seg = np.random.normal(0.0, 1.0, (150, 4))
        cmd, prob, proba = engine.predict_raw(test_seg)
        self.assertIn(cmd, self.commands)
        self.assertTrue(0.0 <= prob <= 1.0)
        self.assertEqual(len(proba), len(self.commands))


if __name__ == "__main__":
    unittest.main()
