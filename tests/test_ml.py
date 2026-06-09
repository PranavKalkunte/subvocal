"""End-to-end integration tests for sEMG DSP and ML components."""

import glob
import os
import unittest

import numpy as np

from subvocal.emg_core import config
from subvocal.emg_core.dsp.features import extract_features
from subvocal.emg_core.dsp.filters import preprocess_multichannel
from subvocal.emg_core.ml.benchmark import run_benchmark
from subvocal.emg_core.ml.export import export_to_onnx, quantize_model_int8
from subvocal.emg_core.ml.infer import InferenceEngine
from subvocal.emg_core.ml.model_io import get_model_path
from subvocal.emg_core.ml.train import calibrate_model, train_model


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
        # Create 8 samples per class = 32 segments total (enough for train/test split, faster on CI)
        for i, cmd in enumerate(cls.commands):
            for _ in range(8):
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

        # Clean up files for calibration user
        calib_user_data = os.path.join(config.DATA_DIR, "test_user_calib_calib.npz")
        if os.path.exists(calib_user_data):
            os.remove(calib_user_data)

        # Remove all model artifacts for the test users (glob catches any extension or naming variant)
        for uid in [cls.user_id, "test_user_calib"]:
            for path in glob.glob(os.path.join(config.MODELS_DIR, f"{uid}_model_*")):
                os.remove(path)

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

    def test_05_svm_pipeline(self):
        """Train and evaluate SVM baseline model."""
        res = train_model(self.user_id, model_type="svm", test_size=0.25)
        self.assertIn("accuracy", res)
        self.assertIn("confusion_matrix", res)
        self.assertEqual(res["labels"], sorted(self.commands))

        # Verify inference
        engine = InferenceEngine(self.user_id, model_type="svm")
        test_seg = np.random.normal(0.0, 1.0, (150, 4))
        cmd, prob, proba = engine.predict_raw(test_seg)
        self.assertIn(cmd, self.commands)
        self.assertTrue(0.0 <= prob <= 1.0)
        self.assertEqual(len(proba), len(self.commands))

    def test_06_transformer_pipeline(self):
        """Train and evaluate PyTorch Transformer model."""
        res = train_model(self.user_id, model_type="transformer", test_size=0.25)
        self.assertIn("accuracy", res)
        self.assertIn("confusion_matrix", res)

        # Verify inference
        engine = InferenceEngine(self.user_id, model_type="transformer")
        test_seg = np.random.normal(0.0, 1.0, (150, 4))
        cmd, prob, proba = engine.predict_raw(test_seg)
        self.assertIn(cmd, self.commands)
        self.assertTrue(0.0 <= prob <= 1.0)
        self.assertEqual(len(proba), len(self.commands))

    def test_07_calibration_pipeline(self):
        """Test per-user calibration / fine-tuning routine."""
        pretrained_path = get_model_path("pretrained", "cnn")
        if not os.path.exists(pretrained_path):
            self.skipTest("Pretrained CNN weights not found; calibration transfer-learning path cannot be tested. Run sdk/emg_core/ml/pretrain.py first.")

        # Create a calibration user dataset
        calib_user = "test_user_calib"
        calib_data_path = os.path.join(config.DATA_DIR, f"{calib_user}_calib.npz")
        
        # 10 segments per class = 40 segments total
        segments = []
        labels = []
        for i, cmd in enumerate(self.commands):
            for _ in range(10):
                seg = np.random.normal(0.0, 1.0, (150, 4))
                t = np.linspace(0, 2*np.pi, 150)
                freq = (i + 1) * 5.0
                pattern = np.sin(t * freq)[:, np.newaxis]
                seg += pattern * 2.0
                segments.append(seg)
                labels.append(cmd)
        np.savez(calib_data_path, segments=segments, labels=labels)

        # Calibrate a model (fine-tunes base model pretrained on pretrained baseline dataset)
        res = calibrate_model(calib_user, pretrained_model_type="cnn")
        self.assertIn("accuracy", res)
        self.assertEqual(res["labels"], sorted(self.commands))
        
        # Verify inference of calibrated model
        engine = InferenceEngine(calib_user, model_type="cnn")
        test_seg = np.random.normal(0.0, 1.0, (150, 4))
        cmd, prob, proba = engine.predict_raw(test_seg)
        self.assertIn(cmd, self.commands)
        
        # Clean up calibration dataset file
        if os.path.exists(calib_data_path):
            os.remove(calib_data_path)

    def test_08_onnx_export(self):
        """Test model export to ONNX."""
        onnx_path = os.path.join(config.MODELS_DIR, f"{self.user_id}_model_cnn.onnx")
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
            
        try:
            export_to_onnx(self.user_id, "cnn", onnx_path)
            self.assertTrue(os.path.exists(onnx_path))
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"Skipping ONNX export test due to missing packages: {e}")

    def test_09_quantization(self):
        """Test int8 dynamic quantization with accuracy regression checks."""
        res = quantize_model_int8(self.user_id, "cnn", threshold=0.1)
        self.assertIn(res["status"], ["APPROVED", "SKIPPED_NO_ENGINE"])
        
        # Test inference loading the quantized model if approved
        if res["status"] == "APPROVED":
            engine = InferenceEngine(self.user_id, model_type="cnn_quantized")
            test_seg = np.random.normal(0.0, 1.0, (150, 4))
            cmd, prob, proba = engine.predict_raw(test_seg)
            self.assertIn(cmd, self.commands)

    def test_10_benchmarking_harness(self):
        """Test inference benchmarking harness (latency, params, flops, energy)."""
        metrics = run_benchmark(self.user_id, model_type="cnn", num_runs=10)
        self.assertEqual(metrics["model_type"], "cnn")
        self.assertGreater(metrics["latency_mean_ms"], 0.0)
        self.assertGreater(metrics["parameter_count"], 0)
        self.assertGreater(metrics["estimated_flops"], 0)
        self.assertGreater(metrics["estimated_energy_uj"], 0.0)


if __name__ == "__main__":
    unittest.main()

