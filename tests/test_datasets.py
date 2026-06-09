"""Unit and integration tests for academic dataset drivers (Ninapro, PutEMG, CSL-HDEMG).
"""

import os
import tempfile
import unittest

import numpy as np
import scipy.io

from subvocal.hardware.datasets import CSLHDEMGDriver, NinaproDriver, PutEMGDriver


class TestDatasetDrivers(unittest.TestCase):
    """Test suite for replaying academic sEMG datasets."""

    def test_ninapro_driver(self):
        """Test NinaproDriver loading and streaming MAT files."""
        # 1. Create a mock Ninapro MAT file
        mock_emg = np.random.randn(100, 10).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            scipy.io.savemat(tmp.name, {"emg": mock_emg})
            tmp_path = tmp.name

        try:
            # 2. Initialize driver
            driver = NinaproDriver(file_path=tmp_path, fs=100.0, loop=True)
            self.assertEqual(driver.num_channels, 10)
            self.assertEqual(driver._emg_data.shape[0], 100)

            # 3. Read frame
            driver.start()
            frame = driver.read_frame(window_ms=100)  # 10 samples
            self.assertEqual(len(frame.samples), 10)
            self.assertEqual(len(frame.samples[0].channels), 10)
            driver.stop()

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_csl_hdemg_npy_driver(self):
        """Test CSLHDEMGDriver loading and streaming NumPy .npy files."""
        # 1. Create mock NumPy data
        mock_emg = np.random.randn(100, 16).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp.name, mock_emg)
            tmp_path = tmp.name

        try:
            # 2. Initialize driver
            driver = CSLHDEMGDriver(file_path=tmp_path, fs=100.0, loop=True)
            self.assertEqual(driver.num_channels, 16)

            # 3. Read frame
            driver.start()
            frame = driver.read_frame(window_ms=200)  # 20 samples
            self.assertEqual(len(frame.samples), 20)
            driver.stop()

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_csl_hdemg_bin_driver(self):
        """Test CSLHDEMGDriver loading and streaming raw binary float files."""
        # 1. Create raw float binary data
        mock_emg = np.random.randn(100, 8).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            mock_emg.tofile(tmp.name)
            tmp_path = tmp.name

        try:
            # 2. Initialize driver (must specify num_channels for raw binary)
            driver = CSLHDEMGDriver(file_path=tmp_path, fs=100.0, loop=True, num_channels=8)
            self.assertEqual(driver.num_channels, 8)

            # 3. Read frame
            driver.start()
            frame = driver.read_frame(window_ms=100)
            self.assertEqual(len(frame.samples), 10)
            driver.stop()

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_putemg_driver_integration(self):
        """Test PutEMGDriver HDF5 loading, looping and fallback behaviors."""
        try:
            import h5py
        except ImportError:
            # Skip actual file test if h5py is not installed
            self.skipTest("h5py not installed, skipping HDF5 parsing test")

        # Create mock H5 structure
        mock_emg = np.random.randn(100, 12).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            with h5py.File(tmp.name, "w") as f:
                # PutEMGDriver searches for a 2D dataset in the file
                f.create_dataset("emg", data=mock_emg)
            tmp_path = tmp.name

        try:
            driver = PutEMGDriver(file_path=tmp_path, fs=100.0, loop=True)
            self.assertEqual(driver.num_channels, 12)

            driver.start()
            frame = driver.read_frame(window_ms=150)  # 15 samples
            self.assertEqual(len(frame.samples), 15)
            
            # Read enough to trigger looping
            frame_loop = driver.read_frame(window_ms=900)  # 90 samples (15 + 90 = 105 samples > 100)
            self.assertEqual(len(frame_loop.samples), 90)
            
            driver.stop()

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
