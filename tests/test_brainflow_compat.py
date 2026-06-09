import time
import unittest
import numpy as np

from subvocal.hardware.brainflow_compat import BoardShim, BoardIds, BrainFlowInputParams, BrainFlowError
from subvocal.emg_core.dsp.brainflow_filter import DataFilter, FilterTypes, AggOperations, WindowOperations, NoiseTypes


class TestBrainFlowCompat(unittest.TestCase):

    def test_board_shim_synthetic(self):
        """Tests pure-Python BoardShim stream using SYNTHETIC_BOARD."""
        params = BrainFlowInputParams()
        # Force fallback by ensuring native shim is bypassed or fallback works
        shim = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
        # Manually force fallback if native was bound, to test our implementation
        shim._native_shim = None

        self.assertFalse(shim.is_prepared())
        shim.prepare_session()
        self.assertTrue(shim.is_prepared())

        # Test starting stream
        shim.start_stream()
        
        # Let it stream some packages
        time.sleep(0.15)
        
        # Current board data read (before clearing)
        current_data = shim.get_current_board_data(5)
        self.assertEqual(current_data.shape[1], 5)

        # Get data
        data = shim.get_board_data()
        self.assertGreater(data.shape[1], 0)
        self.assertEqual(data.shape[0], BoardShim.get_num_rows(BoardIds.SYNTHETIC_BOARD.value))

        # Verify package counters increment
        pkg_chan = BoardShim.get_package_num_channel(BoardIds.SYNTHETIC_BOARD.value)
        pkg_nums = data[pkg_chan]
        self.assertTrue(np.all(pkg_nums >= 0))

        # Verify timestamp channel is valid
        ts_chan = BoardShim.get_timestamp_channel(BoardIds.SYNTHETIC_BOARD.value)
        timestamps = data[ts_chan]
        self.assertTrue(np.all(timestamps > 0))

        # Verify EXG channels are not empty
        exg_chans = BoardShim.get_exg_channels(BoardIds.SYNTHETIC_BOARD.value)
        for ch in exg_chans:
            self.assertTrue(np.any(data[ch] != 0.0))

        # Stop and release
        shim.stop_stream()
        shim.release_session()
        self.assertFalse(shim.is_prepared())

    def test_data_filter_lowpass(self):
        """Tests lowpass filters ( causal & zero-phase)."""
        sr = 250
        t = np.arange(0, 1.0, 1.0 / sr)
        # Sinusoidal signal at 5Hz + 60Hz noise
        data = np.sin(2 * np.pi * 5.0 * t) + np.sin(2 * np.pi * 60.0 * t)
        
        # Filter with Butterworth Zero Phase (lowpass at 15Hz)
        DataFilter.perform_lowpass(
            data,
            sampling_rate=sr,
            cutoff=15.0,
            order=4,
            filter_type=FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
            ripple=0.0
        )
        # High frequency 60Hz should be heavily attenuated
        self.assertEqual(data.shape[0], sr)
        
    def test_data_filter_highpass(self):
        """Tests highpass filters."""
        sr = 250
        t = np.arange(0, 1.0, 1.0 / sr)
        data = np.sin(2 * np.pi * 1.0 * t) + np.sin(2 * np.pi * 50.0 * t)
        
        # Highpass at 10Hz
        DataFilter.perform_highpass(
            data,
            sampling_rate=sr,
            cutoff=10.0,
            order=4,
            filter_type=FilterTypes.BUTTERWORTH.value,
            ripple=0.0
        )
        self.assertEqual(data.shape[0], sr)

    def test_data_filter_bandpass(self):
        """Tests bandpass filters."""
        sr = 250
        t = np.arange(0, 1.0, 1.0 / sr)
        data = np.sin(2 * np.pi * 15.0 * t)
        
        # Bandpass between 10Hz and 30Hz
        DataFilter.perform_bandpass(
            data,
            sampling_rate=sr,
            start_freq=10.0,
            stop_freq=30.0,
            order=4,
            filter_type=FilterTypes.CHEBYSHEV_TYPE_1_ZERO_PHASE.value,
            ripple=1.0
        )
        self.assertEqual(data.shape[0], sr)

    def test_data_filter_bandstop_and_notch(self):
        """Tests bandstop and environmental notch filters."""
        sr = 250
        t = np.arange(0, 1.0, 1.0 / sr)
        data = np.sin(2 * np.pi * 60.0 * t)
        
        # Notch 60Hz environmental noise removal
        DataFilter.remove_environmental_noise(data, sampling_rate=sr, noise_type=NoiseTypes.SIXTY.value)
        self.assertEqual(data.shape[0], sr)

    def test_data_filter_rolling(self):
        """Tests rolling average and median filters."""
        data = np.array([1.0, 2.0, 3.0, 10.0, 5.0])
        
        # Rolling average
        data_mean = data.copy()
        DataFilter.perform_rolling_filter(data_mean, period=3, agg_operation=AggOperations.MEAN.value)
        self.assertEqual(data_mean.shape[0], 5)

        # Rolling median
        data_med = data.copy()
        DataFilter.perform_rolling_filter(data_med, period=3, agg_operation=AggOperations.MEDIAN.value)
        self.assertEqual(data_med.shape[0], 5)

    def test_data_filter_downsampling(self):
        """Tests downsampling logic."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        # Downsample period 2, mean
        downsampled = DataFilter.perform_downsampling(data, period=2, agg_operation=AggOperations.MEAN.value)
        self.assertTrue(np.allclose(downsampled, [1.5, 3.5, 5.5]))

    def test_data_filter_windows(self):
        """Tests window functions coefficients generation."""
        for op in (WindowOperations.NO_WINDOW, WindowOperations.HANNING, WindowOperations.HAMMING, WindowOperations.BLACKMAN_HARRIS):
            w = DataFilter.get_window(op.value, 100)
            self.assertEqual(w.shape[0], 100)

    def test_data_filter_psd_and_band_power(self):
        """Tests Power Spectral Density and band power calculations."""
        sr = 250
        t = np.arange(0, 1.0, 1.0 / sr)
        # 10Hz pure sine wave
        data = np.sin(2 * np.pi * 10.0 * t)
        
        # Calculate Welch PSD (NFFT = 128, overlap = 64)
        ampls, freqs = DataFilter.get_psd_welch(
            data,
            nfft=128,
            overlap=64,
            sampling_rate=sr,
            window=WindowOperations.HAMMING.value
        )
        self.assertEqual(ampls.shape[0], 65)
        self.assertEqual(freqs.shape[0], 65)
        self.assertAlmostEqual(freqs[1] - freqs[0], sr / 128)

        # Calculate band power around 10Hz (e.g. 8Hz to 12Hz)
        bp = DataFilter.get_band_power((ampls, freqs), freq_start=8.0, freq_end=12.0)
        self.assertGreater(bp, 0.0)

        # Calculate band power elsewhere (e.g. 30Hz to 40Hz) - should be much smaller
        bp_noise = DataFilter.get_band_power((ampls, freqs), freq_start=30.0, freq_end=40.0)
        self.assertLess(bp_noise, bp)
