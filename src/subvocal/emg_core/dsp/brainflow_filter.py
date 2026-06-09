import enum
import logging
import math

import numpy as np

from typing import Any
from subvocal.exceptions import MissingDependencyError

logger = logging.getLogger("subvocal.emg_core.dsp.brainflow_filter")

# We handle imports of scipy dynamically to keep core dependencies zero-install if not used.
try:
    import scipy.signal as scipy_signal
except ImportError:
    scipy_signal = None  # type: ignore

scipy_sig: Any = scipy_signal


class FilterTypes(enum.IntEnum):
    BUTTERWORTH = 0
    CHEBYSHEV_TYPE_1 = 1
    BESSEL = 2
    BUTTERWORTH_ZERO_PHASE = 3
    CHEBYSHEV_TYPE_1_ZERO_PHASE = 4
    BESSEL_ZERO_PHASE = 5


class AggOperations(enum.IntEnum):
    MEAN = 0
    MEDIAN = 1
    EACH = 2


class WindowOperations(enum.IntEnum):
    NO_WINDOW = 0
    HANNING = 1
    HAMMING = 2
    BLACKMAN_HARRIS = 3


class DetrendOperations(enum.IntEnum):
    NO_DETREND = 0
    CONSTANT = 1
    LINEAR = 2


class NoiseTypes(enum.IntEnum):
    FIFTY = 0
    SIXTY = 1
    FIFTY_AND_SIXTY = 2


def _check_scipy():
    if scipy_sig is None:
        raise MissingDependencyError(
            "SciPy is required to use the DataFilter signal processing APIs. "
            "Please install it using: pip install scipy"
        )


class DataFilter:
    """Pure Python/SciPy re-implementation of BrainFlow's DataFilter processing methods."""

    @classmethod
    def perform_lowpass(
        cls,
        data: np.ndarray,
        sampling_rate: int,
        cutoff: float,
        order: int,
        filter_type: int,
        ripple: float,
    ) -> None:
        _check_scipy()
        sos = cls._design_lowpass(sampling_rate, cutoff, order, filter_type, ripple)
        if filter_type in (
            FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
            FilterTypes.CHEBYSHEV_TYPE_1_ZERO_PHASE.value,
            FilterTypes.BESSEL_ZERO_PHASE.value,
        ):
            filtered = scipy_sig.sosfiltfilt(sos, data)
        else:
            filtered = scipy_sig.sosfilt(sos, data)
        data[:] = filtered

    @classmethod
    def perform_highpass(
        cls,
        data: np.ndarray,
        sampling_rate: int,
        cutoff: float,
        order: int,
        filter_type: int,
        ripple: float,
    ) -> None:
        _check_scipy()
        sos = cls._design_highpass(sampling_rate, cutoff, order, filter_type, ripple)
        if filter_type in (
            FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
            FilterTypes.CHEBYSHEV_TYPE_1_ZERO_PHASE.value,
            FilterTypes.BESSEL_ZERO_PHASE.value,
        ):
            filtered = scipy_sig.sosfiltfilt(sos, data)
        else:
            filtered = scipy_sig.sosfilt(sos, data)
        data[:] = filtered

    @classmethod
    def perform_bandpass(
        cls,
        data: np.ndarray,
        sampling_rate: int,
        start_freq: float,
        stop_freq: float,
        order: int,
        filter_type: int,
        ripple: float,
    ) -> None:
        _check_scipy()
        sos = cls._design_bandpass(sampling_rate, start_freq, stop_freq, order, filter_type, ripple)
        if filter_type in (
            FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
            FilterTypes.CHEBYSHEV_TYPE_1_ZERO_PHASE.value,
            FilterTypes.BESSEL_ZERO_PHASE.value,
        ):
            filtered = scipy_sig.sosfiltfilt(sos, data)
        else:
            filtered = scipy_sig.sosfilt(sos, data)
        data[:] = filtered

    @classmethod
    def perform_bandstop(
        cls,
        data: np.ndarray,
        sampling_rate: int,
        start_freq: float,
        stop_freq: float,
        order: int,
        filter_type: int,
        ripple: float,
    ) -> None:
        _check_scipy()
        sos = cls._design_bandstop(sampling_rate, start_freq, stop_freq, order, filter_type, ripple)
        if filter_type in (
            FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
            FilterTypes.CHEBYSHEV_TYPE_1_ZERO_PHASE.value,
            FilterTypes.BESSEL_ZERO_PHASE.value,
        ):
            filtered = scipy_sig.sosfiltfilt(sos, data)
        else:
            filtered = scipy_sig.sosfilt(sos, data)
        data[:] = filtered

    @classmethod
    def remove_environmental_noise(cls, data: np.ndarray, sampling_rate: int, noise_type: int) -> None:
        _check_scipy()
        if noise_type == NoiseTypes.FIFTY.value:
            cls.perform_bandstop(
                data, sampling_rate, 48.0, 52.0, 4, FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0.0
            )
        elif noise_type == NoiseTypes.SIXTY.value:
            cls.perform_bandstop(
                data, sampling_rate, 58.0, 62.0, 4, FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0.0
            )
        elif noise_type == NoiseTypes.FIFTY_AND_SIXTY.value:
            cls.perform_bandstop(
                data, sampling_rate, 48.0, 52.0, 4, FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0.0
            )
            cls.perform_bandstop(
                data, sampling_rate, 58.0, 62.0, 4, FilterTypes.BUTTERWORTH.value, 0.0
            )

    @classmethod
    def perform_rolling_filter(cls, data: np.ndarray, period: int, agg_operation: int) -> None:
        _check_scipy()
        if period <= 0:
            raise ValueError("Period must be > 0.")
        if agg_operation == AggOperations.MEAN.value:
            # Moving average using scipy convolve or uniform filter
            kernel = np.ones(period) / period
            # Mode='constant' with padding matches custom loop
            filtered = scipy_sig.lfilter(kernel, 1, data)
            # Replicate custom RollingFilter warm-up matching BrainFlow:
            # The custom rolling average uses historical buffer; in SciPy, lfilter handles it causal-wise.
            data[:] = filtered
        elif agg_operation == AggOperations.MEDIAN.value:
            # Causal running median: we compute running window medians
            filtered = np.zeros_like(data)
            for i in range(len(data)):
                start = max(0, i - period + 1)
                filtered[i] = np.median(data[start : i + 1])
            data[:] = filtered
        elif agg_operation == AggOperations.EACH.value:
            pass

    @classmethod
    def perform_downsampling(cls, data: np.ndarray, period: int, agg_operation: int) -> np.ndarray:
        if period <= 0:
            raise ValueError("Period must be > 0.")
        num_values = len(data) // period
        output = np.zeros(num_values)
        for i in range(num_values):
            chunk = data[i * period : (i + 1) * period]
            if agg_operation == AggOperations.MEAN.value:
                output[i] = np.mean(chunk)
            elif agg_operation == AggOperations.MEDIAN.value:
                output[i] = np.median(chunk)
            else:
                output[i] = chunk[0]
        return output

    @classmethod
    def get_window(cls, window_function: int, window_len: int) -> np.ndarray:
        if window_len <= 0:
            raise ValueError("Window length must be > 0.")
        wind = np.zeros(window_len)
        if window_function == WindowOperations.NO_WINDOW.value:
            wind[:] = 1.0
        elif window_function == WindowOperations.HANNING.value:
            for i in range(window_len):
                wind[i] = 0.5 - 0.5 * math.cos(2.0 * math.pi * i / window_len)
        elif window_function == WindowOperations.HAMMING.value:
            for i in range(window_len):
                wind[i] = 0.54 - 0.46 * math.cos(2.0 * math.pi * i / window_len)
        elif window_function == WindowOperations.BLACKMAN_HARRIS.value:
            for i in range(window_len):
                wind[i] = (
                    0.355768
                    - 0.487396 * math.cos(2.0 * math.pi * i / window_len)
                    + 0.144232 * math.cos(4.0 * math.pi * i / window_len)
                    - 0.012604 * math.cos(6.0 * math.pi * i / window_len)
                )
        return wind

    @classmethod
    def get_psd(cls, data: np.ndarray, sampling_rate: int, window: int) -> tuple[np.ndarray, np.ndarray]:
        if len(data) % 2 != 0:
            raise ValueError("Data length must be even for FFT estimation.")

        window_coeffs = cls.get_window(window, len(data))
        windowed_data = data * window_coeffs

        # Compute RFFT
        fft_vals = np.fft.rfft(windowed_data)
        re = fft_vals.real
        im = fft_vals.imag

        # PSD calculation
        ampls = (re**2 + im**2) / (sampling_rate * len(data))
        ampls[1:-1] *= 2.0  # Scale for one-sided FFT (excluding DC and Nyquist)

        freq_res = sampling_rate / len(data)
        freqs = np.arange(len(ampls)) * freq_res

        return ampls, freqs

    @classmethod
    def get_psd_welch(
        cls, data: np.ndarray, nfft: int, overlap: int, sampling_rate: int, window: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if nfft % 2 != 0:
            raise ValueError("nfft must be even.")
        if overlap < 0 or overlap > nfft:
            raise ValueError("overlap must be between 0 and nfft.")

        num_points = nfft // 2 + 1
        avg_ampls = np.zeros(num_points)
        freqs = np.zeros(num_points)

        counter = 0
        step = nfft - overlap
        if step <= 0:
            raise ValueError("overlap must be strictly less than nfft.")

        pos = 0
        while pos + nfft <= len(data):
            chunk = data[pos : pos + nfft]
            ampls, freqs = cls.get_psd(chunk, sampling_rate, window)
            avg_ampls += ampls
            counter += 1
            pos += step

        if counter == 0:
            raise ValueError("nfft must be less than or equal to data length.")

        avg_ampls /= counter
        return avg_ampls, freqs

    @classmethod
    def get_band_power(
        cls, psd: tuple[np.ndarray, np.ndarray], freq_start: float, freq_end: float
    ) -> float:
        ampl, freq = psd
        if len(freq) < 2:
            raise ValueError("PSD must have at least 2 points.")

        freq_res = freq[1] - freq[0]
        res = 0.0
        counter = 0

        for i in range(len(freq) - 1):
            if freq[i] > freq_end:
                break
            if freq[i] >= freq_start:
                res += 0.5 * freq_res * (ampl[i] + ampl[i + 1])
                counter += 1

        if counter == 0:
            raise ValueError("No data in the specified frequency band.")

        return res

    # Private Helpers for filter coefficients design
    @classmethod
    def _design_lowpass(
        cls, sampling_rate: int, cutoff: float, order: int, filter_type: int, ripple: float
    ):
        nyq = sampling_rate / 2.0
        wn = cutoff / nyq
        ftype = filter_type % 3  # Normalize zero-phase enum offset

        if ftype == 0:  # Butterworth
            return scipy_sig.butter(order, wn, btype="low", output="sos")
        elif ftype == 1:  # Chebyshev
            return scipy_sig.cheby1(order, rp=ripple, Wn=wn, btype="low", output="sos")
        else:  # Bessel
            return scipy_sig.bessel(order, wn, btype="low", output="sos", norm="phase")

    @classmethod
    def _design_highpass(
        cls, sampling_rate: int, cutoff: float, order: int, filter_type: int, ripple: float
    ):
        nyq = sampling_rate / 2.0
        wn = min(0.999, max(0.001, cutoff / nyq))
        ftype = filter_type % 3

        if ftype == 0:
            return scipy_sig.butter(order, wn, btype="high", output="sos")
        elif ftype == 1:
            return scipy_sig.cheby1(order, rp=ripple, Wn=wn, btype="high", output="sos")
        else:
            return scipy_sig.bessel(order, wn, btype="high", output="sos", norm="phase")

    @classmethod
    def _design_bandpass(
        cls,
        sampling_rate: int,
        start_freq: float,
        stop_freq: float,
        order: int,
        filter_type: int,
        ripple: float,
    ):
        nyq = sampling_rate / 2.0
        wn = [max(0.001, start_freq / nyq), min(0.999, stop_freq / nyq)]
        ftype = filter_type % 3

        if ftype == 0:
            return scipy_sig.butter(order, wn, btype="bandpass", output="sos")
        elif ftype == 1:
            return scipy_sig.cheby1(order, rp=ripple, Wn=wn, btype="bandpass", output="sos")
        else:
            return scipy_sig.bessel(order, wn, btype="bandpass", output="sos", norm="phase")

    @classmethod
    def _design_bandstop(
        cls,
        sampling_rate: int,
        start_freq: float,
        stop_freq: float,
        order: int,
        filter_type: int,
        ripple: float,
    ):
        nyq = sampling_rate / 2.0
        wn = [max(0.001, start_freq / nyq), min(0.999, stop_freq / nyq)]
        ftype = filter_type % 3

        if ftype == 0:
            return scipy_sig.butter(order, wn, btype="bandstop", output="sos")
        elif ftype == 1:
            return scipy_sig.cheby1(order, rp=ripple, Wn=wn, btype="bandstop", output="sos")
        else:
            return scipy_sig.bessel(order, wn, btype="bandstop", output="sos", norm="phase")
