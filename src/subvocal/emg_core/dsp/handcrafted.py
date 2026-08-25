"""Handcrafted sEMG feature extraction (112-dim).

Implements the 112-dimensional handcrafted feature set used for silent-speech
sEMG, following:

* Mohapatra et al., ACL 2025 — EMG-to-LLM adaptor with handcrafted vs
  speech-encoder features; TD-style windowing and spectral descriptors.
* Jou et al., 2006 — temporal sEMG features (MAV, ZC, WL, SSC, WAMP) for
  EMG-to-speech; thresholded zero-crossing / Willison amplitude.
* Gaddy & Klein, 2020 — spectral sEMG descriptors (mean/median frequency,
  bandpower) for silent EMG ASR.

Per-channel 28 features × 4 channels = 112 dimensions::

    temporal (11): MAV, RMS, VAR, WL, ZC, SSC, WAMP, IEMG, SSI, DASDV, LOGVAR
    stats    (7):  mean, std, min, max, peak-to-peak, skewness, kurtosis
    spectral (10): MNF, MDF, centroid, bandpower, bandpower-low/high,
                   peak-freq, entropy, spread, rolloff (85%)

Spectral features use :mod:`numpy.fft` (and :func:`scipy.signal.welch` when
available) — no :mod:`torch` required. Imports are lazy / guarded.

References
----------
* Mohapatra et al., "Bridging EMG and LLMs", ACL 2025.
* Jou et al., "EMG-Based Silent Speech Interface", 2006.
* Gaddy & Klein, "Digital Voicing of Silent Speech", EMNLP 2020.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["extract_handcrafted_features", "extract_handcrafted_timevarying"]

# ---------------------------------------------------------------------------
# per-channel helpers
# ---------------------------------------------------------------------------

_THRESH_ZC: float = 1e-4
_THRESH_SSC: float = 1e-4
_THRESH_WAMP: float = 0.01
_EPS: float = 1e-12
_TARGET_DIM: int = 112
_PER_CHANNEL: int = 28


def _temporal_stats(x: np.ndarray) -> list[float]:
    """Compute temporal / statistical scalar features for 1-D signal *x*."""
    n = x.size
    if n == 0:
        return [0.0] * 18  # 11 temporal + 7 stats placeholder

    mean_v = float(np.mean(x))
    std_v = float(np.std(x))
    var_v = float(np.var(x))
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    ptp_v = float(max_v - min_v)

    # MAV / RMS
    mav = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    iemg = float(np.sum(np.abs(x)))
    ssi = float(np.sum(x * x))
    # WL
    wl = float(np.sum(np.abs(np.diff(x)))) if n > 1 else 0.0
    # DASDV
    if n > 1:
        dasdv = float(np.sqrt(np.mean(np.diff(x) ** 2)))
    else:
        dasdv = 0.0
    # LOGVAR
    logvar = float(np.log(var_v + _EPS))

    # ZC – normalised rate
    if n > 1:
        prod = x[:-1] * x[1:]
        amp = np.abs(x[:-1] - x[1:]) >= _THRESH_ZC
        zc = float(np.sum((prod < 0) & amp) / (n - 1))
    else:
        zc = 0.0

    # SSC – slope sign change, normalised
    if n > 2:
        d = np.diff(x)
        prod_d = d[:-1] * d[1:]
        # threshold on difference of successive diffs
        cond = np.abs(d[:-1] - d[1:]) >= _THRESH_SSC
        ssc = float(np.sum((prod_d < 0) & cond) / (n - 2))
    else:
        ssc = 0.0

    # WAMP – Willison amplitude, normalised rate
    if n > 1:
        wamp = float(np.sum(np.abs(np.diff(x)) >= _THRESH_WAMP) / (n - 1))
    else:
        wamp = 0.0

    # skewness / kurtosis (excess)
    if std_v > _EPS:
        norm = (x - mean_v) / std_v
        skew = float(np.mean(norm ** 3))
        kurt = float(np.mean(norm ** 4) - 3.0)
    else:
        skew = 0.0
        kurt = -3.0

    # order: MAV, RMS, VAR, WL, ZC, SSC, WAMP, IEMG, SSI, DASDV, LOGVAR,
    #        mean, std, min, max, ptp, skew, kurt  (18)
    return [
        mav, rms, var_v, wl, zc, ssc, wamp, iemg, ssi, dasdv, logvar,
        mean_v, std_v, min_v, max_v, ptp_v, skew, kurt,
    ]


def _spectral_features(x: np.ndarray, fs: float) -> list[float]:
    """10 spectral features via FFT (Welch if scipy available and N sufficient)."""
    n = x.size
    if n < 4 or fs <= 0:
        return [0.0] * 10

    # Try Welch for more stable PSD when segment is long enough
    freqs: np.ndarray
    psd: np.ndarray
    try:
        from scipy.signal import welch as _welch  # lazy import

        nperseg = min(n, max(16, int(fs * 0.1)))  # ~100 ms window
        if n >= nperseg:
            freqs, psd = _welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            # welch returns PSD (V**2/Hz); keep as is
        else:
            raise ValueError("short segment – fall back to FFT")
    except Exception:
        # Fallback: periodogram via FFT
        X = np.fft.rfft(x)
        psd = (np.abs(X) ** 2) / max(n, 1)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Guard against all-zero PSD
    total = float(np.sum(psd))
    if total < _EPS:
        return [0.0] * 10

    # Mean / median frequency
    mnf = float(np.sum(freqs * psd) / total)
    # centroid – same as MNF for magnitude spectrum; keep duplicate per spec
    centroid = mnf
    cumsum = np.cumsum(psd)
    half = total * 0.5
    mdf_idx = int(np.searchsorted(cumsum, half))
    mdf_idx = min(mdf_idx, freqs.size - 1)
    mdf = float(freqs[mdf_idx])

    bandpower = float(total)
    # low <30 Hz, high >=30 Hz (covers ALS band split)
    low_mask = freqs < 30.0
    bandpower_low = float(np.sum(psd[low_mask]))
    bandpower_high = float(total - bandpower_low)

    peak_freq = float(freqs[int(np.argmax(psd))])

    # spectral entropy
    p_norm = psd / total
    # avoid log(0)
    p_norm = p_norm[p_norm > 0]
    entropy = float(-np.sum(p_norm * np.log(p_norm + _EPS)))

    # spectral spread (variance around centroid)
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / total))

    # roll-off 85%
    roll_thresh = total * 0.85
    roll_idx = int(np.searchsorted(cumsum, roll_thresh))
    roll_idx = min(roll_idx, freqs.size - 1)
    rolloff = float(freqs[roll_idx])

    return [
        mnf, mdf, centroid, bandpower, bandpower_low,
        bandpower_high, peak_freq, entropy, spread, rolloff,
    ]


def _extract_per_channel(x: np.ndarray, fs: float) -> np.ndarray:
    """28-d feature vector for a single channel."""
    # x is 1-D (T,)
    temporal = _temporal_stats(x)  # 18
    spectral = _spectral_features(x, fs)  # 10
    feats = np.array(temporal + spectral, dtype=np.float64)
    # Ensure exactly 28
    if feats.size != _PER_CHANNEL:
        logger.warning("per-channel feature count %d != %d – padding/truncating", feats.size, _PER_CHANNEL)
        if feats.size < _PER_CHANNEL:
            feats = np.pad(feats, (0, _PER_CHANNEL - feats.size))
        else:
            feats = feats[:_PER_CHANNEL]
    # sanitize
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def extract_handcrafted_features(segment: np.ndarray, fs: float = 250) -> np.ndarray:
    """Extract 112 handcrafted sEMG features for a segment.

    Args:
        segment: Array of shape ``(T, C)`` – ``T`` time samples, ``C`` channels.
            Typical subvocal segment is ``(150, 4)`` at ``fs=250 Hz``.
        fs: Sampling rate in Hz.

    Returns:
        Array of shape ``(112,)`` (``C * 28`` for ``C=4``). For ``C != 4``
        the output is ``(C*28,)`` padded/truncated to 112 to satisfy the
        adaptor ``input_dim=112`` contract.

    References:
        Mohapatra et al., ACL 2025; Jou et al., 2006; Gaddy & Klein, 2020.
    """
    if not isinstance(segment, np.ndarray):
        raise TypeError(f"segment must be np.ndarray, got {type(segment)}")
    if segment.ndim != 2:
        raise ValueError(f"segment must be 2-D (T, C), got shape {segment.shape}")
    if segment.shape[0] == 0 or segment.shape[1] == 0:
        raise ValueError(f"segment has empty dimension: {segment.shape}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    n_channels = segment.shape[1]
    logger.debug("extract_handcrafted_features: segment=%s fs=%.1f", segment.shape, fs)

    # Lazily ensure numpy/scipy only – no torch
    seg = segment.astype(np.float64, copy=False)

    per_ch = [_extract_per_channel(seg[:, ch], fs) for ch in range(n_channels)]
    feats = np.concatenate(per_ch, axis=0)

    # Contract: 112 for standard 4-channel subvocal segment
    if n_channels == 4:
        # must be exactly 112
        if feats.size != _TARGET_DIM:
            logger.warning("feature dim %d != %d for 4-channel input", feats.size, _TARGET_DIM)
            if feats.size < _TARGET_DIM:
                feats = np.pad(feats, (0, _TARGET_DIM - feats.size))
            else:
                feats = feats[:_TARGET_DIM]
    else:
        # For non-standard channel counts, pad/truncate to 112 so
        # EMGAdaptor(input_dim=112) can still consume the vector.
        if feats.size != _TARGET_DIM:
            logger.debug("adapting feature dim %d -> %d for non-4-channel input", feats.size, _TARGET_DIM)
            if feats.size < _TARGET_DIM:
                feats = np.pad(feats, (0, _TARGET_DIM - feats.size))
            else:
                feats = feats[:_TARGET_DIM]

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    return feats


def extract_handcrafted_timevarying(
    segment: np.ndarray,
    fs: float = 250,
    window_ms: int = 50,
    step_ms: int = 20,
) -> np.ndarray:
    """Time-varying 112-d features via sliding windows.

    Splits *segment* into overlapping windows of ``window_ms`` with step
    ``step_ms`` and extracts :func:`extract_handcrafted_features` per window.

    Args:
        segment: Array of shape ``(T, C)``.
        fs: Sampling rate in Hz.
        window_ms: Window length in milliseconds.
        step_ms: Step / hop in milliseconds.

    Returns:
        Array of shape ``(num_windows, 112)``.

    References:
        Mohapatra et al., ACL 2025; Jou et al., 2006.
    """
    if not isinstance(segment, np.ndarray):
        raise TypeError(f"segment must be np.ndarray, got {type(segment)}")
    if segment.ndim != 2:
        raise ValueError(f"segment must be 2-D (T, C), got shape {segment.shape}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if window_ms <= 0 or step_ms <= 0:
        raise ValueError(f"window_ms and step_ms must be positive, got {window_ms}, {step_ms}")

    n_samples = segment.shape[0]
    window_samples = max(1, int(round(fs * window_ms / 1000.0)))
    step_samples = max(1, int(round(fs * step_ms / 1000.0)))

    logger.debug(
        "extract_handcrafted_timevarying: segment=%s fs=%.1f window=%d (%d samp) step=%d (%d samp)",
        segment.shape, fs, window_ms, window_samples, step_ms, step_samples,
    )

    if n_samples < window_samples:
        # Pad with edge/zeros to produce at least one window
        pad_len = window_samples - n_samples
        logger.debug("segment shorter than window (%d < %d) – padding %d samples", n_samples, window_samples, pad_len)
        # pad with zeros (or edge) – zeros keeps features stable
        segment_padded = np.pad(segment, ((0, pad_len), (0, 0)), mode="constant")
        feats = extract_handcrafted_features(segment_padded, fs=fs)
        return feats[np.newaxis, :]

    num_windows = 1 + (n_samples - window_samples) // step_samples
    # Guard: at least one window
    num_windows = max(1, num_windows)

    windows = np.empty((num_windows, _TARGET_DIM), dtype=np.float64)
    for i in range(num_windows):
        start = i * step_samples
        end = start + window_samples
        window = segment[start:end, :]
        windows[i] = extract_handcrafted_features(window, fs=fs)

    return windows
