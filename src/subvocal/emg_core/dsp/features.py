"""Feature extraction for sEMG segments.

Implements the TD0/TD10 feature pipeline (validated on the EMG-UKA corpus).
This decomposes the raw sEMG signal into:
1. Low-frequency articulatory gestures (using a double moving average filter).
2. High-frequency muscular activity.

Features are computed over sliding window frames, stacked with context,
and averaged over the entire segment to create robust representations.
"""

import numpy as np


def _double_moving_average(x: np.ndarray, window: int = 9) -> np.ndarray:
    """9-point double moving average to extract low-frequency component.

    First pass: forward moving average. Second pass: moving average of the
    result. This extracts the slow articulation movement component.
    """
    kernel = np.ones(window) / window
    w1 = np.convolve(x, kernel, mode='same')
    w = np.convolve(w1, kernel, mode='same')
    return w


def _compute_td0_channel(
    signal: np.ndarray,
    fs: int,
    frame_size_ms: int = 27,
    frame_shift_ms: int = 10,
) -> np.ndarray:
    """Compute 5 TD0 features per frame for a single channel.

    Features:
    1. w_bar: mean of low-frequency component (articulatory position)
    2. P_w: power of low-frequency component (articulatory energy)
    3. P_r: power of rectified high-frequency component (muscle activity energy)
    4. z_p: zero-crossing rate of high-frequency component
    5. r_bar: mean of rectified high-frequency component

    Returns:
        (num_frames, 5) array of features.
    """
    frame_size = max(int(fs * frame_size_ms / 1000), 1)
    frame_shift = max(int(fs * frame_shift_ms / 1000), 1)

    if len(signal) < frame_size:
        signal = np.pad(signal, (0, frame_size - len(signal)), mode='constant')

    w = _double_moving_average(signal)
    p = signal - w
    r = np.abs(p)

    num_frames = max(1, 1 + (len(signal) - frame_size) // frame_shift)
    td0 = np.zeros((num_frames, 5))

    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_size
        if end > len(signal):
            end = len(signal)

        w_frame = w[start:end]
        p_frame = p[start:end]
        r_frame = r[start:end]

        td0[i, 0] = np.mean(w_frame)
        td0[i, 1] = np.mean(w_frame ** 2)
        td0[i, 2] = np.mean(r_frame ** 2)

        if len(p_frame) > 1:
            sign_changes = np.abs(np.diff(np.sign(p_frame)))
            td0[i, 3] = np.sum(sign_changes > 0) / len(p_frame)
        else:
            td0[i, 3] = 0.0

        td0[i, 4] = np.mean(r_frame)

    return td0


def _stack_context(td0_frames: np.ndarray, context: int = 10) -> np.ndarray:
    """Stack ±context adjacent frames to construct TD10 features.

    For each frame, concatenates neighboring frames. Edge frames are padded.
    """
    num_frames, num_feats = td0_frames.shape
    total_width = 2 * context + 1

    padded = np.pad(td0_frames, ((context, context), (0, 0)), mode='edge')
    td10 = np.zeros((num_frames, num_feats * total_width))

    for i in range(num_frames):
        td10[i] = padded[i:i + total_width].ravel()

    return td10


def extract_features_td10(
    segment: np.ndarray,
    sample_rate: float = 250.0,
    frame_size_ms: int = 27,
    frame_shift_ms: int = 10,
    context: int = 10,
) -> np.ndarray:
    """Extract TD10 features for all channels.

    Returns:
        2D array: (num_frames, num_channels * 5 * (2*context + 1))
    """
    fs = int(sample_rate)
    num_channels = segment.shape[1]

    channel_td0s = []
    for ch in range(num_channels):
        td0 = _compute_td0_channel(
            segment[:, ch],
            fs,
            frame_size_ms=frame_size_ms,
            frame_shift_ms=frame_shift_ms,
        )
        channel_td0s.append(td0)

    min_frames = min(t.shape[0] for t in channel_td0s)
    channel_td0s = [t[:min_frames] for t in channel_td0s]

    td0_all = np.concatenate(channel_td0s, axis=1)
    td10 = _stack_context(td0_all, context=context)
    return td10


def extract_features_td10_segment(
    segment: np.ndarray,
    sample_rate: float = 250.0,
    frame_size_ms: int = 27,
    frame_shift_ms: int = 10,
    context: int = 10,
) -> np.ndarray:
    """Aggregate frame-level TD10 features into a single 1D segment feature vector.

    Computes mean and standard deviation across frames.
    For 4 channels with context=10: 4 * 5 * 21 * 2 = 840 dimensional vector.
    """
    td10 = extract_features_td10(
        segment,
        sample_rate,
        frame_size_ms=frame_size_ms,
        frame_shift_ms=frame_shift_ms,
        context=context,
    )
    frame_mean = np.mean(td10, axis=0)
    frame_std = np.std(td10, axis=0)
    return np.concatenate([frame_mean, frame_std]).astype(np.float64)


def extract_features(segment: np.ndarray, sample_rate: float = 250.0) -> np.ndarray:
    """Extract features from a multi-channel segment using TD10 pipeline."""
    return extract_features_td10_segment(segment, sample_rate)


def extract_features_batch(
    segments: list[np.ndarray],
    sample_rate: float = 250.0,
) -> np.ndarray:
    """Extract features for a batch of segments."""
    return np.array([extract_features(seg, sample_rate) for seg in segments])
