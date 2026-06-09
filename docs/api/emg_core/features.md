---
title: emg_core.dsp.features
sidebar_label: features
---

Feature extraction for sEMG segments.

Implements the TD0/TD10 feature pipeline (validated on the EMG-UKA corpus).
This decomposes the raw sEMG signal into:
1. Low-frequency articulatory gestures (using a double moving average filter).
2. High-frequency muscular activity.

Features are computed over sliding window frames, stacked with context,
and averaged over the entire segment to create robust representations.

## Functions

### `extract_features_td10`

```python
def extract_features_td10(segment: np.ndarray, sample_rate: float = 250.0, frame_size_ms: int = 27, frame_shift_ms: int = 10, context: int = 10) -> np.ndarray
```

Extract TD10 features for all channels.


**Returns:**

- 2D array: (num_frames, num_channels * 5 * (2*context + 1))

### `extract_features_td10_segment`

```python
def extract_features_td10_segment(segment: np.ndarray, sample_rate: float = 250.0, frame_size_ms: int = 27, frame_shift_ms: int = 10, context: int = 10) -> np.ndarray
```

Aggregate frame-level TD10 features into a single 1D segment feature vector.

Computes mean and standard deviation across frames.
For 4 channels with context=10: 4 * 5 * 21 * 2 = 840 dimensional vector.

### `extract_features`

```python
def extract_features(segment: np.ndarray, sample_rate: float = 250.0) -> np.ndarray
```

Extract features from a multi-channel segment using TD10 pipeline.

### `extract_features_batch`

```python
def extract_features_batch(segments: list[np.ndarray], sample_rate: float = 250.0) -> np.ndarray
```

Extract features for a batch of segments.
