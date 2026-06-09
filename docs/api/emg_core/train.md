---
title: emg_core.ml.train
sidebar_label: train
---

Training pipeline for sEMG command classifiers.

Supports:
1. RandomForest (Standard ML baseline using TD10 aggregated features).
2. SVM (Feature-based Support Vector Machine baseline).
3. 1D CNN (Deep learning model operating on raw time-series segments).
4. GRU (Bidirectional temporal RNN operating on raw time-series segments).
5. Transformer (Attention-based sequential encoder operating on raw segments).

## Classes

### `class EMG1DCNN(nn.Module)`

1D CNN for sEMG segment classification.

Applies convolutions over the temporal dimension of raw multichannel sEMG.

#### Methods

##### `__init__`

```python
def __init__(self, num_channels: int, num_classes: int, segment_length: int = 150)
```

No description.

##### `forward`

```python
def forward(self, x)
```

No description.

### `class EMGGRU(nn.Module)`

Bidirectional GRU for temporal tracking of sEMG gestures.

#### Methods

##### `__init__`

```python
def __init__(self, num_channels: int, num_classes: int, hidden_size: int = 64, num_layers: int = 2)
```

No description.

##### `forward`

```python
def forward(self, x)
```

No description.

### `class EMGTransformer(nn.Module)`

Small Transformer encoder for raw multichannel sEMG sequence classification.

#### Methods

##### `__init__`

```python
def __init__(self, num_channels: int, num_classes: int, segment_length: int = 150, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128)
```

No description.

##### `forward`

```python
def forward(self, x)
```

No description.

## Functions

### `load_dataset`

```python
def load_dataset(user_id: str) -> tuple[list[np.ndarray], list[str]]
```

Load the calibration dataset for a user.


**Returns:**

- segments: list of raw 2D arrays, each (segment_length, num_channels)
- labels: list of label strings corresponding to each segment

### `preprocess_segments`

```python
def preprocess_segments(segments: list[np.ndarray], fs: float, low: float, high: float) -> np.ndarray
```

Preprocess raw segments and stack into a 3D numpy array of shape (N, C, T).

### `train_model`

```python
def train_model(user_id: str, model_type: str = 'rf', test_size: float = 0.2, config_obj: TrainingConfig | None = None) -> dict[str, Any]
```

Train a sEMG model for a user and save it to disk.


**Arguments:**

- ` user_id `: The ID of the user.
- ` model_type `: "rf", "svm", "cnn", "gru", or "transformer".
- ` test_size `: Split ratio for testing/validation.
- ` config_obj `: Optional TrainingConfig parameters object.


**Returns:**

- A dictionary containing training metrics and results.

### `calibrate_model`

```python
def calibrate_model(user_id: str, pretrained_model_type: str, calibration_config: TrainingConfig | None = None) -> dict[str, Any]
```

Calibrate / fine-tune a pre-trained model for a new user.

For PyTorch models (CNN, GRU, Transformer), it loads pre-trained weights,
replaces/adjusts the classification head to match the user's classes,
freezes base layers (optionally), and fine-tunes on user calibration data.
For classical models (RF, SVM), it fits a new pipeline on user data.
