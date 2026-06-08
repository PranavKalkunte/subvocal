---
title: emg_core.ml.export
sidebar_label: export
---

Model export and quantization pipeline for sEMG classifiers.

## Functions

### `export_to_onnx`

```python
def export_to_onnx(user_id: str, model_type: str, export_path: str) -> str
```

Export a trained PyTorch model to ONNX format.


**Arguments:**

- ` user_id `: The ID of the user.
- ` model_type `: "cnn", "gru", or "transformer".
- ` export_path `: Destination path for the exported ONNX model.

### `export_to_coreml`

```python
def export_to_coreml(user_id: str, model_type: str, export_path: str) -> bool
```

Compile PyTorch/ONNX model to Apple Core ML format.

Note: requires coremltools. If missing, handles gracefully.

### `export_to_tflite`

```python
def export_to_tflite(user_id: str, model_type: str, export_path: str) -> bool
```

Compile PyTorch/ONNX model to TFLite format.

Note: requires tensorflow. If missing, handles gracefully.

### `quantize_model_int8`

```python
def quantize_model_int8(user_id: str, model_type: str, threshold: float = 0.05) -> Dict[str, Any]
```

Perform dynamic int8 quantization on a PyTorch model and run accuracy regression checks.


**Arguments:**

- ` user_id `: User calibration dataset ID to evaluate against.
- ` model_type `: "cnn", "gru", or "transformer".
- ` threshold `: Maximum allowed accuracy drop (e.g. 0.05 = 5%).


**Returns:**

- Dict detailing quantization metrics.
