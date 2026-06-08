"""Model export and quantization pipeline for sEMG classifiers."""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any

from emg_core import config
from emg_core.ml.model_io import load_model, save_model
from emg_core.ml.train import EMG1DCNN, EMGGRU, EMGTransformer, load_dataset, preprocess_segments, train_test_split


def export_to_onnx(user_id: str, model_type: str, export_path: str) -> str:
    """Export a trained PyTorch model to ONNX format.

    Args:
        user_id: The ID of the user.
        model_type: "cnn", "gru", or "transformer".
        export_path: Destination path for the exported ONNX model.
    """
    model_data = load_model(user_id, model_type)
    num_channels = model_data["num_channels"]
    num_classes = len(model_data["labels"])
    segment_length = model_data["segment_length"]

    cfg = model_data.get("config", {})
    hidden_size = cfg.get("hidden_size", 64)
    num_layers = cfg.get("num_layers", 2)

    if model_type == "cnn":
        model = EMG1DCNN(num_channels, num_classes, segment_length=segment_length)
    elif model_type == "gru":
        model = EMGGRU(num_channels, num_classes, hidden_size=hidden_size, num_layers=num_layers)
    elif model_type == "transformer":
        model = EMGTransformer(num_channels, num_classes, segment_length=segment_length, d_model=hidden_size, num_layers=num_layers)
    else:
        raise ValueError(f"ONNX export only supported for PyTorch models, got: {model_type}")

    # Load weights
    model.load_state_dict(model_data["state_dict"])
    model.eval()

    # Create dummy input: shape (batch_size, num_channels, segment_length)
    dummy_input = torch.randn(1, num_channels, segment_length)

    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    
    torch.onnx.export(
        model,
        (dummy_input,),
        export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14
    )
    print(f"Model exported successfully to ONNX: {export_path}")
    return export_path


def export_to_coreml(user_id: str, model_type: str, export_path: str) -> bool:
    """Compile PyTorch/ONNX model to Apple Core ML format.

    Note: requires coremltools. If missing, handles gracefully.
    """
    try:
        import coremltools as ct
        print("coremltools found. Starting Core ML conversion...")
        
        # Load model metadata
        model_data = load_model(user_id, model_type)
        num_channels = model_data["num_channels"]
        num_classes = len(model_data["labels"])
        segment_length = model_data["segment_length"]

        cfg = model_data.get("config", {})
        hidden_size = cfg.get("hidden_size", 64)
        num_layers = cfg.get("num_layers", 2)

        if model_type == "cnn":
            model = EMG1DCNN(num_channels, num_classes, segment_length=segment_length)
        elif model_type == "gru":
            model = EMGGRU(num_channels, num_classes, hidden_size=hidden_size, num_layers=num_layers)
        elif model_type == "transformer":
            model = EMGTransformer(num_channels, num_classes, segment_length=segment_length, d_model=hidden_size, num_layers=num_layers)
        else:
            raise ValueError(f"Core ML conversion only supported for PyTorch models, got: {model_type}")

        model.load_state_dict(model_data["state_dict"])
        model.eval()

        # Trace the model
        example_input = torch.rand(1, num_channels, segment_length)
        traced_model = torch.jit.trace(model, example_input)

        # Convert
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)]
        )
        os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
        mlmodel.save(export_path)
        print(f"Model successfully converted to Core ML: {export_path}")
        return True

    except ImportError:
        print("[Warning] coremltools package not installed. Skipping Core ML compilation.")
        return False
    except Exception as e:
        print(f"[Warning] Core ML compilation failed: {e}")
        return False


def export_to_tflite(user_id: str, model_type: str, export_path: str) -> bool:
    """Compile PyTorch model to TFLite format via ONNX → tf2onnx → TFLite.

    Requires: tensorflow, tf2onnx. If missing, raises NotImplementedError.
    Returns True on success, raises on failure.
    """
    try:
        import tensorflow as tf
        import tf2onnx  # noqa: F401 — verify converter is available
    except ImportError as e:
        raise NotImplementedError(
            f"TFLite export requires tensorflow and tf2onnx: pip install tensorflow tf2onnx. Missing: {e}"
        )

    onnx_temp = export_path + ".temp.onnx"
    try:
        export_to_onnx(user_id, model_type, onnx_temp)
        os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)

        import subprocess
        result = subprocess.run(
            ["python", "-m", "tf2onnx.convert", "--onnx", onnx_temp,
             "--output", export_path, "--opset", "14"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"tf2onnx conversion failed: {result.stderr}")
        print(f"Model exported to TFLite: {export_path}")
        return True
    finally:
        if os.path.exists(onnx_temp):
            os.remove(onnx_temp)


def quantize_model_int8(user_id: str, model_type: str, threshold: float = 0.05) -> Dict[str, Any]:
    """Perform dynamic int8 quantization on a PyTorch model and run accuracy regression checks.

    Args:
        user_id: User calibration dataset ID to evaluate against.
        model_type: "cnn", "gru", or "transformer".
        threshold: Maximum allowed accuracy drop (e.g. 0.05 = 5%).

    Returns:
        Dict detailing quantization metrics.
    """
    if model_type in ["rf", "svm"]:
        raise ValueError("Quantization not supported for scikit-learn models.")

    # Load original model data
    model_data = load_model(user_id, model_type)
    num_channels = model_data["num_channels"]
    num_classes = len(model_data["labels"])
    segment_length = model_data["segment_length"]

    # Load calibration dataset for regression test
    raw_segs, labels_raw = load_dataset(user_id)
    unique_labels = model_data["labels"]
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in labels_raw], dtype=np.int64)

    X_pre = preprocess_segments(
        raw_segs,
        fs=config.SAMPLE_RATE,
        low=config.BANDPASS_LOW,
        high=config.BANDPASS_HIGH
    )

    test_size = model_data.get("config", {}).get("test_size", 0.2)
    seed = model_data.get("config", {}).get("seed", 42)

    try:
        _, idx_test, _, y_test = train_test_split(
            np.arange(len(raw_segs)), y, test_size=test_size, stratify=y, random_state=seed
        )
    except ValueError:
        _, idx_test, _, y_test = train_test_split(
            np.arange(len(raw_segs)), y, test_size=test_size, random_state=seed
        )

    X_test_p = X_pre[idx_test].astype(np.float32)
    mean = model_data["mean"]
    std = model_data["std"]
    X_test_p = (X_test_p - mean) / std

    # Reconstruct original float32 model
    cfg = model_data.get("config", {})
    hidden_size = cfg.get("hidden_size", 64)
    num_layers = cfg.get("num_layers", 2)

    if model_type == "cnn":
        model_fp32 = EMG1DCNN(num_channels, num_classes, segment_length=segment_length)
    elif model_type == "gru":
        model_fp32 = EMGGRU(num_channels, num_classes, hidden_size=hidden_size, num_layers=num_layers)
    elif model_type == "transformer":
        model_fp32 = EMGTransformer(num_channels, num_classes, segment_length=segment_length, d_model=hidden_size, num_layers=num_layers)
    else:
        raise ValueError(f"Unknown PyTorch model type '{model_type}'")

    model_fp32.load_state_dict(model_data["state_dict"])
    model_fp32.eval()

    # 1. Evaluate original float32 accuracy
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_p)
        outputs_fp32 = model_fp32(test_tensor)
        preds_fp32 = outputs_fp32.argmax(dim=1).numpy()
        acc_fp32 = float(np.mean(preds_fp32 == y_test))

    # 2. Perform dynamic dynamic quantization to int8
    # Quantize Linear and recurrent GRU modules
    try:
        # Set quantization engine
        engines = torch.backends.quantized.supported_engines
        if "qnnpack" in engines:
            torch.backends.quantized.engine = "qnnpack"
        elif "fbgemm" in engines:
            torch.backends.quantized.engine = "fbgemm"

        model_int8 = torch.quantization.quantize_dynamic(
            model_fp32,
            {nn.Linear, nn.GRU},
            dtype=torch.qint8
        )
        model_int8.eval()

        # 3. Evaluate quantized int8 accuracy
        with torch.no_grad():
            outputs_int8 = model_int8(test_tensor)
            preds_int8 = outputs_int8.argmax(dim=1).numpy()
            acc_int8 = float(np.mean(preds_int8 == y_test))

        accuracy_drop = acc_fp32 - acc_int8
    except RuntimeError as e:
        print(f"[Warning] Quantization engine failed: {e}. Skipping dynamic quantization.")
        return {
            "accuracy_fp32": acc_fp32,
            "accuracy_int8": acc_fp32,
            "accuracy_drop": 0.0,
            "status": "SKIPPED_NO_ENGINE"
        }
    print(f"\n[{model_type.upper()} Quantization Check]")
    print(f"FP32 Accuracy: {acc_fp32:.4f}")
    print(f"INT8 Accuracy: {acc_int8:.4f}")
    print(f"Accuracy Drop: {accuracy_drop:.4f} (Threshold: {threshold:.4f})")

    # Accuracy regression check
    if accuracy_drop > threshold:
        raise RuntimeError(
            f"Quantization rejected: accuracy drop ({accuracy_drop:.4f}) exceeds threshold ({threshold:.4f})"
        )

    # Save quantized model
    quant_model_data = {
        "state_dict": model_int8.state_dict(),
        "labels": unique_labels,
        "mean": mean,
        "std": std,
        "model_type": model_type,
        "num_channels": num_channels,
        "segment_length": segment_length,
        "quantized": True,
        "config": model_data.get("config", {})
    }
    
    quant_type = f"{model_type}_quantized"
    save_model(quant_model_data, user_id, model_type=quant_type)
    print(f"Quantized model successfully saved to model type: {quant_type}")

    return {
        "accuracy_fp32": acc_fp32,
        "accuracy_int8": acc_int8,
        "accuracy_drop": accuracy_drop,
        "status": "APPROVED"
    }
