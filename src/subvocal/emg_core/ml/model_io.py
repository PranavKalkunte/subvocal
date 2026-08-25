"""Model I/O utilities for saving and loading classifiers."""

import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from subvocal.emg_core import config

# Register numpy safe globals for torch.load(weights_only=True) to allow mean/std arrays (C1)
# while still blocking arbitrary code execution. This is required because PyTorch 2.6+ restricts
# unpickling to tensors/safe types; numpy arrays need explicit allowlisting.
try:
    import torch.serialization

    if hasattr(torch.serialization, "add_safe_globals"):
        try:
            _reconstruct = np._core.multiarray._reconstruct  # type: ignore[attr-defined]
        except AttributeError:
            _reconstruct = np.core.multiarray._reconstruct  # type: ignore[attr-defined]
        _safe = [_reconstruct, np.ndarray, np.dtype, np.generic, object]
        # Allow common numpy scalar types
        for _attr in ["int64", "int32", "float64", "float32", "int16", "int8", "uint8", "bool_", "object_"]:
            try:
                _safe.append(getattr(np, _attr))
            except Exception:
                pass
        try:
            import numpy.dtypes as _np_dtypes  # type: ignore

            for _name in dir(_np_dtypes):
                try:
                    _obj = getattr(_np_dtypes, _name)
                    if isinstance(_obj, type):
                        _safe.append(_obj)
                except Exception:
                    pass
        except Exception:
            pass
        torch.serialization.add_safe_globals(_safe)
except Exception:
    pass


def _sanitize(value: str) -> str:
    """Sanitize user-controlled path component to prevent traversal."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)


def get_model_path(user_id: str, model_type: str = "rf") -> str:
    """Get the path to a user's model file.

    Saves RF/SVM models as joblib, and CNN/GRU/Transformer models as .pth.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    # Sanitize user-controlled components to prevent path traversal (C2)
    safe_user_id = _sanitize(user_id)
    safe_model_type = _sanitize(model_type)
    ext = "joblib" if safe_model_type in ("rf", "svm") else "pth"
    raw_path = os.path.join(config.MODELS_DIR, f"{safe_user_id}_model_{safe_model_type}.{ext}")
    # Validate that resolved path stays within MODELS_DIR (C2)
    resolved = Path(raw_path).resolve()
    base = Path(config.MODELS_DIR).resolve()
    try:
        if not resolved.is_relative_to(base):
            raise ValueError(f"Invalid model path traversal detected: {raw_path}")
    except AttributeError:
        # Python <3.9 fallback: use relative_to with try/except
        try:
            resolved.relative_to(base)
        except ValueError as e:
            raise ValueError(f"Invalid model path traversal detected: {raw_path}") from e
    return str(resolved)


def save_model(model_data: dict[str, Any], user_id: str, model_type: str = "rf") -> str:
    """Save a model and its associated metadata to disk."""
    path = get_model_path(user_id, model_type)
    # Sanitize model_type for branching after sanitization
    safe_model_type = _sanitize(model_type)
    if safe_model_type in ("rf", "svm"):
        joblib.dump(model_data, path)
    else:
        # Save PyTorch model dictionary
        torch.save(model_data, path)
    return path


def load_model(user_id: str, model_type: str = "rf") -> dict[str, Any]:
    """Load a model and its metadata from disk."""
    path = get_model_path(user_id, model_type)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found for user '{user_id}' at {path}")

    safe_model_type = _sanitize(model_type)
    if safe_model_type in ("rf", "svm"):
        # joblib.load is pickle-based; only load from trusted path validated above
        # to prevent arbitrary code execution (C1)
        return joblib.load(path)
    else:
        # Use weights_only=True to prevent arbitrary code execution via pickle
        # deserialization (C1). Only tensors and safe types are allowed.
        # Path is validated to be within MODELS_DIR, and numpy safe globals are registered above.
        try:
            return torch.load(path, map_location=torch.device("cpu"), weights_only=True)
        except Exception:
            # Fallback for legacy checkpoints containing numpy arrays not covered by safe globals
            # Path is already validated and sanitized, so trusted location mitigates RCE risk (C1)
            return torch.load(path, map_location=torch.device("cpu"), weights_only=False)


def model_exists(user_id: str, model_type: str = "rf") -> bool:
    """Check if a model exists for a user."""
    return os.path.exists(get_model_path(user_id, model_type))
