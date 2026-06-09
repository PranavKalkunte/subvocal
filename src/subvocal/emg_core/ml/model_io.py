"""Model I/O utilities for saving and loading classifiers."""

import os
import joblib
import torch
from typing import Any, Dict

from subvocal.emg_core import config


def get_model_path(user_id: str, model_type: str = "rf") -> str:
    """Get the path to a user's model file.

    Saves RF/SVM models as joblib, and CNN/GRU/Transformer models as .pth.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    ext = "joblib" if model_type in ("rf", "svm") else "pth"
    return os.path.join(config.MODELS_DIR, f"{user_id}_model_{model_type}.{ext}")


def save_model(model_data: Dict[str, Any], user_id: str, model_type: str = "rf") -> str:
    """Save a model and its associated metadata to disk."""
    path = get_model_path(user_id, model_type)
    if model_type in ("rf", "svm"):
        joblib.dump(model_data, path)
    else:
        # Save PyTorch model dictionary
        torch.save(model_data, path)
    return path


def load_model(user_id: str, model_type: str = "rf") -> Dict[str, Any]:
    """Load a model and its metadata from disk."""
    path = get_model_path(user_id, model_type)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found for user '{user_id}' at {path}")

    if model_type in ("rf", "svm"):
        return joblib.load(path)
    else:
        # Load PyTorch model dictionary
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)


def model_exists(user_id: str, model_type: str = "rf") -> bool:
    """Check if a model exists for a user."""
    return os.path.exists(get_model_path(user_id, model_type))
