"""Pre-training script for deep neural sEMG classifiers on public baseline datasets."""

import logging
import os
import re
from pathlib import Path

import numpy as np

from subvocal.emg_core import config
from subvocal.emg_core.ml.train import TrainingConfig, train_model


def _sanitize_user_id(user_id: str) -> str:
    """Sanitize user_id to prevent path traversal (C2)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", user_id)

logger = logging.getLogger(__name__)


def generate_synthetic_dataset(user_id: str = "pretrained", num_classes: int = 5, segments_per_class: int = 40):
    """Generate a reproducible multi-channel sEMG dataset to simulate a public dataset.

    Each class is synthesized with unique harmonic and frequency patterns.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    classes = [f"CMD_{i}" for i in range(num_classes)]
    segments = []
    labels = []

    np.random.seed(1337)
    for i, cmd in enumerate(classes):
        for _ in range(segments_per_class):
            # Baseline physiological noise
            seg = np.random.normal(0.0, 0.8, (150, 4))
            
            # Insert distinct frequency profile for each class
            t = np.linspace(0, 2 * np.pi, 150)
            freq = (i + 1) * 4.5
            pattern = np.sin(t * freq)[:, np.newaxis]
            
            # Amplitude envelope modulation
            envelope = np.exp(-((t - np.pi) ** 2) / 2.0)[:, np.newaxis]
            seg += pattern * envelope * 3.5  # Clear signature pattern
            
            segments.append(seg)
            labels.append(cmd)

    # Sanitize user_id and validate resolved path stays within DATA_DIR (C2)
    safe_user_id = _sanitize_user_id(user_id)
    data_path = os.path.join(config.DATA_DIR, f"{safe_user_id}_calib.npz")
    resolved = Path(data_path).resolve()
    base = Path(config.DATA_DIR).resolve()
    try:
        if not resolved.is_relative_to(base):
            raise ValueError(f"Invalid dataset path traversal detected: {data_path}")
    except AttributeError:
        try:
            resolved.relative_to(base)
        except ValueError:
            raise ValueError(f"Invalid dataset path traversal detected: {data_path}")
    data_path = str(resolved)
    np.savez(data_path, segments=segments, labels=labels)
    logger.info("Synthetic pre-training dataset saved to %s (%d segments, %d classes)", data_path, len(segments), num_classes)


def run_pretraining():
    """Run pre-training pipeline for all deep architectures."""
    user_id = "pretrained"
    generate_synthetic_dataset(user_id=user_id)

    # Custom pre-training configuration (fewer epochs for baseline pretraining)
    base_cfg = {
        "seed": 1337,
        "epochs": 20,
        "batch_size": 16,
        "lr": 1e-3,
        "test_size": 0.2
    }

    for model_type in ["cnn", "gru", "transformer"]:
        logger.info("Pre-training model: %s", model_type.upper())
        cfg = TrainingConfig(model_type=model_type, **base_cfg)
        metrics = train_model(user_id, model_type=model_type, config_obj=cfg)
        logger.info("Pre-training %s accuracy: %.4f", model_type, metrics["accuracy"])


if __name__ == "__main__":
    run_pretraining()
