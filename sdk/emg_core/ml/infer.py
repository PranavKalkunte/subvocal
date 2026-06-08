"""Real-time inference engine for sEMG classifications.

Loads scikit-learn or PyTorch models and classifies incoming raw segments
with confidence gating and command cooldowns.
"""

import time
import numpy as np
import torch
from typing import Optional, List, Dict, Any, Tuple

from emg_core import config
from emg_core.dsp.features import extract_features
from emg_core.dsp.filters import preprocess_multichannel
from emg_core.ml.model_io import load_model


class EMGPrediction:
    """Lightweight prediction object representing a successfully classified command."""
    def __init__(self, t: float, cmd: str, p: float, cooldown_ms: int):
        self.t = t
        self.cmd = cmd
        self.p = p
        self.cooldown_ms = cooldown_ms

    def __repr__(self) -> str:
        return f"EMGPrediction(cmd={self.cmd}, p={self.p:.3f}, t={self.t:.3f})"


class InferenceEngine:
    """Inference engine with debounce and confidence thresholding."""

    def __init__(
        self,
        user_id: str,
        model_type: str = "rf",
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        cooldown_ms: int = config.COOLDOWN_MS,
    ):
        self._model_type = model_type
        self._threshold = confidence_threshold
        self._cooldown_ms = cooldown_ms
        self._last_fired: Dict[str, float] = {}

        # Load saved model dictionary
        model_data = load_model(user_id, model_type)
        self._labels: List[str] = model_data["labels"]

        if model_type == "rf":
            self._model = model_data["model"]
        else:
            # Reconstruct PyTorch model
            from emg_core.ml.train import EMG1DCNN, EMGGRU

            num_channels = model_data["num_channels"]
            num_classes = len(self._labels)
            segment_length = model_data["segment_length"]

            if model_type == "cnn":
                self._model = EMG1DCNN(num_channels, num_classes, segment_length=segment_length)
            elif model_type == "gru":
                self._model = EMGGRU(num_channels, num_classes)
            else:
                raise ValueError(f"Unknown PyTorch model type '{model_type}'")

            self._model.load_state_dict(model_data["state_dict"])
            self._model.eval()

            self._mean = model_data["mean"]
            self._std = model_data["std"]
            self._device = torch.device("cpu")
            self._model = self._model.to(self._device)

    @property
    def labels(self) -> List[str]:
        return self._labels

    def predict(self, segment: np.ndarray) -> Optional[EMGPrediction]:
        """Classify a segment and apply gating logic.

        Args:
            segment: 2D array of shape (segment_length, num_channels)

        Returns:
            EMGPrediction if successful, else None.
        """
        now = time.time()
        cmd, prob, proba = self.predict_raw(segment)

        # Confidence gate
        if prob < self._threshold:
            return None

        # Cooldown gate
        last = self._last_fired.get(cmd, 0.0)
        if (now - last) * 1000 < self._cooldown_ms:
            return None

        self._last_fired[cmd] = now
        return EMGPrediction(
            t=now,
            cmd=cmd,
            p=prob,
            cooldown_ms=self._cooldown_ms
        )

    def predict_raw(self, segment: np.ndarray) -> Tuple[str, float, List[float]]:
        """Classify a segment without debounce gates.

        Returns:
            (best_command, probability, all_probabilities)
        """
        # 1. Preprocess
        seg_pre = preprocess_multichannel(
            segment,
            fs=config.SAMPLE_RATE,
            low=config.BANDPASS_LOW,
            high=config.BANDPASS_HIGH
        )

        if self._model_type == "rf":
            # Extract features for RF
            features = extract_features(seg_pre, sample_rate=config.SAMPLE_RATE).reshape(1, -1)
            proba = self._model.predict_proba(features)[0]
        else:
            # PyTorch inference
            # Transpose to (num_channels, segment_length) and add batch dimension
            seg_t = seg_pre.T[np.newaxis, ...].astype(np.float32)

            # Normalization
            seg_t = (seg_t - self._mean) / self._std

            with torch.no_grad():
                tensor_input = torch.tensor(seg_t).to(self._device)
                logits = self._model(tensor_input)
                proba_tensor = torch.softmax(logits, dim=1)[0]
                proba = proba_tensor.cpu().numpy()

        best_idx = int(np.argmax(proba))
        return self._labels[best_idx], float(proba[best_idx]), proba.tolist()
