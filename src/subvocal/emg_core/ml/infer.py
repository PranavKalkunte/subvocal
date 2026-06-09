import time
import numpy as np
import torch
from typing import Optional, List, Dict, Any, Tuple, Union

from subvocal.emg_core import config
from subvocal.emg_core.dsp.features import extract_features
from subvocal.emg_core.dsp.filters import preprocess_multichannel
from subvocal.emg_core.ml.model_io import load_model
from subvocal.core.interfaces import Classifier
from subvocal.core.models import Frame, CommandToken


class InferenceEngine(Classifier):
    """Inference engine with debounce and confidence thresholding.

    Implements the core.interfaces.Classifier interface.
    """

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

        if model_type in ["rf", "svm"]:
            self._model = model_data["model"]
        else:
            # Reconstruct PyTorch model
            from subvocal.emg_core.ml.train import EMG1DCNN, EMGGRU, EMGTransformer

            num_channels = model_data["num_channels"]
            num_classes = len(self._labels)
            segment_length = model_data["segment_length"]

            cfg = model_data.get("config", {})
            hidden_size = cfg.get("hidden_size", 64)
            num_layers = cfg.get("num_layers", 2)

            base_model_type = model_type.replace("_quantized", "")
            if base_model_type == "cnn":
                self._model = EMG1DCNN(num_channels, num_classes, segment_length=segment_length)
            elif base_model_type == "gru":
                self._model = EMGGRU(num_channels, num_classes, hidden_size=hidden_size, num_layers=num_layers)
            elif base_model_type == "transformer":
                self._model = EMGTransformer(num_channels, num_classes, segment_length=segment_length, d_model=hidden_size, num_layers=num_layers)
            else:
                raise ValueError(f"Unknown PyTorch model type '{model_type}'")


            # Handle dynamically quantized weights loading
            if model_data.get("quantized", False):
                self._model = torch.quantization.quantize_dynamic(
                    self._model,
                    {torch.nn.Linear, torch.nn.GRU},
                    dtype=torch.qint8
                )

            self._model.load_state_dict(model_data["state_dict"])
            self._model.eval()

            self._mean = model_data["mean"]
            self._std = model_data["std"]
            self._device = torch.device("cpu")
            self._model = self._model.to(self._device)

    @property
    def labels(self) -> List[str]:
        return self._labels

    def predict(self, frame: Union[Frame, np.ndarray]) -> Optional[CommandToken]:
        """Classify a segment and apply gating logic.

        Args:
            frame: A Frame object or 2D array of shape (segment_length, num_channels)

        Returns:
            CommandToken if successful, else None.
        """
        now = time.time()
        cmd, prob, proba = self.predict_raw(frame)

        # Confidence gate
        if prob < self._threshold:
            return None

        # Cooldown gate
        last = self._last_fired.get(cmd, 0.0)
        if (now - last) * 1000 < self._cooldown_ms:
            return None

        self._last_fired[cmd] = now
        return CommandToken(
            text=cmd,
            confidence=prob,
            timestamp=now,
            metadata={"cooldown_ms": self._cooldown_ms, "probabilities": proba}
        )

    def predict_raw(self, frame: Union[Frame, np.ndarray]) -> Tuple[str, float, List[float]]:
        """Classify a segment/frame without debounce gates.

        Returns:
            (best_command, probability, all_probabilities)
        """
        # 1. Convert Frame to NumPy array if needed
        if isinstance(frame, Frame):
            segment = frame.to_numpy()
        else:
            segment = frame

        # 2. Preprocess
        seg_pre = preprocess_multichannel(
            segment,
            fs=config.SAMPLE_RATE,
            low=config.BANDPASS_LOW,
            high=config.BANDPASS_HIGH
        )

        if self._model_type in ["rf", "svm"]:
            # Extract features for scikit-learn models
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

