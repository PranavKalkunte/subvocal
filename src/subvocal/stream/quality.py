import logging
import threading
from collections.abc import Callable

import numpy as np

from subvocal.core.models import Frame

logger = logging.getLogger(__name__)


class SignalQualityScorer:
    """Scores sEMG signal quality using physical metrics (saturation, drift, dropouts, SNR).

    Outputs a MOS-like score mapped to EXCELLENT, GOOD, POOR, and LOST states.
    Equivalent to LiveKit's connectionquality Scorer.
    """

    QUALITY_LOST = "lost"
    QUALITY_POOR = "poor"
    QUALITY_GOOD = "good"
    QUALITY_EXCELLENT = "excellent"

    # Hysteresis transition scores
    QUALITY_SCORES = {
        QUALITY_EXCELLENT: 4.0,
        QUALITY_GOOD: 3.0,
        QUALITY_POOR: 2.0,
        QUALITY_LOST: 1.0,
    }

    def __init__(self, clip_max: float = 5.0, increase_factor: float = 0.4, decrease_factor: float = 0.8):
        self.clip_max = clip_max
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor

        self._lock = threading.Lock()
        self._score = 4.5  # Max MOS
        self._quality = self.QUALITY_EXCELLENT
        self._last_notified_quality = self.QUALITY_EXCELLENT
        self._on_status_changed: Callable[[str], None] | None = None

    @property
    def score(self) -> float:
        """Returns the current MOS-like quality score (1.0 - 4.5)."""
        with self._lock:
            return self._score

    @property
    def quality(self) -> str:
        """Returns the current mapped quality state."""
        with self._lock:
            return self._quality

    def on_status_changed(self, callback: Callable[[str], None]) -> None:
        """Registers a callback to be run on quality state transitions."""
        with self._lock:
            self._on_status_changed = callback

    def update(self, frame: Frame) -> None:
        """Evaluates a Frame, computes metrics, updates the quality score, and checks transitions."""
        arr = frame.to_numpy()
        if arr.size == 0:
            self._apply_score(1.0)
            return

        # 1. Saturation / Clipping: Percent of samples close to clip_max
        sat_mask = np.abs(arr) >= (self.clip_max * 0.98)
        saturation_rate = float(np.mean(sat_mask))

        # 2. Baseline Drift: Variance of the sample mean over time/channels (ideal is 0)
        mean_per_channel = np.mean(arr, axis=0)
        drift_val = float(np.var(mean_per_channel))

        # 3. Dropout check: standard deviation is close to 0 (flatline) on any channel
        std_per_channel = np.std(arr, axis=0)
        dropout_channels = np.sum(std_per_channel < 0.005)
        dropout_rate = float(dropout_channels / arr.shape[1])

        # 4. SNR Proxy: ratio of mean absolute value to std dev (signal strength to noise)
        mav = np.mean(np.abs(arr))
        noise_floor = np.mean(std_per_channel)
        snr_ratio = float(mav / noise_floor) if noise_floor > 0 else 5.0

        # Compute current calculated score (Start at 4.5)
        calculated = 4.5

        # Penalize dropouts severely (up to -3.0 MOS)
        calculated -= dropout_rate * 3.0

        # Penalize saturation (up to -2.0 MOS)
        calculated -= saturation_rate * 2.0

        # Penalize baseline drift (up to -1.5 MOS)
        calculated -= min(drift_val * 1.5, 1.5)

        # Penalize low SNR (under 1.0 ratio, up to -1.0 MOS)
        if snr_ratio < 1.0:
            calculated -= (1.0 - snr_ratio) * 1.0

        # Bound the score
        calculated = max(1.0, min(4.5, calculated))

        # Apply conservative rise / aggressive drop factor
        with self._lock:
            factor = self.increase_factor if calculated >= self._score else self.decrease_factor
            self._score = factor * calculated + (1.0 - factor) * self._score

            # Map score to quality state
            self._quality = self._score_to_quality(self._score)

            notify = False
            quality_to_notify = None
            if self._quality != self._last_notified_quality:
                notify = True
                quality_to_notify = self._quality
                self._last_notified_quality = self._quality

        if notify and quality_to_notify is not None and self._on_status_changed is not None:
            try:
                self._on_status_changed(quality_to_notify)
            except Exception:
                logger.exception("SignalQualityScorer status changed callback error")

    def _apply_score(self, target_score: float) -> None:
        notify = False
        quality_to_notify = None

        with self._lock:
            factor = self.increase_factor if target_score >= self._score else self.decrease_factor
            self._score = factor * target_score + (1.0 - factor) * self._score
            self._quality = self._score_to_quality(self._score)
            if self._quality != self._last_notified_quality:
                notify = True
                quality_to_notify = self._quality
                self._last_notified_quality = self._quality

        if notify and quality_to_notify is not None and self._on_status_changed is not None:
            try:
                self._on_status_changed(quality_to_notify)
            except Exception:
                logger.exception("SignalQualityScorer status changed callback error")

    def _score_to_quality(self, score: float) -> str:
        if score >= 4.0:
            return self.QUALITY_EXCELLENT
        elif score >= 3.0:
            return self.QUALITY_GOOD
        elif score >= 2.0:
            return self.QUALITY_POOR
        else:
            return self.QUALITY_LOST
