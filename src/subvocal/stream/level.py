import numpy as np

from subvocal.core.models import Frame


class SignalLevel:
    """Tracks and smooths the biological/electrical signal level of sEMG channels.

    Equivalent to LiveKit's AudioLevel.
    """

    def __init__(
        self,
        active_level: float = 0.1,      # minimum MAV amplitude to be considered active
        min_percentile: float = 40.0,    # percent of window that must exceed active_level
        update_interval_ms: int = 400,   # length of analysis window in milliseconds
        smooth_intervals: int = 2,       # smoothing intervals for EMA
    ):
        self.active_level = active_level
        self.min_percentile = min_percentile
        self.update_interval_ms = update_interval_ms
        self.smooth_intervals = smooth_intervals

        self.min_active_duration_ms = (min_percentile * update_interval_ms) / 100.0
        if smooth_intervals > 0:
            self.smooth_factor = 2.0 / (smooth_intervals + 1)
        else:
            self.smooth_factor = 1.0

        self.smoothed_level = 0.0
        self.max_observed_level = 0.0
        self.active_duration_ms = 0.0
        self.observed_duration_ms = 0.0
        self.last_observed_time = 0.0

    def observe(self, frame: Frame) -> None:
        """Observes a new Frame, computes its MAV, and updates sliding window metrics."""
        now = frame.end_time
        self.last_observed_time = now

        # Compute MAV (Mean Absolute Value) of sEMG signal across all channels
        arr = frame.to_numpy()
        if arr.size == 0:
            return

        mav = float(np.mean(np.abs(arr)))
        duration_ms = (frame.end_time - frame.start_time) * 1000.0
        self.observed_duration_ms += duration_ms

        if mav >= self.active_level:
            self.active_duration_ms += duration_ms
            if mav > self.max_observed_level:
                self.max_observed_level = mav

        # If analysis window is complete, update the smoothed level and reset metrics
        if self.observed_duration_ms >= self.update_interval_ms:
            smoothed = 0.0
            if self.active_duration_ms >= self.min_active_duration_ms:
                # Adjust by how active the window was
                activity_ratio = self.active_duration_ms / self.observed_duration_ms
                adjusted_level = self.max_observed_level * activity_ratio
                # Exponential moving average (EMA)
                smoothed = self.smoothed_level + (adjusted_level - self.smoothed_level) * self.smooth_factor

            self.smoothed_level = smoothed
            self.max_observed_level = 0.0
            self.active_duration_ms = 0.0
            self.observed_duration_ms = 0.0

    def get_level(self, now: float) -> tuple[float, bool]:
        """Returns the current smoothed signal level and active status.

        Resets level to 0 if signal is stale (no updates received for 2x update_interval).
        """
        # Stale check
        if (now - self.last_observed_time) * 1000.0 > 2 * self.update_interval_ms:
            self.smoothed_level = 0.0
            self.max_observed_level = 0.0
            self.active_duration_ms = 0.0
            self.observed_duration_ms = 0.0

        is_active = self.smoothed_level >= self.active_level
        return self.smoothed_level, is_active
