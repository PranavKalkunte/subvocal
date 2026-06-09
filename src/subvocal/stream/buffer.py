import collections
import threading
import time
from typing import Any

from subvocal.core.models import Frame


class FrameRing:
    """Thread-safe circular queue (ring buffer) storing biometric Frame segments."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._deque = collections.deque(maxlen=max_size)
        self._lock = threading.Lock()

    def push(self, frame: Frame) -> None:
        """Pushes a new Frame onto the ring buffer, evicting the oldest if full."""
        with self._lock:
            self._deque.append(frame)

    def pop(self) -> Frame | None:
        """Pops and returns the oldest Frame from the ring buffer, or None if empty."""
        with self._lock:
            if not self._deque:
                return None
            return self._deque.popleft()

    def clear(self) -> None:
        """Clears the ring buffer."""
        with self._lock:
            self._deque.clear()

    def get_all(self) -> list[Frame]:
        """Returns a copy of all current frames in chronological order."""
        with self._lock:
            return list(self._deque)

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)


class StreamStats:
    """Windowed tracking for frame arrivals, sample count, gaps, and inter-frame arrival jitter."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_frames = 0
        self.total_samples = 0
        self.total_gaps = 0
        self.jitter = 0.0

        self._last_arrival_time = None
        self._last_frame_end = None

    def observe(self, frame: Frame) -> None:
        """Observes a new frame and updates statistics."""
        with self._lock:
            now = time.time()
            self.total_frames += 1
            num_samples = len(frame.samples)
            self.total_samples += num_samples

            # 1. Detect dropout gap based on timestamps
            if self._last_frame_end is not None:
                # If there's a gap between the end of the last frame and the start of the current
                # frame larger than 1.5 times the nominal sample spacing, count it as a gap.
                expected_delta = 1.5 / frame.fs if frame.fs > 0 else 0.01
                if frame.start_time - self._last_frame_end > expected_delta:
                    self.total_gaps += 1

            self._last_frame_end = frame.end_time

            # 2. Update RFC 3550-inspired inter-frame jitter
            if self._last_arrival_time is not None:
                # Actual arrival delay minus nominal duration
                actual_delay = now - self._last_arrival_time
                nominal_delay = (frame.end_time - frame.start_time)
                delay_diff = abs(actual_delay - nominal_delay)
                # Filter/EMA updates
                self.jitter = self.jitter + (delay_diff - self.jitter) / 16.0

            self._last_arrival_time = now

    def get_stats(self) -> dict[str, Any]:
        """Returns a snapshot of the current stats."""
        with self._lock:
            return {
                "total_frames": self.total_frames,
                "total_samples": self.total_samples,
                "total_gaps": self.total_gaps,
                "jitter_seconds": self.jitter,
            }

    def reset(self) -> None:
        """Resets all stats to zero."""
        with self._lock:
            self.total_frames = 0
            self.total_samples = 0
            self.total_gaps = 0
            self.jitter = 0.0
            self._last_arrival_time = None
            self._last_frame_end = None
