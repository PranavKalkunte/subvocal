import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class StreamTracker:
    """Monitors whether a biometric sEMG stream is active or stopped.

    Applies a hysteresis threshold to prevent bouncing between states. Equivalent
    to LiveKit's StreamTracker.
    """

    STATUS_STOPPED = "stopped"
    STATUS_ACTIVE = "active"

    def __init__(
        self,
        samples_required: int = 3,  # Consecutive active cycles to mark ACTIVE
        cycles_required: int = 5,   # Consecutive inactive cycles to mark STOPPED
    ):
        self.samples_required = samples_required
        self.cycles_required = cycles_required

        self._lock = threading.Lock()
        self._status = self.STATUS_STOPPED
        self._last_notified_status = self.STATUS_STOPPED
        self._on_status_changed: Callable[[str], None] | None = None

        self._active_consecutive = 0
        self._inactive_consecutive = 0

        self._paused = False
        self._generation = 0
        self._is_stopped = False

    @property
    def status(self) -> str:
        """Returns the current status (active/stopped)."""
        with self._lock:
            return self._status

    @property
    def generation(self) -> int:
        """Returns the current tracker worker generation."""
        with self._lock:
            return self._generation

    def on_status_changed(self, callback: Callable[[str], None]) -> None:
        """Registers a callback to execute on status transitions."""
        with self._lock:
            self._on_status_changed = callback

    def observe(self, has_activity: bool) -> None:
        """Observes an activity tick and updates status state using hysteresis."""
        notify = False
        status_to_notify = None

        with self._lock:
            if self._is_stopped or self._paused:
                return

            if has_activity:
                self._active_consecutive += 1
                self._inactive_consecutive = 0
                
                # Check transition to ACTIVE
                if self._status == self.STATUS_STOPPED and self._active_consecutive >= self.samples_required:
                    self._status = self.STATUS_ACTIVE
            else:
                self._inactive_consecutive += 1
                self._active_consecutive = 0

                # Check transition to STOPPED
                if self._status == self.STATUS_ACTIVE and self._inactive_consecutive >= self.cycles_required:
                    self._status = self.STATUS_STOPPED

            # Notify only on change
            if self._status != self._last_notified_status:
                notify = True
                status_to_notify = self._status
                self._last_notified_status = self._status

        if notify and status_to_notify is not None and self._on_status_changed is not None:
            try:
                self._on_status_changed(status_to_notify)
            except Exception:
                logger.exception("StreamTracker status changed callback error")

    def set_paused(self, paused: bool) -> None:
        """Pauses or resumes tracking.

        Increments generation to invalidate older checks.
        """
        notify = False
        status_to_notify = None

        with self._lock:
            self._paused = paused
            self._generation += 1
            if paused:
                self._status = self.STATUS_STOPPED
                self._active_consecutive = 0
                self._inactive_consecutive = 0
                if self._status != self._last_notified_status:
                    notify = True
                    status_to_notify = self._status
                    self._last_notified_status = self._status

        if notify and status_to_notify is not None and self._on_status_changed is not None:
            try:
                self._on_status_changed(status_to_notify)
            except Exception:
                logger.exception("StreamTracker status changed callback error")

    def reset(self) -> None:
        """Resets the tracker to initial stopped state."""
        notify = False
        status_to_notify = None

        with self._lock:
            self._generation += 1
            self._status = self.STATUS_STOPPED
            self._active_consecutive = 0
            self._inactive_consecutive = 0
            if self._status != self._last_notified_status:
                notify = True
                status_to_notify = self._status
                self._last_notified_status = self._status

        if notify and status_to_notify is not None and self._on_status_changed is not None:
            try:
                self._on_status_changed(status_to_notify)
            except Exception:
                logger.exception("StreamTracker status changed callback error")

    def stop(self) -> None:
        """Permanently stops the tracker."""
        with self._lock:
            self._is_stopped = True
            self._generation += 1
