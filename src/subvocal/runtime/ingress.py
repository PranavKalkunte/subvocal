import logging

from subvocal.core.interfaces import HardwareSource

logger = logging.getLogger("subvocal.runtime.ingress")


class IngressManager:
    """Manages active hardware source registration, monitoring, and failovers."""

    def __init__(self):
        self._sources: dict[str, HardwareSource] = {}
        self._fallback_name: str | None = None
        self._active_name: str | None = None

    def register_source(self, name: str, source: HardwareSource, is_fallback: bool = False) -> None:
        """Registers a biometric input source."""
        self._sources[name] = source
        if is_fallback:
            self._fallback_name = name
        if self._active_name is None and not is_fallback:
            self._active_name = name

    def start(self) -> None:
        """Starts the active ingress source stream."""
        if not self._active_name:
            raise ValueError("No ingress sources registered.")
        logger.info("Starting active ingress source '%s'", self._active_name)
        self._sources[self._active_name].start()

    def stop(self) -> None:
        """Stops all registered ingress sources."""
        if self._active_name and self._active_name in self._sources:
            try:
                self._sources[self._active_name].stop()
            except Exception:
                pass
        if self._fallback_name and self._fallback_name in self._sources:
            try:
                self._sources[self._fallback_name].stop()
            except Exception:
                pass

    @property
    def active_name(self) -> str | None:
        return self._active_name

    def get_active_source(self) -> HardwareSource | None:
        if self._active_name:
            return self._sources.get(self._active_name)
        return None

    def trigger_failover(self) -> None:
        """Fails over dynamically to the registered fallback source."""
        if not self._fallback_name:
            logger.warning("Failover triggered but no fallback source registered.")
            return
        if self._active_name == self._fallback_name:
            return

        logger.warning(
            "Failing over from active source '%s' to fallback source '%s'",
            self._active_name,
            self._fallback_name,
        )

        # Stop old active source
        if self._active_name and self._active_name in self._sources:
            try:
                self._sources[self._active_name].stop()
            except Exception:
                pass

        # Start and switch to fallback source
        self._active_name = self._fallback_name
        self._sources[self._active_name].start()
