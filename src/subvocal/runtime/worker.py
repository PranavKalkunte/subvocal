import logging
import threading
from typing import Any

from subvocal.config import SubvocalConfig
from subvocal.runtime.session import Session

logger = logging.getLogger(__name__)


class SessionWorker:
    """Manages active sessions and load-tracking across a pool of silent speech sessions.

    Equivalent to LiveKit's SessionWorker/AgentWorker.
    """

    def __init__(self, config: SubvocalConfig, max_sessions: int = 10, store: Any | None = None):
        self.config = config
        self.max_sessions = max_sessions

        from subvocal.runtime.store import InMemorySessionStore
        self.store = store or InMemorySessionStore()

        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    @property
    def load(self) -> float:
        """Returns the current load fraction of the worker (0.0 to 1.0)."""
        with self._lock:
            if self.max_sessions <= 0:
                return 0.0
            return len(self._sessions) / self.max_sessions

    @property
    def status(self) -> str:
        """Returns status string based on load."""
        curr_load = self.load
        if curr_load >= 1.0:
            return "full"
        elif curr_load > 0.0:
            return "active"
        return "idle"

    def create_session(self, session_id: str, *args, **kwargs) -> Session:
        """Creates, registers, and returns a new Session.

        Raises ValueError if session pool capacity is exhausted.
        """
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise ValueError(f"SessionWorker full. Capacity is {self.max_sessions} active sessions.")
            if session_id in self._sessions:
                raise ValueError(f"Session with ID '{session_id}' already exists.")

            # Create session
            session = Session(session_id, self.config, *args, **kwargs)
            self._sessions[session_id] = session

            # Persist configuration and state snapshot to store
            import yaml
            config_yaml = yaml.dump(self.config.model_dump())
            self.store.save_session(session_id, session.state, config_yaml, {"created_by": "worker"})

            return session

    def get_session(self, session_id: str) -> Session | None:
        """Retrieves a registered Session by ID, or None if not found."""
        with self._lock:
            return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> Session | None:
        """Stops and unregisters a Session by ID."""
        session = None
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions.pop(session_id)
                self.store.delete_session(session_id)

        if session:
            session.stop()
        return session


    def start_session(self, session_id: str) -> None:
        """Starts a registered session by ID."""
        session = self.get_session(session_id)
        if session:
            session.start()

    def close(self) -> None:
        """Gracefully closes all sessions and stops the worker."""
        sessions_to_stop = []
        with self._lock:
            sessions_to_stop = list(self._sessions.values())
            self._sessions.clear()

        for s in sessions_to_stop:
            try:
                s.stop()
            except Exception:
                logger.exception("Error stopping session %s during worker close", s.id)
