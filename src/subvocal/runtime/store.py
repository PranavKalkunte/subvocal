import json
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("subvocal.runtime.store")


class SessionStore(ABC):
    """Abstract interface for session configuration and state storage."""

    @abstractmethod
    def save_session(self, session_id: str, state: str, config_yaml: str, metadata: dict[str, Any]) -> None:
        """Saves or updates a session configuration and state snapshot."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieves a session snapshot by ID, returning metadata, config, and state."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Deletes a session snapshot by ID."""
        pass

    @abstractmethod
    def list_sessions(self) -> list[dict[str, Any]]:
        """Lists all stored sessions."""
        pass


class InMemorySessionStore(SessionStore):
    """In-memory session registry for lightweight, transient deployments."""

    def __init__(self):
        self._store: dict[str, dict[str, Any]] = {}

    def save_session(self, session_id: str, state: str, config_yaml: str, metadata: dict[str, Any]) -> None:
        self._store[session_id] = {
            "session_id": session_id,
            "state": state,
            "config_yaml": config_yaml,
            "metadata": metadata,
            "updated_at": time.time(),
        }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self._store.get(session_id)

    def delete_session(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        return list(self._store.values())


class SQLiteSessionStore(SessionStore):
    """SQLite-backed session registry for persistent edge configurations."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    config_yaml TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()

    def save_session(self, session_id: str, state: str, config_yaml: str, metadata: dict[str, Any]) -> None:
        metadata_str = json.dumps(metadata)
        updated_at = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, state, config_yaml, metadata, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, state, config_yaml, metadata_str, updated_at),
            )
            conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT state, config_yaml, metadata, updated_at FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "session_id": session_id,
                "state": row[0],
                "config_yaml": row[1],
                "metadata": json.loads(row[2]),
                "updated_at": row[3],
            }

    def delete_session(self, session_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    def list_sessions(self) -> list[dict[str, Any]]:
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id, state, config_yaml, metadata, updated_at FROM sessions")
            for row in cursor.fetchall():
                results.append({
                    "session_id": row[0],
                    "state": row[1],
                    "config_yaml": row[2],
                    "metadata": json.loads(row[3]),
                    "updated_at": row[4],
                })
        return results
