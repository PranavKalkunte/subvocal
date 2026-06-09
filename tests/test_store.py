import os
import tempfile
import unittest

from subvocal.config import load_config
from subvocal.core.testing import MockActionExecutor, MockContextProvider, MockHardwareSource, MockLLMProvider
from subvocal.runtime.store import InMemorySessionStore, SQLiteSessionStore
from subvocal.runtime.worker import SessionWorker


class TestSessionStores(unittest.TestCase):

    def setUp(self):
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        # Close file descriptor so sqlite can open it cleanly
        os.close(self.temp_db_fd)

    def tearDown(self):
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_in_memory_store_lifecycle(self):
        """Verify basic InMemorySessionStore operations."""
        store = InMemorySessionStore()
        metadata = {"creator": "test"}
        store.save_session("session-1", "active", "hardware: fs: 250", metadata)

        # Retrieve
        s = store.get_session("session-1")
        self.assertIsNotNone(s)
        self.assertEqual(s["state"], "active")
        self.assertEqual(s["config_yaml"], "hardware: fs: 250")
        self.assertEqual(s["metadata"], metadata)

        # List
        all_sessions = store.list_sessions()
        self.assertEqual(len(all_sessions), 1)

        # Delete
        store.delete_session("session-1")
        self.assertIsNone(store.get_session("session-1"))
        self.assertEqual(len(store.list_sessions()), 0)

    def test_sqlite_store_lifecycle(self):
        """Verify basic SQLiteSessionStore operations."""
        store = SQLiteSessionStore(self.temp_db_path)
        metadata = {"version": "2.0"}
        store.save_session("session-2", "degraded", "classifier: type: rf", metadata)

        # Retrieve
        s = store.get_session("session-2")
        self.assertIsNotNone(s)
        self.assertEqual(s["state"], "degraded")
        self.assertEqual(s["config_yaml"], "classifier: type: rf")
        self.assertEqual(s["metadata"], metadata)

        # Verify SQL persistence
        store2 = SQLiteSessionStore(self.temp_db_path)
        s2 = store2.get_session("session-2")
        self.assertIsNotNone(s2)
        self.assertEqual(s2["state"], "degraded")

        # List
        self.assertEqual(len(store2.list_sessions()), 1)

        # Delete
        store2.delete_session("session-2")
        self.assertIsNone(store.get_session("session-2"))
        self.assertEqual(len(store.list_sessions()), 0)

    def test_worker_store_integration(self):
        """Verify SessionWorker correctly saves and deletes from a SessionStore."""
        config = load_config()
        store = InMemorySessionStore()
        worker = SessionWorker(config, max_sessions=2, store=store)

        hw = MockHardwareSource()
        llm = MockLLMProvider()
        ctx = MockContextProvider()
        executor = MockActionExecutor()

        worker.create_session("sess-xyz", hw, lambda f: None, llm, ctx, executor)

        # Verify session is persisted in store
        self.assertEqual(len(store.list_sessions()), 1)
        s = store.get_session("sess-xyz")
        self.assertIsNotNone(s)
        self.assertEqual(s["metadata"]["created_by"], "worker")

        # Verify removal deletes it from store
        worker.remove_session("sess-xyz")
        self.assertEqual(len(store.list_sessions()), 0)


if __name__ == "__main__":
    unittest.main()
