import threading
import time
import unittest

from subvocal.utils import ChangeNotifier, ChangeNotifierManager, Debouncer, IncrementalDispatcher, OpsQueue


class TestConcurrency(unittest.TestCase):
    def test_ops_queue_execution_order(self):
        """Verify OpsQueue executes enqueued tasks sequentially on a single thread."""
        q = OpsQueue("test_order", flush_on_stop=True)
        q.start()

        results = []
        threads = set()

        def job(val):
            results.append(val)
            threads.add(threading.current_thread().ident)
            time.sleep(0.01)

        for i in range(5):
            q.enqueue(job, i)

        q.stop()  # waits for completion since flush_on_stop is True

        self.assertEqual(results, [0, 1, 2, 3, 4])
        self.assertEqual(len(threads), 1)  # all jobs executed on the same background thread
        # check that background thread is not the main thread
        self.assertNotIn(threading.current_thread().ident, threads)

    def test_ops_queue_flush_on_stop_false(self):
        """Verify OpsQueue discards remaining queued tasks if flush_on_stop is False."""
        q = OpsQueue("test_no_flush", flush_on_stop=False)
        q.start()

        results = []

        def slow_job(val):
            time.sleep(0.05)
            results.append(val)

        q.enqueue(slow_job, 1)
        q.enqueue(slow_job, 2)
        q.enqueue(slow_job, 3)

        time.sleep(0.01)  # allow first job to start
        q.stop()  # should terminate worker after first job completes, discarding the rest

        self.assertEqual(results, [1])

    def test_incremental_dispatcher(self):
        """Verify IncrementalDispatcher yields items to multiple consumers and terminates on done."""
        dispatcher = IncrementalDispatcher()
        consumer1_results = []
        consumer2_results = []

        def run_consumer(res):
            dispatcher.for_each(res.append)

        t1 = threading.Thread(target=run_consumer, args=(consumer1_results,))
        t2 = threading.Thread(target=run_consumer, args=(consumer2_results,))

        t1.start()
        t2.start()

        dispatcher.add("hello")
        dispatcher.add("world")
        time.sleep(0.05)

        self.assertEqual(consumer1_results, ["hello", "world"])
        self.assertEqual(consumer2_results, ["hello", "world"])

        dispatcher.add("final")
        dispatcher.done()

        t1.join()
        t2.join()

        self.assertEqual(consumer1_results, ["hello", "world", "final"])
        self.assertEqual(consumer2_results, ["hello", "world", "final"])

    def test_change_notifier(self):
        """Verify ChangeNotifier registers and notifies observers asynchronously."""
        notifier = ChangeNotifier()
        calls = []
        event = threading.Event()

        def observer1():
            calls.append("obs1")
            if len(calls) == 2:
                event.set()

        def observer2():
            calls.append("obs2")
            if len(calls) == 2:
                event.set()

        notifier.add_observer("k1", observer1)
        notifier.add_observer("k2", observer2)

        self.assertTrue(notifier.has_observers())

        notifier.notify_changed()
        
        # wait up to 1 second for async delivery
        self.assertTrue(event.wait(timeout=1.0))
        self.assertIn("obs1", calls)
        self.assertIn("obs2", calls)

        notifier.remove_observer("k1")
        self.assertTrue(notifier.has_observers())

        notifier.remove_observer("k2")
        self.assertFalse(notifier.has_observers())

    def test_change_notifier_manager(self):
        """Verify ChangeNotifierManager lifecycle management."""
        mgr = ChangeNotifierManager()
        n1 = mgr.get_or_create_notifier("test")
        n2 = mgr.get_or_create_notifier("test")
        self.assertIs(n1, n2)

        n1.add_observer("obs", lambda: None)
        mgr.remove_notifier("test", force=False)
        # Should not be removed because it has observers
        self.assertIsNotNone(mgr.get_notifier("test"))

        mgr.remove_notifier("test", force=True)
        self.assertIsNone(mgr.get_notifier("test"))

    def test_debouncer(self):
        """Verify Debouncer filters consecutive rapid calls and executes only the last one."""
        debouncer = Debouncer(after_seconds=0.05)
        calls = []
        event = threading.Event()

        def task():
            calls.append(True)
            event.set()

        debouncer.add(task)
        debouncer.add(task)
        debouncer.add(task)

        # Event should fire only once after ~50ms
        self.assertTrue(event.wait(timeout=0.2))
        self.assertEqual(len(calls), 1)

        # Cancel works
        event.clear()
        debouncer.add(task)
        debouncer.cancel()
        time.sleep(0.08)
        self.assertFalse(event.is_set())
