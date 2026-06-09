import logging
import queue
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class OpsQueue:
    """A thread-safe serialized work queue running on a single background worker thread.

    Equivalent to LiveKit's OpsQueue.
    """

    def __init__(self, name: str, min_size: int = 128, flush_on_stop: bool = False, custom_logger: logging.Logger | None = None):
        self.name = name
        self.min_size = min_size
        self.flush_on_stop = flush_on_stop
        self.logger = custom_logger or logging.getLogger(f"subvocal.utils.opsqueue.{name}")

        self._queue = queue.Queue()
        self._thread = None
        self._started = False
        self._stopped = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Starts the background processing thread if not already started."""
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(target=self._process, name=f"OpsQueue-{self.name}", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stops the queue.

        If flush_on_stop is True, all currently queued operations are completed
        before the processing thread exits.
        """
        with self._lock:
            if self._stopped or not self._started:
                return
            self._stopped = True
            # Enqueue sentinel to request thread termination
            self._queue.put(None)

        if self._thread:
            self._thread.join()

    def enqueue(self, fn: Callable, *args, **kwargs) -> None:
        """Enqueues an operation to be run on the background processing thread."""
        with self._lock:
            if self._stopped:
                return
            self._queue.put((fn, args, kwargs))

    def _process(self) -> None:
        while True:
            try:
                item = self._queue.get()
                
                # Check stop state before executing
                with self._lock:
                    if self._stopped and not self.flush_on_stop:
                        self._queue.task_done()
                        break

                if item is None:
                    # Sentinel wakeup, stop requested
                    if self.flush_on_stop:
                        # Process remaining queued items
                        while not self._queue.empty():
                            rem_item = self._queue.get_nowait()
                            if rem_item is not None:
                                self._run_op(rem_item)
                            self._queue.task_done()
                    self._queue.task_done()
                    break

                self._run_op(item)
                self._queue.task_done()
            except Exception:
                self.logger.exception("OpsQueue internal error in worker thread loop")

    def _run_op(self, item: tuple[Callable, tuple, dict]) -> None:
        fn, args, kwargs = item
        try:
            fn(*args, **kwargs)
        except Exception:
            self.logger.exception("Error executing queued operation in OpsQueue '%s'", self.name)


class IncrementalDispatcher:
    """Allows multiple consumers to consume items thread-safely as they become available.

    Producers can add items at any time. Equivalent to LiveKit's IncrementalDispatcher.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._items = []
        self._done = False

    def add(self, item) -> None:
        """Appends an item and broadcasts to waiting consumers."""
        with self._cond:
            if self._done:
                return
            self._items.append(item)
            self._cond.notify_all()

    def done(self) -> None:
        """Signals that no more items will be added and wakes up all waiting consumers."""
        with self._cond:
            if self._done:
                return
            self._done = True
            self._cond.notify_all()

    def for_each(self, callback: Callable) -> None:
        """Iterates over all items (including future ones) and invokes the callback on them."""
        idx = 0
        while True:
            items_to_dispatch = []
            with self._cond:
                while idx < len(self._items):
                    items_to_dispatch.append(self._items[idx])
                    idx += 1

                if self._done and idx == len(self._items):
                    # Dispatch final batch and terminate
                    for item in items_to_dispatch:
                        try:
                            callback(item)
                        except Exception:
                            logger.exception("IncrementalDispatcher callback error")
                    break

            for item in items_to_dispatch:
                try:
                    callback(item)
                except Exception:
                    logger.exception("IncrementalDispatcher callback error")

            with self._cond:
                if self._done and idx == len(self._items):
                    break
                if idx == len(self._items):
                    self._cond.wait()


class ChangeNotifier:
    """Registry for keyed observers to receive change notifications asynchronously.

    Equivalent to LiveKit's ChangeNotifier.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._observers = {}

    def add_observer(self, key: str, on_changed: Callable[[], None]) -> None:
        """Registers a callback with a unique key."""
        with self._lock:
            self._observers[key] = on_changed

    def remove_observer(self, key: str) -> None:
        """Unregisters a callback by its key."""
        with self._lock:
            self._observers.pop(key, None)

    def has_observers(self) -> bool:
        """Returns True if there is at least one observer registered."""
        with self._lock:
            return len(self._observers) > 0

    def notify_changed(self) -> None:
        """Asynchronously dispatches notifications to all registered observers."""
        with self._lock:
            observers = list(self._observers.values())
        if not observers:
            return

        def run_observers():
            for f in observers:
                try:
                    f()
                except Exception:
                    logger.exception("ChangeNotifier observer callback error")

        threading.Thread(target=run_observers, name="ChangeNotifier-Dispatcher", daemon=True).start()


class ChangeNotifierManager:
    """Manages a registry of named ChangeNotifier instances.

    Equivalent to LiveKit's ChangeNotifierManager.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._notifiers = {}

    def get_notifier(self, key: str) -> ChangeNotifier | None:
        """Retrieves a notifier by key, or returns None if it doesn't exist."""
        with self._lock:
            return self._notifiers.get(key)

    def get_or_create_notifier(self, key: str) -> ChangeNotifier:
        """Retrieves a notifier by key, creating a new one if it doesn't exist."""
        with self._lock:
            if key not in self._notifiers:
                self._notifiers[key] = ChangeNotifier()
            return self._notifiers[key]

    def remove_notifier(self, key: str, force: bool = False) -> None:
        """Removes a notifier by key.

        If force is False, the notifier will only be removed if it has no observers.
        """
        with self._lock:
            if key in self._notifiers:
                if force or not self._notifiers[key].has_observers():
                    del self._notifiers[key]


class Debouncer:
    """Resettable delay timer to debounce fast-firing events.

    Equivalent to LiveKit's Debouncer.
    """

    def __init__(self, after_seconds: float):
        self._lock = threading.Lock()
        self.after_seconds = after_seconds
        self._timer = None

    def add(self, fn: Callable[[], None]) -> None:
        """Cancels any pending callback timer and schedules a new one to execute fn."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()

            def run():
                try:
                    fn()
                except Exception:
                    logger.exception("Debouncer function execution error")

            self._timer = threading.Timer(self.after_seconds, run)
            self._timer.name = "Debouncer-Timer"
            self._timer.start()

    def set_duration(self, after_seconds: float) -> None:
        """Updates the delay duration for subsequent debounce invocations."""
        with self._lock:
            self.after_seconds = after_seconds

    def cancel(self) -> None:
        """Cancels any pending timer execution."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
