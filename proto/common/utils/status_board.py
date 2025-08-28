# pip install rich
from rich.live import Live
from rich.table import Table
from rich.console import Console
import threading
import time
import queue

class StatusBoard:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:  # Thread-safe instance creation
            if cls._instance is None:
                cls._instance = super(StatusBoard, cls).__new__(cls)
                # Initialize the instance
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Ensure initialization happens only once
        if not hasattr(self, '_initialized') or not self._initialized:
            self.data = {}
            self._q = queue.Queue()
            self._instance_lock = threading.Lock()  # For thread-safe operations
            # other initialization...
            self._initialized = True

            threading.Thread(target=self.ui_loop, daemon=True).start()

    def set(self, key, value):
        with self._instance_lock:
            self._q.put((key, value))

    def get(self):
        try:
            return self._q.get_nowait()
        except Exception:
            return None, None

    def render(self):
        t = Table(title="Status", expand=True, show_edge=False)
        t.add_column("State"); t.add_column("Value")
        t.add_row("Local time", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for k, v in self.data.items():
            t.add_row(k, str(v))
        return t

    def ui_loop(self):
        console = Console()
        with Live(self.render(), console=console, refresh_per_second=12) as live:
            while True:
                try:
                    # drain updates quickly
                    while True:
                        k, v = self._q.get_nowait()
                        self.data[k] = v
                    # no break; keep draining
                except queue.Empty:
                    pass
                live.update(self.render())
                time.sleep(0.05)



board = StatusBoard()

