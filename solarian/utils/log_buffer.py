import logging
import multiprocessing
import os
import sys
import collections
import queue
import traceback
from datetime import datetime
from typing import Optional

from rich.panel import Panel
from rich.text import Text


LEVEL_STYLES = {
    logging.DEBUG: "dim",
    logging.INFO: "green",
    logging.WARNING: "yellow",
    logging.ERROR: "bold red",
    logging.CRITICAL: "bold white on red",
}


class LogRecord:
    """Lightweight, picklable snapshot of a logging.LogRecord."""
    __slots__ = ("timestamp", "level", "name", "message", "pid")

    def __init__(self, timestamp: float, level: int, name: str, message: str, pid: int):
        self.timestamp = timestamp
        self.level = level
        self.name = name
        self.message = message
        self.pid = pid

    @classmethod
    def from_logging_record(cls, record: logging.LogRecord) -> "LogRecord":
        message = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            message = f"{message}: {exception_summary(record.exc_info[1])}"
        return cls(
            timestamp=record.created,
            level=record.levelno,
            name=record.name,
            message=message,
            pid=record.process or os.getpid(),
        )


def exception_summary(exc: BaseException) -> str:
    """``Type: message (file:line in func)`` of the frame that raised.

    The panel has one line per record, so a ``logger.exception`` keeps the
    part of its traceback that usually names the bug.
    """
    text = f"{type(exc).__name__}: {exc}"
    tb = traceback.extract_tb(exc.__traceback__)
    if tb:
        frame = tb[-1]
        text += f" ({os.path.basename(frame.filename)}:{frame.lineno} in {frame.name})"
    return text


class LogBuffer:
    """Thread-safe ring buffer that renders as a Rich Panel.

    Call ``drain(queue)`` periodically to pull :class:`LogRecord` objects from
    a :class:`multiprocessing.Queue` into the fixed-size buffer.
    """

    def __init__(self, maxlen: int = 200):
        self._buf: collections.deque[LogRecord] = collections.deque(maxlen=maxlen)

    def drain(self, q: multiprocessing.Queue) -> list[LogRecord]:
        """Move all pending records from *q* into the ring buffer.

        Returns the list of newly drained records so callers can print them.
        """
        new: list[LogRecord] = []
        while True:
            try:
                rec = q.get_nowait()
                self._buf.append(rec)
                new.append(rec)
            except (queue.Empty, EOFError):
                break
        return new

    @staticmethod
    def format_record(rec: "LogRecord") -> Text:
        ts = datetime.fromtimestamp(rec.timestamp).strftime("%H:%M:%S.%f")[:-3]
        lvl = logging.getLevelName(rec.level)[:4]
        style = LEVEL_STYLES.get(rec.level, "")
        line = Text()
        line.append(f"{ts} ", style="dim")
        line.append(f"{lvl:<4} ", style=style)
        line.append(f"[pid {rec.pid}] ", style="dim cyan")
        line.append(f"{rec.name}: ", style="bold")
        line.append(rec.message)
        return line

    def render(self, max_lines: int = 20) -> Panel:
        lines = Text()
        tail = list(self._buf)[-max_lines:]
        for rec in tail:
            line = self.format_record(rec)
            line.append("\n")
            lines.append_text(line)
        return Panel(lines or Text("(no logs yet)", style="dim"), title="Logs", border_style="cyan")


class QueueLogHandler(logging.Handler):
    """A :class:`logging.Handler` that serialises records into a
    :class:`multiprocessing.Queue` so they can be drained by the
    :class:`LogBuffer` living in the main process.

    Usage (in any process)::

        handler = QueueLogHandler(log_queue)
        logging.getLogger().addHandler(handler)
    """

    def __init__(self, log_queue: multiprocessing.Queue, level: int = logging.NOTSET):
        super().__init__(level)
        self._queue = log_queue

    def emit(self, record: logging.LogRecord):
        try:
            self._queue.put_nowait(LogRecord.from_logging_record(record))
        except Exception:
            self.handleError(record)


class FallbackStreamHandler(logging.StreamHandler):
    """StreamHandler used as a console fallback before StatusBoard takes over.

    Identified by class so it can be selectively removed once the
    StatusBoard starts draining the log queue.
    """
    pass


_log_queue: Optional[multiprocessing.Queue] = None


def get_log_queue() -> multiprocessing.Queue:
    """Return the global log queue, creating it on first call.

    The queue is safe to pass to child processes (it is a
    :class:`multiprocessing.Queue`).
    """
    global _log_queue
    if _log_queue is None:
        _log_queue = multiprocessing.Queue()
    return _log_queue


def setup_process_logging(
    level: int = logging.DEBUG,
    logger_name: Optional[str] = None,
    log_queue: Optional[multiprocessing.Queue] = None,
    install_fallback: bool = True,
):
    """Wire the given (or root) logger so all records go to the shared log queue.

    Args:
        log_queue: Explicit queue to use.  Pass the parent's queue into child
            processes so they share the same pipe instead of creating a new one.
        install_fallback: If *True* (default), add a :class:`FallbackStreamHandler`
            so logs are visible on stderr before the :class:`StatusBoard` takes
            over.  Set to *False* in subprocesses where the parent already
            drains the queue via the StatusBoard.
    """
    if log_queue is not None:
        global _log_queue
        _log_queue = log_queue
    log_q = get_log_queue()
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if not any(isinstance(h, QueueLogHandler) for h in logger.handlers):
        logger.addHandler(QueueLogHandler(log_q, level))
    if install_fallback and not any(isinstance(h, FallbackStreamHandler) for h in logger.handlers):
        fmt = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-4.4s [pid %(process)d] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        sh = FallbackStreamHandler(sys.stderr)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # Root is often DEBUG; quiet chatty third-party loggers (e.g. picamera2
    # "Execute job" on every capture) while keeping their INFO+ messages.
    for noisy in ("picamera2",):
        logging.getLogger(noisy).setLevel(logging.INFO)


def remove_fallback_handlers(logger_name: Optional[str] = None):
    """Remove all :class:`FallbackStreamHandler` instances from the logger.

    Called by :class:`StatusBoard` when it starts draining the log queue so
    that log lines are no longer duplicated on the console.
    """
    logger = logging.getLogger(logger_name)
    for h in logger.handlers[:]:
        if isinstance(h, FallbackStreamHandler):
            logger.removeHandler(h)
