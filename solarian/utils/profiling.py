from bisect import bisect_right
from collections import defaultdict
from functools import wraps
import time
import inspect
import atexit
import math

#: Upper edges of the histogram bins, in ms: 32 per decade from 1 us to 100 s,
#: so a percentile is good to about 7.5%.
_EDGES = [10 ** (k / 32.0) for k in range(-3 * 32, 5 * 32 + 1)]


class Timings:
    """Count, mean, extremes and a log-binned histogram of durations in ms.

    Fixed size however long the run: hot paths are timed millions of times in
    an hour, and a forked worker ships its timings to the parent through a pipe.
    """

    __slots__ = ("count", "total", "min", "max", "bins")

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.min = math.inf
        self.max = 0.0
        self.bins = [0] * (len(_EDGES) + 1)

    def add(self, ms: float) -> None:
        self.count += 1
        self.total += ms
        if ms < self.min:
            self.min = ms
        if ms > self.max:
            self.max = ms
        self.bins[bisect_right(_EDGES, ms)] += 1

    def merge(self, other: "Timings") -> None:
        self.count += other.count
        self.total += other.total
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        for i, n in enumerate(other.bins):
            self.bins[i] += n

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def percentile(self, q: float) -> float:
        """The upper edge of the bin holding the ``q``-th percentile, at most ``max``."""
        if not self.count:
            return 0.0
        rank = q / 100.0 * self.count
        seen = 0
        for i, n in enumerate(self.bins):
            seen += n
            if seen >= rank:
                return min(_EDGES[i] if i < len(_EDGES) else self.max, self.max)
        return self.max

    def summary(self) -> str:
        return (
            f"n={self.count} avg={self.mean:.3f} p50={self.percentile(50):.3f} "
            f"p99={self.percentile(99):.3f} max={self.max:.3f} ms"
        )


func_stats: "defaultdict[str, Timings]" = defaultdict(Timings)
verbose = False
_suppress_atexit = False


@atexit.register
def print_stats():
    if _suppress_atexit or not func_stats:
        return
    print("\n=== Function Profiling Summary ===")
    for func_name, timings in sorted(func_stats.items(), key=lambda item: item[0]):
        print(f"- {func_name}: {timings.summary()}")


def suppress_atexit_summary():
    """Prevent the profiling summary from printing at exit."""
    global _suppress_atexit
    _suppress_atexit = True


def profile(func):
    # Resolved once: getfullargspec costs more than the body of a hot
    # function like get_observation, and only ``verbose`` ever reads it.
    arg_names = inspect.getfullargspec(func).args
    name = func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter_ns() - start_time) * 1e-6

        func_stats[name].add(elapsed)

        if verbose:
            args_list = [f"{n}={v}" for n, v in zip(arg_names, args)]
            kwargs_list = [f"{k}={v}" for k, v in kwargs.items()]
            arg_str = ", ".join(args_list + kwargs_list)
            print(f"Function '{name}'({arg_str}) took {elapsed:.6f} ms.")

        return result
    return wrapper


def manual_profile(func_name, elapsed):
    """Record ``elapsed`` nanoseconds under ``func_name``; over 1 s is discarded."""
    if elapsed > 1e9:
        return

    elapsed *= 1e-6
    func_stats[func_name].add(elapsed)
    if verbose:
        print(f"Function '{func_name}' took {elapsed:.6f} ms.")


def get_stats_snapshot() -> dict[str, Timings]:
    """Return a plain-dict copy of the current per-function timing data."""
    return dict(func_stats)


def merge_stats(other: dict[str, Timings]):
    """Merge timing data (e.g. from a child process) into the global stats."""
    for name, timings in other.items():
        func_stats[name].merge(timings)
