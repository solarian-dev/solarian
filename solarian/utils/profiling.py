from collections import defaultdict
from functools import wraps, cache
import time
import inspect
import atexit

func_stats = defaultdict(list)
verbose = False
_suppress_atexit = False

@atexit.register
def print_stats():
    if _suppress_atexit or not func_stats:
        return
    print("\n=== Function Profiling Summary ===")
    for func_name, times in sorted(func_stats.items(), key=lambda item: item[0]):
        min_time = min(times)
        max_time = max(times)
        avg_time = sum(times) / len(times)
        print(f"- {func_name}: called {len(times)} times | "
              f"min={min_time:.6f}ms, max={max_time:.6f}ms, avg={avg_time:.6f}ms")


def suppress_atexit_summary():
    """Prevent the profiling summary from printing at exit."""
    global _suppress_atexit
    _suppress_atexit = True


def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time_ns()
        result = func(*args, **kwargs)
        end_time = time.time_ns()
        elapsed = (end_time - start_time)*1e-6
        
        # Store elapsed time
        func_stats[func.__name__].append(elapsed)

        # Get argument names and values
        arg_names = inspect.getfullargspec(func).args
        arg_values = args[:len(arg_names)]
        args_list = [f"{name}={value}" for name, value in zip(arg_names, arg_values)]
        kwargs_list = [f"{k}={v}" for k, v in kwargs.items()]
        all_args = args_list + kwargs_list
        arg_str = ", ".join(all_args)

        if verbose:
            print(f"Function '{func.__name__}'({arg_str}) took {elapsed:.6f} ms.")

        return result
    return wrapper


def manual_profile(func_name, elapsed):
    # if elapsed is more than 2s, then discard
    if elapsed > 1e9:
        return
    
    elapsed *= 1e-6
    func_stats[func_name].append(elapsed)
    if verbose:
        print(f"Function '{func_name}' took {elapsed:.6f} ms.")


def get_stats_snapshot() -> dict[str, list[float]]:
    """Return a plain-dict copy of the current per-function timing data."""
    return dict(func_stats)


def merge_stats(other: dict[str, list[float]]):
    """Merge timing data (e.g. from a child process) into the global stats."""
    for name, times in other.items():
        func_stats[name].extend(times)
