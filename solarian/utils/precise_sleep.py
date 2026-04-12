"""
Precise timing utilities for real-time control loops.

These functions provide sub-millisecond timing accuracy by using
a hybrid approach: sleeping for most of the duration, then spinning
for the remaining time to minimize jitter.
"""

import time


def precise_sleep(dt: float, slack_time: float = 0.001, time_func=time.monotonic):
    """
    Sleep for a precise duration with minimal jitter.
    
    Uses hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    
    Args:
        dt: Duration to sleep in seconds
        slack_time: Time to spin at the end for precision (default 1ms)
        time_func: Time function to use (default time.monotonic)
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    """
    Wait until a specific time with minimal jitter.
    
    Uses hybrid of time.sleep and spinning to minimize jitter.
    
    Args:
        t_end: Target end time (in time_func units)
        slack_time: Time to spin at the end for precision (default 1ms)
        time_func: Time function to use (default time.monotonic)
    """
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return

