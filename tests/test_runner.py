"""Runner machinery: the same worker protocol under process, thread, and inline."""

import multiprocessing as mp
import time

import pytest

from solarian.core.execution import ExecutionMode, WorkerExit
from solarian.core.runner import (
    InlineModuleRunner,
    ProcessModuleRunner,
    ThreadModuleRunner,
    make_runner,
)


class Counter:
    """Minimal Runnable that records its lifecycle in host-visible state."""

    execution_mode = ExecutionMode.INLINE

    def __init__(self, name="counter", frequency=200.0, max_ticks=None, boom=False):
        self.name = name
        self.frequency = frequency
        self.launch_timeout = 5.0
        self.max_ticks = max_ticks
        self.boom = boom
        self.ticks = 0
        self.setup_reconnect = None
        self.exit_reason = None

    def worker_setup(self, reconnect):
        self.setup_reconnect = reconnect

    def worker_tick(self):
        if self.boom:
            raise RuntimeError("worker exploded")
        self.ticks += 1
        return self.max_ticks is None or self.ticks < self.max_ticks

    def worker_teardown(self, exit_reason):
        self.exit_reason = exit_reason


class ProcessCounter(Counter):
    """Reports back across the process boundary via shared state."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shared_ticks = mp.Value("i", 0)
        self.shared_reconnect = mp.Value("i", -1)

    def worker_setup(self, reconnect):
        super().worker_setup(reconnect)
        self.shared_reconnect.value = 1 if reconnect else 0

    def worker_tick(self):
        keep_running = super().worker_tick()
        self.shared_ticks.value = self.ticks
        return keep_running


def test_make_runner_matches_execution_mode():
    assert isinstance(make_runner(Counter()), InlineModuleRunner)
    assert isinstance(
        make_runner(Counter(), ExecutionMode.THREAD), ThreadModuleRunner
    )
    assert isinstance(
        make_runner(Counter(), ExecutionMode.PROCESS), ProcessModuleRunner
    )
    assert isinstance(make_runner(Counter(), "thread"), ThreadModuleRunner)


class TestInline:
    def test_host_drives_each_tick(self):
        target = Counter()
        runner = InlineModuleRunner(target)
        runner.start()

        assert target.setup_reconnect is False
        assert target.ticks == 0

        runner.tick()
        runner.tick()
        assert target.ticks == 2

    def test_finishing_stops_further_ticks(self):
        target = Counter(max_ticks=2)
        runner = InlineModuleRunner(target)
        runner.start()

        runner.tick()
        runner.tick()
        runner.tick()

        assert target.ticks == 2
        assert target.exit_reason is WorkerExit.COMPLETED
        assert not runner.is_alive()

    def test_crash_is_captured_not_raised(self):
        target = Counter(boom=True)
        runner = InlineModuleRunner(target)
        runner.start()
        runner.tick()

        error = runner.get_error()
        assert error is not None
        assert "worker exploded" in error[1]
        assert target.exit_reason is WorkerExit.FAILED

    def test_stop_reports_stopped(self):
        target = Counter()
        runner = InlineModuleRunner(target)
        runner.start()
        runner.stop()
        assert target.exit_reason is WorkerExit.STOPPED


class TestThread:
    def test_runs_and_stops(self):
        target = Counter()
        runner = ThreadModuleRunner(target)
        runner.start()

        assert runner.is_ready
        assert target.setup_reconnect is False

        runner.stop()
        assert not runner.is_alive()
        assert target.ticks > 0
        assert target.exit_reason is WorkerExit.STOPPED

    def test_finishes_on_its_own(self):
        target = Counter(max_ticks=3)
        runner = ThreadModuleRunner(target)
        runner.start()
        runner.join(timeout=5)

        assert target.ticks == 3
        assert target.exit_reason is WorkerExit.COMPLETED

    def test_crash_surfaces_to_host(self):
        target = Counter(boom=True)
        runner = ThreadModuleRunner(target)
        runner.start()
        runner.join(timeout=5)

        error = runner.get_error()
        assert error is not None
        assert "worker exploded" in error[1]


class TestProcess:
    def test_runs_in_a_separate_process(self):
        target = ProcessCounter(name="proc")
        runner = ProcessModuleRunner(target)
        runner.start()

        assert runner.is_ready
        assert runner.pid is not None
        assert runner.pid != mp.current_process().pid
        # Setup must be told it crossed a process boundary so the module
        # rebuilds hardware handles instead of reusing the parent's.
        assert target.shared_reconnect.value == 1

        runner.stop()
        assert target.shared_ticks.value > 0
        # The parent's own copy never advanced: only the child ran the loop.
        assert target.ticks == 0

    def test_finishes_on_its_own(self):
        target = ProcessCounter(max_ticks=3)
        runner = ProcessModuleRunner(target)
        runner.start()
        runner.join(timeout=10)

        assert target.shared_ticks.value == 3
        assert runner.exitcode == 0

    def test_crash_reaches_the_parent(self):
        target = ProcessCounter(boom=True)
        runner = ProcessModuleRunner(target)
        runner.start(wait=False)
        runner.join(timeout=10)

        deadline = time.monotonic() + 5
        error = None
        while error is None and time.monotonic() < deadline:
            error = runner.get_error()
        assert error is not None
        assert "worker exploded" in error[1]

    def test_a_forked_worker_exits_despite_the_timings_its_parent_took(self, monkeypatch):
        """Its snapshot is read only after it exits, so an inherited one bigger
        than the pipe would hold that exit until the host killed it."""
        import solarian.core.runner as runner_module
        from solarian.utils.profiling import func_stats

        monkeypatch.setattr(runner_module, "Process", mp.get_context("fork").Process)
        monkeypatch.setitem(func_stats, "parent_loop", [0.1] * 200_000)
        runner = ProcessModuleRunner(ProcessCounter(max_ticks=1))
        runner.start()
        runner.join(timeout=3)
        exitcode = runner.exitcode
        runner.terminate()

        assert exitcode == 0
