"""Where a robot module's initialize() runs: in the process that drives it."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from solarian.core.execution import ExecutionMode
from solarian.core.module import Module
from solarian.core.robot import Robot


def test_robot_initializes_only_the_modules_it_drives_itself():
    inline = MagicMock(execution_mode=ExecutionMode.INLINE)
    process = MagicMock(execution_mode=ExecutionMode.PROCESS)
    Robot.initialize(SimpleNamespace(modules={"inline": inline, "process": process}))
    inline.initialize.assert_called_once()
    process.initialize.assert_not_called()


def _worker(**attrs):
    module = MagicMock(
        _status_mp_q=None, _log_queue=None, _module_obs_cls=None, _clock_sync_connection=None
    )
    module.get_calibration.return_value = {}
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_a_process_module_initializes_after_reconnecting_in_its_own_process():
    module = _worker()
    Module.worker_setup(module, reconnect=True)
    calls = [c[0] for c in module.method_calls]
    assert calls.index("connect") < calls.index("initialize")


def test_a_module_in_the_host_process_is_not_initialized_twice():
    module = _worker()
    Module.worker_setup(module, reconnect=False)
    module.initialize.assert_not_called()
