"""Motor speed caps that follow the planned speed, and the joint-speed limit on
Cartesian moves."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from solarian.cli.commands._calibration import save_module_field, speed_limits_lifted
from solarian.core.arm_module import ArmModule
from solarian.core.ik.bus import IKActuatorBus


class _Module:
    """Records the configured speed each time the backstop is applied."""

    name = "gripper"

    def __init__(self):
        self.cfg = SimpleNamespace(max_speed=40.0, max_accel=[0.1, 0.6])
        self.applied: list[float] = []

    def apply_speed_backstop(self):
        self.applied.append(self.cfg.max_speed)

    def _module_yaml_path(self):
        return None


def test_lifted_limits_are_in_force_inside_and_restored_after():
    module = _Module()
    with speed_limits_lifted(module, max_speed=160.0, max_accel=None):
        assert module.cfg.max_speed == 160.0
        assert module.cfg.max_accel is None
    assert module.cfg.max_speed == 40.0
    assert module.cfg.max_accel == [0.1, 0.6]
    # The motor cap follows the lift, and comes back down with it.
    assert module.applied == [160.0, 40.0]


def test_lifted_limits_are_restored_when_the_ramp_fails():
    module = _Module()
    with pytest.raises(RuntimeError):
        with speed_limits_lifted(module, max_speed=160.0):
            raise RuntimeError("bus dropped")
    assert module.cfg.max_speed == 40.0
    assert module.applied[-1] == 40.0


def test_saving_a_speed_moves_the_motor_cap_and_a_latency_does_not():
    module = _Module()
    save_module_field(module, "max_speed", 120.0)
    assert module.applied == [120.0]
    save_module_field(module, "action_latency", 0.08)
    assert module.applied == [120.0]


def _ik_bus(max_joint_speed):
    bus = IKActuatorBus.__new__(IKActuatorBus)
    bus.port = "test"
    bus.max_joint_speed = lambda: max_joint_speed
    bus._last_q_cmd = None
    bus._joint_limit_binding = False
    return bus


def test_ik_step_over_the_joint_limit_is_scaled_uniformly():
    bus = _ik_bus(max_joint_speed=1.0)
    q_now = np.zeros(3)
    q = bus._limit_joint_step(q_now, np.array([0.1, 0.05, -0.02]), dt=0.01)
    # 0.1 rad in 10 ms is 10 rad/s; scaled to 1 rad/s, every joint by the same 1/10.
    np.testing.assert_allclose(q, [0.01, 0.005, -0.002])
    assert bus._joint_limit_binding


def test_ik_step_continues_from_the_last_command_not_the_lagging_measurement():
    bus = _ik_bus(max_joint_speed=1.0)
    bus._limit_joint_step(np.zeros(1), np.array([1.0]), dt=0.01)
    # The arm has not moved yet; the next step still advances from 0.01.
    q = bus._limit_joint_step(np.zeros(1), np.array([1.0]), dt=0.01)
    np.testing.assert_allclose(q, [0.02])


def test_ik_step_under_the_limit_or_without_one_is_untouched():
    q_next = np.array([0.001, 0.0])
    np.testing.assert_allclose(
        _ik_bus(max_joint_speed=1.0)._limit_joint_step(np.zeros(2), q_next, dt=0.01), q_next
    )
    np.testing.assert_allclose(
        _ik_bus(max_joint_speed=None)._limit_joint_step(np.zeros(2), q_next * 100, dt=0.01),
        q_next * 100,
    )


def _arm(max_joint_speed, gain):
    """An arm whose one joint turns *gain* rad per metre of x."""
    arm = ArmModule.__new__(ArmModule)
    arm.cfg = SimpleNamespace(max_joint_speed=max_joint_speed)
    arm._ik_joint_names = lambda: ["j"]
    arm._read_joints_rad = lambda: np.zeros(1)
    arm.reachable_pose = lambda target, q_seed=None: (
        np.asarray(target), [], np.array([gain * float(target[0])])
    )
    return arm


def test_movel_is_slowed_when_its_path_needs_a_faster_joint():
    arm = _arm(max_joint_speed=0.5, gain=10.0)
    start, target = np.zeros(6), np.array([0.1, 0, 0, 0, 0, 0])
    # 0.1 m/s along x turns the joint at 1 rad/s: twice its limit.
    factor = arm._joint_speed_factor(start, target, max_pos_speed=0.1, max_rot_speed=1.0)
    assert factor == pytest.approx(0.5)


def test_movel_within_the_joint_limit_or_without_one_is_not_slowed():
    start, target = np.zeros(6), np.array([0.1, 0, 0, 0, 0, 0])
    assert _arm(max_joint_speed=2.0, gain=10.0)._joint_speed_factor(
        start, target, max_pos_speed=0.1, max_rot_speed=1.0
    ) == 1.0
    assert _arm(max_joint_speed=None, gain=10.0)._joint_speed_factor(
        start, target, max_pos_speed=0.1, max_rot_speed=1.0
    ) == 1.0
