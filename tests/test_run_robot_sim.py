"""Sim is chosen per run with ``run-robot --sim``; hardware is the default."""

from types import SimpleNamespace

import pytest

from solarian.cli.commands.run_robot import RunRobotCommand
from solarian.core.robot import Robot


def test_sim_flag_is_consumed_and_overrides_pass_through():
    cmd = RunRobotCommand()
    assert cmd.parse_args(["--sim", "fps=60"]) == ["fps=60"]
    assert cmd.sim is True


def test_hardware_is_the_default():
    cmd = RunRobotCommand()
    assert cmd.parse_args(["fps=60"]) == ["fps=60"]
    assert cmd.sim is False


class _SceneRobot(Robot):
    scene = object()


def _build(robot_cls, cfg):
    # Robot.__new__ hands back a process-wide singleton; skip it.
    robot = object.__new__(robot_cls)
    Robot.__init__(robot, cfg)
    return robot


@pytest.mark.parametrize(
    "cfg, expected",
    [
        (SimpleNamespace(sim=True), True),
        (SimpleNamespace(sim=False), False),
        (SimpleNamespace(sim=None), False),
        (SimpleNamespace(), False),
    ],
)
def test_is_sim_follows_the_config(cfg, expected):
    assert _build(_SceneRobot, cfg).is_sim is expected


def test_sim_without_a_scene_is_refused():
    class _NoScene(Robot):
        pass

    with pytest.raises(ValueError, match="--sim"):
        _build(_NoScene, SimpleNamespace(sim=True))
