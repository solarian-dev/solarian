"""``enabled: false`` in a module yaml leaves the module out of the run, not out of the code."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from solarian.core.action import Action, ActionType
from solarian.core.actuator_module import ActuatorModule
from solarian.core.configs import ModuleConfig, SensorConfig, SimSensorConfig
from solarian.core.robot import Robot
from solarian.core.skill import SkillStatus
from solarian.sim import CameraMount, ModuleAttachment, SceneConfig
from solarian.sim.scene import scene_without

from test_robot_skills import FakeQueue, FakeRobot, start
from test_skill_controller import RIGHT_ARM, RIGHT_GRIPPER, HoldSkill, TeleopCtrl

LEFT_ARM = "left_arm"


SCENE = SceneConfig(
    attachments=[
        ModuleAttachment(module="right_arm", mjcf="arm.xml", prefix="right/"),
        ModuleAttachment(
            module="right_gripper",
            mjcf="gripper.xml",
            parent_site="right/flange",
            child_site="flange",
            prefix="right_gripper/",
        ),
        ModuleAttachment(module="left_arm", mjcf="arm.xml", prefix="left/"),
    ],
    cameras=[CameraMount(name="wrist_cam", site="right_gripper/camera")],
)


# -- Sim scene --------------------------------------------------------------


def test_a_disabled_module_leaves_the_scene():
    pruned = scene_without(SCENE, {"left_arm"})
    assert [a.module for a in pruned.attachments] == ["right_arm", "right_gripper"]
    assert [c.name for c in pruned.cameras] == ["wrist_cam"]


def test_what_is_mounted_on_a_disabled_module_leaves_with_it():
    pruned = scene_without(SCENE, {"right_arm"})
    assert [a.module for a in pruned.attachments] == ["left_arm"]
    assert pruned.cameras == []


def test_the_declared_scene_is_left_alone():
    scene_without(SCENE, {"right_arm"})
    assert len(SCENE.attachments) == 3


# -- Robot ------------------------------------------------------------------


def _build(robot_cls, cfg, configs, monkeypatch):
    """A robot whose module yamls are ``configs``, past the process-wide singleton."""
    monkeypatch.setattr(robot_cls, "load_module_config", lambda self, name, cls: configs[name])
    robot = object.__new__(robot_cls)
    Robot.__init__(robot, cfg)
    return robot


class TwoArmRobot(Robot):
    right_arm = ActuatorModule()
    left_arm = ActuatorModule()


class SimRobot(Robot):
    right_arm = ActuatorModule()
    right_gripper = ActuatorModule()
    right_wrist_camera = ActuatorModule()
    scene = SCENE


def test_a_disabled_module_gets_no_instance_buffer_or_queue(monkeypatch):
    robot = _build(
        TwoArmRobot,
        SimpleNamespace(),
        {
            "right_arm": ModuleConfig(frequency=30, enabled=False),
            "left_arm": ModuleConfig(frequency=30, enabled=False),
        },
        monkeypatch,
    )
    assert robot.disabled_modules == {"right_arm", "left_arm"}
    assert dict(robot.modules) == {}
    assert robot.observation_buffers == {} and robot.action_queues == {}
    # The declaration stays, so observation.py and action.py still import.
    assert set(TwoArmRobot.modules) == {"right_arm", "left_arm"}


def test_in_sim_a_module_mounted_on_a_disabled_one_is_disabled_too(caplog, monkeypatch):
    camera = SensorConfig.model_construct(port="0", sim=SimSensorConfig(camera="wrist_cam"))
    with caplog.at_level("WARNING"):
        robot = _build(
            SimRobot,
            SimpleNamespace(sim=True),
            {
                "right_arm": ModuleConfig(frequency=30, enabled=False),
                "right_gripper": ModuleConfig(frequency=30),
                "right_wrist_camera": ModuleConfig.model_construct(
                    frequency=30, enabled=True, sensors={"camera": camera}
                ),
            },
            monkeypatch,
        )
    assert robot.disabled_modules == {"right_arm", "right_gripper", "right_wrist_camera"}
    assert [a.module for a in robot.scene.attachments] == ["left_arm"]
    assert "right_gripper, right_wrist_camera" in caplog.text


def module_action(*names):
    return Action(
        action_type=ActionType.MODULE_ACTION,
        data={name: {"action_type": int(ActionType.SERVO_L), "data": {}} for name in names},
    )


class SizedQueue(FakeQueue):
    def qsize(self):
        return len(self.actions)


def test_an_action_for_a_disabled_module_is_dropped_and_the_rest_delivered(caplog):
    robot = FakeRobot()
    robot.action_queues = {"arm": SizedQueue()}
    with caplog.at_level("WARNING"):
        robot.process_action(module_action("arm", "gripper"))
        robot.process_action(module_action("arm", "gripper"))
    assert len(robot.action_queues["arm"].actions) == 2
    assert caplog.text.count("'gripper'") == 1


class StatusRows:
    def __init__(self):
        self.rows = []

    def put(self, **row):
        self.rows.append(row)


def test_the_robot_refuses_a_skill_writing_a_disabled_module():
    robot = FakeRobot()
    robot.disabled_modules = {"arm"}
    robot.skills_status_buffer = StatusRows()

    start(robot, "arm_wave", run_id=7, params={"arm": "arm"})

    assert robot._local_skills == {}
    assert robot.skills_status_buffer.rows[0]["skill_id"] == 7
    assert robot.skills_status_buffer.rows[0]["status"] == int(SkillStatus.FAILED)
    assert "arm" in robot._skill_failure_reason(SimpleNamespace(skill_id=7))


# -- Controller -------------------------------------------------------------


def test_the_controller_refuses_a_skill_writing_a_disabled_module(caplog):
    ctrl = TeleopCtrl()
    ctrl.disabled_modules = {RIGHT_GRIPPER}
    with caplog.at_level("ERROR"):
        ctrl.press("m")
    assert ctrl.running_skills == {}
    assert "right_gripper" in caplog.text


def test_the_robots_report_overrides_the_controllers_yaml(caplog):
    ctrl = TeleopCtrl()
    ctrl.hold_everything()
    with caplog.at_level("WARNING"):
        ctrl._adopt_robot_disabled({RIGHT_ARM})
        ctrl._adopt_robot_disabled({RIGHT_ARM})
    assert caplog.text.count("disabled") == 1
    assert ctrl.disabled_modules == {RIGHT_ARM}

    ctrl._stop_skills_on_disabled_modules()
    assert ctrl.running_skills == {}
    with pytest.raises(Exception, match="right_arm"):
        ctrl.start_skill(HoldSkill("hold", {"arm": RIGHT_ARM}))


def test_a_module_the_robot_runs_again_is_usable_again():
    ctrl = TeleopCtrl()
    ctrl._adopt_robot_disabled({LEFT_ARM})
    ctrl._adopt_robot_disabled(set())
    assert ctrl.disabled_modules == set()
    ctrl.start_skill(HoldSkill("hold", {"arm": LEFT_ARM}))
    assert "hold" in ctrl.running_skills


def test_the_agent_is_not_offered_tools_bound_to_a_disabled_module(monkeypatch):
    import solarian.core.ai.agent as agent

    monkeypatch.setattr(agent, "AgentHarness", lambda host, cfg, tools: tools)
    ctrl = TeleopCtrl()
    ctrl.cfg = SimpleNamespace(agent=object())
    ctrl.disabled_modules = {RIGHT_GRIPPER}

    assert list(ctrl._build_agent()) == []
