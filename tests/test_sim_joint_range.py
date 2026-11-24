"""The sim maps a servo's normalized range onto its joint's limits in the model."""

from __future__ import annotations

import math
import threading
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from solarian.sim import ModuleAttachment, SceneConfig
from solarian.sim.mujoco_world import MujocoWorldServer

GRIPPER_MJCF = """\
<mujoco model="gripper">
  <worldbody>
    <body name="base">
      <geom type="box" size="0.02 0.02 0.02"/>
      <body name="left">
        <joint name="left" type="slide" axis="0 1 0" range="0 0.05"/>
        <geom type="box" size="0.01 0.01 0.01"/>
      </body>
      <body name="right">
        <joint name="right" type="slide" axis="0 1 0" range="-0.05 0"/>
        <geom type="box" size="0.01 0.01 0.01"/>
      </body>
      <body name="spin">
        <joint name="spin" type="hinge" axis="0 0 1" limited="false"/>
        <geom type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="left" joint="left" kp="50"/>
    <position name="right" joint="right" kp="50"/>
    <position name="spin" joint="spin" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def server(tmp_path: Path) -> MujocoWorldServer:
    mjcf = tmp_path / "gripper.xml"
    mjcf.write_text(GRIPPER_MJCF)
    specs = {
        "gripper/jaw": {
            "kind": "joint",
            "joint": "g/left",
            "actuator": "g/left",
            "mode": "range_m100_100",
            "range_override": (None, None),
            "drive_mode": 0,
            "followers": [{"joint": "g/right", "actuator": "g/right", "drive_mode": 0}],
        },
        "gripper/fitted": {
            "kind": "joint",
            "joint": "g/right",
            "actuator": "g/right",
            "mode": "range_m100_100",
            "range_override": (-0.04, None),
            "drive_mode": 0,
            "followers": [],
        },
        "gripper/spin": {
            "kind": "joint",
            "joint": "g/spin",
            "actuator": "g/spin",
            "mode": "range_m100_100",
            "range_override": (None, None),
            "drive_mode": 0,
            "followers": [],
        },
    }
    scene = SceneConfig(
        backdrop=False,
        attachments=[ModuleAttachment(module="gripper", mjcf=str(mjcf), prefix="g/")],
    )
    world = MujocoWorldServer(scene, specs, {}, {})
    world._build()
    world._lock = threading.Lock()
    return world


def test_a_joint_and_its_follower_span_their_own_model_limits(server):
    jaw = server._component_specs["gripper/jaw"]
    assert (jaw["range_min"], jaw["range_max"]) == pytest.approx((0.0, 0.05))
    follower = jaw["followers"][0]
    assert (follower["range_min"], follower["range_max"]) == pytest.approx((-0.05, 0.0))


def test_an_unlimited_joint_spans_one_turn(server):
    spin = server._component_specs["gripper/spin"]
    assert (spin["range_min"], spin["range_max"]) == pytest.approx((-math.pi, math.pi))


def test_an_override_replaces_only_the_end_it_sets(server):
    fitted = server._component_specs["gripper/fitted"]
    assert (fitted["range_min"], fitted["range_max"]) == pytest.approx((-0.04, 0.0))


def test_an_override_is_logged_beside_the_model_limits(server, caplog):
    with caplog.at_level("INFO", logger="mujoco_world"):
        server._build()
    logged = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("gripper/fitted:") and "-0.04..0 overrides" in m for m in logged)
    assert not any(m.startswith("gripper/jaw:") for m in logged)


def test_a_reading_is_normalized_against_the_model_limits(server):
    server.data.qpos[server._component_specs["gripper/jaw"]["qposadr"]] = 0.0375
    assert server.read("gripper", float, ["jaw"])["jaw"] == pytest.approx(50.0)
