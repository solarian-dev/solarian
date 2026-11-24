"""IKActuatorBus: one joint read per tick, and no solve it can skip."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from solarian.core.data import PosData

_MJCF = """\
<mujoco>
  <worldbody>
    <body name="base">
      <joint name="slide" type="slide" axis="1 0 0" range="-1 1"/>
      <geom type="sphere" size="0.01"/>
      <site name="flange" pos="0 0 0" quat="1 0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""

POSE = ["x", "y", "z", "rx", "ry", "rz"]


class _JointBus:
    """One slide joint in normalized units; counts reads."""

    port = "test"
    is_connected = True

    def __init__(self, norm: float):
        self.norm = norm
        self.reads = 0

    def read(self, data_type, names, normalize=True):
        self.reads += 1
        return {name: self.norm for name in names}

    def write(self, values, normalize=True):
        self.norm = float(values["slide"].value)


@pytest.fixture
def ik(tmp_path: Path):
    pytest.importorskip("mink")
    from solarian.core.ik.bus import IKActuatorBus
    from solarian.core.ik.mink_backend import MinkBackend

    path = tmp_path / "arm.xml"
    path.write_text(_MJCF)
    # A posture cost high enough to hold the solution millimetres off the target.
    backend = MinkBackend(
        mjcf=str(path),
        joint_actuator_names=["slide"],
        joint_model_names=["slide"],
        frame_name="flange",
        posture_cost=0.1,
    )
    solves = []
    solve = backend.solve
    backend.solve = lambda target, dt: solves.append(1) or solve(target, dt)
    inner = _JointBus(norm=30.0)  # 0.3 m
    bus = IKActuatorBus(
        inner,
        {"slide": SimpleNamespace(cfg=SimpleNamespace(mode="range_m100_100"))},
        {name: SimpleNamespace(cfg=SimpleNamespace(id=i)) for i, name in enumerate(POSE)},
        backend,
    )
    return bus, inner, solves


def _pose(x: float) -> dict[str, PosData]:
    return {name: PosData(value=value) for name, value in zip(POSE, [x, 0, 0, 0, 0, 0])}


def test_a_target_the_posture_holds_off_ends_the_solve_early(ik):
    bus, _, solves = ik
    bus.write(_pose(0.3))
    assert 0 < len(solves) < bus.ik_iterations


def test_holding_a_target_reuses_its_solution(ik):
    bus, inner, solves = ik
    bus.write(_pose(0.3))
    commanded, solved = inner.norm, len(solves)
    bus.write(_pose(0.3))
    assert len(solves) == solved
    assert inner.norm == commanded
    bus.write(_pose(0.31))
    assert len(solves) > solved


def test_the_observation_read_serves_the_next_write(ik):
    bus, inner, _ = ik
    observed = bus.read(PosData, ["slide", *POSE])
    assert observed["slide"] == 30.0
    assert observed["x"] == pytest.approx(0.3)
    bus.write(_pose(0.3))
    assert inner.reads == 1
