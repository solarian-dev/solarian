"""MinkBackend applies a rigid tool offset in FK and inverts it for IK."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from solarian.core.ik import FrameOffset
from solarian.core.pose import quat_rotate, rotvec_to_quat

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

OFFSET = FrameOffset(pos=(0.0, 0.0, 0.2), quat=(1.0, 0.0, 0.0, 0.0))


@pytest.fixture
def backends(tmp_path: Path):
    pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    from solarian.core.ik.mink_backend import MinkBackend

    path = tmp_path / "arm.xml"
    path.write_text(_MJCF)
    kwargs = dict(
        mjcf=str(path),
        joint_actuator_names=["slide"],
        joint_model_names=["slide"],
        frame_name="flange",
    )
    flange = MinkBackend(**kwargs)
    tcp = MinkBackend(**kwargs, tcp_offset=OFFSET)
    return flange, tcp


def test_get_pose_composes_offset(backends):
    flange, tcp = backends
    q = np.array([0.3])
    flange.set_q(q)
    tcp.set_q(q)
    fp = flange.get_pose()
    tp = tcp.get_pose()
    expected = fp[:3] + quat_rotate(rotvec_to_quat(fp[3:6]), OFFSET.pos)
    assert tp[:3] == pytest.approx(expected, abs=1e-9)
    assert tp[3:6] == pytest.approx(fp[3:6], abs=1e-9)


def test_solve_inverts_offset(backends):
    flange, tcp = backends
    q = np.array([0.1])
    flange.set_q(q)
    tcp.set_q(q)
    v_tcp = tcp.solve(tcp.get_pose(), dt=0.01)
    v_flange = flange.solve(flange.get_pose(), dt=0.01)
    assert np.linalg.norm(v_tcp) == pytest.approx(np.linalg.norm(v_flange), abs=1e-6)
