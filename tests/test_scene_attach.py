"""Unit tests for ModuleAttachment body-pose and site-mate composition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

mujoco = pytest.importorskip("mujoco")

from solarian.sim import ModuleAttachment, SceneConfig
from solarian.sim.scene import compose_scene

PARENT_MJCF = """\
<mujoco model="arm">
  <worldbody>
    <body name="flange" pos="0 0 1">
      <site name="side_bracket" pos="0.1 0 0" quat="1 0 0 0"/>
      <geom type="sphere" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

CHILD_MJCF = """\
<mujoco model="gripper">
  <worldbody>
    <body name="base" pos="0 0 0">
      <site name="side_bracket" pos="0 0.2 0" quat="0.707107 0 0.707107 0"/>
      <geom type="sphere" size="0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def mjcf_pair(tmp_path: Path) -> tuple[Path, Path]:
    parent = tmp_path / "arm.xml"
    child = tmp_path / "gripper.xml"
    parent.write_text(PARENT_MJCF)
    child.write_text(CHILD_MJCF)
    return parent, child


def _site_pose(model, data, name: str):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    assert sid >= 0, name
    return data.site_xpos[sid].copy(), data.site_xmat[sid].copy()


def test_sites_must_be_paired():
    with pytest.raises(ValidationError, match="parent_site and child_site"):
        ModuleAttachment(
            module="g",
            mjcf="x.xml",
            parent_site="right/side_bracket",
        )
    with pytest.raises(ValidationError, match="parent_site and child_site"):
        ModuleAttachment(
            module="g",
            mjcf="x.xml",
            child_site="side_bracket",
        )


def test_body_mode_still_attaches_at_pos_quat(mjcf_pair, tmp_path, monkeypatch):
    parent, child = mjcf_pair
    monkeypatch.setenv("SOLARIAN_PROJECT_PATH", str(tmp_path))
    scene = SceneConfig(
        attachments=[
            ModuleAttachment(
                module="arm",
                mjcf=str(parent),
                parent_body="world",
                pos=[0.5, 0.0, 0.0],
                prefix="arm/",
            ),
        ]
    )
    model = compose_scene(scene)
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    pos, _ = _site_pose(model, data, "arm/side_bracket")
    np.testing.assert_allclose(pos, [0.6, 0.0, 1.0], atol=1e-9)


def test_site_mate_coincides_frames(mjcf_pair, tmp_path, monkeypatch):
    parent, child = mjcf_pair
    monkeypatch.setenv("SOLARIAN_PROJECT_PATH", str(tmp_path))
    scene = SceneConfig(
        attachments=[
            ModuleAttachment(
                module="arm",
                mjcf=str(parent),
                prefix="arm/",
            ),
            ModuleAttachment(
                module="gripper",
                mjcf=str(child),
                parent_site="arm/side_bracket",
                child_site="side_bracket",
                prefix="gripper/",
            ),
        ]
    )
    model = compose_scene(scene)
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)

    p_pos, p_mat = _site_pose(model, data, "arm/side_bracket")
    c_pos, c_mat = _site_pose(model, data, "gripper/side_bracket")
    np.testing.assert_allclose(c_pos, p_pos, atol=1e-9)
    np.testing.assert_allclose(c_mat, p_mat, atol=1e-9)


def test_site_mate_applies_offset_in_parent_site_frame(
    mjcf_pair, tmp_path, monkeypatch
):
    parent, child = mjcf_pair
    monkeypatch.setenv("SOLARIAN_PROJECT_PATH", str(tmp_path))
    scene = SceneConfig(
        attachments=[
            ModuleAttachment(module="arm", mjcf=str(parent), prefix="arm/"),
            ModuleAttachment(
                module="gripper",
                mjcf=str(child),
                parent_site="arm/side_bracket",
                child_site="side_bracket",
                pos=[0.0, 0.0, 0.05],
                prefix="gripper/",
            ),
        ]
    )
    model = compose_scene(scene)
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)

    p_pos, _ = _site_pose(model, data, "arm/side_bracket")
    c_pos, _ = _site_pose(model, data, "gripper/side_bracket")
    # parent site axes are identity in world after compose; +Z offset → +0.05 world Z
    np.testing.assert_allclose(c_pos, p_pos + [0.0, 0.0, 0.05], atol=1e-9)


def test_missing_parent_site_raises(mjcf_pair, tmp_path, monkeypatch):
    _, child = mjcf_pair
    monkeypatch.setenv("SOLARIAN_PROJECT_PATH", str(tmp_path))
    scene = SceneConfig(
        attachments=[
            ModuleAttachment(
                module="gripper",
                mjcf=str(child),
                parent_site="missing/side_bracket",
                child_site="side_bracket",
                prefix="gripper/",
            ),
        ]
    )
    with pytest.raises(ValueError, match="missing/side_bracket"):
        compose_scene(scene)
