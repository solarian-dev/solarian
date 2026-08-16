"""Unit tests for ee_site placement (body+axis and site+offset modes)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from solarian.sim.onshape import (
    EESiteSpec,
    SolarianExportSpec,
    _add_ee_site,
    _site_pose_with_offset,
)


SAMPLE_RAW = """\
<?xml version='1.0' encoding='utf-8'?>
<mujoco model="test">
  <worldbody>
    <body name="wrist">
      <site name="flange" pos="0.01 0.02 0.03" quat="1 0 0 0" />
      <body name="child">
        <site name="side_bracket" pos="0 0 0" quat="0.5 0.5 0.5 0.5" />
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _worldbody(xml: str = SAMPLE_RAW):
    return ET.fromstring(xml).find("worldbody")


def test_ee_site_spec_requires_body_or_site() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        EESiteSpec(name="gripper")
    with pytest.raises(ValueError, match="exactly one"):
        EESiteSpec(name="gripper", body="wrist", site="flange")


def test_ee_site_spec_offset_must_be_xyz() -> None:
    with pytest.raises(ValueError, match="length-3"):
        EESiteSpec(name="gripper", site="flange", offset=(1.0, 2.0))


def test_ee_site_spec_quat_must_be_wxyz() -> None:
    with pytest.raises(ValueError, match="length-4"):
        EESiteSpec(name="gripper", site="flange", quat=(1.0, 0.0, 0.0))


def test_from_config_parses_site_offset() -> None:
    spec = SolarianExportSpec.from_config(
        {
            "solarian": {
                "ee_site": {
                    "name": "gripper",
                    "site": "flange",
                    "offset": [0.0, 0.0, 0.05],
                }
            }
        }
    )
    assert spec.ee_site is not None
    assert spec.ee_site.site == "flange"
    assert spec.ee_site.body is None
    assert spec.ee_site.offset == (0.0, 0.0, 0.05)
    assert spec.ee_site.quat is None


def test_from_config_parses_site_quat() -> None:
    spec = SolarianExportSpec.from_config(
        {
            "solarian": {
                "ee_site": {
                    "name": "gripper",
                    "site": "flange",
                    "offset": [-0.03946, 0.00394, 0.1538],
                    "quat": [0.0, 0.0, 0.0, 1.0],
                }
            }
        }
    )
    assert spec.ee_site is not None
    assert spec.ee_site.offset == (-0.03946, 0.00394, 0.1538)
    assert spec.ee_site.quat == (0.0, 0.0, 0.0, 1.0)


def test_site_pose_with_offset_identity_quat() -> None:
    site = ET.Element("site", pos="1 0 0", quat="1 0 0 0")
    pos, quat = _site_pose_with_offset(site, (0.0, 0.1, 0.2))
    np.testing.assert_allclose(pos, [1.0, 0.1, 0.2])
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0])


def test_site_pose_with_offset_rotates_in_site_frame() -> None:
    # 90° about z: site-local +x maps to body +y.
    site = ET.Element("site", pos="0 0 0", quat="0.70710678118 0 0 0.70710678118")
    pos, _ = _site_pose_with_offset(site, (0.1, 0.0, 0.0))
    np.testing.assert_allclose(pos, [0.0, 0.1, 0.0], atol=1e-6)


def test_site_pose_with_offset_right_composes_local_quat() -> None:
    # Identity site, Rz(180) local: origin unchanged, X/Y flip, Z same.
    site = ET.Element("site", pos="1 2 3", quat="1 0 0 0")
    pos, quat = _site_pose_with_offset(site, (0.1, 0.0, 0.0), local_quat=(0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(pos, [1.1, 2.0, 3.0])
    from transforms3d.quaternions import quat2mat

    R = quat2mat(quat)
    np.testing.assert_allclose(R, np.diag([-1.0, -1.0, 1.0]), atol=1e-6)


def test_site_pose_local_quat_does_not_move_origin_after_offset() -> None:
    # Offset is applied in the *source* site frame, then quat only spins axes.
    site = ET.Element("site", pos="0 0 0", quat="1 0 0 0")
    pos_no_q, _ = _site_pose_with_offset(site, (0.0, 0.0, 0.05))
    pos_q, _ = _site_pose_with_offset(
        site, (0.0, 0.0, 0.05), local_quat=(0.0, 0.0, 0.0, 1.0)
    )
    np.testing.assert_allclose(pos_q, pos_no_q)


def test_add_ee_site_from_site_plus_offset() -> None:
    worldbody = _worldbody()
    spec = SolarianExportSpec(
        ee_site=EESiteSpec(
            name="gripper",
            site="flange",
            offset=(0.0, 0.0, 0.05),
        )
    )
    _add_ee_site(worldbody, spec, mesh_dir="/unused")

    wrist = next(b for b in worldbody.iter("body") if b.get("name") == "wrist")
    tip = next(s for s in wrist.findall("site") if s.get("name") == "gripper")
    # flange at (0.01, 0.02, 0.03), identity quat, +0.05 along site z
    assert tip.get("pos") == "0.010000 0.020000 0.080000"
    assert tip.get("quat") == "1.000000 0.000000 0.000000 0.000000"
    assert tip.get("group") == "1"


def test_add_ee_site_from_site_plus_offset_and_quat() -> None:
    worldbody = _worldbody()
    spec = SolarianExportSpec(
        ee_site=EESiteSpec(
            name="gripper",
            site="flange",
            offset=(0.0, 0.0, 0.05),
            quat=(0.0, 0.0, 0.0, 1.0),
        )
    )
    _add_ee_site(worldbody, spec, mesh_dir="/unused")

    wrist = next(b for b in worldbody.iter("body") if b.get("name") == "wrist")
    tip = next(s for s in wrist.findall("site") if s.get("name") == "gripper")
    assert tip.get("pos") == "0.010000 0.020000 0.080000"
    q = np.array([float(x) for x in tip.get("quat").split()])
    from transforms3d.quaternions import quat2mat

    np.testing.assert_allclose(quat2mat(q), np.diag([-1.0, -1.0, 1.0]), atol=1e-6)


def test_add_ee_site_missing_site_raises() -> None:
    worldbody = _worldbody()
    spec = SolarianExportSpec(
        ee_site=EESiteSpec(name="gripper", site="does_not_exist", offset=(0, 0, 0.01))
    )
    with pytest.raises(ValueError, match="does_not_exist"):
        _add_ee_site(worldbody, spec, mesh_dir="/unused")


def test_clean_writes_site_offset_ee(tmp_path: Path) -> None:
    from solarian.sim.onshape import _clean

    raw = tmp_path / "raw.xml"
    raw.write_text(SAMPLE_RAW, encoding="utf-8")
    out = tmp_path / "out" / "model.xml"
    spec = SolarianExportSpec(
        attach=False,
        strip_inertials=False,
        ee_site=EESiteSpec(name="gripper", site="flange", offset=(0.0, 0.05, 0.0)),
    )
    _clean(str(raw), str(out), spec, meshdir="meshes")

    root = ET.parse(out).getroot()
    tip = next(s for s in root.iter("site") if s.get("name") == "gripper")
    assert tip.get("pos") == "0.010000 0.070000 0.030000"
