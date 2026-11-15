"""URDF → raw MJCF for cad-import, then the clean stage. No network, no MuJoCo."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from solarian.sim.cad_urdf import write_raw_mjcf
from solarian.sim.cad import SolarianExportSpec, clean_raw, export


URDF = """\
<?xml version="1.0"?>
<robot name="gripper">
  <link name="link_base">
    <visual>
      <geometry><mesh filename="package://demo/meshes/base.stl"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="package://demo/meshes/base.stl"/></geometry>
    </collision>
  </link>
  <link name="jaw">
    <visual>
      <geometry><mesh filename="meshes/jaw.stl"/></geometry>
    </visual>
  </link>
  <link name="roll"/>
  <link name="follower"/>
  <link name="horn">
    <visual>
      <origin xyz="0.01 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="meshes/horn.stl"/></geometry>
    </visual>
  </link>
  <link name="frame_flange"/>
  <link name="frame_tcp"/>
  <link name="loop_child"/>

  <joint name="dof_jaw" type="prismatic">
    <origin xyz="0 0.1 0" rpy="0 0 0"/>
    <parent link="link_base"/>
    <child link="jaw"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="0.0545"/>
  </joint>
  <joint name="dof_wrist_roll_inv" type="revolute">
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <parent link="link_base"/>
    <child link="roll"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="1.5"/>
  </joint>
  <joint name="dof_follower" type="prismatic">
    <origin xyz="0 -0.1 0" rpy="0 0 0"/>
    <parent link="link_base"/>
    <child link="follower"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.0545" upper="0"/>
    <mimic joint="dof_jaw" multiplier="-1" offset="0"/>
  </joint>
  <joint name="fix_horn" type="fixed">
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <parent link="link_base"/>
    <child link="horn"/>
  </joint>
  <joint name="frame_flange" type="fixed">
    <origin xyz="0 0 0.01" rpy="0 0 0"/>
    <parent link="link_base"/>
    <child link="frame_flange"/>
  </joint>
  <joint name="frame_tcp" type="fixed">
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
    <parent link="link_base"/>
    <child link="frame_tcp"/>
  </joint>
  <joint name="closing_loop" type="revolute">
    <origin xyz="0.3 0 0" rpy="0 0 0"/>
    <parent link="link_base"/>
    <child link="loop_child"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1" upper="1"/>
  </joint>
</robot>
"""


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split()]


def _assembly(tmp_path: Path) -> Path:
    assembly = tmp_path / "gripper_description"
    meshes = assembly / "meshes"
    meshes.mkdir(parents=True)
    for name in ("base.stl", "jaw.stl", "horn.stl"):
        (meshes / name).write_bytes(b"solid " + name.encode())
    (assembly / "robot.urdf").write_text(URDF, encoding="utf-8")
    (assembly / "config.json").write_text(
        """{
            "robot_name": "gripper",
            "output_filename": "gripper_raw",
            "assets_directory": "meshes",
            "solarian": {
                "dof_mates": ["jaw"],
                "base_dof": "jaw",
                "output_xml": "../gripper.xml",
                "meshdir": "gripper_description/meshes",
                "density": 522
            }
        }
        """,
        encoding="utf-8",
    )
    return assembly


def test_prefixes_mimic_merge_and_frames_survive_the_clean_stage(tmp_path: Path):
    assembly = _assembly(tmp_path)
    raw = write_raw_mjcf(str(assembly), {
        "robot_name": "gripper",
        "output_filename": "gripper_raw",
        "assets_directory": "meshes",
    })
    root = ET.parse(raw).getroot()

    assert (assembly / "meshes" / "base.stl").read_bytes().startswith(b"solid ")
    assert (assembly / "meshes" / "jaw.stl").is_file()
    assert (assembly / "meshes" / "horn.stl").is_file()

    bodies = {body.get("name") for body in root.iter("body")}
    assert "base" in bodies
    assert "link_base" not in bodies
    assert "horn" not in bodies
    assert "frame_tcp" not in bodies

    joints = {joint.get("name"): joint for joint in root.find("worldbody").iter("joint")}
    assert set(joints) == {"jaw", "wrist_roll", "follower"}
    assert joints["jaw"].get("type") == "slide"
    assert _floats(joints["jaw"].get("axis")) == [0, 0, 1]
    low, high = _floats(joints["jaw"].get("range"))
    assert low == 0
    assert abs(high - 0.0545) < 1e-9

    assert _floats(joints["wrist_roll"].get("axis")) == [0, 0, -1]
    low, high = _floats(joints["wrist_roll"].get("range"))
    assert abs(low - -1.5) < 1e-9
    assert abs(high - 0.5) < 1e-9

    base = next(body for body in root.iter("body") if body.get("name") == "base")
    horn = next(geom for geom in base.findall("geom") if geom.get("mesh") == "horn")
    hx, hy, hz = _floats(horn.get("pos"))
    assert abs(hx - 0.01) < 1e-9 and hy == 0 and abs(hz - 0.05) < 1e-9
    assert {geom.get("class") for geom in base.findall("geom") if geom.get("mesh") == "horn"} == {
        "visual",
        "collision",
    }

    sites = {site.get("name") for site in root.iter("site")}
    assert {"flange", "tcp", "closing_loop_1", "closing_loop_2"} <= sites

    actuators = {act.get("joint") for act in root.find("actuator")}
    assert actuators == {"jaw", "wrist_roll"}

    equality = root.find("equality")
    mimic = next(item for item in equality if item.tag == "joint")
    assert mimic.get("joint1") == "follower"
    assert mimic.get("joint2") == "jaw"
    poly = _floats(mimic.get("polycoef"))
    assert poly[0] == 0 and poly[1] == -1
    connect = next(item for item in equality if item.tag == "connect")
    assert connect.get("site1") == "closing_loop_1"
    assert connect.get("site2") == "closing_loop_2"

    out = tmp_path / "gripper.xml"
    clean_raw(
        raw,
        str(out),
        SolarianExportSpec(density=522, meshdir="gripper_description/meshes"),
        "gripper_description/meshes",
    )
    cleaned = ET.parse(out).getroot()
    assert cleaned.find(".//freejoint") is None
    collision = next(
        geom
        for default in cleaned.iter("default")
        if default.get("class") == "collision"
        for geom in default.findall("geom")
    )
    assert float(collision.get("density")) == 522


def test_a_zip_download_is_unpacked_next_to_config(tmp_path: Path):
    assembly = tmp_path / "from_zip"
    assembly.mkdir()
    staging = tmp_path / "staging"
    (staging / "urdf").mkdir(parents=True)
    (staging / "meshes").mkdir()
    (staging / "meshes" / "base.stl").write_bytes(b"solid base")
    (staging / "urdf" / "robot.urdf").write_text(
        """<?xml version="1.0"?>
        <robot name="arm">
          <link name="link_base">
            <visual><geometry><mesh filename="../meshes/base.stl"/></geometry></visual>
          </link>
        </robot>
        """,
        encoding="utf-8",
    )
    zip_path = tmp_path / "arm.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())

    raw = write_raw_mjcf(
        str(assembly),
        {"robot_name": "arm", "output_filename": "arm_raw", "assets_directory": "meshes"},
        urdf_path=str(zip_path),
    )
    root = ET.parse(raw).getroot()
    assert (assembly / "meshes" / "base.stl").read_bytes() == b"solid base"
    body = next(body for body in root.iter("body"))
    assert body.get("name") == "base"
    assert body.find("freejoint") is not None


def test_directory_name_and_density_need_no_config(tmp_path: Path):
    assembly = _assembly(tmp_path)
    (assembly / "config.json").unlink()
    out = export(str(assembly), density=522)
    assert Path(out) == (tmp_path / "gripper.xml").resolve()
    assert (assembly / "gripper_raw.xml").is_file()
    assert (assembly / "meshes" / "jaw.stl").is_file()
    collision = next(
        geom
        for default in ET.parse(out).getroot().iter("default")
        if default.get("class") == "collision"
        for geom in default.findall("geom")
    )
    assert float(collision.get("density")) == 522


def test_reimport_replaces_a_mesh_the_previous_import_copied(tmp_path: Path):
    assembly = tmp_path / "gripper_description"
    (assembly / "meshes").mkdir(parents=True)
    (assembly / "meshes" / "base.stl").write_bytes(b"old")
    (assembly / "fresh" / "meshes").mkdir(parents=True)
    (assembly / "fresh" / "meshes" / "base.stl").write_bytes(b"new")
    (assembly / "robot.urdf").write_text(
        """<?xml version="1.0"?>
        <robot name="arm">
          <link name="base">
            <visual><geometry><mesh filename="package://fresh/meshes/base.stl"/></geometry></visual>
          </link>
        </robot>
        """,
        encoding="utf-8",
    )
    write_raw_mjcf(str(assembly), {"assets_directory": "meshes", "output_filename": "arm_raw", "robot_name": "arm"})
    assert (assembly / "meshes" / "base.stl").read_bytes() == b"new"


def test_two_different_meshes_with_one_name_are_an_error(tmp_path: Path):
    assembly = tmp_path / "gripper_description"
    (assembly / "a").mkdir(parents=True)
    (assembly / "b").mkdir(parents=True)
    (assembly / "a" / "base.stl").write_bytes(b"one")
    (assembly / "b" / "base.stl").write_bytes(b"two")
    (assembly / "robot.urdf").write_text(
        """<?xml version="1.0"?>
        <robot name="arm">
          <link name="base">
            <visual><geometry><mesh filename="package://a/base.stl"/></geometry></visual>
          </link>
          <link name="tip">
            <visual><geometry><mesh filename="package://b/base.stl"/></geometry></visual>
          </link>
          <joint name="fix_tip" type="fixed">
            <parent link="base"/><child link="tip"/>
          </joint>
        </robot>
        """,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="both named base.stl"):
        write_raw_mjcf(str(assembly), {"assets_directory": "meshes", "output_filename": "arm_raw", "robot_name": "arm"})


def test_config_yaml_is_written_once_and_survives_the_next_import(tmp_path: Path):
    assembly = _assembly(tmp_path)
    (assembly / "config.json").unlink()
    export(str(assembly))
    path = assembly / "config.yaml"
    edited = path.read_text(encoding="utf-8").replace("kp: 50", "kp: 200", 1)
    path.write_text(edited, encoding="utf-8")
    export(str(assembly))
    assert path.read_text(encoding="utf-8") == edited
    position = next(
        element
        for default in ET.parse(tmp_path / "gripper.xml").getroot().iter("default")
        for element in default.findall("position")
    )
    assert position.get("kp") == "200"
