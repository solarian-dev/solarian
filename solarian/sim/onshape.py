"""Export an Onshape assembly into an attachable MuJoCo (MJCF) sim model.

This generalizes the two bespoke scripts originally written for the SO-106 arm
(``export_so106.py`` + ``clean_mjcf.py``) into a single, config-driven pipeline
that any assembly can reuse. It is the engine behind the ``solarian
onshape-export`` CLI command.

The pipeline has two stages:

1. **Raw export** (needs the Onshape API + network). Uses
   `onshape-to-robot <https://onshape-to-robot.readthedocs.io>`_ to pull the
   assembly and write a raw MJCF + STL meshes. Two things onshape-to-robot does
   not do out of the box are patched in, driven by the ``solarian`` block of the
   assembly's ``config.json``:

   * onshape-to-robot only treats a mate as a *joint* when its name is prefixed
     with ``dof_``. Rather than editing the CAD, the ``dof_mates`` are renamed to
     ``dof_<name>`` on the fly in the API responses (both the assembly and the
     features response, the latter being where joint limits are read).
   * the kinematic tree is rooted at the assembly's base (``base_dof``) so the
     model attaches to the world / a parent body at its base, not at the tool.

   Free-floating shells that carry no mate (duplicate CAD bodies) are dropped so
   the exported model is a single connected mechanism.

2. **Clean** (no network). Turns the raw MJCF into an *attachable* model:
   drops the base ``<freejoint>`` so it can be composed into a scene, infers
   link mass/inertia from the collision meshes at a fixed density (the CAD
   assigns no materials), points ``meshdir`` at the exported meshes, makes the
   mechanism's own collision geoms non-self-colliding, and optionally adds an
   end-effector ``<site>`` at a flange.

Credentials are read by onshape-to-robot from the environment
(``ONSHAPE_API``, ``ONSHAPE_ACCESS_KEY``, ``ONSHAPE_SECRET_KEY``); the CLI
command sets these up (see :mod:`solarian.cli.commands.onshape_export`).

MuJoCo / onshape-to-robot / numpy-stl are imported lazily inside the functions,
so importing this module never requires the ``[sim]`` extras.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional

import commentjson


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

#: Uniform density (kg/m^3) used to infer link masses from solid mesh volumes
#: when the CAD assigns no materials. ~1400 gives a ~1 kg SO-ARM-sized part.
DEFAULT_DENSITY = 1400.0


@dataclass
class EESiteSpec:
    """An end-effector ``<site>`` added to a body of the cleaned model.

    The site is placed at the centre of the body's far face along ``axis`` (a
    tool flange / grasp frame). IK backends reference it by ``name``.
    """

    #: Site name (referenced by e.g. ``MinkIK(frame=...)``).
    name: str
    #: Body whose collision geometry defines the flange.
    body: str
    #: Tool axis pointing out of the flange: ``x``/``y``/``z`` picks the extreme
    #: face with the larger absolute extent (sign auto-detected); prefix with
    #: ``+``/``-`` (e.g. ``-z``) to force a side.
    axis: str = "z"
    #: Visual marker size and viewer group.
    size: float = 0.005
    group: int = 1


@dataclass
class SolarianExportSpec:
    """The ``solarian`` block of an assembly ``config.json``.

    Everything specific to turning an Onshape assembly into an attachable
    Solarian sim model. All fields have defaults that reproduce the SO-106 arm's
    original behaviour.
    """

    #: Revolute / prismatic mates (by CAD name) to expose as actuated joints.
    dof_mates: list[str] = field(default_factory=list)
    #: Root the kinematic tree at the body of this DOF that no other DOF
    #: touches (the mechanism base). ``None`` keeps onshape-to-robot's default.
    base_dof: Optional[str] = None
    #: Drop links that have no mate at all (duplicate free-floating shells).
    drop_unmated_links: bool = True
    #: Where to write the cleaned, attachable MJCF, relative to the assembly
    #: dir. Defaults to ``../<robot_name>.xml`` (i.e. the project ``assets/``).
    output_xml: Optional[str] = None
    #: ``meshdir`` written into the cleaned MJCF (relative to ``output_xml``).
    #: Defaults to ``<assembly_dir_name>/<assets_directory>``.
    meshdir: Optional[str] = None
    #: Density for mass/inertia inference from the collision meshes.
    density: float = DEFAULT_DENSITY
    #: Drop the base ``<freejoint>`` so the model attaches rigidly into a scene.
    attach: bool = True
    #: Strip the placeholder inertials and infer mass/inertia from geometry.
    strip_inertials: bool = True
    #: Allow the mechanism's own links to collide with each other. Off by
    #: default: full-link collision meshes overlap by design at every joint and
    #: MuJoCo does not auto-exclude parent/child contacts, so self-collision
    #: produces permanent interpenetrations that fight the actuators.
    self_collision: bool = False
    #: Optional end-effector site.
    ee_site: Optional[EESiteSpec] = None

    @classmethod
    def from_config(cls, config: dict) -> "SolarianExportSpec":
        block = dict(config.get("solarian", {}))
        ee = block.pop("ee_site", None)
        spec = cls(**block)
        if ee is not None:
            spec.ee_site = EESiteSpec(**ee)
        return spec


# --------------------------------------------------------------------------- #
# Stage 1: raw Onshape export (patched)                                       #
# --------------------------------------------------------------------------- #

def _prefix_dof_mates(assembly: dict, features: dict, dof_mates: set[str]):
    """Rename the ``dof_mates`` to ``dof_<name>`` in the API responses so
    onshape-to-robot treats them as joints (in-place)."""
    for feature in assembly.get("rootAssembly", {}).get("features", []):
        data = feature.get("featureData", {})
        if data.get("name") in dof_mates:
            data["name"] = f"dof_{data['name']}"
    for feature in features.get("features", []):
        msg = feature.get("message", {})
        if msg.get("name") in dof_mates:
            msg["name"] = f"dof_{msg['name']}"


@contextlib.contextmanager
def _onshape_patches(spec: SolarianExportSpec) -> Iterator[None]:
    """Monkeypatch the onshape-to-robot client/assembly for the duration of an
    export: expose ``dof_mates`` as joints and (optionally) root the tree at
    ``base_dof``. Everything is restored on exit so the patch never leaks."""
    from onshape_to_robot.onshape_api.client import Client
    from onshape_to_robot.assembly import Assembly

    dof_mates = set(spec.dof_mates)
    base_dof = spec.base_dof

    orig_get_assembly = Client.get_assembly
    orig_get_features = Client.get_features
    orig_process_mates = Assembly.process_mates

    def patched_get_assembly(self, *args, **kwargs):
        result = orig_get_assembly(self, *args, **kwargs)
        _prefix_dof_mates(result, {"features": []}, dof_mates)
        return result

    def patched_get_features(self, *args, **kwargs):
        result = orig_get_features(self, *args, **kwargs)
        _prefix_dof_mates({}, result, dof_mates)
        return result

    def patched_process_mates(self):
        orig_process_mates(self)
        if base_dof is None:
            return
        # Root the tree at the base: the body of ``base_dof`` that no other DOF
        # touches. build_trees() starts at the first body in instance_body, so
        # reorder to put the base first.
        dof_bodies = [b for dof in self.dofs for b in (dof.body1_id, dof.body2_id)]
        base_candidates = [
            b
            for dof in self.dofs
            if dof.name == base_dof
            for b in (dof.body1_id, dof.body2_id)
            if dof_bodies.count(b) == 1
        ]
        if base_candidates:
            base_body = base_candidates[0]
            reordered = {
                occ: bid for occ, bid in self.instance_body.items() if bid == base_body
            }
            for occ, bid in self.instance_body.items():
                if bid != base_body:
                    reordered[occ] = bid
            self.instance_body = reordered
            print(f"* Rooting kinematic tree at base body {base_body} ({base_dof})")

    Client.get_assembly = patched_get_assembly
    Client.get_features = patched_get_features
    Assembly.process_mates = patched_process_mates
    try:
        yield
    finally:
        Client.get_assembly = orig_get_assembly
        Client.get_features = orig_get_features
        Assembly.process_mates = orig_process_mates


def _reachable_links(robot, root) -> set[int]:
    seen: set[int] = set()
    stack = [root]
    while stack:
        link = stack.pop()
        if id(link) in seen:
            continue
        seen.add(id(link))
        for joint in robot.get_link_joints(link):
            stack.append(joint.child)
    return seen


def _drop_unmated_links(robot) -> None:
    """Keep only the single connected mechanism rooted at the base link,
    dropping free-floating duplicate shells that carry no mate."""
    chain_root = robot.base_links[0]
    chain_ids = _reachable_links(robot, chain_root)
    for link in robot.base_links:
        if id(link) not in chain_ids:
            print(f"* Dropping unmated link {link.name} {[p.name for p in link.parts]}")
    robot.base_links = [chain_root]
    robot.links = [link for link in robot.links if id(link) in chain_ids]


def _export_raw(assembly_dir: str, spec: SolarianExportSpec) -> tuple[str, "object"]:
    """Run onshape-to-robot to produce the raw MJCF. Returns (raw_xml_path, config)."""
    from onshape_to_robot.config import Config
    from onshape_to_robot.robot_builder import RobotBuilder
    from onshape_to_robot.exporter_mujoco import ExporterMuJoCo

    config = Config(assembly_dir)
    if config.output_format != "mujoco":
        raise ValueError(
            f"solarian onshape-export requires output_format=\"mujoco\" "
            f"(got {config.output_format!r})."
        )

    exporter = ExporterMuJoCo(config)

    with _onshape_patches(spec):
        robot = RobotBuilder(config).robot

    if spec.drop_unmated_links:
        _drop_unmated_links(robot)

    for processor in config.processors:
        processor.process(robot)

    raw_xml = os.path.join(
        config.output_directory, config.output_filename + "." + exporter.ext
    )
    exporter.write_xml(robot, raw_xml)
    print(f"* Wrote raw export {raw_xml}")
    return raw_xml, config


# --------------------------------------------------------------------------- #
# Stage 2: clean into an attachable MJCF                                       #
# --------------------------------------------------------------------------- #

_AXES = {"x": 0, "y": 1, "z": 2}


def _parse_axis(axis: str) -> tuple[int, int]:
    """Return (index, sign) for an axis like ``z``, ``+z`` or ``-z``. Sign 0
    means auto-detect the extreme face with the larger absolute extent."""
    axis = axis.strip().lower()
    sign = 0
    if axis[0] in "+-":
        sign = 1 if axis[0] == "+" else -1
        axis = axis[1:]
    if axis not in _AXES:
        raise ValueError(f"ee_site.axis must be one of x/y/z (got {axis!r})")
    return _AXES[axis], sign


def _flange_tip(body, mesh_dir: str, axis: str):
    """Centre of ``body``'s far face along ``axis`` (in the body frame),
    computed from its collision meshes."""
    import numpy as np
    from stl import mesh as stl_mesh
    from transforms3d.quaternions import quat2mat

    idx, sign = _parse_axis(axis)
    verts = []
    for geom in body.findall("geom"):
        if geom.get("class") != "collision":
            continue
        path = os.path.join(mesh_dir, geom.get("mesh") + ".stl")
        v = stl_mesh.Mesh.from_file(path).vectors.reshape(-1, 3)
        pos = np.array([float(x) for x in geom.get("pos", "0 0 0").split()])
        quat = np.array([float(x) for x in geom.get("quat", "1 0 0 0").split()])
        verts.append((quat2mat(quat) @ v.T).T + pos)
    pts = np.vstack(verts)
    coord = pts[:, idx]
    if sign > 0:
        tip = coord.max()
    elif sign < 0:
        tip = coord.min()
    else:
        tip = coord.min() if abs(coord.min()) > abs(coord.max()) else coord.max()
    near = pts[np.abs(coord - tip) < 0.005]
    center = near.mean(axis=0)
    center[idx] = tip
    return center


def _clean(raw_xml: str, out_xml: str, spec: SolarianExportSpec, meshdir: str) -> None:
    """Transform the raw MJCF into the attachable model at ``out_xml``."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(raw_xml)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", meshdir)
        if spec.strip_inertials:
            compiler.set("inertiafromgeom", "auto")
            # Restrict inertia inference to the collision group so visual
            # duplicates are not double-counted.
            compiler.set("inertiagrouprange", "3 3")

    # Density on the collision default so inferred link masses are non-zero;
    # disjoint contype/conaffinity so the mechanism's own links don't
    # self-collide (see SolarianExportSpec.self_collision).
    for d in root.iter("default"):
        if d.get("class") == "collision":
            g = d.find("geom")
            if g is None:
                continue
            if spec.strip_inertials:
                g.set("density", str(spec.density))
            if not spec.self_collision:
                g.set("contype", "2")
                g.set("conaffinity", "1")

    worldbody = root.find("worldbody")
    for body in worldbody.iter("body"):
        if spec.attach:
            for fj in body.findall("freejoint"):
                body.remove(fj)
        if spec.strip_inertials:
            for inert in body.findall("inertial"):
                body.remove(inert)

    if spec.ee_site is not None:
        ee = spec.ee_site
        target = None
        for body in worldbody.iter("body"):
            if body.get("name") == ee.body:
                target = body
        if target is None:
            raise ValueError(f"ee_site.body {ee.body!r} not found in the exported model")
        tip = _flange_tip(target, os.path.dirname(out_xml) + os.sep + meshdir, ee.axis)
        site = ET.SubElement(target, "site")
        site.set("name", ee.name)
        site.set("pos", "%.6f %.6f %.6f" % tuple(tip))
        site.set("group", str(ee.group))
        site.set("size", str(ee.size))
        print(f"* {ee.name} site at {ee.body} frame pos {tuple(round(float(x), 4) for x in tip)}")

    # Drop empty <equality/>.
    for eq in root.findall("equality"):
        if len(eq) == 0:
            root.remove(eq)

    os.makedirs(os.path.dirname(os.path.abspath(out_xml)), exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(out_xml, xml_declaration=True, encoding="utf-8")
    print(f"* Wrote {out_xml}")


# --------------------------------------------------------------------------- #
# Entry points                                                                 #
# --------------------------------------------------------------------------- #

def _load_config(assembly_dir: str) -> dict:
    path = assembly_dir
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No config.json found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return commentjson.load(f)


def _resolve_defaults(assembly_dir: str, config: dict, spec: SolarianExportSpec) -> tuple[str, str]:
    """Fill in ``output_xml`` / ``meshdir`` defaults; return absolute out path
    and the meshdir string to embed."""
    robot_name = config.get("robot_name") or os.path.basename(os.path.abspath(assembly_dir))
    assets_dir = config.get("assets_directory", "assets")

    out_rel = spec.output_xml or os.path.join("..", f"{robot_name}.xml")
    out_xml = os.path.normpath(os.path.join(os.path.abspath(assembly_dir), out_rel))

    if spec.meshdir is not None:
        meshdir = spec.meshdir
    else:
        # Default: point back at the assembly dir's assets from output_xml's dir.
        rel = os.path.relpath(
            os.path.join(os.path.abspath(assembly_dir), assets_dir),
            os.path.dirname(out_xml),
        )
        meshdir = rel
    return out_xml, meshdir


def _raw_xml_path(assembly_dir: str, config: dict) -> str:
    output_filename = config.get("output_filename", "robot")
    output_filename = "".join(
        c for c in output_filename if c.isalnum() or c in ("_", "-")
    ).rstrip()
    return os.path.join(os.path.abspath(assembly_dir), f"{output_filename}.xml")


def export(assembly_dir: str, *, clean_only: bool = False) -> str:
    """Run the Onshape → attachable-MJCF pipeline for ``assembly_dir``.

    ``assembly_dir`` holds a ``config.json`` (onshape-to-robot fields plus a
    ``solarian`` block, see :class:`SolarianExportSpec`). Returns the path to the
    cleaned MJCF.

    With ``clean_only=True`` the network export is skipped and only the clean
    stage is re-run on the previously-exported raw MJCF — handy for iterating on
    the cleaning options without re-hitting the Onshape API.
    """
    config = _load_config(assembly_dir)
    spec = SolarianExportSpec.from_config(config)
    out_xml, meshdir = _resolve_defaults(assembly_dir, config, spec)

    if clean_only:
        raw_xml = _raw_xml_path(assembly_dir, config)
        if not os.path.exists(raw_xml):
            raise FileNotFoundError(
                f"--clean-only needs a previous raw export at {raw_xml}; "
                f"run without --clean-only first."
            )
    else:
        raw_xml, _ = _export_raw(assembly_dir, spec)

    _clean(raw_xml, out_xml, spec, meshdir)
    return out_xml


def clean_raw(raw_xml: str, out_xml: str, spec: SolarianExportSpec, meshdir: str) -> str:
    """Run only the clean stage on an explicit raw MJCF path (used by tests)."""
    _clean(raw_xml, out_xml, spec, meshdir)
    return out_xml
