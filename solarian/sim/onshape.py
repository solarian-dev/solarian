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
   * stale ``joint_properties.*.limits`` in ``config.json`` are ignored during
     the pull (Onshape mate limits win); after cleaning, those CAD limits are
     written back into ``config.json``. Set ``solarian.prefer_config_limits``
     to keep config.json as the source of truth instead.

   Free-floating shells that carry no mate (duplicate CAD bodies) are dropped so
   the exported model is a single connected mechanism.

2. **Clean** (no network). Turns the raw MJCF into an *attachable* model:
   drops the base ``<freejoint>`` so it can be composed into a scene, infers
   link mass/inertia from the collision meshes at a fixed density (the CAD
   assigns no materials), points ``meshdir`` at the exported meshes, and makes
   the mechanism's own collision geoms non-self-colliding. CAD Frames
   (``flange``, ``tcp``, ``camera``, …) are already ``<site>`` elements in the
   raw export — the clean stage does not invent extra sites.

Credentials are read by onshape-to-robot from the environment
(``ONSHAPE_API``, ``ONSHAPE_ACCESS_KEY``, ``ONSHAPE_SECRET_KEY``); the CLI
command sets these up (see :mod:`solarian.cli.commands.onshape_export`).

MuJoCo / onshape-to-robot / numpy-stl are imported lazily inside the functions,
so importing this module never requires the ``[sim]`` extras.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
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
    #: When ``False`` (default), ``joint_properties.*.limits`` in config.json are
    #: ignored during the Onshape pull so mate limits from the assembly win;
    #: after export those CAD limits are written back into config.json. Set
    #: ``True`` only when you intentionally want config.json to override CAD.
    prefer_config_limits: bool = False

    @classmethod
    def from_config(cls, config: dict) -> "SolarianExportSpec":
        return cls(**dict(config.get("solarian", {})))


def clear_joint_property_limits(joint_properties: dict) -> list[str]:
    """Remove ``limits`` overrides from a ``joint_properties`` mapping (in-place).

    Returns the joint names (including ``default`` / ``*``) whose ``limits``
    entry was cleared. Leaves other properties (``range``, effort, …) intact so
    onshape-to-robot still emits ``range=`` from the CAD mate limits.
    """
    cleared: list[str] = []
    if not isinstance(joint_properties, dict):
        return cleared
    for name, entry in joint_properties.items():
        if not isinstance(entry, dict) or "limits" not in entry:
            continue
        del entry["limits"]
        cleared.append(name)
    return cleared


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

    # onshape-to-robot treats joint_properties.*.limits as an override of the
    # mate limits. Default workflow: drop those overrides so the assembly wins,
    # then rewrite config.json from the exported MJCF after cleaning.
    if not spec.prefer_config_limits:
        cleared = clear_joint_property_limits(config.joint_properties)
        if cleared:
            print(
                "* Using Onshape mate limits "
                f"(cleared joint_properties limits for: {', '.join(cleared)})"
            )
    else:
        print("* prefer_config_limits=true: joint_properties.*.limits override CAD")

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

    # Drop empty <equality/>.
    for eq in root.findall("equality"):
        if len(eq) == 0:
            root.remove(eq)

    os.makedirs(os.path.dirname(os.path.abspath(out_xml)), exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(out_xml, xml_declaration=True, encoding="utf-8")
    print(f"* Wrote {out_xml}")


# --------------------------------------------------------------------------- #
# Sync MJCF joint ranges → module YAML + config.json                          #
# --------------------------------------------------------------------------- #

#: Actuator keys that are never MJCF joints (Cartesian IK components).
_CARTESIAN_ACTUATORS = frozenset({"x", "y", "z", "rx", "ry", "rz"})

_ACTUATOR_KEY_RE = re.compile(r"^(\s*)([A-Za-z_][\w]*)\s*:\s*(?:#.*)?$")
_SIM_JOINT_RE = re.compile(
    r"""^(\s*)joint\s*:\s*["']?([^"'#\n]+?)["']?\s*(?:#.*)?$"""
)
_RANGE_MIN_RE = re.compile(r"^(\s*)range_min\s*:\s*[^\n#]*")
_RANGE_MAX_RE = re.compile(r"^(\s*)range_max\s*:\s*[^\n#]*")


def _format_limit(value: float) -> str:
    """Compact float formatting that matches typical MJCF / yaml limit style."""
    text = f"{value:.10g}"
    if text in ("-0", "-0.0"):
        return "0"
    return text


def _bare_joint_name(joint_ref: str) -> str:
    """Strip an attach prefix: ``right/shoulder_pan`` → ``shoulder_pan``."""
    return joint_ref.strip().strip("\"'").rsplit("/", 1)[-1]


def _split_newline(raw: str) -> tuple[str, str]:
    if raw.endswith("\r\n"):
        return raw[:-2], "\r\n"
    if raw.endswith("\n"):
        return raw[:-1], "\n"
    return raw, ""


_JOINT_TAG_RE = re.compile(r"<joint\b([^>]*)/?>", re.IGNORECASE)
_ATTR_NAME_RE = re.compile(r"""\bname\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_ATTR_RANGE_RE = re.compile(r"""\brange\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


def read_joint_ranges(mjcf_path: str) -> dict[str, tuple[float, float]]:
    """Parse ``<joint name=... range="lo hi">`` from an MJCF file.

    Joints without a ``range`` attribute are skipped. Returns
    ``{joint_name: (lo, hi)}``.

    Uses a regex over joint tags rather than a full XML parse so hand-edited
    MJCF with non-strict comments (e.g. ``--`` inside ``<!-- ... -->``) still
    works — MuJoCo accepts those files; ElementTree does not.
    """
    with open(mjcf_path, "r", encoding="utf-8") as f:
        text = f.read()
    ranges: dict[str, tuple[float, float]] = {}
    for tag in _JOINT_TAG_RE.finditer(text):
        attrs = tag.group(1)
        name_m = _ATTR_NAME_RE.search(attrs)
        range_m = _ATTR_RANGE_RE.search(attrs)
        if not name_m or not range_m:
            continue
        parts = range_m.group(1).split()
        if len(parts) != 2:
            continue
        ranges[name_m.group(1)] = (float(parts[0]), float(parts[1]))
    return ranges


def _discover_yaml_actuator_joints(lines: list[str]) -> dict[str, str]:
    """Map actuator key → bare MJCF joint name for actuators with sim ranges."""
    from solarian.core.module_yaml import discover_actuator_sim_joints

    return discover_actuator_sim_joints(lines)


def sync_module_yaml(
    yaml_path: str, ranges: dict[str, tuple[float, float]]
) -> list[str]:
    """Surgically update ``sim_range_min`` / ``sim_range_max`` in a module yaml."""
    from solarian.core.module_yaml import sync_sim_ranges

    return sync_sim_ranges(yaml_path, ranges)


def sync_joint_properties(
    config_path: str, ranges: dict[str, tuple[float, float]]
) -> list[str]:
    """Rewrite ``joint_properties.<name>.limits`` from MJCF ranges.

    Only updates joints that already exist in ``joint_properties``; does not
    invent new entries. Returns the list of updated joint names.
    """
    path = os.path.abspath(config_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config.json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = commentjson.load(f)

    jp = config.get("joint_properties")
    if not isinstance(jp, dict):
        return []

    updated: list[str] = []
    for name, (lo, hi) in ranges.items():
        entry = jp.get(name)
        if not isinstance(entry, dict):
            continue
        entry["limits"] = [lo, hi]
        updated.append(name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        f.write("\n")
    return updated


def sync_limits_from_mjcf(
    mjcf_path: str,
    *,
    config_path: str,
    yaml_path: str | None = None,
) -> dict[str, list[str]]:
    """Sync MJCF joint ranges into the assembly config.json (and optionally a module yaml).

    ``config.json`` ``joint_properties.*.limits`` are always rewritten from the
    MJCF so they mirror the Onshape mate limits that produced the export.
    When ``yaml_path`` is set, ``sim_range_min`` / ``sim_range_max`` are updated too.

    Returns ``{"yaml": [...], "config": [...], "unmatched_mjcf": [...],
    "unmatched_yaml": [...]}`` for the caller to print a summary.
    """
    ranges = read_joint_ranges(mjcf_path)
    config_updated = sync_joint_properties(config_path, ranges)
    yaml_updated: list[str] = []
    unmatched_mjcf: list[str] = []
    unmatched_yaml: list[str] = []

    if yaml_path is not None:
        yaml_updated = sync_module_yaml(yaml_path, ranges)
        with open(yaml_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines(keepends=True)
        actuator_joints = _discover_yaml_actuator_joints(lines)
        yaml_target_joints = set(actuator_joints.values())

        unmatched_mjcf = sorted(name for name in ranges if name not in yaml_target_joints)
        unmatched_yaml = sorted(
            actuator
            for actuator, joint in actuator_joints.items()
            if joint not in ranges
            and joint not in _CARTESIAN_ACTUATORS
            and actuator not in _CARTESIAN_ACTUATORS
        )

    return {
        "yaml": yaml_updated,
        "config": config_updated,
        "unmatched_mjcf": unmatched_mjcf,
        "unmatched_yaml": unmatched_yaml,
    }


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


def export(
    assembly_dir: str,
    *,
    clean_only: bool = False,
    update_yaml: str | None = None,
) -> str:
    """Run the Onshape → attachable-MJCF pipeline for ``assembly_dir``.

    ``assembly_dir`` holds a ``config.json`` (onshape-to-robot fields plus a
    ``solarian`` block, see :class:`SolarianExportSpec`). Returns the path to the
    cleaned MJCF.

    With ``clean_only=True`` the network export is skipped and only the clean
    stage is re-run on the previously-exported raw MJCF — handy for iterating on
    the cleaning options without re-hitting the Onshape API.

    Joint limits flow **CAD → MJCF → config.json**: stale
    ``joint_properties.*.limits`` are ignored during the Onshape pull (unless
    ``solarian.prefer_config_limits`` is set), and after cleaning the MJCF ranges
    are always written back into ``config.json``. With ``update_yaml`` set to a
    module conf path (e.g. ``conf/right_arm.yaml``), those same ranges are also
    written into that yaml's ``sim_range_min`` / ``sim_range_max``.
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

    config_path = os.path.join(os.path.abspath(assembly_dir), "config.json")
    yaml_path = None
    if update_yaml:
        yaml_path = os.path.abspath(update_yaml)
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"--update-yaml path not found: {yaml_path}")

    result = sync_limits_from_mjcf(
        out_xml, config_path=config_path, yaml_path=yaml_path
    )
    print(f"* Synced joint limits from {out_xml}")
    if result["config"]:
        print(f"  config.json joint_properties: {', '.join(result['config'])}")
    else:
        print("  config.json joint_properties: no matching entries updated")
    if yaml_path:
        if result["yaml"]:
            print(f"  yaml ({yaml_path}): {', '.join(result['yaml'])}")
        else:
            print(f"  yaml ({yaml_path}): no matching actuators updated")
        for name in result["unmatched_mjcf"]:
            print(f"  warning: MJCF joint {name!r} has no matching yaml actuator")
        for name in result["unmatched_yaml"]:
            print(f"  warning: yaml actuator {name!r} has no MJCF joint range")

    return out_xml


def clean_raw(raw_xml: str, out_xml: str, spec: SolarianExportSpec, meshdir: str) -> str:
    """Run only the clean stage on an explicit raw MJCF path (used by tests)."""
    _clean(raw_xml, out_xml, spec, meshdir)
    return out_xml
