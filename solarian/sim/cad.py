"""Turn a downloaded CAD URDF into an attachable MuJoCo (MJCF) sim model.

This is the engine behind the ``solarian cad-import`` command. The URDF and
its meshes are files already downloaded from a CAD tool (see
:mod:`solarian.sim.cad_urdf`). The clean stage then drops the base
``<freejoint>``, infers link mass and inertia from the collision meshes at a
fixed density, points ``meshdir`` at those meshes, and makes the mechanism's
own collision geoms non-self-colliding. Frames (``flange``, ``tcp``,
``camera``, …) are already ``<site>`` elements in the raw MJCF.

``commentjson`` is imported at module level. MuJoCo is not: importing this
module never requires the ``[sim]`` extra.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import commentjson


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

#: Uniform density (kg/m^3) used to infer link masses from solid mesh volumes
#: when the CAD assigns no materials. ~1400 gives a ~1 kg SO-ARM-sized part.
DEFAULT_DENSITY = 1400.0
#: Position-actuator stiffness. N/m on a slide, N·m/rad on a hinge.
DEFAULT_KP = 50.0
DEFAULT_DAMPRATIO = 1.0
#: Joint friction loss. N on a slide, N·m on a hinge.
DEFAULT_FRICTIONLOSS = 0.1
DEFAULT_ARMATURE = 0.005


@dataclass
class SolarianExportSpec:
    """The ``solarian`` block of an assembly ``config.json``.

    Everything specific to turning a downloaded CAD URDF into an attachable
    Solarian sim model. All fields have defaults that reproduce the Solarian Arm 1's
    original behaviour.
    """

    #: Unused. Kept so a config.json written for the old Onshape API export still parses.
    dof_mates: list[str] = field(default_factory=list)
    #: Unused. The URDF's root link is the MJCF base.
    base_dof: Optional[str] = None
    #: Unused. A URDF already contains only the links its joints connect.
    drop_unmated_links: bool = True
    #: Where to write the cleaned, attachable MJCF, relative to the assembly
    #: dir. Defaults to ``../<robot_name>.xml`` (i.e. the project ``assets/``).
    output_xml: Optional[str] = None
    #: ``meshdir`` written into the cleaned MJCF (relative to ``output_xml``).
    #: Defaults to ``<assembly_dir_name>/<assets_directory>``.
    meshdir: Optional[str] = None
    #: Density for mass/inertia inference from the collision meshes.
    density: float = DEFAULT_DENSITY
    #: Position-actuator stiffness written onto the robot default class.
    kp: float = DEFAULT_KP
    #: Damping ratio of those position actuators.
    dampratio: float = DEFAULT_DAMPRATIO
    #: Joint friction loss written onto the robot default class.
    frictionloss: float = DEFAULT_FRICTIONLOSS
    #: Reflected rotor inertia written onto the robot default class.
    armature: float = DEFAULT_ARMATURE
    #: Drop the base ``<freejoint>`` so the model attaches rigidly into a scene.
    attach: bool = True
    #: Strip the placeholder inertials and infer mass/inertia from geometry.
    strip_inertials: bool = True
    #: Allow the mechanism's own links to collide with each other. Off by
    #: default: full-link collision meshes overlap by design at every joint and
    #: MuJoCo does not auto-exclude parent/child contacts, so self-collision
    #: produces permanent interpenetrations that fight the actuators.
    self_collision: bool = False
    #: Unused. Joint limits in the imported MJCF are written back into config.json.
    prefer_config_limits: bool = False

    @classmethod
    def from_config(cls, config: dict) -> "SolarianExportSpec":
        return cls(**dict(config.get("solarian", {})))


def clear_joint_property_limits(joint_properties: dict) -> list[str]:
    """Remove ``limits`` overrides from a ``joint_properties`` mapping (in-place).

    Returns the joint names (including ``default`` / ``*``) whose ``limits``
    entry was cleared. Leaves other properties (``range``, effort, …) intact.
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
        joint = d.find("joint")
        if joint is not None:
            joint.set("frictionloss", str(spec.frictionloss))
            joint.set("armature", str(spec.armature))
        position = d.find("position")
        if position is not None:
            position.set("kp", str(spec.kp))
            position.set("dampratio", str(spec.dampratio))

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
# Sync MJCF joint ranges → config.json                                        #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Entry points                                                                 #
# --------------------------------------------------------------------------- #

def model_name(assembly_dir: str) -> str:
    """``gripper1_description`` → ``gripper1``. A directory without that suffix keeps its name."""
    name = os.path.basename(os.path.abspath(assembly_dir))
    suffix = "_description"
    if name.endswith(suffix) and name != suffix:
        return name[: -len(suffix)]
    return name


def _load_config(assembly_dir: str) -> dict:
    path = assembly_dir
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return commentjson.load(f)


_CONFIG_YAML_NAME = "config.yaml"
_CONFIG_YAML_KEYS = (
    "density",
    "kp",
    "dampratio",
    "frictionloss",
    "armature",
    "attach",
    "strip_inertials",
    "self_collision",
)
_DEFAULT_CONFIG_YAML = """\
# cad-import reads this file and does not rewrite it.
# Edit a value, then run the import again.

# kg/m³ used to infer link mass from the collision meshes.
density: 1400

# Position-actuator stiffness. N/m on a slide, N·m/rad on a hinge.
kp: 50
# Damping ratio of the position actuators.
dampratio: 1
# Joint friction loss. N on a slide, N·m on a hinge.
frictionloss: 0.1
# Rotor inertia reflected onto the joint.
armature: 0.005

# Drop the base freejoint so the model attaches into a scene.
attach: true
# Infer mass from geometry instead of the URDF inertials.
strip_inertials: true
# Let the mechanism's own links collide with each other.
self_collision: false
"""


def _config_yaml_path(assembly_dir: str) -> str:
    return os.path.join(os.path.abspath(assembly_dir), _CONFIG_YAML_NAME)


def _load_config_yaml(assembly_dir: str) -> dict:
    """Read ``config.yaml``, writing the defaults when the file is absent.

    An existing file is returned unchanged, including keys this version does
    not understand. Missing keys keep the :class:`SolarianExportSpec` defaults.
    """
    import yaml

    path = _config_yaml_path(assembly_dir)
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(_DEFAULT_CONFIG_YAML)
        print(f"* Wrote {path}")
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping")
    return loaded


def _apply_config_yaml(spec: SolarianExportSpec, data: dict) -> None:
    for key in _CONFIG_YAML_KEYS:
        if key not in data or data[key] is None:
            continue
        setattr(spec, key, data[key])


def _fill_names(assembly_dir: str, config: dict) -> None:
    name = model_name(assembly_dir)
    config.setdefault("robot_name", name)
    config.setdefault("assets_directory", "meshes")
    config.setdefault("output_filename", f"{name}_raw")


def _resolve_defaults(assembly_dir: str, config: dict, spec: SolarianExportSpec) -> tuple[str, str]:
    """Fill in ``output_xml`` / ``meshdir`` defaults; return absolute out path
    and the meshdir string to embed."""
    robot_name = config.get("robot_name") or model_name(assembly_dir)
    assets_dir = config.get("assets_directory") or "meshes"

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
    urdf: str | None = None,
    density: float | None = None,
) -> str:
    """Import a downloaded URDF into an attachable MJCF for ``assembly_dir``.

    The model name is the assembly directory with a trailing ``_description``
    removed. Meshes go in ``meshes/``, the raw MJCF is ``<name>_raw.xml``, and
    the attachable model is ``../<name>.xml``. ``config.yaml`` in the assembly
    directory holds density, actuator gain, and the other clean-stage values.
    The first import writes that file with the defaults; later imports read it
    and do not change it. ``density`` overrides the file's density for this
    run only.

    With ``clean_only=True`` only the clean stage re-runs on the raw MJCF this
    command already wrote. ``urdf`` is a ``.urdf`` or a ``.zip`` when the
    download is not already laid out in ``assembly_dir``.

    When ``config.json`` exists, MJCF joint ranges are written back into its
    ``joint_properties``.
    """
    if clean_only and urdf:
        raise ValueError("--clean-only does not read a URDF; pass only one of them.")
    config = _load_config(assembly_dir)
    _fill_names(assembly_dir, config)
    spec = SolarianExportSpec.from_config(config)
    _apply_config_yaml(spec, _load_config_yaml(assembly_dir))
    if density is not None:
        spec.density = density
    out_xml, meshdir = _resolve_defaults(assembly_dir, config, spec)

    if clean_only:
        raw_xml = _raw_xml_path(assembly_dir, config)
        if not os.path.exists(raw_xml):
            raise FileNotFoundError(
                f"--clean-only needs a previous raw MJCF at {raw_xml}; "
                f"run without --clean-only first."
            )
    else:
        from solarian.sim.cad_urdf import write_raw_mjcf

        raw_xml = write_raw_mjcf(
            assembly_dir,
            config,
            urdf_path=urdf,
            frictionloss=spec.frictionloss,
            armature=spec.armature,
            kp=spec.kp,
            dampratio=spec.dampratio,
        )

    _clean(raw_xml, out_xml, spec, meshdir)

    config_path = os.path.join(os.path.abspath(assembly_dir), "config.json")
    if os.path.isfile(config_path):
        updated = sync_joint_properties(config_path, read_joint_ranges(out_xml))
        print(f"* Synced joint limits from {out_xml}")
        if updated:
            print(f"  config.json joint_properties: {', '.join(updated)}")
        else:
            print("  config.json joint_properties: no matching entries updated")

    return out_xml


def clean_raw(raw_xml: str, out_xml: str, spec: SolarianExportSpec, meshdir: str) -> str:
    """Run only the clean stage on an explicit raw MJCF path (used by tests)."""
    _clean(raw_xml, out_xml, spec, meshdir)
    return out_xml
