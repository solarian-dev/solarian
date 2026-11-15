"""Turn a downloaded URDF and its meshes into the raw MJCF ``cad-import`` cleans.

The name prefixes are the ones ``onshape-export`` used in the CAD model.
``dof_``, ``frame_``, ``fix_``, ``link_``, and ``closing_`` are stripped so the
MJCF joint, site, and body names stay the bare names. A joint or link without
a prefix keeps its name.
"""

from __future__ import annotations

import math
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

_MESH_EXTS = {".stl", ".obj"}
_DAE_EXTS = {".dae"}


@dataclass
class _Geom:
    cls: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    mesh: str | None = None
    kind: str = "mesh"
    size: tuple[float, ...] = ()
    rgba: str | None = None


@dataclass
class _Link:
    name: str
    geoms: list[_Geom] = field(default_factory=list)


@dataclass
class _Mimic:
    joint: str
    multiplier: float
    offset: float


@dataclass
class _Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None
    upper: float | None
    mimic: _Mimic | None


def write_raw_mjcf(
    assembly_dir: str,
    config: dict,
    urdf_path: str | None = None,
    *,
    frictionloss: float = 0.1,
    armature: float = 0.005,
    kp: float = 50,
    dampratio: float = 1,
) -> str:
    """Write the raw MJCF and copy meshes into the assembly's assets directory.

    ``urdf_path`` is a ``.urdf`` or a ``.zip``. When it is omitted, the single
    ``*.urdf`` in ``assembly_dir`` or in its ``urdf/`` directory is used.
    """
    assembly = Path(assembly_dir).resolve()
    if urdf_path is not None and Path(urdf_path).suffix.lower() == ".zip":
        _extract_zip(Path(urdf_path), assembly)
        urdf_file = _find_urdf(assembly)
    elif urdf_path is not None:
        urdf_file = Path(urdf_path).resolve()
        if not urdf_file.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_file}")
    else:
        urdf_file = _find_urdf(assembly)

    assets_rel = config.get("assets_directory") or "meshes"
    assets_dir = (assembly / assets_rel).resolve()
    robot_name = config.get("robot_name") or _robot_name(urdf_file)
    links, joints, root_name = _parse_urdf(urdf_file)
    mesh_names = _copy_meshes(links, urdf_file.parent, assembly, assets_dir)

    raw_xml = _raw_xml_path(assembly, config)
    _write_mjcf(
        raw_xml,
        robot_name=robot_name,
        meshdir=assets_rel,
        links=links,
        joints=joints,
        root_name=root_name,
        mesh_names=mesh_names,
        frictionloss=frictionloss,
        armature=armature,
        kp=kp,
        dampratio=dampratio,
    )
    print(f"* Wrote raw export {raw_xml}")
    return raw_xml


def _robot_name(urdf_file: Path) -> str:
    root = ET.parse(urdf_file).getroot()
    return root.get("name") or urdf_file.stem


def _raw_xml_path(assembly: Path, config: dict) -> str:
    output_filename = config.get("output_filename") or "robot"
    output_filename = "".join(
        c for c in output_filename if c.isalnum() or c in ("_", "-")
    ).rstrip()
    return str(assembly / f"{output_filename}.xml")


def _extract_zip(zip_path: Path, dest: Path) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"URDF zip not found: {zip_path}")
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"zip entry escapes the assembly directory: {info.filename}")
        zf.extractall(dest)
    print(f"* Unpacked {zip_path} into {dest}")


def _find_urdf(assembly: Path) -> Path:
    """One URDF in the directory, else in ``urdf/``, else the only one below."""
    direct = sorted(assembly.glob("*.urdf"))
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        raise FileNotFoundError(
            "several URDF files in "
            f"{assembly}: {', '.join(p.name for p in direct)}. Pass --urdf."
        )
    nested_dir = assembly / "urdf"
    nested = sorted(nested_dir.glob("*.urdf")) if nested_dir.is_dir() else []
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        raise FileNotFoundError(
            "several URDF files in "
            f"{nested_dir}: {', '.join(p.name for p in nested)}. Pass --urdf."
        )
    deeper = sorted(p for p in assembly.rglob("*.urdf") if p.is_file())
    if len(deeper) == 1:
        return deeper[0]
    if not deeper:
        raise FileNotFoundError(
            f"no .urdf in {assembly} or {nested_dir}. "
            "Unzip the CAD download there, or pass --urdf."
        )
    raise FileNotFoundError(
        "several URDF files under "
        f"{assembly}: {', '.join(str(p.relative_to(assembly)) for p in deeper)}. Pass --urdf."
    )


def _vec3(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not text:
        return default
    parts = text.split()
    if len(parts) != 3:
        raise ValueError(f"expected 3 numbers, got {text!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _origin(element: ET.Element | None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if element is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return _vec3(element.get("xyz"), (0.0, 0.0, 0.0)), _vec3(element.get("rpy"), (0.0, 0.0, 0.0))


def _parse_urdf(urdf_file: Path) -> tuple[dict[str, _Link], list[_Joint], str]:
    root = ET.parse(urdf_file).getroot()
    if root.tag != "robot":
        raise ValueError(f"{urdf_file} is not a URDF (root element is {root.tag!r})")
    colors = _material_colors(root)
    links: dict[str, _Link] = {}
    for link_el in root.findall("link"):
        name = link_el.get("name")
        if not name:
            raise ValueError(f"{urdf_file} has a link with no name")
        links[name] = _Link(name=name, geoms=_link_geoms(link_el, colors))
    joints: list[_Joint] = []
    children: set[str] = set()
    for joint_el in root.findall("joint"):
        joint = _parse_joint(joint_el)
        joints.append(joint)
        children.add(joint.child)
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise ValueError(
            f"{urdf_file} has {len(roots)} root links ({', '.join(roots) or 'none'}); "
            "a URDF needs one link that no joint owns."
        )
    return links, joints, roots[0]


def _material_colors(robot: ET.Element) -> dict[str, str]:
    colors: dict[str, str] = {}
    for material in robot.findall("material"):
        name = material.get("name")
        color = material.find("color")
        if name and color is not None and color.get("rgba"):
            colors[name] = color.get("rgba") or ""
    return colors


def _rgba_of(element: ET.Element, colors: dict[str, str]) -> str | None:
    material = element.find("material")
    if material is None:
        return None
    color = material.find("color")
    if color is not None and color.get("rgba"):
        return color.get("rgba")
    name = material.get("name")
    if name and name in colors:
        return colors[name]
    return None


def _link_geoms(link_el: ET.Element, colors: dict[str, str]) -> list[_Geom]:
    geoms: list[_Geom] = []
    for tag, cls in (("visual", "visual"), ("collision", "collision")):
        for element in link_el.findall(tag):
            geom = _geom_from(element, cls, colors)
            if geom is not None:
                geoms.append(geom)
    return geoms


def _geom_from(element: ET.Element, cls: str, colors: dict[str, str]) -> _Geom | None:
    geometry = element.find("geometry")
    if geometry is None:
        return None
    xyz, rpy = _origin(element.find("origin"))
    rgba = _rgba_of(element, colors)
    mesh = geometry.find("mesh")
    if mesh is not None:
        filename = mesh.get("filename")
        if not filename:
            raise ValueError("URDF mesh has no filename")
        return _Geom(cls=cls, xyz=xyz, rpy=rpy, mesh=filename, rgba=rgba)
    box = geometry.find("box")
    if box is not None:
        size = _vec3(box.get("size"), (0.0, 0.0, 0.0))
        # URDF box size is full edge lengths. MuJoCo size is half-extents.
        return _Geom(cls=cls, xyz=xyz, rpy=rpy, kind="box", size=tuple(v / 2 for v in size), rgba=rgba)
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        radius = float(cylinder.get("radius") or 0)
        length = float(cylinder.get("length") or 0)
        return _Geom(
            cls=cls, xyz=xyz, rpy=rpy, kind="cylinder", size=(radius, length / 2), rgba=rgba
        )
    sphere = geometry.find("sphere")
    if sphere is not None:
        radius = float(sphere.get("radius") or 0)
        return _Geom(cls=cls, xyz=xyz, rpy=rpy, kind="sphere", size=(radius,), rgba=rgba)
    raise ValueError(f"unsupported URDF geometry under <{element.tag}>")


def _parse_joint(joint_el: ET.Element) -> _Joint:
    name = joint_el.get("name")
    jtype = joint_el.get("type")
    parent = joint_el.find("parent")
    child = joint_el.find("child")
    if not name or not jtype or parent is None or child is None:
        raise ValueError(f"URDF joint is missing name, type, parent, or child: {name!r}")
    if jtype in ("floating", "planar"):
        raise ValueError(f"URDF joint {name!r} has type {jtype!r}, which MJCF cannot import")
    xyz, rpy = _origin(joint_el.find("origin"))
    # URDF's default axis is +X when the element is omitted.
    axis = _vec3((joint_el.find("axis").get("xyz") if joint_el.find("axis") is not None else None), (1.0, 0.0, 0.0))
    lower = upper = None
    limit = joint_el.find("limit")
    if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
        lower = float(limit.get("lower") or 0)
        upper = float(limit.get("upper") or 0)
    mimic_el = joint_el.find("mimic")
    mimic = None
    if mimic_el is not None and mimic_el.get("joint"):
        mimic = _Mimic(
            joint=mimic_el.get("joint") or "",
            multiplier=float(mimic_el.get("multiplier") or 1),
            offset=float(mimic_el.get("offset") or 0),
        )
    return _Joint(
        name=name,
        type=jtype,
        parent=parent.get("link") or "",
        child=child.get("link") or "",
        xyz=xyz,
        rpy=rpy,
        axis=axis,
        lower=lower,
        upper=upper,
        mimic=mimic,
    )


def _copy_meshes(
    links: dict[str, _Link], urdf_dir: Path, assembly: Path, assets_dir: Path
) -> dict[str, str]:
    """Copy each referenced mesh into ``assets_dir``. Returns filename → mesh name.

    A visual COLLADA file is replaced by that link's STL or OBJ collision mesh.
    MuJoCo does not load COLLADA.
    """
    copied: dict[str, str] = {}
    # Keyed by basename. Two sources in this import that share a name must be
    # the same bytes. A file already in assets_dir is the previous import.
    seen: dict[str, Path] = {}
    for link in links.values():
        collision_mesh = next(
            (g.mesh for g in link.geoms if g.cls == "collision" and g.mesh and _ext(g.mesh) in _MESH_EXTS),
            None,
        )
        for geom in link.geoms:
            if geom.mesh is None:
                continue
            filename = geom.mesh
            if _ext(filename) in _DAE_EXTS:
                if collision_mesh is None:
                    raise ValueError(
                        f"link {link.name!r} visual mesh {filename!r} is COLLADA, "
                        "which MuJoCo cannot load. Re-export that mesh as STL or OBJ."
                    )
                filename = collision_mesh
                geom.mesh = filename
            if _ext(filename) not in _MESH_EXTS:
                raise ValueError(
                    f"link {link.name!r} mesh {filename!r} is not STL or OBJ. "
                    "Re-export the CAD meshes as STL."
                )
            if filename in copied:
                continue
            src = _resolve_mesh(filename, urdf_dir, assembly)
            prior = seen.get(src.name)
            if (
                prior is not None
                and prior.resolve() != src.resolve()
                and prior.read_bytes() != src.read_bytes()
            ):
                raise ValueError(f"two different meshes are both named {src.name}")
            seen[src.name] = src
            assets_dir.mkdir(parents=True, exist_ok=True)
            dest = assets_dir / src.name
            if dest.resolve() != src.resolve() and (
                not dest.exists() or dest.read_bytes() != src.read_bytes()
            ):
                shutil.copyfile(src, dest)
            copied[filename] = src.name
            print(f"* Mesh {src.name}")
    return copied


def _ext(filename: str) -> str:
    path = filename.split("?", 1)[0]
    if path.startswith("package://"):
        path = path[len("package://"):]
    elif path.startswith("file://"):
        path = urlparse(path).path
    return Path(path).suffix.lower()


def _resolve_mesh(filename: str, urdf_dir: Path, assembly: Path) -> Path:
    relatives: list[str] = []
    if filename.startswith("package://"):
        rest = filename[len("package://"):]
        relatives.append(rest)
        parts = rest.split("/", 1)
        if len(parts) == 2:
            relatives.append(parts[1])
        relatives.append(Path(rest).name)
    elif filename.startswith("file://"):
        parsed = Path(unquote(urlparse(filename).path))
        if parsed.is_file():
            return parsed
        relatives.append(parsed.name)
    else:
        relatives.append(filename)
        relatives.append(Path(filename).name)
    for rel in relatives:
        for base in (urdf_dir, assembly, assembly / "meshes"):
            candidate = (base / rel).resolve()
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"mesh {filename!r} is not next to {urdf_dir} or under {assembly}"
    )


def _rpy_matrix(rpy: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
    """URDF rpy is extrinsic XYZ: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _apply(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def _quat(m) -> tuple[float, float, float, float]:
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)


def _as_matrix(rotation):
    if rotation and isinstance(rotation[0], tuple):
        return rotation
    return _rpy_matrix(rotation)


def _compose(parent_xyz, parent_rot, child_xyz, child_rot):
    parent_m = _as_matrix(parent_rot)
    child_m = _as_matrix(child_rot)
    rotated = _apply(parent_m, child_xyz)
    return (
        (parent_xyz[0] + rotated[0], parent_xyz[1] + rotated[1], parent_xyz[2] + rotated[2]),
        _mul(parent_m, child_m),
    )


def _fmt(values) -> str:
    return " ".join(f"{v:.8g}" for v in values)


def _pos_quat(xyz, rotation) -> dict[str, str]:
    quat = _quat(_as_matrix(rotation))
    return {"pos": _fmt(xyz), "quat": _fmt(quat)}


def _bare_link(name: str) -> str:
    if name.startswith("link_") and len(name) > len("link_"):
        return name[len("link_"):]
    return name


def _bare_joint(name: str) -> tuple[str, bool]:
    """Return ``(mjcf_name, flip_axis)``. ``_inv`` negates the axis and the limits."""
    flip = False
    if name.endswith("_inverted"):
        name = name[: -len("_inverted")]
        flip = True
    elif name.endswith("_inv"):
        name = name[: -len("_inv")]
        flip = True
    if name.startswith("dof_") and len(name) > len("dof_"):
        name = name[len("dof_"):]
    return name, flip


def _is_continuous(joint: _Joint, mjcf_name: str) -> bool:
    return joint.type == "continuous" or "wheel" in mjcf_name or "continuous" in mjcf_name


def _mjcf_type(joint: _Joint) -> str:
    if joint.type in ("revolute", "continuous", "cylindrical"):
        return "hinge"
    if joint.type in ("prismatic", "slider"):
        return "slide"
    raise ValueError(f"URDF joint {joint.name!r} has type {joint.type!r}")


def _classify(joint: _Joint, child: _Link) -> str:
    if joint.name.startswith("closing_") or child.name.startswith("closing_"):
        return "closing"
    if joint.name.startswith("frame_") or child.name.startswith("frame_"):
        return "frame"
    if joint.name.startswith("fix_") or (joint.type == "fixed" and child.geoms):
        return "merge"
    if joint.type == "fixed" and not child.geoms:
        return "frame"
    return "dof"


def _site_name(joint: _Joint, child: _Link) -> str:
    raw = child.name if child.name.startswith("frame_") else joint.name
    if raw.startswith("frame_") and len(raw) > len("frame_"):
        return raw[len("frame_"):]
    return raw


def _limits(joint: _Joint, flip: bool) -> tuple[float, float] | None:
    if joint.lower is None or joint.upper is None:
        return None
    if flip:
        return (-joint.upper, -joint.lower)
    return (joint.lower, joint.upper)


def _write_mjcf(
    raw_xml: str,
    *,
    robot_name: str,
    meshdir: str,
    links: dict[str, _Link],
    joints: list[_Joint],
    root_name: str,
    mesh_names: dict[str, str],
    frictionloss: float = 0.1,
    armature: float = 0.005,
    kp: float = 50,
    dampratio: float = 1,
) -> None:
    children: dict[str, list[_Joint]] = {}
    for joint in joints:
        children.setdefault(joint.parent, []).append(joint)
    mujoco = ET.Element("mujoco", model=robot_name)
    ET.SubElement(mujoco, "compiler", angle="radian", meshdir=meshdir, autolimits="true")
    outer = ET.SubElement(ET.SubElement(mujoco, "default"), "default", **{"class": robot_name})
    ET.SubElement(outer, "joint", frictionloss=str(frictionloss), armature=str(armature))
    ET.SubElement(outer, "position", kp=str(kp), dampratio=str(dampratio))
    ET.SubElement(ET.SubElement(outer, "default", **{"class": "visual"}), "geom",
                  type="mesh", contype="0", conaffinity="0", group="2")
    ET.SubElement(ET.SubElement(outer, "default", **{"class": "collision"}), "geom", group="3")
    world = ET.SubElement(mujoco, "worldbody")
    actuators: list[tuple[str, bool]] = []
    equalities: list[ET.Element] = []
    used_meshes: list[str] = []
    materials: dict[str, str] = {}
    joint_names: dict[str, str] = {}

    def remember_joint(urdf_name: str, mjcf_name: str) -> None:
        previous = joint_names.get(mjcf_name)
        if previous is not None and previous != urdf_name:
            raise ValueError(
                f"URDF joints {previous!r} and {urdf_name!r} both become MJCF joint {mjcf_name!r}"
            )
        joint_names[mjcf_name] = urdf_name

    def add_geoms(host: ET.Element, geoms: list[_Geom], frame_xyz, frame_rpy) -> None:
        present = {g.cls for g in geoms}
        emitted = list(geoms)
        if present == {"visual"} or present == {"collision"}:
            only = next(iter(present))
            other = "collision" if only == "visual" else "visual"
            emitted += [_Geom(**{**g.__dict__, "cls": other}) for g in geoms]
        for geom in emitted:
            xyz, rpy = _compose(frame_xyz, frame_rpy, geom.xyz, geom.rpy)
            attrs = {"class": geom.cls, **_pos_quat(xyz, rpy)}
            if geom.kind == "mesh":
                file_name = mesh_names[geom.mesh or ""]
                mesh_name = Path(file_name).stem
                attrs["mesh"] = mesh_name
                if file_name not in used_meshes:
                    used_meshes.append(file_name)
                if geom.rgba:
                    material = f"{mesh_name}_material"
                    materials[material] = geom.rgba
                    attrs["material"] = material
            else:
                attrs["type"] = geom.kind
                attrs["size"] = _fmt(geom.size)
            ET.SubElement(host, "geom", attrs)

    def add_body(
        link_name: str,
        parent: ET.Element,
        pose: dict[str, str],
        joint: _Joint | None,
        flip: bool,
    ) -> ET.Element:
        body_name = _bare_link(link_name)
        body = ET.SubElement(parent, "body", name=body_name, **pose)
        if parent.tag == "worldbody":
            body.set("childclass", robot_name)
            ET.SubElement(body, "freejoint", name=f"{body_name}_freejoint")
        if joint is not None:
            mjcf_name, _ = _bare_joint(joint.name)
            remember_joint(joint.name, mjcf_name)
            axis = tuple(-v for v in joint.axis) if flip else joint.axis
            attrs = {"name": mjcf_name, "type": _mjcf_type(joint), "axis": _fmt(axis)}
            limits = None if _is_continuous(joint, mjcf_name) else _limits(joint, flip)
            if limits is not None:
                attrs["range"] = _fmt(limits)
            ET.SubElement(body, "joint", attrs)
            if joint.mimic is not None:
                source, _ = _bare_joint(joint.mimic.joint)
                equalities.append(ET.Element(
                    "joint",
                    joint1=mjcf_name,
                    joint2=source,
                    polycoef=_fmt((joint.mimic.offset, joint.mimic.multiplier, 0, 0, 0)),
                ))
            elif not joint.name.startswith("closing_"):
                actuators.append((mjcf_name, limits is not None))
        link = links[link_name]
        add_geoms(body, link.geoms, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for child in children.get(link_name, []):
            walk(child, body, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        return body

    def walk(joint: _Joint, host: ET.Element, frame_xyz, frame_rpy) -> None:
        child = links[joint.child]
        kind = _classify(joint, child)
        child_xyz, child_rpy = _compose(frame_xyz, frame_rpy, joint.xyz, joint.rpy)
        if kind == "frame":
            ET.SubElement(
                host, "site", name=_site_name(joint, child), group="3", **_pos_quat(child_xyz, child_rpy)
            )
            return
        if kind == "merge":
            add_geoms(host, child.geoms, child_xyz, child_rpy)
            for grandchild in children.get(child.name, []):
                walk(grandchild, host, child_xyz, child_rpy)
            return
        if kind == "closing":
            body = add_body(child.name, host, _pos_quat(child_xyz, child_rpy), None, False)
            # No joint: the child is fixed to its parent. The equality is the closure.
            parent_name = host.get("name") or ""
            child_name = body.get("name") or ""
            if joint.type == "fixed":
                equalities.append(ET.Element("weld", body1=parent_name, body2=child_name))
            else:
                bare, _ = _bare_joint(joint.name)
                ET.SubElement(host, "site", name=f"{bare}_1", group="3", **_pos_quat(child_xyz, child_rpy))
                ET.SubElement(body, "site", name=f"{bare}_2", group="3", **_pos_quat((0, 0, 0), (0, 0, 0)))
                equalities.append(ET.Element("connect", site1=f"{bare}_1", site2=f"{bare}_2"))
            return
        mjcf_name, flip = _bare_joint(joint.name)
        remember_joint(joint.name, mjcf_name)
        placed = _Joint(
            name=joint.name,
            type=joint.type,
            parent=joint.parent,
            child=joint.child,
            xyz=child_xyz,
            rpy=child_rpy,
            axis=joint.axis,
            lower=joint.lower,
            upper=joint.upper,
            mimic=joint.mimic,
        )
        add_body(child.name, host, _pos_quat(placed.xyz, placed.rpy), placed, flip)

    add_body(root_name, world, _pos_quat((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), None, False)

    asset = ET.SubElement(mujoco, "asset")
    for file_name in used_meshes:
        ET.SubElement(asset, "mesh", file=file_name)
    for material, rgba in materials.items():
        ET.SubElement(asset, "material", name=material, rgba=rgba)

    actuator = ET.SubElement(mujoco, "actuator")
    for name, has_range in actuators:
        attrs = {"class": robot_name, "name": name, "joint": name}
        if has_range:
            attrs["inheritrange"] = "1"
        ET.SubElement(actuator, "position", attrs)
    if equalities:
        equality = ET.SubElement(mujoco, "equality")
        for element in equalities:
            equality.append(element)

    tree = ET.ElementTree(mujoco)
    ET.indent(tree, space="  ")
    os.makedirs(os.path.dirname(os.path.abspath(raw_xml)), exist_ok=True)
    tree.write(raw_xml, xml_declaration=True, encoding="utf-8")
