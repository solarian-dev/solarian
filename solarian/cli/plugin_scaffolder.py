from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Dict, List, Tuple

# importlib.resources works with files bundled in the package
try:  # Py 3.11+
    from importlib.resources import files as pkg_files
except ImportError:  # Py 3.9-3.10
    from importlib_resources import files as pkg_files  # type: ignore

import solarian  # this package, used for locating templates


#: Component types that can be scaffolded into a plugin.
ALLOWED_TYPES = {"actuator", "sensor", "module", "robot", "command", "policy", "op"}


@dataclass(frozen=True)
class FileSpec:
    template_relpath: Path     # path inside solarian/templates/plugin
    render_relpath: str        # output path inside the plugin root (str.format template)


#: Base files created once per plugin (never overwritten — a re-run that adds a
#: new component type leaves these untouched).
BASE_FILES: Tuple[FileSpec, ...] = (
    FileSpec(Path("gitignore.tmpl"), ".gitignore"),
    FileSpec(Path("pyproject.toml.tmpl"), "pyproject.toml"),
    FileSpec(Path("README.md.tmpl"), "README.md"),
    FileSpec(Path("__init__.py.tmpl"), "{package_name}/__init__.py"),
)


#: Source files written per component type.
COMPONENT_FILES: Dict[str, Tuple[FileSpec, ...]] = {
    "actuator": (
        FileSpec(Path("actuator/actuator.py.tmpl"), "{package_name}/actuator.py"),
        FileSpec(Path("actuator/bus.py.tmpl"), "{package_name}/bus.py"),
    ),
    "sensor": (
        FileSpec(Path("sensor/sensor.py.tmpl"), "{package_name}/sensor.py"),
        FileSpec(Path("sensor/sensor_bus.py.tmpl"), "{package_name}/sensor_bus.py"),
    ),
    "module": (
        FileSpec(Path("module/module.py.tmpl"), "{package_name}/module.py"),
        FileSpec(Path("module/conf.yaml.tmpl"), "{package_name}/conf.yaml"),
    ),
    "robot": (
        FileSpec(Path("robot/robot.py.tmpl"), "{package_name}/robot.py"),
    ),
    "command": (
        FileSpec(Path("command/__init__.py.tmpl"), "{package_name}/commands/__init__.py"),
        FileSpec(Path("command/command.py.tmpl"), "{package_name}/commands/{plugin_name}_command.py"),
    ),
    "policy": (
        FileSpec(Path("policy/trainer.py.tmpl"), "{package_name}/trainer.py"),
        FileSpec(Path("policy/policy_server.py.tmpl"), "{package_name}/policy_server.py"),
        FileSpec(Path("policy/skill.py.tmpl"), "{package_name}/skill.py"),
    ),
    "op": (
        FileSpec(Path("op/op.py.tmpl"), "{package_name}/op.py"),
    ),
}


#: Entry points (group, key, target) declared per component type.  ``key`` and
#: ``target`` are str.format templates rendered with the plugin context.
ENTRY_POINTS: Dict[str, List[Tuple[str, str, str]]] = {
    "actuator": [
        ("solarian.actuators", "{plugin_name}", "{package_name}.actuator:{class_prefix}Servo"),
        ("solarian.actuator_buses", "{plugin_name}", "{package_name}.bus:{class_prefix}Bus"),
    ],
    "sensor": [
        ("solarian.sensors", "{plugin_name}", "{package_name}.sensor:{class_prefix}Sensor"),
        ("solarian.sensor_buses", "{plugin_name}", "{package_name}.sensor_bus:{class_prefix}SensorBus"),
    ],
    "module": [
        ("solarian.modules", "{plugin_name}", "{package_name}.module:{class_prefix}Module"),
    ],
    "robot": [
        ("solarian.robots", "{plugin_name}", "{package_name}.robot:{class_prefix}Robot"),
    ],
    "command": [
        ("solarian.commands", "{plugin_name}-command",
         "{package_name}.commands.{plugin_name}_command:{class_prefix}Command"),
    ],
    "policy": [
        ("solarian.trainers", "{plugin_name}", "{package_name}.trainer:{class_prefix}Trainer"),
        ("solarian.skills", "{plugin_name}", "{package_name}.skill:{class_prefix}Skill"),
    ],
    "op": [
        ("solarian.dataset_ops", "{plugin_name}", "{package_name}.op:{class_prefix}Op"),
    ],
}


def _normalize_names(name: str) -> Tuple[str, str, str]:
    """Return (short_name, dist_name, package_name) from the user-supplied name.

    ``feetech``          -> ("feetech", "solarian-feetech", "solarian_feetech")
    ``solarian-feetech`` -> ("feetech", "solarian-feetech", "solarian_feetech")
    """
    name = name.strip()
    if name.startswith("solarian-"):
        dist_name = name
        short = name[len("solarian-"):]
    elif name.startswith("solarian_"):
        dist_name = name.replace("_", "-")
        short = name[len("solarian_"):]
    else:
        short = name
        dist_name = f"solarian-{short}"
    package_name = dist_name.replace("-", "_")
    return short, dist_name, package_name


def _camel(name: str) -> str:
    """Convert a plugin short name to a CamelCase class prefix."""
    parts = [p for p in re.split(r"[-_\s]+", name) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _render(text: str, context: dict) -> str:
    """Small templating using string.Template with ${...} placeholders."""
    return Template(text).safe_substitute(context)


def _read_template(relpath: Path) -> str:
    root = pkg_files(solarian) / "templates" / "plugin"
    return (root / relpath.as_posix()).read_text(encoding="utf-8")


def _write_if_absent(dst: Path, content: str, force: bool) -> bool:
    """Write *content* to *dst* unless it already exists (and not force).

    Returns True if the file was written, False if it was left untouched.
    """
    if dst.exists() and not force:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")
    return True


def _render_specs(
    plugin_root: Path,
    specs: Tuple[FileSpec, ...],
    ctx: dict,
    force: bool,
) -> List[str]:
    """Render and write a set of FileSpecs; return the relpaths actually written."""
    written: List[str] = []
    for spec in specs:
        text = _read_template(spec.template_relpath)
        rendered = _render(text, ctx)
        render_relpath = spec.render_relpath.format(**ctx)
        if _write_if_absent(plugin_root / render_relpath, rendered, force):
            written.append(render_relpath)
    return written


def _merge_one_entry_point(text: str, group: str, key: str, target: str) -> str:
    """Idempotently add ``key = "target"`` under ``[project.entry-points."group"]``."""
    header = f'[project.entry-points."{group}"]'
    line = f'{key} = "{target}"'
    lines = text.split("\n")

    header_idx = None
    for i, l in enumerate(lines):
        if l.strip() == header:
            header_idx = i
            break

    if header_idx is None:
        # Append a brand new entry-point table at the end.
        return text.rstrip("\n") + "\n\n" + header + "\n" + line + "\n"

    # Determine the extent of this table (up to the next table header / EOF).
    end_idx = len(lines)
    for j in range(header_idx + 1, len(lines)):
        if lines[j].lstrip().startswith("["):
            end_idx = j
            break

    for j in range(header_idx + 1, end_idx):
        stripped = lines[j].strip()
        if not stripped or stripped.startswith("#"):
            continue
        existing_key = stripped.split("=", 1)[0].strip().strip('"')
        if existing_key == key:
            return text  # already present -> nothing to do

    lines.insert(header_idx + 1, line)
    return "\n".join(lines)


def _apply_entry_points(plugin_root: Path, plugin_type: str, ctx: dict) -> List[str]:
    """Merge the entry points for *plugin_type* into the plugin's pyproject.toml.

    Returns the list of "group -> key" strings that were newly added.
    """
    pyproject = plugin_root / "pyproject.toml"
    if not pyproject.exists():
        return []

    text = pyproject.read_text(encoding="utf-8")
    original = text
    added: List[str] = []
    for group, key_tmpl, target_tmpl in ENTRY_POINTS.get(plugin_type, []):
        key = key_tmpl.format(**ctx)
        target = target_tmpl.format(**ctx)
        new_text = _merge_one_entry_point(text, group, key, target)
        if new_text != text:
            added.append(f"{group}: {key}")
            text = new_text

    if text != original:
        pyproject.write_text(text, encoding="utf-8")
    return added


def scaffold_plugin(
    plugin_type: str,
    name: str,
    target_dir: Path | None = None,
    force: bool = False,
) -> Tuple[Path, bool, List[str]]:
    """
    Scaffold a new Solarian plugin, or add a component type to an existing one.

    Args:
        plugin_type: One of :data:`ALLOWED_TYPES`.
        name: Plugin name (short or ``solarian-`` prefixed).
        target_dir: Parent directory to create the plugin in. Defaults to cwd.
        force: If True, overwrite existing component source files.

    Returns:
        (plugin_root, is_new_plugin, written_relpaths)
    """
    if plugin_type not in ALLOWED_TYPES:
        raise ValueError(
            f"Unknown plugin type '{plugin_type}'. "
            f"Valid types: {', '.join(sorted(ALLOWED_TYPES))}"
        )

    if target_dir is None:
        target_dir = Path(os.getcwd())

    short, dist_name, package_name = _normalize_names(name)
    ctx = {
        "plugin_name": short,
        "dist_name": dist_name,
        "package_name": package_name,
        "class_prefix": _camel(short),
    }

    plugin_root = target_dir / dist_name
    is_new = not plugin_root.exists()
    plugin_root.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    # Base files are created once and never overwritten (protects entry points
    # accumulated across multiple component types in pyproject.toml).
    written += _render_specs(plugin_root, BASE_FILES, ctx, force=False)
    # Component source files honour --force so stubs can be regenerated.
    written += _render_specs(plugin_root, COMPONENT_FILES[plugin_type], ctx, force=force)

    _apply_entry_points(plugin_root, plugin_type, ctx)

    return plugin_root, is_new, written


def _print_plugin_success(
    plugin_type: str,
    name: str,
    plugin_root: Path,
    is_new: bool,
    written: List[str],
) -> None:
    short, dist_name, package_name = _normalize_names(name)

    if is_new:
        print(f"\n✓ Plugin '{dist_name}' created at {plugin_root}")
    else:
        print(f"\n✓ Added '{plugin_type}' component to existing plugin '{dist_name}' at {plugin_root}")

    if written:
        print("\nFiles written:")
        for relpath in written:
            print(f"  {relpath}")
    else:
        print("\nNo new files written (everything already existed). "
              "Use --force to regenerate component files.")

    print("\nNext steps:")
    print(f"  cd {dist_name}")
    print(f"  pip install -e .          # Install the plugin as a package")
    print(f"  # implement the generated {plugin_type} stub(s) in {package_name}/")
    print()
