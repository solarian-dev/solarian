"""Surgical edits to ``conf/<module>.yaml`` that preserve comments."""

from __future__ import annotations

import os
import re
from typing import Any, Optional

_ACTUATOR_KEY_RE = re.compile(r"^(\s*)([A-Za-z_][\w]*)\s*:\s*(?:#.*)?$")
_SIM_JOINT_NESTED_RE = re.compile(
    r"""^(\s*)joint\s*:\s*["']?([^"'#\n]+?)["']?\s*(?:#.*)?$"""
)
_SIM_JOINT_FLAT_RE = re.compile(
    r"""^(\s*)sim_joint\s*:\s*["']?([^"'#\n]+?)["']?\s*(?:#.*)?$"""
)


def _format_limit(value: float) -> str:
    text = f"{value:.10g}"
    if text in ("-0", "-0.0"):
        return "0"
    return text


def _bare_joint_name(joint_ref: str) -> str:
    return joint_ref.strip().strip("\"'").rsplit("/", 1)[-1]


def _split_newline(raw: str) -> tuple[str, str]:
    if raw.endswith("\r\n"):
        return raw[:-2], "\r\n"
    if raw.endswith("\n"):
        return raw[:-1], "\n"
    return raw, ""


def _field_line_re(name: str) -> re.Pattern[str]:
    return re.compile(rf"^(\s*){re.escape(name)}\s*:\s*[^\n#]*")


def module_yaml_path(module_name: str) -> Optional[str]:
    """``conf/<module_name>.yaml``, or ``None`` when the project has no such file."""
    conf_path = os.environ.get(
        "SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf")
    )
    path = os.path.join(conf_path, f"{module_name}.yaml")
    return path if os.path.isfile(path) else None


def read_module_field(module_name: str, key: str, default: float) -> float:
    """One top-level numeric field from a module's yaml.

    For readers outside the module's own process — a skill runs in its own and
    never holds the object. Only the named key is selected, so interpolations
    elsewhere in the file (``${deployment.*}``) are never resolved and no
    deployment tree has to be composed to read one number.
    """
    from omegaconf import OmegaConf

    path = module_yaml_path(module_name)
    if path is None:
        return default
    value = OmegaConf.select(OmegaConf.load(path), key)
    return default if value is None else float(value)


def discover_actuator_sim_joints(lines: list[str]) -> dict[str, str]:
    """Map actuator key → bare MJCF joint name.

    An actuator participates if it has ``sim_joint``, ``sim_range_min``, or a
    leftover nested ``sim:`` block. Default joint name is the actuator key.
    """
    actuators_indent: Optional[int] = None
    actuator_key_indent: Optional[int] = None
    current_actuator: Optional[str] = None
    current_has_sim = False
    current_joint: Optional[str] = None
    in_sim = False
    sim_indent: Optional[int] = None
    mapping: dict[str, str] = {}

    def _commit() -> None:
        nonlocal current_actuator, current_has_sim, current_joint
        if current_actuator is not None and current_has_sim:
            mapping[current_actuator] = current_joint or current_actuator
        current_actuator = None
        current_has_sim = False
        current_joint = None

    for raw in lines:
        body, _ = _split_newline(raw)
        if not body.strip() or body.lstrip().startswith("#"):
            continue
        indent = len(body) - len(body.lstrip(" "))

        if actuators_indent is None:
            if re.match(r"^\s*actuators\s*:", body):
                actuators_indent = indent
            continue

        if indent <= actuators_indent:
            _commit()
            break

        if actuator_key_indent is None:
            m = _ACTUATOR_KEY_RE.match(body)
            if m and len(m.group(1)) > actuators_indent:
                actuator_key_indent = len(m.group(1))

        if actuator_key_indent is not None and indent == actuator_key_indent:
            m = _ACTUATOR_KEY_RE.match(body)
            if m:
                _commit()
                current_actuator = m.group(2)
                in_sim = False
                sim_indent = None
                continue

        if current_actuator is None or actuator_key_indent is None:
            continue

        if re.match(r"^\s*sim\s*:", body) and indent > actuator_key_indent:
            in_sim = True
            sim_indent = indent
            current_has_sim = True
            continue

        if _SIM_JOINT_FLAT_RE.match(body) or re.match(r"^\s*sim_range_min\s*:", body):
            current_has_sim = True
            jm = _SIM_JOINT_FLAT_RE.match(body)
            if jm:
                current_joint = _bare_joint_name(jm.group(2))

        if in_sim and sim_indent is not None:
            if indent <= sim_indent:
                in_sim = False
                sim_indent = None
            else:
                jm = _SIM_JOINT_NESTED_RE.match(body)
                if jm:
                    current_joint = _bare_joint_name(jm.group(2))

    _commit()
    return mapping


def patch_actuator_fields(
    yaml_path: str, updates: dict[str, dict[str, Any]]
) -> list[str]:
    """Set scalar fields on named actuators, preserving comments.

    Existing keys are replaced in place (trailing comments kept). Missing keys
    are appended at the end of that actuator's block.
    """
    path = os.path.abspath(yaml_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Module yaml not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)

    actuators_indent: Optional[int] = None
    actuator_key_indent: Optional[int] = None
    current_actuator: Optional[str] = None
    child_indent: Optional[int] = None
    pending: dict[str, Any] = {}
    written: set[str] = set()
    updated: list[str] = []
    out: list[str] = []

    def _flush_pending(newline: str) -> None:
        nonlocal pending, written
        if current_actuator is None or not pending:
            return
        indent = child_indent if child_indent is not None else (
            (actuator_key_indent + 4) if actuator_key_indent is not None else 4
        )
        pad = " " * indent
        for key, value in list(pending.items()):
            out.append(f"{pad}{key}: {_format_value(value)}{newline}")
            written.add(f"{current_actuator}.{key}")
        pending = {}
        if current_actuator not in updated:
            updated.append(current_actuator)

    for raw in lines:
        body, newline = _split_newline(raw)
        stripped = body.lstrip(" ")
        indent = len(body) - len(stripped) if stripped else len(body)

        if actuators_indent is None:
            if re.match(r"^\s*actuators\s*:", body):
                actuators_indent = indent
            out.append(raw)
            continue

        if indent <= actuators_indent and stripped and not stripped.startswith("#"):
            _flush_pending(newline)
            actuators_indent = None
            current_actuator = None
            out.append(raw)
            continue

        if actuator_key_indent is None:
            m = _ACTUATOR_KEY_RE.match(body)
            if m and actuators_indent is not None and len(m.group(1)) > actuators_indent:
                actuator_key_indent = len(m.group(1))

        if (
            actuator_key_indent is not None
            and indent == actuator_key_indent
            and stripped
            and not stripped.startswith("#")
        ):
            m = _ACTUATOR_KEY_RE.match(body)
            if m:
                _flush_pending(newline)
                current_actuator = m.group(2)
                pending = dict(updates.get(current_actuator) or {})
                child_indent = None
            out.append(raw)
            continue

        if (
            current_actuator is not None
            and pending
            and stripped
            and not stripped.startswith("#")
            and actuator_key_indent is not None
            and indent > actuator_key_indent
        ):
            if child_indent is None:
                child_indent = indent
            if indent == child_indent:
                trailing = ""
                comment_idx = body.find(" #")
                if comment_idx >= 0:
                    trailing = body[comment_idx:]
                for key in list(pending):
                    aliases = (key,) + (("homing_offset",) if key == "center_offset" else ())
                    if any(_field_line_re(alias).match(body) for alias in aliases):
                        out.append(
                            f"{body[:indent]}{key}: {_format_value(pending.pop(key))}"
                            f"{trailing}{newline}"
                        )
                        written.add(f"{current_actuator}.{key}")
                        if current_actuator not in updated:
                            updated.append(current_actuator)
                        break
                else:
                    out.append(raw)
                continue

        out.append(raw)

    _flush_pending("\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)
    return updated


def patch_top_level_value(yaml_path: str, key: str, value: Any) -> None:
    """Set a top-level scalar (or flow list) key, preserving comments.

    Only column zero counts as top level, so writing ``max_speed`` cannot land
    on an actuator that happens to declare one. An existing key keeps its
    trailing comment; a new one goes in above the hardware map, which is where
    a reader looks for the module's own settings.
    """
    path = os.path.abspath(yaml_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Module yaml not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)

    key_re = re.compile(rf"^{re.escape(key)}\s*:\s*[^\n#]*")
    # A camera declares sensors and no actuators.
    hardware_re = re.compile(r"^(actuators|sensors)\s*:")
    newline = _split_newline(lines[0])[1] if lines else "\n"
    rendered = f"{key}: {_format_value(value)}"

    for i, raw in enumerate(lines):
        body, line_newline = _split_newline(raw)
        if not key_re.match(body):
            continue
        # Kept verbatim, spacing included: a trailing comment is usually the
        # provenance of the number being overwritten.
        comment = re.match(r"^[^#]*?(\s*#.*)$", body)
        trailing = comment.group(1) if comment else ""
        lines[i] = f"{rendered}{trailing}{line_newline or newline}"
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return

    block = [f"{rendered}{newline}"]
    for i, raw in enumerate(lines):
        if hardware_re.match(_split_newline(raw)[0]):
            lines[i:i] = block
            break
    else:
        if lines and not lines[-1].endswith(("\n", "\r\n")):
            lines[-1] = lines[-1] + newline
        lines.extend(block)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def patch_top_level_mapping(
    yaml_path: str, key: str, values: dict[str, Any]
) -> None:
    """Replace or insert a top-level mapping (e.g. ``home:``), preserving comments."""
    path = os.path.abspath(yaml_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Module yaml not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)

    key_re = re.compile(rf"^(\s*){re.escape(key)}\s*:")
    actuators_re = re.compile(r"^(\s*)actuators\s*:")
    start: Optional[int] = None
    key_indent: Optional[int] = None
    end: Optional[int] = None
    actuators_at: Optional[int] = None
    file_indent = 4

    for i, raw in enumerate(lines):
        body, _ = _split_newline(raw)
        if actuators_re.match(body) and actuators_at is None:
            actuators_at = i
        m = key_re.match(body)
        if m and start is None:
            start = i
            key_indent = len(m.group(1))
            continue
        if start is not None and key_indent is not None and end is None:
            stripped = body.lstrip(" ")
            indent = len(body) - len(stripped) if stripped else len(body)
            if stripped and not stripped.startswith("#") and indent <= key_indent:
                end = i
                break
    if start is not None and end is None:
        end = len(lines)

    newline = "\n"
    if lines:
        _, newline = _split_newline(lines[0])
    child_pad = " " * (
        (key_indent + file_indent) if key_indent is not None else file_indent
    )
    block = [f"{key}:{newline}"]
    for name, value in values.items():
        block.append(f"{child_pad}{name}: {_format_value(value)}{newline}")

    if start is not None:
        out = lines[:start] + block + lines[end:]
    elif actuators_at is not None:
        out = lines[:actuators_at] + block + [newline] + lines[actuators_at:]
    else:
        if lines and not lines[-1].endswith(("\n", "\r\n")):
            lines[-1] = lines[-1] + newline
        if lines and lines[-1].strip():
            lines.append(newline)
        out = lines + block

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _format_limit(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    return str(value)


def sync_sim_ranges(
    yaml_path: str, ranges: dict[str, tuple[float, float]]
) -> list[str]:
    """Update ``sim_range_min`` / ``sim_range_max`` (or leftover nested names)."""
    path = os.path.abspath(yaml_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Module yaml not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)

    actuator_joints = discover_actuator_sim_joints(lines)
    update_for: dict[str, tuple[float, float]] = {
        actuator: ranges[joint]
        for actuator, joint in actuator_joints.items()
        if joint in ranges
    }
    field_updates = {
        name: {
            "sim_range_min": lo,
            "sim_range_max": hi,
        }
        for name, (lo, hi) in update_for.items()
    }
    # Also rewrite leftover nested ``range_min`` / ``range_max`` under ``sim:``.
    nested = _rewrite_nested_sim_ranges(lines, update_for)
    if nested is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(nested)
    return patch_actuator_fields(path, field_updates)


def _rewrite_nested_sim_ranges(
    lines: list[str], update_for: dict[str, tuple[float, float]]
) -> list[str] | None:
    """Replace ``range_min`` / ``range_max`` inside leftover ``sim:`` blocks."""
    range_min_re = re.compile(r"^(\s*)range_min\s*:\s*[^\n#]*")
    range_max_re = re.compile(r"^(\s*)range_max\s*:\s*[^\n#]*")
    actuators_indent: Optional[int] = None
    actuator_key_indent: Optional[int] = None
    current_actuator: Optional[str] = None
    in_sim = False
    sim_indent: Optional[int] = None
    changed = False
    out: list[str] = []

    for raw in lines:
        body, newline = _split_newline(raw)
        stripped = body.lstrip(" ")
        indent = len(body) - len(stripped) if stripped else len(body)

        if actuators_indent is None:
            if re.match(r"^\s*actuators\s*:", body):
                actuators_indent = indent
            out.append(raw)
            continue

        if indent <= actuators_indent and stripped and not stripped.startswith("#"):
            actuators_indent = None
            current_actuator = None
            in_sim = False
            out.append(raw)
            continue

        if actuator_key_indent is None:
            m = _ACTUATOR_KEY_RE.match(body)
            if m and actuators_indent is not None and len(m.group(1)) > actuators_indent:
                actuator_key_indent = len(m.group(1))

        if (
            actuator_key_indent is not None
            and indent == actuator_key_indent
            and stripped
            and not stripped.startswith("#")
        ):
            m = _ACTUATOR_KEY_RE.match(body)
            if m:
                current_actuator = m.group(2)
                in_sim = False
                sim_indent = None
            out.append(raw)
            continue

        if current_actuator is not None and re.match(r"^\s*sim\s*:", body):
            if actuator_key_indent is None or indent > actuator_key_indent:
                in_sim = True
                sim_indent = indent
                out.append(raw)
                continue

        if in_sim and sim_indent is not None:
            if stripped and not stripped.startswith("#") and indent <= sim_indent:
                in_sim = False
                sim_indent = None
            elif current_actuator in update_for:
                lo, hi = update_for[current_actuator]
                trailing = ""
                comment_idx = body.find(" #")
                if comment_idx >= 0:
                    trailing = body[comment_idx:]
                if range_min_re.match(body):
                    out.append(
                        f"{body[:indent]}range_min: {_format_limit(lo)}{trailing}{newline}"
                    )
                    changed = True
                    continue
                if range_max_re.match(body):
                    out.append(
                        f"{body[:indent]}range_max: {_format_limit(hi)}{trailing}{newline}"
                    )
                    changed = True
                    continue

        out.append(raw)

    return out if changed else None
