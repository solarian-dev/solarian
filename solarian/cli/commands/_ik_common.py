"""Shared helpers for the IK sim<->real alignment commands.

Keeps ``verify-ik-calibration``, ``derive-model-limits`` and ``calibrate-ik``
DRY: discover the project's IK module(s), build the kinematics backend (for the
model joint limits), and resolve encoder counts-per-revolution offline via the
servo bus's optional angular hook.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from solarian.utils import get_class


def _collect_modules_from_class(cls) -> dict[str, type]:
    """Return ``{attr_name: ModuleSubclass}`` for every Module descriptor on ``cls``."""
    from solarian.core.module import Module

    modules: dict[str, type] = {}
    for name in cls.__dict__:
        val = getattr(cls, name, None)
        if isinstance(val, Module):
            modules[name] = val.__class__
    return modules


def collect_ik_modules() -> dict[str, type]:
    """Find project modules that declare an IK backend (``ik_backend`` set).

    Looks on the project's ``ROBOT_CLASS`` (and ``CONTROLLER_CLASS``) declared in
    settings. Keyed by the module's attribute name (which is also its yaml name).
    """
    from solarian.conf import settings

    found: dict[str, type] = {}
    for class_attr in ("ROBOT_CLASS", "CONTROLLER_CLASS"):
        class_path = getattr(settings, class_attr, None)
        if not class_path:
            continue
        parent_cls = get_class(class_path)
        for name, module_cls in _collect_modules_from_class(parent_cls).items():
            if getattr(module_cls, "ik_backend", None) is not None:
                found[name] = module_cls
    return found


def select_ik_module(module_name: Optional[str] = None) -> tuple[str, type]:
    """Resolve which IK module to operate on (explicit, single, or prompt)."""
    modules = collect_ik_modules()
    if not modules:
        raise SystemExit(
            "No IK modules found. A module must declare a class-level `ik_backend` "
            "(e.g. `ik_backend = MinkIK(...)`) and Cartesian IK actuators."
        )
    if module_name:
        if module_name not in modules:
            raise SystemExit(f"Module '{module_name}' is not an IK module. Options: {list(modules)}")
        return module_name, modules[module_name]
    if len(modules) == 1:
        name = next(iter(modules))
        return name, modules[name]

    entries = list(modules.items())
    print("\n=== Available IK modules ===")
    for idx, (name, cls) in enumerate(entries, 1):
        print(f"{idx}. {name} ({cls.__name__})")
    while True:
        raw = input(f"Select module (1-{len(entries)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(entries):
            return entries[int(raw) - 1]
        print("Invalid selection.")


def load_module(module_name: str, module_class: type):
    """Instantiate a module from its ``conf/<name>.yaml`` (not connected)."""
    from solarian.utils.configs import load_config_yaml

    conf_path = os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))
    yaml_path = os.path.join(conf_path, f"{module_name}.yaml")
    module_cfg = load_config_yaml(yaml_path, module_class.config_class)
    return module_class(module_cfg, module_name)


def build_backend(module_class: type):
    """Build the module's differential IK backend (needs the ``[sim]`` extra).

    Exposes ``joint_actuator_names`` and ``joint_limits`` (the model ``jnt_range``)
    without touching hardware.
    """
    spec = module_class.ik_backend
    if hasattr(spec, "build"):
        return spec.build()
    if callable(spec):
        return spec()
    return spec


def counts_per_rev_resolver(module) -> Optional[dict[str, float]]:
    """Best-effort offline ``{joint: counts/rev}`` via the servo bus angular hook.

    Constructs each port's joint bus without opening the serial port (the
    resolution comes from the motor model table), so it works from a JSON file
    with no hardware attached. Returns ``None`` if nothing could be resolved.
    """
    from solarian.core.ik.actuator import CartesianIKActuator

    resolved: dict[str, float] = {}
    for port, names in module.actuators_by_port.items():
        components = {
            n: module.actuators[n]
            for n in names
            if not isinstance(module.actuators[n], CartesianIKActuator)
        }
        if not components:
            continue
        first = next(iter(components.values()))
        bus_class = getattr(first, "bus_class", None)
        if bus_class is None:
            continue
        try:
            bus = bus_class(port=port, components=components, calibration=None)
        except Exception:
            continue
        for name in components:
            try:
                cpr = bus.counts_per_revolution(name)
            except Exception:
                cpr = None
            if cpr:
                resolved[name] = float(cpr)
    return resolved or None


def load_calibration_dict(module) -> dict[str, Any]:
    """Load the module's calibration JSON ({joint: MotorCalibration}) or empty."""
    module.load_calibration()
    return module.calibration or {}
