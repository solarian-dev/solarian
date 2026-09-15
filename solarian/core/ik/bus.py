"""Composite actuator bus that adds Cartesian (IK) control to a joint bus.

An :class:`IKActuatorBus` wraps the *inner* joint bus a port would otherwise use
(a serial ``FeetechBus`` on real hardware, or a ``BusClient`` proxy to the
MuJoCo world in simulation) and layers differential inverse kinematics on top:

* Joint-space reads/writes pass straight through to the inner bus.
* Cartesian-space reads run forward kinematics (FK) on the current joint
  positions and return the requested pose components.
* Cartesian-space writes run one differential IK step (solve + integrate) and
  write the resulting joint targets to the inner bus.

Joints are exchanged with the inner bus in the normalized ``[-100, 100]`` space
(so sim and real stay transparent) and converted to/from radians -- which the IK
backend reasons in -- using the model joint limits the backend exposes.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import numpy as np

from solarian.core.actuator_bus import ActuatorBus
from solarian.core.data import Data, PosData
from solarian.core.ik.base import DifferentialIKBackend

import logging

logger = logging.getLogger(__name__)

# Opt-in diagnostics: set SOLARIAN_IK_DEBUG=1 to log commanded vs. measured
# Cartesian pose (and joint angles) each write. Off by default so normal
# (and sim) runs are unaffected.
_IK_DEBUG = os.environ.get("SOLARIAN_IK_DEBUG", "0").lower() not in ("0", "", "false", "no")
# Minimum seconds between debug log lines so a fast control loop doesn't flood.
_IK_DEBUG_PERIOD = 0.2

# Bounds on the differential-IK integration timestep, measured from the actual
# inter-write cadence. Keeps a stalled loop from feeding a wildly out-of-range dt
# into the solver.
_MIN_DT = 1.0 / 250.0
_MAX_DT = 1.0 / 30.0
_DEFAULT_DT = 1.0 / 120.0


class IKActuatorBus(ActuatorBus):
    """Wrap a joint bus so a module's Cartesian actuators are driven by IK.

    Args:
        inner_bus: The joint bus for the port (``FeetechBus``, ``BusClient``,
            ...). Owns all hardware/joint I/O and calibration.
        joint_components: All non-Cartesian actuators on the port (incl. e.g. a
            gripper that IK does not control), keyed by name.
        cartesian_components: The Cartesian actuators, keyed by name. Each maps
            to a pose index (0..5 == x, y, z, rx, ry, rz) via its ``cfg.id``.
        backend: The differential IK backend. Its
            :attr:`~solarian.core.ik.base.DifferentialIKBackend.joint_actuator_names`
            select and order the subset of joints IK actually controls.
    """

    def __init__(
        self,
        inner_bus: Any,
        joint_components: Dict[str, Any],
        cartesian_components: Dict[str, Any],
        backend: DifferentialIKBackend,
    ):
        self.inner_bus = inner_bus
        self.joint_components = dict(joint_components)
        self.cartesian_components = dict(cartesian_components)
        self.backend = backend

        self.port = getattr(inner_bus, "port", None)
        self.components = {**self.joint_components, **self.cartesian_components}

        # Pose index (0..5) each Cartesian actuator drives.
        self.cartesian_index = {
            name: int(comp.cfg.id) for name, comp in self.cartesian_components.items()
        }
        # Joints IK controls (ordered as the backend expects q).
        self.ik_joint_names = list(backend.joint_actuator_names)
        self._limits = dict(backend.joint_limits)

        self._last_write_time: float | None = None
        self._dt_ema: float = _DEFAULT_DT
        self._last_debug_time: float = 0.0

        # Differential IK converges to the target by iterating solve+integrate
        # within a single control tick (the mink-recommended usage) instead of
        # taking one velocity step per tick. One step badly undershoots -- worst
        # in ill-conditioned directions and with high ``lm_damping`` -- which
        # shows up as imprecise Cartesian tracking. We iterate the model to the
        # target and early-exit once the pose error is within tolerance, so a
        # nearby (normal teleop) target costs only a couple of iterations.
        self.ik_iterations = 12
        self.ik_pos_tol = 1e-4  # metres
        self.ik_ori_tol = 1e-3  # radians

    # -- connection (delegate to inner bus) --------------------------------
    @staticmethod
    def _is_connected(bus: Any) -> bool:
        # ``is_connected`` is a property on some buses and a method on others.
        attr = getattr(bus, "is_connected", False)
        return bool(attr() if callable(attr) else attr)

    @property
    def is_connected(self) -> bool:
        return self._is_connected(self.inner_bus)

    def connect(self):
        if not self._is_connected(self.inner_bus):
            self.inner_bus.connect()

    def disconnect(self):
        self.inner_bus.disconnect()

    # -- optional angular hook (delegated to the inner joint bus) -----------
    def counts_per_revolution(self, name: str):
        fn = getattr(self.inner_bus, "counts_per_revolution", None)
        return fn(name) if callable(fn) else None

    def center_offset_in_hardware(self, name: str):
        fn = getattr(self.inner_bus, "center_offset_in_hardware", None)
        return bool(fn(name)) if callable(fn) else False

    def max_center_offset_abs(self, name: str):
        fn = getattr(self.inner_bus, "max_center_offset_abs", None)
        return fn(name) if callable(fn) else None

    def max_homing_offset_abs(self, name: str):
        fn = getattr(self.inner_bus, "max_center_offset_abs", None) or getattr(
            self.inner_bus, "max_homing_offset_abs", None
        )
        return fn(name) if callable(fn) else None

    def raw_to_radians(self, name: str, raw: float):
        fn = getattr(self.inner_bus, "raw_to_radians", None)
        if callable(fn):
            return fn(name, raw)
        return super().raw_to_radians(name, raw)

    # -- normalized <-> radian joint conversion ----------------------------
    # NOTE: ``drive_mode`` (joint direction sign) is owned by the *inner* joint
    # bus, which already applies it when normalizing raw encoder units to the
    # ``[-100, 100]`` space (real: ``ActuatorBus._normalize``; sim: the
    # MuJoCo world's ``_normalize``). This layer must NOT re-apply it -- doing so
    # would cancel it out (double flip), silently ignoring the sign set during
    # IK sim<->real alignment. So the conversions below are pure normalized<->
    # radian maps over the model limits, matching the composed formula in
    # ``solarian.core.ik.alignment``.
    def _norm_to_rad(self, name: str, norm: float) -> float:
        mode = self.joint_components[name].cfg.mode
        lo, hi = self._limits[name]
        if mode == "range_m100_100":
            v = min(100.0, max(-100.0, norm))
            return ((v + 100.0) / 200.0) * (hi - lo) + lo
        if mode == "range_0_100":
            v = min(100.0, max(0.0, norm))
            return (v / 100.0) * (hi - lo) + lo
        return norm

    def _rad_to_norm(self, name: str, rad: float) -> float:
        mode = self.joint_components[name].cfg.mode
        lo, hi = self._limits[name]
        if hi == lo:
            return 0.0
        if mode == "range_m100_100":
            bounded = min(hi, max(lo, rad))
            return (((bounded - lo) / (hi - lo)) * 200.0) - 100.0
        if mode == "range_0_100":
            bounded = min(hi, max(lo, rad))
            return ((bounded - lo) / (hi - lo)) * 100.0
        return rad

    def _read_ik_joint_radians(self) -> np.ndarray:
        norm = self.inner_bus.read(PosData, self.ik_joint_names, normalize=True)
        return np.array(
            [self._norm_to_rad(name, float(norm[name])) for name in self.ik_joint_names]
        )

    def _current_pose(self) -> np.ndarray:
        self.backend.set_q(self._read_ik_joint_radians())
        return np.asarray(self.backend.get_pose(), dtype=float)

    def _tick_dt(self) -> float:
        now = time.monotonic()
        if self._last_write_time is not None:
            measured = now - self._last_write_time
            if measured > 0:
                self._dt_ema = 0.2 * measured + 0.8 * self._dt_ema
        self._last_write_time = now
        return max(_MIN_DT, min(_MAX_DT, self._dt_ema))

    # -- read / write ------------------------------------------------------
    def read(self, data_type: type[Data], component_names: list[str], normalize: bool = True) -> Dict[str, Any]:
        joint_names = [n for n in component_names if n in self.joint_components]
        cart_names = [n for n in component_names if n in self.cartesian_components]

        result: Dict[str, Any] = {}
        if joint_names:
            result.update(self.inner_bus.read(data_type, joint_names, normalize=normalize))

        if cart_names and issubclass(data_type, PosData):
            pose = self._current_pose()
            for name in cart_names:
                result[name] = float(pose[self.cartesian_index[name]])
        elif cart_names:
            # Only position (pose) reads are meaningful for Cartesian actuators.
            for name in cart_names:
                result[name] = 0.0
        return result

    def write(self, values: Dict[str, Data], normalize: bool = True):
        joint_vals = {n: v for n, v in values.items() if n in self.joint_components}
        cart_vals = {n: v for n, v in values.items() if n in self.cartesian_components}

        # Direct joint commands first; a Cartesian command then overrides the
        # IK-controlled arm joints for this tick.
        if joint_vals:
            self.inner_bus.write(joint_vals, normalize=normalize)
        if cart_vals:
            self._write_cartesian(cart_vals)

    def _write_cartesian(self, cart_vals: Dict[str, Data]):
        dt = self._tick_dt()

        q_now = self._read_ik_joint_radians()
        self.backend.set_q(q_now)
        current_pose = np.asarray(self.backend.get_pose(), dtype=float).copy()
        target = current_pose.copy()
        for name, data in cart_vals.items():
            target[self.cartesian_index[name]] = float(data.value)

        # Iterate solve+integrate to drive the model onto the target this tick
        # (see __init__: ik_iterations). Each pass feeds the integrated joints
        # back via set_q so the next solve sees the reduced error; we stop early
        # once within tolerance. The backend's configuration is left at the
        # converged solution -- fine, since the next tick re-seeds it from the
        # freshly read joint positions.
        q_next = q_now
        for _ in range(max(1, int(self.ik_iterations))):
            v = self.backend.solve(target, dt)
            q_next = np.asarray(self.backend.integrate(v, dt), dtype=float)
            self.backend.set_q(q_next)
            pose = np.asarray(self.backend.get_pose(), dtype=float)
            if (
                np.linalg.norm(pose[:3] - target[:3]) < self.ik_pos_tol
                and np.linalg.norm(pose[3:] - target[3:]) < self.ik_ori_tol
            ):
                break

        out = {
            name: PosData(value=self._rad_to_norm(name, float(rad)))
            for name, rad in zip(self.ik_joint_names, q_next)
        }
        self.inner_bus.write(out, normalize=True)

        if _IK_DEBUG:
            self._debug_log(cart_vals, current_pose, target, q_now, q_next, dt)

    def _debug_log(self, cart_vals, current_pose, target, q_now, q_next, dt):
        now = time.monotonic()
        if now - self._last_debug_time < _IK_DEBUG_PERIOD:
            return
        self._last_debug_time = now
        commanded = {name: round(float(data.value), 4) for name, data in cart_vals.items()}
        # Cartesian delta the solver is trying to achieve this tick (target minus
        # the pose measured from the current joint angles).
        pose_err = np.round(target - current_pose, 4)
        logger.info(
            "IK[%s] dt=%.4f cmd=%s\n"
            "    measured_pose xyz=%s rxyz=%s\n"
            "    target_pose   xyz=%s rxyz=%s\n"
            "    pose_err(t-m) xyz=%s rxyz=%s\n"
            "    q_now(rad)=%s\n"
            "    q_next(rad)=%s\n"
            "    joints=%s",
            self.port, dt, commanded,
            np.round(current_pose[:3], 4), np.round(current_pose[3:], 4),
            np.round(target[:3], 4), np.round(target[3:], 4),
            pose_err[:3], pose_err[3:],
            np.round(np.asarray(q_now, dtype=float), 4),
            np.round(np.asarray(q_next, dtype=float), 4),
            self.ik_joint_names,
        )

    # -- hardware / calibration ops (delegate joint names only) ------------
    def _joint_subset(self, component_names: list[str]) -> list[str]:
        return [n for n in component_names if n in self.joint_components]

    def configure(self, params: dict[str, int], component_names: list[str]):
        names = self._joint_subset(component_names)
        if names:
            self.inner_bus.configure(params, names)

    def enable_torque(self, component_names: list[str]):
        names = self._joint_subset(component_names)
        if names:
            self.inner_bus.enable_torque(names)

    def disable_torque(self, component_names: list[str]):
        names = self._joint_subset(component_names)
        if names:
            self.inner_bus.disable_torque(names)

    def is_calibrated(self) -> bool:
        return self.inner_bus.is_calibrated()

    def reset_calibration(self, component_names: list[str]):
        names = self._joint_subset(component_names)
        if names:
            self.inner_bus.reset_calibration(names)

    def read_calibration(self) -> dict[str, Any]:
        return self.inner_bus.read_calibration()

    def write_calibration(self, calibration_dict: dict[str, Any]) -> dict[str, Any]:
        return self.inner_bus.write_calibration(calibration_dict)

    def start_calibration_mode(self):
        return self.inner_bus.start_calibration_mode()

    def end_calibration_mode(self):
        return self.inner_bus.end_calibration_mode()

    def set_half_turn_homings(self, actuator_names: list[str]) -> Dict[str, int]:
        names = self._joint_subset(actuator_names)
        return self.inner_bus.set_half_turn_homings(names) if names else {}

    def drive_to_encoder_center(self, component_names: list[str]) -> Dict[str, int]:
        names = self._joint_subset(component_names)
        return self.inner_bus.drive_to_encoder_center(names) if names else {}

    def record_ranges_of_motion(self, actuator_names: list[str]):
        names = self._joint_subset(actuator_names)
        return self.inner_bus.record_ranges_of_motion(names)

    def setup_component(self, component_name: str, configs: Dict[str, Any]):
        return self.inner_bus.setup_component(component_name, configs)
