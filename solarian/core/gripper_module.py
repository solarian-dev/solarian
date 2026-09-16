"""Reusable gripper module: a scalar interpolator driven by ``MOVE_J`` / ``SERVO_J`` / ``SCHEDULE_J``.

Concrete grippers declare a single actuator (``scalar_actuator``) and are
commanded by finger aperture in millimetres. This module converts that aperture
into the servo's normalized ``[-100, 100]`` space; each backend then maps
normalized onto its own target (servo counts on hardware, a prismatic joint in
simulation).

The conversion is the measured :class:`~solarian.core.aperture_table.ApertureTable`
when ``cfg.aperture_table`` is set, and a straight line between
``aperture_closed_mm`` and ``aperture_open_mm`` otherwise. Linkage geometry makes
most parallel grippers non-linear, so the straight line is a starting point, not
a description of the hardware — run ``solarian calibrate-gripper-ranges`` to
replace it.

``moveJ`` / ``servoJ`` / ``scheduleJ`` are the same names as on
:class:`~solarian.core.arm_module.ArmModule`; here they drive the scalar
interpolator.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from solarian.core.action import ActionType, ModuleAction, action_handler
from solarian.core.actuator_module import ActuatorModule
from solarian.core.aperture_table import ApertureTable
from solarian.core.configs import ModuleConfig
from solarian.core.data import PosData
from solarian.utils.status_board import StatusBoard
from solarian.utils.trajectory_interpolator import ScalarTrajectoryInterpolator

logger = logging.getLogger("module")

#: Physical unit of the gripper action space. Apertures are commanded, reported
#: and calibrated in millimetres throughout.
APERTURE_UNIT = "mm"


class GripperConfig(ModuleConfig):
    """Module yaml for a gripper commanded by finger aperture in millimetres.

    ``aperture_closed_mm`` / ``aperture_open_mm`` are the physical fingertip
    separation at the two ends of servo travel, and must agree with the motor
    calibration (``real_range_*``) and the ``sim_range_*`` joint range.
    ``aperture_table``, once measured, supersedes them.
    """

    aperture_closed_mm: float = 0.0
    aperture_open_mm: float = 0.0
    #: Ruler-measured aperture curve from ``solarian calibrate-gripper-ranges``.
    aperture_table: ApertureTable | None = None


class GripperModule(ActuatorModule):
    """Actuator module with a 1-D interpolator over finger aperture (mm)."""

    config_class = GripperConfig

    # Name of the actuator driven by the scalar interpolator. Defaults to the
    # sole non-Cartesian actuator when unset.
    scalar_actuator: str | None = None

    # Default exclusive-move speed in mm/s, used when ``params.speed`` is
    # absent. ``None`` uses ``cfg.max_speed``.
    default_speed: float | None = None

    def scalar_limits(self) -> tuple[float, float]:
        """Closed / open ends of travel, in mm.

        A measured table defines the travel it was swept over, so it wins over
        the configured endpoints rather than being clipped by them.
        """
        if self.cfg.aperture_table is not None:
            return self.cfg.aperture_table.limits
        return float(self.cfg.aperture_closed_mm), float(self.cfg.aperture_open_mm)

    def _scalar_to_actuator(self, value: float) -> float:
        """Aperture (mm) -> the servo's normalized command [-100, 100]."""
        table = self.cfg.aperture_table
        if table is not None:
            return table.to_norm(value)
        lo, hi = self.scalar_limits()
        norm = (float(value) - lo) / (hi - lo) * 200.0 - 100.0
        return max(-100.0, min(100.0, norm))

    def _actuator_to_scalar(self, value: float) -> float:
        """Inverse of :meth:`_scalar_to_actuator` (normalized -> mm)."""
        table = self.cfg.aperture_table
        if table is not None:
            return table.to_aperture(value)
        lo, hi = self.scalar_limits()
        return lo + ((float(value) + 100.0) / 200.0) * (hi - lo)

    def read_aperture(self) -> float:
        """Live measured finger aperture in mm."""
        return self._read_current_scalar()

    def _scalar_value(self, q) -> float:
        return float(np.asarray(q, dtype=np.float64).reshape(-1)[0])

    def scheduleJ(
        self, q, target_time_wall: float, t_now: float | None = None
    ) -> None:
        """Queue a timestamped scalar waypoint."""
        t = float(t_now if t_now is not None else time.monotonic())
        self._schedule_scalar_waypoint(self._scalar_value(q), float(target_time_wall), t)

    def _default_speed(self) -> float:
        if self.default_speed is not None:
            return float(self.default_speed)
        max_speed = self.cfg.max_speed
        if isinstance(max_speed, (int, float)):
            return float(max_speed)
        return float(max_speed[0])

    def _write_scalar(self, value: float) -> None:
        name = self._scalar_actuator_name()
        if not name:
            return
        servo = float(self._scalar_to_actuator(value))
        self.write_actuators({name: PosData(value=servo)})
        StatusBoard.set(f"{self.name} servo cmd", round(servo, 1))

    def moveJ(
        self, q, speed: float | None = None, t_now: float | None = None
    ) -> None:
        """Speed-limited go-to *q* (mm of aperture)."""
        value = self._scalar_value(q)
        t = float(t_now if t_now is not None else time.monotonic())
        try:
            current = self._read_current_scalar()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Module '%s': moveJ could not read current scalar (%s); "
                "seeding interpolator.",
                self.name,
                exc,
            )
            self._ensure_scalar_interpolator(t)
            current = (
                float(self.interp(t)) if self.interp is not None else float(value)
            )
        interp = ScalarTrajectoryInterpolator(
            np.array([t]), np.array([current], dtype=np.float64)
        )
        self.interp = interp.drive_to_waypoint(
            value=value,
            time=t,
            curr_time=t,
            max_speed=float(speed) if speed is not None else self._default_speed(),
        )
        self.last_waypoint_time = float(self.interp.times[-1])

    def servoJ(
        self, q, duration: float = 0.1, t_now: float | None = None
    ) -> None:
        """Streaming scalar target over ``duration`` seconds."""
        value = self._scalar_value(q)
        t = float(t_now if t_now is not None else time.monotonic())
        self._ensure_scalar_interpolator(t)
        assert self.interp is not None
        duration = self._min_servo_duration(duration)
        self.interp = self.interp.drive_to_waypoint(
            value=value,
            time=t + duration,
            curr_time=t,
            **self._get_max_speed_kwargs(),
        )
        self.last_waypoint_time = t + duration

    def _publish_aperture_status(
        self, *, measured: float | None = None, cmd: float | None = None
    ) -> None:
        """Push live aperture (and command) onto the StatusBoard."""
        if measured is not None:
            StatusBoard.set(
                f"{self.name} aperture ({APERTURE_UNIT})", round(float(measured), 2)
            )
        if cmd is not None:
            StatusBoard.set(
                f"{self.name} aperture cmd ({APERTURE_UNIT})", round(float(cmd), 2)
            )

    def _scalar_from_obs(self, obs) -> float | None:
        name = self._scalar_actuator_name()
        if not name:
            return None
        fields = getattr(obs, "_fields", None)
        if not isinstance(fields, dict) or name not in fields:
            return None
        field = fields[name]
        raw = float(field["value"] if isinstance(field, dict) else field.value)
        return self._actuator_to_scalar(raw)

    def _write_aperture_obs_field(self, obs, measured: float) -> None:
        """Put millimetres on ``obs['aperture_mm']`` so it shows in Observation."""
        getter = getattr(type(obs), "get_calculated_fields_spec", None)
        if getter is None:
            return
        spec = getter().get("aperture_mm")
        if spec is None:
            return
        obs["aperture_mm"] = spec.output.duplicate({"value": float(measured)})

    def enrich_observation(self, obs):
        obs = super().enrich_observation(obs)
        measured = self._scalar_from_obs(obs)
        if measured is not None:
            self._publish_aperture_status(measured=measured)
            self._write_aperture_obs_field(obs, measured)
        return obs

    def get_initial_action_values(self) -> np.ndarray:
        try:
            return np.array([self._read_current_scalar()], dtype=np.float64)
        except Exception:
            return np.array([0.0], dtype=np.float64)

    def control_step(self):
        self._ensure_scalar_interpolator()
        cmd = None
        if self.interp is not None:
            cmd = float(self.interp(time.monotonic()))
            self._write_scalar(cmd)
        self._publish_aperture_status(cmd=cmd)

    @action_handler(ActionType.MOVE_J)
    def _handle_move_j(self, action: ModuleAction, t_now: float) -> bool:
        extracted = self._extract_scalar_waypoint(action)
        if extracted is None:
            return True
        speed = action.params.get("speed")
        self.moveJ(
            extracted[0],
            speed=float(speed) if speed is not None else None,
            t_now=t_now,
        )
        return True

    @action_handler(ActionType.SERVO_J)
    def _handle_servo_j(self, action: ModuleAction, t_now: float) -> bool:
        extracted = self._extract_scalar_waypoint(action)
        if extracted is None:
            return True
        self.servoJ(
            extracted[0],
            duration=self._adaptive_drive_horizon(t_now),
            t_now=t_now,
        )
        return True

    @action_handler(ActionType.SCHEDULE_J)
    def _handle_schedule_j(self, action: ModuleAction, t_now: float) -> bool:
        extracted = self._extract_scalar_waypoint(action)
        if extracted is None:
            return True
        self.scheduleJ(extracted[0], extracted[1], t_now=t_now)
        return True
