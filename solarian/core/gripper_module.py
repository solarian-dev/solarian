"""Reusable gripper module: a scalar interpolator driven by ``MOVE_J`` / ``SERVO_J`` / ``SCHEDULE_J``.

Concrete grippers declare a single actuator (``scalar_actuator``) and optionally
override :meth:`_scalar_to_actuator` when the action space differs from the
servo command space (e.g. aperture cm → normalized ``[-100, 100]``).

:attr:`open_value`, or ``cfg.aperture_open_cm`` when that config field is
present, is the open end of travel that bounds every commanded aperture.

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
from solarian.core.data import PosData
from solarian.utils.status_board import StatusBoard
from solarian.utils.trajectory_interpolator import ScalarTrajectoryInterpolator

logger = logging.getLogger("module")


class GripperModule(ActuatorModule):
    """Actuator module with a 1-D interpolator (aperture, jaw width, ...)."""

    # Name of the actuator driven by the scalar interpolator. Defaults to the
    # sole non-Cartesian actuator when unset.
    scalar_actuator: str | None = None

    # Fully-open end of travel in interpolator/action units. ``None`` falls
    # back to ``cfg.aperture_open_cm`` when present.
    open_value: float | None = None
    # Default exclusive-move speed in interpolator units / second, used when
    # ``params.speed`` is absent. ``None`` uses ``cfg.max_speed`` (same units as
    # the scalar interpolator, e.g. cm/s).
    default_speed: float | None = None

    def _open_scalar(self) -> float:
        if self.open_value is not None:
            return float(self.open_value)
        cfg_open = getattr(self.cfg, "aperture_open_cm", None)
        if cfg_open is not None:
            return float(cfg_open)
        return 0.0

    def scalar_limits(self) -> tuple[float, float]:
        """Closed / open ends of travel in interpolator (action) units.

        Uses ``aperture_closed_cm`` / ``aperture_open_cm`` when both are set;
        otherwise ``(0.0, open)``.
        """
        closed = getattr(self.cfg, "aperture_closed_cm", None)
        opened = getattr(self.cfg, "aperture_open_cm", None)
        if closed is not None and opened is not None:
            return float(closed), float(opened)
        return 0.0, self._open_scalar()

    def read_scalar(self) -> float:
        """Live measured scalar (same units as :meth:`scheduleJ`)."""
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
        max_speed = getattr(self.cfg, "max_speed", 1.0)
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
        """Speed-limited go-to *q* (interpolator units)."""
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

    def aperture_unit_suffix(self) -> str:
        """`" (cm)"` when the config names a physical aperture, else `""`.

        Only grippers configured in centimetres can label their scalar; a plain
        `open_value` gripper is in unnamed action units.
        """
        if getattr(self.cfg, "aperture_open_cm", None) is not None:
            return " (cm)"
        return ""

    def _publish_aperture_status(
        self, *, measured: float | None = None, cmd: float | None = None
    ) -> None:
        """Push live aperture (and command) onto the StatusBoard."""
        suffix = self.aperture_unit_suffix()
        if measured is not None:
            StatusBoard.set(f"{self.name} aperture{suffix}", round(float(measured), 2))
        if cmd is not None:
            StatusBoard.set(f"{self.name} aperture cmd{suffix}", round(float(cmd), 2))

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
        """Put centimetres on ``obs['aperture_cm']`` so it shows in Observation."""
        getter = getattr(type(obs), "get_calculated_fields_spec", None)
        if getter is None:
            return
        spec = getter().get("aperture_cm")
        if spec is None:
            return
        obs["aperture_cm"] = spec.output.duplicate({"value": float(measured)})

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
