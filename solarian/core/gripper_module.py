"""Reusable gripper module: scalar interpolator and SET_HOME → fully open.

Concrete grippers declare a single actuator (``scalar_actuator``) and optionally
override :meth:`_scalar_to_actuator` when the action space differs from the
servo command space (e.g. aperture cm → normalized ``[-100, 100]``).

``SET_HOME`` drives the scalar interpolator to :attr:`home_value`, or to
``cfg.aperture_open_cm`` when that config field is present.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from solarian.core.action import ActionType, ModuleAction, action_handler
from solarian.core.actuator_module import ActuatorModule
from solarian.core.clock_sync import synced_time
from solarian.core.data import PosData

logger = logging.getLogger("module")


class GripperModule(ActuatorModule):
    """Actuator module with a 1-D interpolator (aperture, jaw width, ...)."""

    # Name of the actuator driven by the scalar interpolator. Defaults to the
    # sole non-Cartesian actuator when unset.
    scalar_actuator: str | None = None

    # Target for SET_HOME in interpolator/action units. ``None`` falls back to
    # ``cfg.aperture_open_cm`` when present.
    home_value: float | None = None

    def _scalar_actuator_name(self) -> str | None:
        explicit = getattr(type(self), "scalar_actuator", None)
        if explicit:
            return str(explicit)
        from solarian.core.ik.actuator import CartesianIKActuator

        names = [
            n
            for n, act in self.actuators.items()
            if not isinstance(act, CartesianIKActuator)
        ]
        return names[0] if names else None

    def _scalar_to_actuator(self, value: float) -> float:
        """Map interpolator/action units to ``write_actuators`` units."""
        return float(value)

    def _actuator_to_scalar(self, value: float) -> float:
        """Inverse of :meth:`_scalar_to_actuator` (for seeding from sensors)."""
        return float(value)

    def _home_scalar(self) -> float:
        if self.home_value is not None:
            return float(self.home_value)
        cfg_open = getattr(self.cfg, "aperture_open_cm", None)
        if cfg_open is not None:
            return float(cfg_open)
        return 0.0

    def _read_current_scalar(self) -> float:
        name = self._scalar_actuator_name()
        if not name:
            return 0.0
        readings = self.read_sensors(PosData, [name])
        entry = readings[name]
        raw = float(entry["value"] if isinstance(entry, dict) else entry.value)
        return self._actuator_to_scalar(raw)

    def _ensure_scalar_interpolator(self, t_now: float | None = None) -> None:
        if self.interp is not None:
            return
        t = float(t_now if t_now is not None else time.monotonic())
        try:
            value = self._read_current_scalar()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Module '%s': could not seed scalar interpolator (%s); using 0.",
                self.name,
                exc,
            )
            value = 0.0
        self.interp = self._create_interpolator(
            np.array([t]), np.array([value], dtype=np.float64)
        )
        self.last_waypoint_time = t

    def _extract_scalar_waypoint(
        self, action: ModuleAction
    ) -> tuple[float, float] | None:
        name = self._scalar_actuator_name()
        if not name:
            return None
        fields = action.values()
        if name not in fields:
            if len(fields) == 1:
                field = next(iter(fields.values()))
                return float(field.value), float(field.timestamp)
            logger.warning(
                "Module '%s': SCHEDULE_WAYPOINT missing scalar field %r; ignoring.",
                self.name,
                name,
            )
            return None
        field = fields[name]
        return float(field.value), float(field.timestamp)

    def _schedule_scalar_waypoint(
        self,
        value: float,
        target_time_wall: float,
        t_now: float,
        drop_timestamps: bool = False,
    ) -> None:
        self._ensure_scalar_interpolator(t_now)
        assert self.interp is not None
        if drop_timestamps:
            target_time = t_now + self._adaptive_drive_horizon(t_now)
            self.interp = self.interp.drive_to_waypoint(
                value=float(value),
                time=target_time,
                curr_time=t_now,
                **self._get_max_speed_kwargs(),
            )
            self.last_waypoint_time = target_time
            return

        target_time = time.monotonic() - synced_time() + float(target_time_wall)
        target_time = self._resolve_waypoint_time(target_time, t_now)
        self.interp = self.interp.schedule_waypoint(
            value=float(value),
            time=target_time,
            curr_time=t_now,
            last_waypoint_time=self.last_waypoint_time,
            **self._get_max_speed_kwargs(),
        )
        self.last_waypoint_time = target_time

    def _write_scalar(self, value: float) -> None:
        name = self._scalar_actuator_name()
        if not name:
            return
        self.write_actuators({name: PosData(value=self._scalar_to_actuator(value))})

    def get_initial_action_values(self) -> np.ndarray:
        try:
            return np.array([self._read_current_scalar()], dtype=np.float64)
        except Exception:
            return np.array([0.0], dtype=np.float64)

    def control_step(self):
        self._ensure_scalar_interpolator()
        if self.interp is not None:
            self._write_scalar(float(self.interp(time.monotonic())))

    @action_handler(ActionType.SCHEDULE_WAYPOINT)
    def _handle_schedule_waypoint(self, action: ModuleAction, t_now: float) -> bool:
        extracted = self._extract_scalar_waypoint(action)
        if extracted is None:
            return True
        value, target_time_wall = extracted
        self._schedule_scalar_waypoint(
            value,
            target_time_wall,
            t_now,
            drop_timestamps=self._drop_timestamps(action),
        )
        return True

    @action_handler(ActionType.SET_HOME)
    def _handle_set_home(self, action: ModuleAction, t_now: float) -> bool:
        del action
        self._schedule_scalar_waypoint(
            self._home_scalar(), t_now, t_now, drop_timestamps=True
        )
        return True
