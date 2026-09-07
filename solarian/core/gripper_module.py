"""Reusable gripper module: a scalar interpolator driven by ``SCHEDULE_WAYPOINT``.

Concrete grippers declare a single actuator (``scalar_actuator``) and optionally
override :meth:`_scalar_to_actuator` when the action space differs from the
servo command space (e.g. aperture cm → normalized ``[-100, 100]``).

:attr:`open_value`, or ``cfg.aperture_open_cm`` when that config field is
present, is the open end of travel that bounds every commanded aperture.
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
        """Live measured scalar (same units as :meth:`schedule_scalar`)."""
        return self._read_current_scalar()

    def schedule_scalar(self, value: float, target_time_wall: float) -> None:
        """Queue a timestamped scalar waypoint (never drops client timestamps).

        ``drop_timestamps=False`` so teleop ``drive_waypoints: true`` cannot
        collapse a calibration sinusoid into adaptive "drive now".
        """
        self._schedule_scalar_waypoint(
            float(value),
            float(target_time_wall),
            time.monotonic(),
            drop_timestamps=False,
        )

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

    def _aperture_status_suffix(self) -> str:
        if getattr(self.cfg, "aperture_open_cm", None) is not None:
            return " (cm)"
        return ""

    def _publish_aperture_status(
        self, *, measured: float | None = None, cmd: float | None = None
    ) -> None:
        """Push live aperture (and command) onto the StatusBoard."""
        suffix = self._aperture_status_suffix()
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

    @action_handler(ActionType.SCHEDULE_WAYPOINT)
    def _handle_schedule_waypoint(self, action: ModuleAction, t_now: float) -> bool:
        """Stream a scalar waypoint, or run an exclusive speed-limited move.

        Whoever supplies the timing owns the path. With ``params.speed`` (see
        :meth:`_is_exclusive_move`) the module drives the scalar at a speed
        limit, taking as long as it takes. Otherwise the caller owns timing —
        ``drop_timestamps`` drives over the adaptive horizon, and a plain
        waypoint blends in at its own timestamp.
        """
        params = action.params
        if self._is_exclusive_move(params):
            return self._move_exclusive(action, t_now, params)

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

    def _move_exclusive(
        self, action: ModuleAction, t_now: float, params: dict
    ) -> bool:
        """Speed-limited go-to the action's scalar field.

        Same pattern as ``ArmModule.moveJ``: seed from the live scalar and drive
        at a speed limit, so the duration comes from the distance rather than
        the short teleop drive horizon.
        """
        extracted = self._extract_scalar_waypoint(action)
        if extracted is None:
            return True
        target = extracted[0]

        try:
            current = self._read_current_scalar()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Module '%s': exclusive move could not read current scalar (%s); "
                "seeding interpolator.",
                self.name,
                exc,
            )
            self._ensure_scalar_interpolator(t_now)
            current = (
                float(self.interp(t_now)) if self.interp is not None else float(target)
            )

        speed = params.get("speed")
        interp = ScalarTrajectoryInterpolator(
            np.array([t_now]), np.array([current], dtype=np.float64)
        )
        self.interp = interp.drive_to_waypoint(
            value=float(target),
            time=t_now,
            curr_time=t_now,
            max_speed=float(speed) if speed is not None else self._default_speed(),
        )
        self.last_waypoint_time = float(self.interp.times[-1])
        logger.info(
            "Module '%s': exclusive move %.3f → %.3f in %.2fs",
            self.name,
            current,
            target,
            self.last_waypoint_time - t_now,
        )
        return True
