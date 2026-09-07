"""Actuator-bearing modules: motors, calibration, interpolators.

Cameras and other sensor-only modules stay on :class:`Module`. Anything that
writes actuators (arms, grippers, leader devices) subclasses this.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from solarian.core.action import ActionType, ModuleAction, action_handler
from solarian.core.actuator_bus import ActuatorBus
from solarian.core.calibration import MotorCalibration
from solarian.core.clock_sync import synced_time
from solarian.core.configs import InterpolatorType, ModuleConfig
from solarian.core.data import PosData
from solarian.core.module import Module
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.observation_buffer import ObservationBuffer
from solarian.utils.status_board import StatusBoard
from solarian.utils.trajectory_interpolator import (
    JointTrajectoryInterpolator,
    PoseTrajectoryInterpolator,
    ScalarTrajectoryInterpolator,
    TrajectoryInterpolator,
)

logger = logging.getLogger("module")


class ActuatorModule(Module):
    """Module with actuators: torque, calibration sweep, trajectory interpolators."""

    def __init__(
        self,
        cfg: ModuleConfig | None = None,
        name: str | None = None,
        observation_buffer: ObservationBuffer | None = None,
        action_queue: ModuleActionQueue | None = None,
    ):
        super().__init__(cfg, name, observation_buffer, action_queue)
        if cfg is None:
            return

        # Interpolator runtime (pose / scalar / joint).
        self.interp: TrajectoryInterpolator | None = None
        self.last_waypoint_time: float = 0.0
        self.speed_kwargs: dict[str, Any] = {}
        self._late_waypoint_warn_t: float = 0.0
        self._late_waypoint_suppressed: int = 0
        self._last_action_time: float | None = None
        self._action_interval_ema: float = float(
            getattr(self.cfg, "drive_horizon_min_s", 1.0 / 30.0)
        )

    def _hardware_actuators_by_port(self) -> dict[str, list[str]]:
        """Port → hardware actuator names (excludes virtual IK Cartesian axes)."""
        from solarian.core.ik.actuator import CartesianIKActuator

        out: dict[str, list[str]] = {}
        for port, names in self.actuators_by_port.items():
            hw = [
                n for n in names
                if not isinstance(self.actuators[n], CartesianIKActuator)
            ]
            if hw:
                out[port] = hw
        return out

    def calibrate(self) -> None:
        """Calibrate hard-stop joints from a range sweep only.

        Flow:
        1. Reset homing offset to 0 and record raw min/max while the operator
           sweeps each joint's full travel (no middle-pose prompt). Glitch
           samples that would expand the span across the encoder wrap are
           ignored so travel near 0/CPR does not require remounting.
        2. Keep ``homing_offset=0``; refuse non-contiguous / over-wide spans.
        3. Ask the operator to move each joint to its **low command** end to set
           ``drive_mode``.
        """
        from solarian.utils.general import enter_pressed
        from solarian.core.joint_calibration import (
            CalibrationSafetyError,
            DerivedHoming,
            EncoderRangeTracker,
            calibration_from_raw_range,
            drive_mode_from_low_end_pose,
            low_command_label,
        )

        if self.cfg.calibration_file is None:
            logger.error(f"No calibration required for module {self.name}")
            return

        hw_by_port = self._hardware_actuators_by_port()
        if not hw_by_port:
            logger.error(f"No hardware actuators to calibrate on module {self.name}")
            return

        # Continuous (no hard-stop) joints are zeroed in ``calibrate-ik``, not here.
        continuous = set(getattr(type(self), "ik_continuous_joints", []) or [])
        if continuous:
            hw_by_port = {
                port: [n for n in names if n not in continuous]
                for port, names in hw_by_port.items()
            }
            hw_by_port = {p: ns for p, ns in hw_by_port.items() if ns}
            print(
                f"Skipping continuous joints (no hard stop; zero them in "
                f"calibrate-ik): {sorted(continuous)}"
            )
            if not hw_by_port:
                logger.error(
                    f"No hard-stop actuators left to calibrate on module {self.name}"
                )
                return

        print(f"\nRunning calibration of {self}")
        print(
            "Homing offset stays 0 (raw encoder). Move each joint through its\n"
            "FULL mechanical travel while recording, then press ENTER.\n"
            "The sweep defines the limits (no span cap). Travel must stay on\n"
            "one side of the encoder wrap (POS must not jump 4095→0). If a\n"
            "joint crosses the wrap, remount that horn, then re-run."
        )

        self.start_calibration_mode()

        for port, actuator_names in hw_by_port.items():
            bus = self.module_bus.component_bus_by_port[port]
            assert isinstance(bus, ActuatorBus)
            bus.reset_calibration(actuator_names)

        trackers: dict[str, EncoderRangeTracker] = {}
        for port, actuator_names in hw_by_port.items():
            bus = self.module_bus.component_bus_by_port[port]
            for name in actuator_names:
                cpr = bus.counts_per_revolution(name)
                if not cpr:
                    print(
                        f"\nCalibration aborted: {name} has no counts_per_revolution."
                    )
                    self.end_calibration_mode()
                    return
                trackers[name] = EncoderRangeTracker(cpr=int(cpr))

        print(
            "\nRecording positions. Move every joint through its entire range of "
            "motion, then press ENTER to stop..."
        )
        # Standalone calibrate-module never starts the robot UI; without this
        # the MIN/POS/MAX writes go nowhere. Stop before the drive-mode prompts
        # (Live would eat ``input()``) and before any later module fork.
        StatusBoard.start_ui()
        try:
            while not enter_pressed():
                status_min, status_pos, status_max = {}, {}, {}
                for port, actuator_names in hw_by_port.items():
                    bus = self.module_bus.component_bus_by_port[port]
                    positions = bus.read(PosData, actuator_names, normalize=False)
                    for name in actuator_names:
                        pos = int(positions[name])
                        trackers[name].update(pos)
                        t = trackers[name]
                        status_pos[name] = pos
                        status_min[name] = t.raw_min
                        status_max[name] = t.raw_max
                        if t.rejected:
                            StatusBoard.set(f"{name}_ignored", t.rejected)
                StatusBoard.set("MIN", status_min)
                StatusBoard.set("POS", status_pos)
                StatusBoard.set("MAX", status_max)
        finally:
            StatusBoard.stop()

        derived: dict[str, DerivedHoming] = {}
        try:
            for name, tracker in trackers.items():
                if tracker.rejected:
                    wrap_note = (
                        f" ({tracker.wrap_rejected} crossed encoder wrap)"
                        if tracker.wrap_rejected
                        else ""
                    )
                    print(
                        f"  {name}: ignored {tracker.rejected} glitch/wrap sample(s)"
                        f"{wrap_note}"
                    )
                raw_lo, raw_hi = tracker.result()
                derived[name] = calibration_from_raw_range(
                    name, raw_lo, raw_hi, tracker.cpr
                )
                d = derived[name]
                print(
                    f"  {name}: raw/Present range {d.range_min} .. {d.range_max} "
                    f"(span {d.range_max - d.range_min}, homing_offset=0)"
                )
                if tracker.wrap_rejected:
                    print(
                        f"  warning: {name} saw wrap-crossing samples during the "
                        f"sweep — if MIN/MAX look incomplete, remount the horn "
                        f"so travel does not cross 0/{tracker.cpr - 1}."
                    )
        except CalibrationSafetyError as exc:
            print(f"\nCalibration aborted (safety): {exc}")
            logger.error("Calibration aborted (safety): %s", exc)
            self.end_calibration_mode()
            return

        print("\n=== Drive mode (command direction) ===")
        drive_modes: dict[str, int] = {}
        for port, actuator_names in hw_by_port.items():
            bus = self.module_bus.component_bus_by_port[port]
            for name in actuator_names:
                mode = self.actuators[name].cfg.mode
                label = low_command_label(mode)
                input(
                    f"Move '{name}' to the pose that should map to command {label}, "
                    "then press ENTER..."
                )
                reading = bus.read(PosData, [name], normalize=False)
                pose = float(reading[name])
                d = derived[name]
                dm = drive_mode_from_low_end_pose(pose, d.range_min, d.range_max)
                drive_modes[name] = dm
                end = "min" if dm == 0 else "max"
                print(
                    f"  {name}: pose={pose:.0f} near encoder {end} → drive_mode={dm}"
                )
                sim = getattr(self.actuators[name].cfg, "sim", None)
                sim_dm = getattr(sim, "drive_mode", None) if sim is not None else None
                if sim_dm is not None and int(sim_dm) != dm:
                    print(
                        f"  warning: sim.drive_mode={sim_dm} disagrees with operator "
                        f"result ({dm}); keeping operator result."
                    )

        preserved = {
            name: entry
            for name, entry in (self.calibration or {}).items()
            if name in continuous
        }
        self.calibration = dict(preserved)
        for port, actuator_names in hw_by_port.items():
            port_calibration = {}
            for name in actuator_names:
                d = derived[name]
                port_calibration[name] = MotorCalibration(
                    id=self.actuators[name].cfg.id,
                    drive_mode=drive_modes[name],
                    homing_offset=d.homing_offset,
                    range_min=d.range_min,
                    range_max=d.range_max,
                )
            port_calibration = self.module_bus.component_bus_by_port[port].write_calibration(
                port_calibration
            )
            self.calibration.update(port_calibration)

        self._save_calibration(self.calibration)
        logger.info(f"Calibration saved to {self.cfg.calibration_file}")
        self.write_calibration()
        logger.info("Calibration written to actuators")
        self.end_calibration_mode()

    def get_initial_action_values(self) -> np.ndarray:
        return np.array([0.0] * max(len(self.actuators), 1))

    # ========= interpolator helpers ============
    def _is_pose_interpolator(self) -> bool:
        return self.cfg.interpolator_type == InterpolatorType.POSE

    def _is_scalar_interpolator(self) -> bool:
        return self.cfg.interpolator_type == InterpolatorType.SCALAR

    def _warn_late_waypoint(
        self, target_time: float, t_now: float, *, clamped: bool = False
    ) -> None:
        late_ms = (t_now - target_time) * 1000.0
        now_wall = time.time()
        if now_wall - self._late_waypoint_warn_t < 1.0:
            self._late_waypoint_suppressed += 1
            return
        suppressed = self._late_waypoint_suppressed
        self._late_waypoint_warn_t = now_wall
        self._late_waypoint_suppressed = 0
        extra = f" ({suppressed} more suppressed)" if suppressed else ""
        action = "clamping" if clamped else "discarding"
        logger.warning(
            "Module '%s': %s late SCHEDULE_WAYPOINT (%.1f ms late)%s",
            self.name,
            action,
            late_ms,
            extra,
        )

    def _resolve_waypoint_time(self, target_time: float, t_now: float) -> float:
        if target_time > t_now:
            return float(target_time)
        self._warn_late_waypoint(target_time, t_now, clamped=True)
        horizon = 1.0 / max(float(self.frequency), 1.0)
        return float(t_now + horizon)

    def _adaptive_drive_horizon(self, t_now: float) -> float:
        min_h = float(getattr(self.cfg, "drive_horizon_min_s", 1.0 / 30.0))
        max_h = float(getattr(self.cfg, "drive_horizon_max_s", 0.12))
        alpha = float(getattr(self.cfg, "drive_horizon_ema_alpha", 0.2))
        mult = float(getattr(self.cfg, "drive_horizon_multiplier", 1.5))
        if self._last_action_time is not None:
            dt = t_now - self._last_action_time
            if dt > 0:
                self._action_interval_ema = (
                    alpha * dt + (1.0 - alpha) * self._action_interval_ema
                )
        self._last_action_time = t_now
        horizon = min(max(self._action_interval_ema * mult, min_h), max_h)
        StatusBoard.set(
            f"{self.name} drive horizon (ms)",
            round(horizon * 1000.0, 1),
        )
        return horizon

    def _is_exclusive_move(self, params: dict) -> bool:
        """True when the module owns the timing of a ``SCHEDULE_WAYPOINT``.

        A ``speed`` key (even ``None``, meaning "use the module default") is a
        velocity limit rather than a deadline, so it makes the module drive the
        target to completion instead of blending it into a streamed trajectory.
        """
        return "speed" in params

    def _create_interpolator(
        self, times: np.ndarray, values: np.ndarray
    ) -> TrajectoryInterpolator:
        interp_type = self.cfg.interpolator_type
        if interp_type == InterpolatorType.SCALAR:
            return ScalarTrajectoryInterpolator(times, values)
        if interp_type == InterpolatorType.JOINT:
            return JointTrajectoryInterpolator(times, values)
        if interp_type == InterpolatorType.POSE:
            return PoseTrajectoryInterpolator(times, values)
        raise ValueError(f"Unknown interpolator type: {interp_type}")

    def _get_max_speed_kwargs(self) -> dict:
        interp_type = self.cfg.interpolator_type
        max_speed = self.cfg.max_speed
        if interp_type == InterpolatorType.SCALAR:
            return {
                "max_speed": max_speed if isinstance(max_speed, (int, float)) else max_speed[0]
            }
        if interp_type == InterpolatorType.JOINT:
            return {"max_joint_speed": max_speed}
        if interp_type == InterpolatorType.POSE:
            if isinstance(max_speed, (int, float)):
                return {"max_pos_speed": max_speed, "max_rot_speed": max_speed}
            return {
                "max_pos_speed": max_speed[0],
                "max_rot_speed": max_speed[1] if len(max_speed) > 1 else max_speed[0],
            }
        raise ValueError(f"Unknown interpolator type: {interp_type}")

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

    # ========= waypoint scheduling ============
    @action_handler(ActionType.SCHEDULE_WAYPOINT)
    def _handle_schedule_waypoint(self, action: ModuleAction, t_now: float) -> bool:
        """Schedule a timestamped waypoint into the module interpolator.

        Handles ``interpolator_type: scalar`` (a single actuator, e.g. a
        gripper); other interpolator types ignore the waypoint. Cartesian and
        joint scheduling live on :class:`ArmModule`, which owns the pose
        interpolator and overrides this.
        """
        if self._is_scalar_interpolator():
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

    def control_step(self):
        """Sample the scalar interpolator when configured.

        Subclasses may override; call ``super().control_step()`` to keep
        upsampling, or replace entirely (ArmModule pose and joint motion).
        """
        if self._is_scalar_interpolator():
            self._ensure_scalar_interpolator()
            if self.interp is not None:
                self._write_scalar(float(self.interp(time.monotonic())))

    def update_observations(self):
        """Publish actuator state, typed when a schema describes this module.

        An actuator with no entry in the robot's ``Observation`` schema — a
        leader arm attached to the controller — still has the one thing worth
        reporting: where it is. Publishing it here rather than leaving skills
        to call ``read_sensors`` means the serial read happens once per tick
        no matter how many skills are following the arm, and they all see it
        at the same instant.
        """
        if self._module_obs_cls is not None:
            super().update_observations()
            return

        if self.observation_buffer is None or not self.is_connected:
            return
        self.observation_buffer.put(self.read_sensors(PosData))
