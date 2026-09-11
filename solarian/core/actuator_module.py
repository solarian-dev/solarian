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
from solarian.core.configs import InterpolatorType, ModuleConfig, SafeState
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
        """Port → hardware actuator names (excludes virtual pose axes)."""
        out: dict[str, list[str]] = {}
        for port, names in self.actuators_by_port.items():
            hw = [
                n for n in names
                if getattr(self.actuators[n], "uses_motor_calibration", True)
            ]
            if hw:
                out[port] = hw
        return out

    def calibrate(self) -> None:
        """Calibrate hard-stop joints from a range sweep.

        Records travel, unwraps a wrapping arc via ``center_offset``, and seeds
        ``drive_mode``. IK modules skip the direction prompt (the live-mirror
        sign check decides). Non-IK modules ask one physical low-command pose.
        Continuous joints are skipped; their zero is set in ``solarian calibrate``
        home.
        """
        from solarian.utils.general import enter_pressed
        from solarian.core.joint_calibration import (
            CalibrationSafetyError,
            DerivedHoming,
            EncoderRangeTracker,
            drive_mode_from_low_end_pose,
            low_command_label,
        )

        if not self.needs_motor_calibration():
            logger.error(f"No calibration required for module {self.name}")
            return

        hw_by_port = self._hardware_actuators_by_port()
        if not hw_by_port:
            logger.error(f"No hardware actuators to calibrate on module {self.name}")
            return

        has_ik = getattr(type(self), "ik_backend", None) is not None
        continuous = {
            n for n, act in self.actuators.items() if getattr(act, "is_continuous", False)
        }
        if continuous:
            hw_by_port = {
                port: [n for n in names if n not in continuous]
                for port, names in hw_by_port.items()
            }
            hw_by_port = {p: ns for p, ns in hw_by_port.items() if ns}
            print(
                f"Continuous joints (no hard stop — skip the limit sweep; "
                f"their zero is set at the home step): {sorted(continuous)}"
            )
            if not hw_by_port:
                print("No hard-stop joints to sweep.")
                return

        print(f"\n=== Step 1: Joint limits ===")
        print(
            "Move each hard-stop joint through its FULL mechanical travel, then press ENTER.\n"
        )

        self.start_calibration_mode()

        for port, actuator_names in hw_by_port.items():
            bus = self.module_bus.component_bus_by_port[port]
            assert isinstance(bus, ActuatorBus)
            bus.reset_calibration(actuator_names)

        trackers: dict[str, EncoderRangeTracker] = {}
        bus_by_name: dict[str, ActuatorBus] = {}
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
                bus_by_name[name] = bus

        
        board = StatusBoard.start_ui()
        board.table_title = "Step 1: Joint limits"
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
                    print(
                        f"  {name}: ignored {tracker.rejected} teleport sample(s)"
                    )
                bus = bus_by_name[name]
                cap = bus.max_center_offset_abs(name)
                derived[name] = tracker.derived(name, max_offset_abs=cap)
                d = derived[name]
                if tracker.wrap_committed:
                    print(
                        f"  {name}: crossed the encoder seam; center_offset={d.center_offset} "
                        f"moves it into the unused gap. Present range "
                        f"{d.range_min} .. {d.range_max} (span {d.range_max - d.range_min})."
                    )
                else:
                    if tracker.wrap_rejected:
                        print(
                            f"  {name}: ignored {tracker.wrap_rejected} wrap-looking "
                            f"glitch(es); travel treated as contiguous."
                        )
                    print(
                        f"  {name}: Present range {d.range_min} .. {d.range_max} "
                        f"(span {d.range_max - d.range_min}, center_offset=0; "
                        f"raw encoder = Present)."
                    )
        except CalibrationSafetyError as exc:
            print(f"\nCalibration aborted (safety): {exc}")
            logger.error("Calibration aborted (safety): %s", exc)
            self.end_calibration_mode()
            return

        drive_modes: dict[str, int] = {}
        for name in derived:
            cfg = self.actuators[name].cfg
            drive_modes[name] = int(getattr(cfg, "sim_drive_mode", 0) or 0)

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
                    homing_offset=d.center_offset,
                    range_min=d.range_min,
                    range_max=d.range_max,
                )
            port_calibration = self.module_bus.component_bus_by_port[port].write_calibration(
                port_calibration
            )
            self.calibration.update(port_calibration)

        if not has_ik:
            print(
                "\n=== Direction (one physical pose) ===\n"
                "drive_mode is the sign between encoder and command. Move the\n"
                "joint to the pose that should be the low command (e.g. fully\n"
                "closed), not 'whichever encoder end is smaller'."
            )
            for port, actuator_names in hw_by_port.items():
                bus = self.module_bus.component_bus_by_port[port]
                for name in actuator_names:
                    label = low_command_label(self.actuators[name].cfg.mode)
                    input(
                        f"Move '{name}' to the physical low-command pose ({label}), "
                        "then press ENTER..."
                    )
                    reading = bus.read(PosData, [name], normalize=False)
                    raw = float(reading[name])
                    present = bus._present_ticks(name, raw)
                    d = derived[name]
                    dm = drive_mode_from_low_end_pose(present, d.range_min, d.range_max)
                    drive_modes[name] = dm
                    self.calibration[name].drive_mode = dm
                    print(f"  {name}: drive_mode={dm}")
            for port, actuator_names in hw_by_port.items():
                subset = {n: self.calibration[n] for n in actuator_names}
                self.module_bus.component_bus_by_port[port].write_calibration(subset)
        else:
            print(
                "\nDirection (drive_mode) is decided in the live-mirror step — "
                "whether the sim joint moves the same way as the real one."
            )

        self._save_calibration(self.calibration)
        self.write_calibration()
        if has_ik:
            print("Saved joint-limit ranges. Next: home position.")
        else:
            print_calibration_output(self)
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

    @action_handler(ActionType.HOLD_SAFE)
    def _handle_hold_safe(self, action: ModuleAction, t_now: float) -> bool:
        """Enter ``cfg.link_lost`` because nothing is commanding this module.

        ``HOLD`` trims the trajectory to now, so ``control_step`` keeps writing
        the target the module is already at instead of continuing toward a
        waypoint whose author is gone. ``RELAX`` cuts torque, and drops the
        trajectory so re-enabling torque later re-seeds from the measured pose
        rather than snapping to where the arm was before it went limp.
        """
        if self.cfg.link_lost == SafeState.RELAX:
            self.interp = None
            self.last_waypoint_time = 0.0
            self.disable_torque()
        elif self.interp is not None:
            self.interp = self.interp.trim(t_now, t_now)
        return True

    @action_handler(ActionType.CANCEL_TRAJECTORY)
    def _handle_cancel_trajectory(self, action: ModuleAction, t_now: float) -> bool:
        """Drop what is still queued and hold where the trajectory is now.

        A chunk outlives its author: seconds of waypoints sit in this
        interpolator after the skill that scheduled them is gone, and whoever
        commands the module next has to out-argue them.
        """
        if self.interp is not None:
            self.interp = self.interp.trim(t_now, t_now)
            self.last_waypoint_time = t_now
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


def print_calibration_output(module) -> None:
    """Print the saved yaml fields in operator language (matches the docs recipe)."""
    print("\n=== Output ===")
    path = module._module_yaml_path()
    print(f"Saved to {path or '(in memory — no module yaml)'}")
    calib = module.calibration or {}
    for name in sorted(calib):
        entry = calib[name]
        print(
            f"  {name}: real_range {entry.range_min} .. {entry.range_max}  "
            f"center_offset={entry.homing_offset}  drive_mode={entry.drive_mode}"
        )
    home = getattr(module.cfg, "home", None) or {}
    if home:
        printed = ", ".join(f"{n}={home[n]:.1f}" for n in home)
        print(f"  home: {printed}")
    print(
        "  real_range_* = Present travel window (after any wrap shift)\n"
        "  center_offset = encoder-zero shift (0 = no shift)\n"
        "  drive_mode = sign vs the model\n"
        "  home = normalized pose you matched (not move.yaml poses.home)"
    )
