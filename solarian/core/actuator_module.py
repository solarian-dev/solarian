"""Actuator-bearing modules: motors, calibration, interpolators.

Cameras and other sensor-only modules stay on :class:`Module`. Anything that
writes actuators (arms, grippers, leader devices) subclasses this.
"""

from __future__ import annotations

import logging

import numpy as np

from solarian.core.actuator_bus import ActuatorBus
from solarian.core.calibration import MotorCalibration
from solarian.core.configs import ModuleConfig
from solarian.core.data import PosData
from solarian.core.module import Module
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.observation_ring_buffer import ObservationRingBuffer
from solarian.utils.status_board import StatusBoard

logger = logging.getLogger("module")


class ActuatorModule(Module):
    """Module with actuators: torque, calibration sweep, trajectory interpolators."""

    def __init__(
        self,
        cfg: ModuleConfig | None = None,
        name: str | None = None,
        observation_buffer: ObservationRingBuffer | None = None,
        action_queue: ModuleActionQueue | None = None,
    ):
        super().__init__(cfg, name, observation_buffer, action_queue)
        if cfg is None:
            return

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
