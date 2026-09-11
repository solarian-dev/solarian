"""IK alignment steps used by ``solarian calibrate`` (home at model q=0, then sign)."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("calibrate")


class _MirrorThread(threading.Thread):
    """Continuously drives the sim to the real arm's present joints (pausable)."""

    def __init__(self, mirror, module, module_name, joints, period=1.0 / 30.0):
        super().__init__(daemon=True)
        self.mirror = mirror
        self.module = module
        self.module_name = module_name
        self.joints = list(joints)
        self.period = period
        # NOTE: do NOT name these `_stop` -- that shadows threading.Thread's
        # internal `_stop()` method and breaks join().
        self._stop_evt = threading.Event()
        self._paused = threading.Event()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        from solarian.sim.mirror import joints_present

        last_err = None
        last_log = 0.0
        while not self._stop_evt.is_set():
            if not self._paused.is_set():
                try:
                    real = joints_present(self.module, self.joints)
                    self.mirror.show(self.module_name, real)
                    self.mirror.update_hud(self.module_name, self.joints, real)
                except Exception as exc:  # noqa: BLE001
                    now = time.monotonic()
                    if str(exc) != last_err or now - last_log > 5.0:
                        logger.warning("Live sim mirror failed: %s", exc)
                        last_err = str(exc)
                        last_log = now
            time.sleep(self.period)


class IKAlignmentSteps:
    """Home at model q=0, then live sign check. Robot stays connected."""

    def run(self, robot, module, name) -> None:
        joints, continuous, ik_bus, model_limits = self._joint_context(module)
        missing = [n for n in joints if n not in (module.calibration or {})]
        if missing:
            print(
                f"\nNote: {missing} have no range yet (typical for continuous "
                f"joints). They will not track in the sim until the home step "
                f"sets their zero."
            )

        mirror = self._open_viewer(robot, module)
        mirror_thread = None
        try:
            if mirror.available:
                mirror.show_model_zero(name, joints)
                mirror.update_hud(name, joints, {})
            self._step_home(module, joints, continuous, ik_bus, model_limits, mirror)
            if mirror.available:
                mirror_thread = _MirrorThread(mirror, module, name, joints)
                mirror_thread.start()
            self._step_sign(module, name, joints, continuous, mirror, mirror_thread)
            self._finish(module, name, mirror_thread)
            if mirror.available:
                self._step_play(mirror, continuous)
        finally:
            self._stop_mirror(mirror, mirror_thread)

    def run_mirror_only(self, robot, module, name) -> None:
        """Live sim↔real view so the operator can confirm tracking after verify."""
        joints, continuous, _, _ = self._joint_context(module)
        mirror = self._open_viewer(robot, module)
        mirror_thread = None
        if mirror.available:
            mirror_thread = _MirrorThread(mirror, module, name, joints)
            mirror_thread.start()
        try:
            extra = f" Continuous joints: {sorted(continuous)}." if continuous else ""
            self._ask(
                mirror,
                "Move the real arm by hand. The sim should follow it 1:1. "
                "On-screen nrm is the real normalized pose; rad is the sim joint angle."
                f"{extra} Press ENTER when you are done.",
            )
        finally:
            self._stop_mirror(mirror, mirror_thread)

    @staticmethod
    def _joint_context(module):
        from solarian.core.ik.actuator import CartesianIKActuator

        joints = [
            n for n in module.actuators
            if not isinstance(module.actuators[n], CartesianIKActuator)
        ]
        continuous = {
            n for n, act in module.actuators.items()
            if getattr(act, "is_continuous", False)
        }
        _, ik_bus = IKAlignmentSteps._find_ik_bus(module)
        model_limits = dict(getattr(ik_bus, "_limits", {}) or {}) if ik_bus is not None else {}
        return joints, continuous, ik_bus, model_limits

    @staticmethod
    def _open_viewer(robot, module):
        from solarian.sim.mirror import SimMirror

        print("\nStarting the sim view (this may take a few seconds)...")
        mirror = SimMirror(robot)
        mirror.start()
        if not mirror.available:
            print(
                "Live sim view is unavailable (no display or MuJoCo). "
                "Answer the prompts in this terminal.\n"
            )
        IKAlignmentSteps._disable_torque(module)
        return mirror

    @staticmethod
    def _stop_mirror(mirror, mirror_thread):
        if mirror_thread is not None:
            mirror_thread.stop()
            mirror_thread.join(timeout=2)
        mirror.close()

    @staticmethod
    def _ask(mirror, text, *, yes_no: bool = False) -> str:
        if mirror is not None:
            return mirror.ask(text, yes_no=yes_no)
        print(f"\n{text}")
        if yes_no:
            return input("[y/n]: ").strip().lower()
        input("Press ENTER...")
        return ""

    # -- steps -------------------------------------------------------------
    @staticmethod
    def _disable_torque(module):
        try:
            module.disable_torque()
            print("\nTorque is OFF -- you can move the arm freely by hand.")
        except Exception as exc:  # noqa: BLE001
            print(f"(could not disable torque automatically: {exc})")

    @staticmethod
    def _find_ik_bus(module):
        """Return ``(cart_port, ik_bus)`` for the port carrying the Cartesian actuators."""
        from solarian.core.ik.actuator import CartesianIKActuator

        for port, names in module.actuators_by_port.items():
            if any(isinstance(module.actuators.get(n), CartesianIKActuator) for n in names):
                return port, module.module_bus.component_bus_by_port.get(port)
        return None, None

    def _anchor_joint(self, module, joint, ik_bus, model_limits) -> bool:
        """Set one continuous joint's range from a single reference read.

        Assumes the joint is already held at the operator-chosen zero pose (the
        caller prompts and waits). Sets the span equal to the model span and
        centres the range so the reference reading maps to the model's zero.
        Used for continuous joints that have no hard stops to sweep; does not
        disturb the sign (``drive_mode``).

        Always re-homes this pose to mid-encoder and persists that
        ``center_offset``. Skipping the re-home when the (already-homed) present
        already fits writes ``center_offset=0`` over the servo's existing
        EEPROM value; Present then jumps and home is no longer model 0 — the
        sim looks constantly rotated even though verify-calibration's span
        table passes.
        """
        from solarian.core.calibration import MotorCalibration
        from solarian.core.data import PosData
        from solarian.core.ik.alignment import continuous_range_window

        calib = module.calibration or {}
        entry = calib.get(joint)
        lim = model_limits.get(joint)
        try:
            cpr = ik_bus.counts_per_revolution(joint)
        except Exception:  # noqa: BLE001
            cpr = None
        if lim is None or not cpr:
            print(f"  {joint}: missing model-limit/encoder-resolution; skipping.")
            return False
        if entry is None:
            # First-time continuous joint (skipped by the hard-stop sweep).
            act = module.actuators.get(joint)
            if act is None:
                print(f"  {joint}: not an actuator on this module; skipping.")
                return False
            entry = MotorCalibration(
                id=int(act.cfg.id),
                drive_mode=0,
                homing_offset=0,
                range_min=0,
                range_max=int(cpr) - 1,
            )
            if module.calibration is None:
                module.calibration = {}
            module.calibration[joint] = entry
            print(f"  {joint}: created calibration entry (continuous joint).")
        lo, hi = lim
        if abs((0.0 - lo) / (hi - lo) - 0.5) > 0.05:
            print(
                f"  {joint}: model range [{lo:.3f}, {hi:.3f}] is not 0-centered; "
                f"home sits on a limit. Use ±(span/2) in the MJCF."
            )

        def _read_center():
            out = ik_bus.read(PosData, [joint], normalize=False)
            raw = out[joint]
            return float(raw["value"] if isinstance(raw, dict) else raw)

        try:
            offsets = ik_bus.set_half_turn_homings([joint])
            entry.homing_offset = int(offsets[joint])
            center = _read_center()
        except Exception as exc:  # noqa: BLE001
            print(f"  {joint}: could not re-home ({exc}); skipping.")
            return False

        win = continuous_range_window(center, lo, hi, cpr)
        entry.range_min = win.range_min
        entry.range_max = win.range_max
        print(
            f"  {joint}: range_min={entry.range_min}, range_max={entry.range_max} "
            f"(span {hi - lo:.3f} rad, center={center:.0f}, "
            f"center_offset={entry.homing_offset})."
        )
        return True

    def _step_home(self, module, joints, continuous, ik_bus, model_limits, mirror):
        """Freedrive the whole arm to the on-screen model-zero pose."""
        print("\n=== Step 2: Home position ===")
        extra = (
            f" Continuous joints get their range from this pose: {sorted(continuous)}."
            if continuous
            else ""
        )
        self._ask(
            mirror,
            "Step 2: Home position. Match the pose on screen (every sim joint at 0). "
            "That is mid-travel for hard-stop joints and the zero we store for "
            "continuous wrists. Leave a little room so any joint can still move a "
            "few degrees either way. Drag to orbit if the arm is off-screen; R resets."
            f"{extra} Press ENTER when the arm matches.",
        )

        if continuous and ik_bus is None:
            print("(skipping continuous-joint zero: IK bus unavailable)")
        elif continuous:
            changed = False
            for joint in sorted(continuous):
                if self._anchor_joint(module, joint, ik_bus, model_limits):
                    changed = True
            if changed:
                self._save(module)
                print("Saved continuous-joint ranges from home.")
        home = self._read_normalized_joints(module, joints)
        if home:
            for joint in sorted(continuous):
                value = home.get(joint)
                if value is not None and abs(value) > 5:
                    print(
                        f"  warning: {joint} home is {value:.1f}, expected ~0. "
                        f"The sim will look rotated; re-run this home step."
                    )
            try:
                module._save_home(home)
                printed = ", ".join(f"{n}={home[n]:.1f}" for n in joints if n in home)
                print(f"Saved home: {printed}")
            except Exception as exc:  # noqa: BLE001
                print(f"(warning: could not persist home: {exc})")

    @staticmethod
    def _read_normalized_joints(module, joints) -> dict[str, float]:
        from solarian.core.data import PosData

        try:
            readings = module.read_sensors(PosData, list(joints), normalize=True)
        except Exception as exc:  # noqa: BLE001
            print(f"(warning: could not read home pose: {exc})")
            return {}
        home: dict[str, float] = {}
        for name in joints:
            entry = readings.get(name)
            if entry is None:
                continue
            raw = entry["value"] if isinstance(entry, dict) else entry
            home[name] = float(raw)
        return home

    def _step_sign(self, module, module_name, joints, continuous, mirror, mirror_thread=None):
        print("\n=== Step 3: Sim mapping ===")
        print("Answer Y/N in the sim window for each joint. The joint axis is a red arrow.")
        changed = False
        try:
            for joint in joints:
                note = " (continuous)" if joint in continuous else ""
                if mirror is not None:
                    mirror.highlight(module_name, joint)
                ans = self._ask(
                    mirror,
                    f"Step 3: Sim mapping. Joint '{joint}'{note} is marked with a red arrow. "
                    "Move it and watch the sim. Does the sim move the SAME "
                    "direction as the real joint?",
                    yes_no=True,
                )
                if ans.startswith("n"):
                    if self._flip_drive_mode(module, joint):
                        changed = True
                        print(f"  Flipped '{joint}' direction. It should now track correctly.")
                    else:
                        print(f"  Could not flip '{joint}' (no calibration entry).")
        finally:
            if mirror is not None:
                mirror.highlight(module_name, None)
        if changed:
            self._save(module, mirror_thread)
            print("\nSaved updated joint directions.")
        else:
            print("\nNo direction changes needed.")

    @staticmethod
    def _flip_drive_mode(module, joint) -> bool:
        calib = module.calibration or {}
        entry = calib.get(joint)
        if entry is None:
            return False
        entry.drive_mode = 0 if int(getattr(entry, "drive_mode", 0)) else 1
        return True

    def _finish(self, module, module_name, mirror_thread=None):
        from solarian.core.actuator_module import print_calibration_output

        self._save(module, mirror_thread)
        print_calibration_output(module)
        print(
            f"\nSaved '{module_name}'. The live mirror stays open so you can "
            "check the mapping. Run `solarian verify-calibration` later for "
            "the span table.\n"
        )

    def _step_play(self, mirror, continuous):
        extra = f" Continuous joints: {sorted(continuous)}." if continuous else ""
        self._ask(
            mirror,
            "Live mirror is on. Move the arm and check the numbers "
            "(nrm = real normalized, rad = sim joint angle)."
            f"{extra} Press ENTER when done.",
        )

    @staticmethod
    def _pause_mirror(mirror_thread):
        if mirror_thread is None:
            return
        mirror_thread.pause()
        time.sleep(max(mirror_thread.period * 2, 0.05))

    @staticmethod
    def _resume_mirror(mirror_thread):
        if mirror_thread is not None:
            mirror_thread.resume()

    def _save(self, module, mirror_thread=None):
        self._pause_mirror(mirror_thread)
        try:
            try:
                module._save_calibration(module.calibration)
                module.write_calibration()
            except Exception as exc:  # noqa: BLE001
                print(f"(warning: could not persist calibration: {exc})")
        finally:
            self._resume_mirror(mirror_thread)
