"""``calibrate-ik`` -- guided sim<->real alignment for an IK arm.

Runs the real arm and a live MuJoCo view side by side and walks a non-expert
through making Cartesian IK track correctly:

1. Live mirror -- move the real arm by hand; the sim must track it 1:1.
2. Sign check -- per joint, confirm the sim moves the SAME way as the real joint;
   flip ``drive_mode`` (and save) when it does not.
3. Cartesian check (optional) -- command pure +X/+Y/+Z moves and confirm the tool
   translates the right way by the right amount.

The easy limit sweep itself stays in ``solarian calibrate-module``; this command
assumes a calibration exists and fixes the sim<->real correspondence on top of it.
"""

from __future__ import annotations

import logging
import threading
import time

from solarian.cli import BaseCommand, CommandRegistry

logger = logging.getLogger("calibrate_ik")


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
                    self.mirror.show(self.module_name, joints_present(self.module, self.joints))
                except Exception as exc:  # noqa: BLE001
                    now = time.monotonic()
                    if str(exc) != last_err or now - last_log > 5.0:
                        logger.warning("Live sim mirror failed: %s", exc)
                        last_err = str(exc)
                        last_log = now
            time.sleep(self.period)


@CommandRegistry.register("calibrate-ik", "robot")
class CalibrateIKCommand(BaseCommand):
    help = "Guided sim<->real alignment for an IK arm (live mirror + sign + Cartesian checks)."
    params = ("module",)

    def handle(self, cfg) -> None:
        from solarian.conf import settings
        from solarian.utils import get_class
        from solarian.core.ik.actuator import CartesianIKActuator
        from solarian.sim.mirror import SimMirror
        from solarian.cli.commands import _ik_common as ik

        module_name = getattr(cfg, "module", None) if cfg is not None else None
        name, _module_class = ik.select_ik_module(module_name)

        # Real robot (force hardware even if the class defaults to sim).
        RobotClass = get_class(settings.ROBOT_CLASS)
        robot = RobotClass(cfg)
        robot.is_sim = False
        print("\nConnecting to the real robot (hardware)...")
        robot.connect()

        module = robot.modules[name]
        joints = [
            n for n in module.actuators
            if not isinstance(module.actuators[n], CartesianIKActuator)
        ]
        continuous = set(getattr(type(module), "ik_continuous_joints", []) or [])
        missing = [n for n in joints if n not in (module.calibration or {})]
        if missing:
            print(
                f"\nNote: {missing} have no calibration yet (typical for continuous "
                f"joints skipped by calibrate-module). They will not track in the\n"
                f"sim until Step 2 sets their zero. The other joints should follow now."
            )

        print("\nStarting the live MuJoCo view (this may take a few seconds)...")
        mirror = SimMirror(robot)
        mirror.start()
        if not mirror.available:
            print(
                "Live sim view is unavailable (no display or MuJoCo). You can still run\n"
                "the sign check by watching the real arm, but the visual mirror is off.\n"
            )

        cart_port, ik_bus = self._find_ik_bus(module)
        model_limits = dict(getattr(ik_bus, "_limits", {}) or {}) if ik_bus is not None else {}

        mirror_thread = None
        try:
            self._disable_torque(module)
            if mirror.available:
                mirror_thread = _MirrorThread(mirror, module, name, joints)
                mirror_thread.start()

            self._step_mirror(joints, continuous)
            self._step_continuous(module, continuous, ik_bus, model_limits, mirror_thread)
            self._step_sign(module, name, joints, continuous, mirror_thread)
            self._maybe_step_cartesian(module, name, mirror_thread, cart_port, ik_bus)
            self._finish(module, name)
        finally:
            if mirror_thread is not None:
                mirror_thread.stop()
                mirror_thread.join(timeout=2)
            mirror.close()
            try:
                robot.disconnect()
            except Exception:  # noqa: BLE001
                pass

    # -- steps -------------------------------------------------------------
    @staticmethod
    def _disable_torque(module):
        try:
            module.disable_torque()
            print("\nTorque is OFF -- you can move the arm freely by hand.")
        except Exception as exc:  # noqa: BLE001
            print(f"(could not disable torque automatically: {exc})")

    @staticmethod
    def _step_mirror(joints, continuous):
        print("\n=== Step 1/4: Live mirror ===")
        print(
            "Move the real arm around by hand. The on-screen sim arm should follow it\n"
            "1:1 -- same joints, same directions. If a joint moves the WRONG way in the\n"
            "sim, note it: we fix that next."
        )
        if continuous:
            print(f"(continuous joints without a hard stop: {sorted(continuous)})")
        input("\nPress ENTER when you are ready to continue to the sign check...")

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

        A continuous joint's zero can sit anywhere on the encoder circle, so the
        model span centred on it may spill outside the servo's valid
        ``[0, cpr-1]`` window. If it does, we re-home the joint so the zero pose
        reads mid-encoder -- recentring the whole range inside the window. This
        matters because the bus clamps the saved range to ``[0, 4095]`` on write:
        an unclamped negative ``range_min`` would otherwise be truncated,
        shifting the mapped zero (the "sim is rotated ~45 deg from my chosen
        zero" symptom) and risking wrapped goal writes at the extremes.
        """
        import math

        from solarian.core.calibration import MotorCalibration
        from solarian.core.data import PosData

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
            # First-time continuous joint (skipped by calibrate-module sweep).
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
        span_ticks = (hi - lo) / (2.0 * math.pi) * cpr
        frac = (0.0 - lo) / (hi - lo) if hi != lo else 0.5

        def _read_center():
            out = ik_bus.read(PosData, [joint], normalize=False)
            raw = out[joint]
            return float(raw["value"] if isinstance(raw, dict) else raw)

        try:
            center = _read_center()
        except Exception as exc:  # noqa: BLE001
            print(f"  {joint}: could not read position ({exc}); skipping.")
            return False

        range_min = center - span_ticks * frac
        range_max = center + span_ticks * (1.0 - frac)

        # Recentre out-of-window ranges by re-homing this pose to mid-encoder.
        if range_min < 0 or range_max > (cpr - 1):
            try:
                offsets = ik_bus.set_half_turn_homings([joint])
                entry.homing_offset = int(offsets[joint])
                center = _read_center()
                range_min = center - span_ticks * frac
                range_max = center + span_ticks * (1.0 - frac)
                print(f"  {joint}: re-homed this pose to mid-encoder to keep its range in bounds.")
            except Exception as exc:  # noqa: BLE001
                print(f"  {joint}: could not re-home ({exc}); saving best-effort range.")

        entry.range_min = int(round(range_min))
        entry.range_max = int(round(range_max))
        print(
            f"  {joint}: range_min={entry.range_min}, range_max={entry.range_max} "
            f"(span {hi - lo:.3f} rad ~= {span_ticks:.0f} ticks, center={center:.0f})."
        )
        return True

    def _step_continuous(self, module, continuous, ik_bus, model_limits, mirror_thread=None):
        """Set zero + range for hard-stop-less joints from one easy reference.

        Jointed (hard-stop) joints need no zero calibration: their two swept
        range endpoints map straight onto the model's ``lo``/``hi``, so the
        midpoint IS the model zero automatically -- nothing to pose. A continuous
        joint (e.g. ``wrist_roll``) has no hard stops to anchor on, so we set its
        range from the model span centred on one operator-chosen zero, using a
        natural, easy-to-reproduce pose (not a mimic of the sim).
        """
        if not continuous:
            return
        if ik_bus is None:
            print("\n(skipping continuous-joint zero: IK bus unavailable)")
            return
        print("\n=== Step 2/4: Continuous-joint zero ===")
        print(
            "Joints with hard stops need no zero step -- their swept range already\n"
            "fixes it. These joints have NO hard stop, so pick a natural zero for each:"
        )
        changed = False
        paused = False
        try:
            for joint in sorted(continuous):
                input(
                    f"\nRotate '{joint}' to the orientation you want as its ZERO "
                    f"(e.g. gripper level/neutral),\nthen press ENTER..."
                )
                # Pause the live mirror before touching the bus: anchoring may
                # re-home the joint (which resets then rewrites its calibration),
                # and concurrent reads on the half-duplex serial line can corrupt
                # those writes. Stay paused through _save so the mirror never
                # renders the transient reset calibration.
                if mirror_thread is not None and not paused:
                    mirror_thread.pause()
                    paused = True
                if self._anchor_joint(module, joint, ik_bus, model_limits):
                    changed = True
            if changed:
                self._save(module)
                print("Saved continuous-joint zero.")
        finally:
            if paused:
                mirror_thread.resume()

    def _step_sign(self, module, module_name, joints, continuous, mirror_thread):
        print("\n=== Step 3/4: Joint direction (sign) check ===")
        print(
            "For each joint, move it by hand and watch the sim. If the sim joint moves\n"
            "the OPPOSITE way, we flip its direction and save."
        )
        changed = False
        for joint in joints:
            note = " (continuous: no hard stop, sign still matters)" if joint in continuous else ""
            input(f"\nJoint '{joint}'{note}: move it now, watch the sim, then press ENTER...")
            ans = input(f"Did the sim '{joint}' move the SAME direction as the real one? [y/n]: ").strip().lower()
            if ans.startswith("n"):
                if self._flip_drive_mode(module, joint):
                    changed = True
                    print(f"  Flipped '{joint}' direction. It should now track correctly.")
                else:
                    print(f"  Could not flip '{joint}' (no calibration entry).")
        if changed:
            self._save(module)
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

    def _maybe_step_cartesian(self, module, module_name, mirror_thread, cart_port, ik_bus):
        print("\n=== Step 4/4: Cartesian axis check (optional) ===")
        if cart_port is None or ik_bus is None:
            print("No Cartesian actuators / IK bus on this module; skipping.")
            return
        ans = input(
            "This ENABLES TORQUE and commands small +X/+Y/+Z tool moves. Keep the\n"
            "workspace clear and a hand near the e-stop. Run it now? [y/N]: "
        ).strip().lower()
        if not ans.startswith("y"):
            print("Skipped the Cartesian check.")
            return

        from solarian.core.data import PosData
        from solarian.core.ik.verify import HumanVerifier, run_cartesian_check

        if mirror_thread is not None:
            mirror_thread.pause()
        try:
            self._arm(module)

            def read_pose():
                vals = ik_bus.read(PosData, ["x", "y", "z"])
                return {a: float(vals[a]) for a in ("x", "y", "z")}

            def command_pose(pose):
                ik_bus.write({a: PosData(value=float(pose[a])) for a in ("x", "y", "z")})

            report = run_cartesian_check(read_pose, command_pose, HumanVerifier(), delta=0.05)
            print("\n" + report.format_summary())
            for hint in report.hints():
                print("  hint: " + hint)
        except Exception as exc:  # noqa: BLE001
            print(f"Cartesian check could not run ({exc}); skipping.")
        finally:
            try:
                module.disable_torque()
            except Exception:  # noqa: BLE001
                pass
            if mirror_thread is not None:
                mirror_thread.resume()

    @staticmethod
    def _arm(module):
        # Bring the arm to operating torque using the module's own safety hooks
        # when present (so106 defines initialize/enter_safe_hold/on_arm).
        for hook in ("initialize", "enter_safe_hold", "on_arm"):
            fn = getattr(module, hook, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass

    def _finish(self, module, module_name):
        self._save(module)
        print(
            f"\nDone. Calibration for '{module_name}' saved. Run\n"
            f"  solarian verify-ik-calibration\n"
            f"to confirm the model limits match, then drive the arm with `solarian run-robot`.\n"
        )

    @staticmethod
    def _save(module):
        try:
            module._save_calibration(module.calibration)
            module.write_calibration()
        except Exception as exc:  # noqa: BLE001
            print(f"(warning: could not persist calibration: {exc})")
