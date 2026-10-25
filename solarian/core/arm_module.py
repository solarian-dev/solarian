"""Reusable arm module: Cartesian IK plus UR-style motion primitives.

Concrete arms declare joint actuators, six ``CartesianIKActuator`` s, and an
``ik_backend`` spec. Motion methods run **inside the module process** (from
action handlers). Controllers send actions; they cannot call these methods
directly.

Units are SI so skills can switch arms: joints in radians, pose in metres +
rotvec, ``moveJ`` speed in rad/s. Feetech conversion uses the IK backend's
joint limits (normalized ``[-100, 100]`` ↔ radians).

A UR5 subclass can override ``moveJ`` / ``moveL`` / ``servoJ`` / ``servoL`` to
talk RTDE instead of the software interpolators.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from solarian.core.action import ActionType, ModuleAction, action_handler
from solarian.core.actuator_module import ActuatorModule
from solarian.core.clock_sync import synced_time
from solarian.core.configs import SafeState
from solarian.core.data import PosData
from solarian.core.pose import compose_ee_delta
from solarian.utils.trajectory_interpolator import JointTrajectoryInterpolator

logger = logging.getLogger("module")

# Default moveJ speed (rad/s), matching UMI's UR5 ``moveJ(..., speed=1.05)``.
_DEFAULT_JOINT_SPEED = 1.05

# Enough solve+integrate passes for `reachable_travel` to settle against the
# model's configuration limits, and the headroom below which a joint counts as
# the thing that stopped it.
_REACH_ITERATIONS = 300
_LIMIT_MARGIN_RAD = np.deg2rad(1.0)

# Latch agreement in raw encoder ticks. STS/SMS are 4096 ticks/turn, so 60
# is about 5 deg: above tremor between two reads, far below a wrap flip.
_LATCH_AGREE_TICKS = 60
_LATCH_CONFIRM_TRIES = 3
# After torque comes on, gravity sag is a few degrees. A wrap-flipped
# wrist_roll is a half turn (~2048 ticks). This is the line between them.
_LATCH_JUMP_TICKS = 120


def _float_param(action: ModuleAction, key: str) -> float | None:
    value = action.params.get(key)
    return float(value) if value is not None else None


def _max_wrapped_abs(delta: np.ndarray, spans: np.ndarray) -> float:
    """Largest per-joint error, wrapping continuous joints onto their circle."""
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    spans = np.asarray(spans, dtype=np.float64).reshape(-1)
    wrapped = delta.copy()
    for i, span in enumerate(spans):
        if span > 0:
            wrapped[i] = (delta[i] + span / 2.0) % span - span / 2.0
    return float(np.max(np.abs(wrapped))) if wrapped.size else 0.0


class ArmModule(ActuatorModule):
    """Actuator module with IK, Cartesian interpolators, and UR-style moves."""

    kind: str = "arm"

    # Optional IK backend spec (``MinkIK(...)``), a ``.build()`` object, or a
    # ready :class:`DifferentialIKBackend`. Required when Cartesian actuators
    # are declared.
    ik_backend = None

    # Joint speed limit (rad/s) for an exclusive move that names no ``speed``.
    default_joint_speed: float = _DEFAULT_JOINT_SPEED

    # Class defaults so ``__new__`` test fixtures can call scheduleL without
    # running ``__init__``. Instance writes in moveJ / servoL override them.
    _exclusive_move = False
    _logged_exclusive_block = False

    def __init__(
        self,
        cfg=None,
        name=None,
        observation_buffer=None,
        action_queue=None,
        *,
        ik_tcp=None,
    ):
        self.ik_tcp = ik_tcp
        spec = type(self).ik_backend
        if spec is not None and ik_tcp is not None:
            from dataclasses import replace

            self.ik_backend = replace(spec, tcp_offset=ik_tcp)
        super().__init__(cfg, name, observation_buffer, action_queue)
        # Set even for a schema-only instance (``cfg is None``), so readers
        # need no guard.
        self._cached_ik_backend = None
        self._joint_interp: JointTrajectoryInterpolator | None = None
        # Exclusive while the go-to interpolator still has future time. Leftover
        # SCHEDULE from a reaped policy must not abort it; a finished go-to
        # must not block the next policy.
        self._exclusive_move = False

    def bind(self, cfg, name):
        return type(self)(cfg, name, ik_tcp=self.ik_tcp)

    # ========= inverse kinematics ============
    def wrap_port_bus(self, port, inner_bus, joint_components, cartesian_components):
        """Wrap a port's joint bus with Cartesian IK control when needed."""
        if not cartesian_components:
            return inner_bus
        from solarian.core.ik.bus import IKActuatorBus

        # Rebuild on every connect (parent vs forked child); cache for moveJ.
        self._cached_ik_backend = None
        return IKActuatorBus(
            inner_bus=inner_bus,
            joint_components=joint_components,
            cartesian_components=cartesian_components,
            backend=self._resolve_ik_backend(),
        )

    def _resolve_ik_backend(self):
        """Build (and cache) the :class:`DifferentialIKBackend` from ``ik_backend``."""
        from solarian.core.ik.base import DifferentialIKBackend

        if self._cached_ik_backend is not None:
            return self._cached_ik_backend

        spec = getattr(self, "ik_backend", None)
        if spec is None:
            raise NotImplementedError(
                f"Module '{self.name}' declares Cartesian IK actuators but no "
                f"`ik_backend` is set. Declare one on the module class, e.g. "
                f"`ik_backend = MinkIK(mjcf=..., joints=..., frame=...)`."
            )
        if isinstance(spec, DifferentialIKBackend):
            backend = spec
        else:
            build = getattr(spec, "build", None)
            if callable(build):
                backend = build()
            elif callable(spec):
                backend = spec()
            else:
                raise TypeError(
                    f"Module '{self.name}'.ik_backend must be a DifferentialIKBackend, a "
                    f"spec with .build() (e.g. MinkIK), or a zero-arg callable; got "
                    f"{type(spec).__name__}."
                )
        self._cached_ik_backend = backend
        return backend

    def _ik_joint_names(self) -> list[str]:
        spec = getattr(self, "ik_backend", None)
        joints = getattr(spec, "joints", None) if spec is not None else None
        if isinstance(joints, dict) and joints:
            return list(joints.keys())
        try:
            return list(self._resolve_ik_backend().joint_actuator_names)
        except Exception:  # noqa: BLE001
            from solarian.core.ik.actuator import CartesianIKActuator

            return [
                n
                for n, act in self.actuators.items()
                if not isinstance(act, CartesianIKActuator)
            ]

    def _joint_limits(self) -> dict[str, tuple[float, float]]:
        return dict(self._resolve_ik_backend().joint_limits)

    def _norm_to_rad(self, name: str, norm: float) -> float:
        mode = self.actuators[name].cfg.mode
        lo, hi = self._joint_limits()[name]
        if mode == "range_m100_100":
            v = min(100.0, max(-100.0, float(norm)))
            return ((v + 100.0) / 200.0) * (hi - lo) + lo
        if mode == "range_0_100":
            v = min(100.0, max(0.0, float(norm)))
            return (v / 100.0) * (hi - lo) + lo
        return float(norm)

    def _rad_to_norm(self, name: str, rad: float) -> float:
        mode = self.actuators[name].cfg.mode
        lo, hi = self._joint_limits()[name]
        if hi == lo:
            return 0.0
        if mode == "range_m100_100":
            bounded = min(hi, max(lo, float(rad)))
            return (((bounded - lo) / (hi - lo)) * 200.0) - 100.0
        if mode == "range_0_100":
            bounded = min(hi, max(lo, float(rad)))
            return ((bounded - lo) / (hi - lo)) * 100.0
        return float(rad)

    def _read_joints_rad(self) -> np.ndarray:
        names = self._ik_joint_names()
        readings = self.read_sensors(PosData, names)
        return np.array(
            [
                self._norm_to_rad(
                    n,
                    float(
                        readings[n]["value"]
                        if isinstance(readings[n], dict)
                        else readings[n].value
                    ),
                )
                for n in names
            ],
            dtype=np.float64,
        )

    def _write_joints_rad(self, q: np.ndarray) -> None:
        names = self._ik_joint_names()
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        self.write_actuators(
            {
                name: PosData(value=self._rad_to_norm(name, float(q[i])))
                for i, name in enumerate(names)
            }
        )

    def _read_joints_raw(self) -> np.ndarray:
        """Present_Position ticks, no drive_mode / offset / radian conversion."""
        names = self._ik_joint_names()
        readings = self.read_sensors(PosData, names, normalize=False)
        return np.array(
            [
                float(
                    readings[n]["value"]
                    if isinstance(readings[n], dict)
                    else readings[n].value
                )
                for n in names
            ],
            dtype=np.float64,
        )

    def _write_joints_raw(self, q: np.ndarray) -> None:
        """Goal_Position := the ticks just read. The servo sees no conversion."""
        names = self._ik_joint_names()
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        self.write_actuators(
            {name: PosData(value=int(round(q[i]))) for i, name in enumerate(names)},
            normalize=False,
        )

    def _joint_raw_wrap_spans(self) -> np.ndarray:
        """Encoder circle for continuous joints, 0 when the joint does not wrap."""
        names = self._ik_joint_names()
        spans = np.zeros(len(names), dtype=np.float64)
        actuators = getattr(self, "actuators", None) or {}
        for i, name in enumerate(names):
            act = actuators.get(name)
            if act is not None and getattr(act, "is_continuous", False):
                spans[i] = 4096.0
        return spans

    # ========= Cartesian pose interpolator ============
    def _cartesian_pose_names(self) -> list[str]:
        """Ordered Cartesian actuator names (by ``cfg.id`` 0..5)."""
        from solarian.core.ik.actuator import CartesianIKActuator

        named: list[tuple[int, str]] = []
        for name, act in self.actuators.items():
            if isinstance(act, CartesianIKActuator):
                named.append((int(act.cfg.id), name))
        named.sort(key=lambda item: item[0])
        return [n for _, n in named]

    def _read_current_pose(self) -> np.ndarray:
        """FK the current end-effector pose via Cartesian actuator reads."""
        names = self._cartesian_pose_names()
        if not names:
            return np.zeros(6, dtype=np.float64)
        readings = self.read_sensors(PosData, names)
        pose = np.zeros(len(names), dtype=np.float64)
        for i, name in enumerate(names):
            entry = readings[name]
            pose[i] = float(entry["value"] if isinstance(entry, dict) else entry.value)
        return pose

    def read_pose(self) -> np.ndarray:
        """Live measured EE pose (metres + rotvec), same frame as :meth:`scheduleL`."""
        return self._read_current_pose()

    def _release_finished_exclusive_move(self, t_now: float) -> None:
        if not self._exclusive_move:
            return
        interp = self._joint_interp if self._joint_interp is not None else self.interp
        if interp is None or float(interp.times[-1]) <= t_now:
            self._exclusive_move = False

    def _ignore_schedule_during_exclusive(self, kind: str) -> None:
        if self._logged_exclusive_block:
            return
        self._logged_exclusive_block = True
        logger.warning(
            "Module '%s': ignoring %s while exclusive move is in progress",
            self.name,
            kind,
        )

    def scheduleL(self, pose, target_time_wall: float, t_now: float | None = None) -> None:
        """Queue a timestamped Cartesian waypoint."""
        t = float(t_now if t_now is not None else time.monotonic())
        self._release_finished_exclusive_move(t)
        if self._exclusive_move:
            self._ignore_schedule_during_exclusive("SCHEDULE_L")
            return
        self._cancel_joint_motion()
        self._schedule_pose_waypoint(
            np.asarray(pose, dtype=np.float64),
            float(target_time_wall),
            t,
        )

    def scheduleJ(self, q, target_time_wall: float, t_now: float | None = None) -> None:
        """Queue a timestamped joint waypoint (radians)."""
        t = float(t_now if t_now is not None else time.monotonic())
        self._release_finished_exclusive_move(t)
        if self._exclusive_move:
            self._ignore_schedule_during_exclusive("SCHEDULE_J")
            return
        self._schedule_joint_waypoint(
            np.asarray(q, dtype=np.float64),
            float(target_time_wall),
            t,
        )

    def hold_measured_pose(self) -> np.ndarray:
        """Copy Present_Position onto Goal_Position, tick for tick.

        Enabling torque with no write is not a hold: Feetech servos keep the
        last Goal_Position while limp, so they run back to wherever they were
        commanded before free-drive -- usually the pose from ``initialize``.
        The write has to happen. What must not happen is converting that
        reading through drive_mode, center_offset, or radians. ``wrist_roll``
        is a full encoder circle; any of those can send Goal to the far side
        of the wrap, and the servo takes the long way around.

        Two reads have to agree first. A grouped sync read that mis-parses
        hands back a plausible number, and that number as a goal is a spin.
        """
        previous = self._read_joints_raw()
        spans = self._joint_raw_wrap_spans()
        for _ in range(_LATCH_CONFIRM_TRIES):
            measured = self._read_joints_raw()
            if _max_wrapped_abs(measured - previous, spans) <= _LATCH_AGREE_TICKS:
                self._write_joints_raw(measured)
                return measured
            previous = measured
        raise RuntimeError(
            "Refusing to latch a hold pose: consecutive joint reads disagree "
            f"by more than {_LATCH_AGREE_TICKS:g} encoder ticks, so the "
            "measurement cannot be trusted and energising onto it could drive "
            "the arm a long way. Check the bus wiring and Return_Delay_Time."
        )

    def enable_torque_holding(self) -> np.ndarray:
        """Energise the arm so it holds the ticks it is on right now.

        The same raw ticks are written again after torque comes on, because
        some Feetech firmware ignores Goal_Position while limp. Not a new
        read: a new read after a spin has started would freeze it mid-turn.
        """
        latched = self.hold_measured_pose()
        self.enable_torque()
        self._write_joints_raw(latched)
        return latched

    def joint_command_delta(self, reference) -> float:
        """How far the joints have moved from *reference*, wrapping circles."""
        return _max_wrapped_abs(
            self._read_joints_raw() - np.asarray(reference, dtype=np.float64),
            self._joint_raw_wrap_spans(),
        )

    def reachable_travel(self, start_pose, axis: int, distance: float) -> tuple[float, list[str]]:
        """Signed travel actually achievable along *axis*, and the joints stopping it.

        Differential IK enforces the model's configuration limits, so a target
        past the arm's reach never reports failure — the solver just converges
        short. Asking for more than this returns clips the motion, and a clipped
        sinusoid cross-correlates to the clip, not to the command pipeline.
        """
        backend = self._resolve_ik_backend()
        target = np.asarray(start_pose, dtype=np.float64).copy()
        target[axis] += distance

        # Leaves the model at the converged configuration: the control path
        # re-seeds set_q from freshly measured joints every tick, so this does
        # not leak into motion.
        backend.set_q(self._read_joints_rad())
        origin = float(backend.get_pose()[axis])
        dt = 1.0 / max(float(self.frequency), 1.0)
        q = np.zeros(0)
        for _ in range(_REACH_ITERATIONS):
            q = np.asarray(backend.integrate(backend.solve(target, dt), dt), dtype=np.float64)
            backend.set_q(q)

        limits = self._joint_limits()
        blocked = [
            name
            for name, value in zip(self._ik_joint_names(), q)
            if min(value - limits[name][0], limits[name][1] - value) < _LIMIT_MARGIN_RAD
        ]
        return float(backend.get_pose()[axis]) - origin, blocked

    def _ensure_pose_interpolator(self, t_now: float | None = None) -> None:
        if self.interp is not None:
            return
        t = float(t_now if t_now is not None else time.monotonic())
        try:
            pose = self._read_current_pose()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Module '%s': could not seed pose interpolator from FK (%s); using zeros.",
                self.name,
                exc,
            )
            pose = np.zeros(6, dtype=np.float64)
        self.interp = self._create_interpolator(
            np.array([t]), np.asarray(pose, dtype=np.float64)[None, :]
        )
        self.last_waypoint_time = t

    def _clear_pose_interpolator(self) -> None:
        """Drop the pose trajectory so the next tick re-seeds from live FK."""
        self.interp = None
        self.last_waypoint_time = 0.0

    def _extract_pose_waypoint(
        self, action: ModuleAction
    ) -> tuple[np.ndarray, float] | None:
        names = self._cartesian_pose_names()
        if not names:
            return None
        fields = action.values()
        if any(n not in fields for n in names):
            return None
        pose = np.array([float(fields[n].value) for n in names], dtype=np.float64)
        target_time_wall = float(fields[names[0]].timestamp)
        return pose, target_time_wall

    def _extract_joint_waypoint(
        self, action: ModuleAction
    ) -> tuple[np.ndarray, float] | None:
        names = self._ik_joint_names()
        if not names:
            return None
        fields = action.values()
        if any(n not in fields for n in names):
            return None
        q = np.array(
            [self._norm_to_rad(n, float(fields[n].value)) for n in names],
            dtype=np.float64,
        )
        target_time_wall = float(fields[names[0]].timestamp)
        return q, target_time_wall

    def _schedule_joint_waypoint(
        self,
        q: np.ndarray,
        target_time_wall: float,
        t_now: float,
    ) -> None:
        self._clear_pose_interpolator()
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if self._joint_interp is None:
            q_now = self._read_joints_rad()
            self._joint_interp = JointTrajectoryInterpolator(
                times=np.array([t_now]), values=q_now[None, :]
            )
        target_time = self._resolve_waypoint_time(
            time.monotonic() - synced_time() + float(target_time_wall),
            t_now,
        )
        if target_time is None:
            return
        if target_time <= self.last_waypoint_time:
            # Unused tail of a previous chunk: splice from live joints so an
            # untracked lookahead cannot yank the command backward.
            q_now = self._read_joints_rad()
            self._joint_interp = JointTrajectoryInterpolator(
                times=np.array([t_now]), values=q_now[None, :]
            )
            self.last_waypoint_time = t_now
        self._joint_interp = self._joint_interp.schedule_waypoint(
            q,
            time=target_time,
            curr_time=t_now,
            last_waypoint_time=self.last_waypoint_time,
            max_joint_speed=np.inf,
        )
        self.last_waypoint_time = target_time

    def _pose_speed_kwargs(
        self,
        max_pos_speed: float | None = None,
        max_rot_speed: float | None = None,
    ) -> dict[str, float]:
        if self._is_pose_interpolator():
            kwargs = dict(self._get_max_speed_kwargs())
        else:
            max_speed = self.cfg.max_speed
            if isinstance(max_speed, (int, float)):
                kwargs = {"max_pos_speed": float(max_speed), "max_rot_speed": float(max_speed)}
            else:
                kwargs = {
                    "max_pos_speed": float(max_speed[0]),
                    "max_rot_speed": float(
                        max_speed[1] if len(max_speed) > 1 else max_speed[0]
                    ),
                }
        if max_pos_speed is not None:
            kwargs["max_pos_speed"] = float(max_pos_speed)
        if max_rot_speed is not None:
            kwargs["max_rot_speed"] = float(max_rot_speed)
        return kwargs

    def _schedule_pose_waypoint(
        self,
        pose: np.ndarray,
        target_time_wall: float,
        t_now: float,
    ) -> None:
        self._ensure_pose_interpolator(t_now)
        assert self.interp is not None
        pose = np.asarray(pose, dtype=np.float64)
        target_time = self._resolve_waypoint_time(
            time.monotonic() - synced_time() + float(target_time_wall),
            t_now,
        )
        if target_time is None:
            return
        if target_time <= self.last_waypoint_time:
            # Unused tail of a previous chunk: splice from live FK so an
            # untracked lookahead cannot yank the command back to the
            # observation pose (the snap at overlapping replan).
            self.interp = None
            self._ensure_pose_interpolator(t_now)
        self.interp = self.interp.schedule_waypoint(
            pose=pose,
            time=target_time,
            curr_time=t_now,
            last_waypoint_time=self.last_waypoint_time,
            **self._pose_speed_kwargs(),
        )
        self.last_waypoint_time = target_time

    def _write_pose(self, pose: np.ndarray) -> None:
        names = self._cartesian_pose_names()
        if not names:
            return
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        self.write_actuators(
            {name: PosData(value=float(pose[i])) for i, name in enumerate(names)}
        )

    def _cancel_joint_motion(self) -> None:
        self._joint_interp = None

    # ========= UR-style motion API (SI units) ============
    def _pose_accel_kwargs(
        self,
        max_pos_acc: float | None = None,
        max_rot_acc: float | None = None,
    ) -> dict[str, float | None]:
        """``cfg.max_accel`` split into position / rotation, call site winning."""
        max_accel = self.cfg.max_accel
        if max_accel is None:
            pos = rot = None
        elif isinstance(max_accel, (int, float)):
            pos = rot = float(max_accel)
        else:
            pos = float(max_accel[0])
            rot = float(max_accel[1] if len(max_accel) > 1 else max_accel[0])
        if max_pos_acc is not None:
            pos = float(max_pos_acc)
        if max_rot_acc is not None:
            rot = float(max_rot_acc)
        return {"max_pos_acc": pos, "max_rot_acc": rot}

    def moveJ(
        self,
        q,
        speed: float = _DEFAULT_JOINT_SPEED,
        acceleration: float | None = None,
        t_now: float | None = None,
    ) -> None:
        """Exclusive joint-space move to ``q`` (radians).

        Stops Cartesian servo (UR ``servoStop`` + ``moveJ``). ``acceleration``
        (rad/s^2, default ``cfg.max_joint_accel``) ramps the move up and down;
        without one it runs at constant ``speed``.
        """
        if acceleration is None:
            acceleration = self.cfg.max_joint_accel
        t = float(t_now if t_now is not None else time.monotonic())
        self._exclusive_move = True
        self._logged_exclusive_block = False
        self._clear_pose_interpolator()
        q_target = np.asarray(q, dtype=np.float64).reshape(-1)
        q_now = self._read_joints_rad()
        interp = JointTrajectoryInterpolator(
            times=np.array([t]), values=q_now[None, :]
        )
        self._joint_interp = interp.drive_to_waypoint(
            q_target,
            time=t,
            curr_time=t,
            max_joint_speed=float(speed),
            max_joint_acc=acceleration,
        )

    def moveL(
        self,
        pose,
        speed: float | None = None,
        acceleration: float | None = None,
        t_now: float | None = None,
        rot_speed: float | None = None,
        rot_acceleration: float | None = None,
    ) -> None:
        """Exclusive linear Cartesian move to ``pose`` (metres + rotvec).

        ``speed`` / ``rot_speed`` override ``cfg.max_speed`` and
        ``acceleration`` / ``rot_acceleration`` override ``cfg.max_accel``.
        With an acceleration the move ramps up from the current motion and
        down to rest, like a UR ``moveL``; without one it is a single
        constant-speed segment.
        """
        t = float(t_now if t_now is not None else time.monotonic())
        self._exclusive_move = True
        self._logged_exclusive_block = False
        self._cancel_joint_motion()
        self._ensure_pose_interpolator(t)
        assert self.interp is not None
        pose = np.asarray(pose, dtype=np.float64)
        self.interp = self.interp.drive_to_waypoint(
            pose=pose,
            time=t,
            curr_time=t,
            **self._pose_speed_kwargs(
                max_pos_speed=float(speed) if speed is not None else None,
                max_rot_speed=float(rot_speed) if rot_speed is not None else None,
            ),
            **self._pose_accel_kwargs(acceleration, rot_acceleration),
        )
        self.last_waypoint_time = float(self.interp.times[-1])

    def stopL(
        self,
        acceleration: float | None = None,
        rot_acceleration: float | None = None,
        t_now: float | None = None,
    ) -> None:
        """Decelerate the Cartesian motion to rest (UR ``stopL``).

        Brakes along the current velocity at ``acceleration`` /
        ``rot_acceleration`` (default ``cfg.max_accel``). With no limit
        configured, holds the commanded pose at once.
        """
        t = float(t_now if t_now is not None else time.monotonic())
        if self._joint_interp is not None:
            self._joint_interp = self._joint_interp.trim(t, t)
        if self.interp is None:
            return
        accel = self._pose_accel_kwargs(acceleration, rot_acceleration)
        if accel["max_pos_acc"] is None and accel["max_rot_acc"] is None:
            self.interp = self.interp.trim(t, t)
        else:
            self._exclusive_move = True
            self._logged_exclusive_block = False
            self.interp = self.interp.stop(
                t,
                np.inf if accel["max_pos_acc"] is None else accel["max_pos_acc"],
                np.inf if accel["max_rot_acc"] is None else accel["max_rot_acc"],
            )
        self.last_waypoint_time = float(self.interp.times[-1])

    def servoJ(self, q, duration: float = 0.1, t_now: float | None = None) -> None:
        """Streaming joint target (radians) over ``duration`` seconds."""
        t = float(t_now if t_now is not None else time.monotonic())
        self._exclusive_move = False
        self._clear_pose_interpolator()
        q_target = np.asarray(q, dtype=np.float64).reshape(-1)
        if self._joint_interp is None:
            q_now = self._read_joints_rad()
            self._joint_interp = JointTrajectoryInterpolator(
                times=np.array([t]), values=q_now[None, :]
            )
        duration = self._min_servo_duration(duration)
        self._joint_interp = self._joint_interp.drive_to_waypoint(
            q_target, time=t + duration, curr_time=t, max_joint_speed=np.inf
        )

    def servoL(self, pose, duration: float = 0.1, t_now: float | None = None) -> None:
        """Streaming Cartesian target over ``duration`` seconds (teleop path)."""
        t = float(t_now if t_now is not None else time.monotonic())
        self._exclusive_move = False
        self._cancel_joint_motion()
        self._ensure_pose_interpolator(t)
        assert self.interp is not None
        pose = np.asarray(pose, dtype=np.float64)
        duration = self._min_servo_duration(duration)
        self.interp = self.interp.drive_to_waypoint(
            pose=pose,
            time=t + duration,
            curr_time=t,
            **self._pose_speed_kwargs(),
        )
        self.last_waypoint_time = t + duration

    def get_initial_action_values(self) -> np.ndarray:
        try:
            return self._read_current_pose()
        except Exception:
            return np.zeros(6, dtype=np.float64)

    def control_step(self):
        """Sample joint motion if active, otherwise the Cartesian interpolator.

        After ``moveJ`` / ``servoJ`` reach the last waypoint the joint
        interpolator is *kept* and continues to write that target. Dropping it
        would re-seed Cartesian hold from the measured (possibly gravity-sagged)
        pose and overwrite the joint goal. Cartesian ``SCHEDULE_L`` /
        ``servoL`` / ``moveL`` cancel joint motion via ``_cancel_joint_motion``.
        """
        t = time.monotonic()
        self._release_finished_exclusive_move(t)
        if self._joint_interp is not None:
            self._write_joints_rad(self._joint_interp(t))
            return
        if self._is_pose_interpolator() or self._cartesian_pose_names():
            self._ensure_pose_interpolator()
            if self.interp is not None:
                self._write_pose(self.interp(t))

    @action_handler(ActionType.HOLD_SAFE)
    def _handle_hold_safe(self, action: ModuleAction, t_now: float) -> bool:
        """Hold or relax, including the joint trajectory ``control_step`` prefers."""
        if self._joint_interp is not None:
            if self.cfg.link_lost == SafeState.RELAX:
                self._cancel_joint_motion()
            else:
                self._joint_interp = self._joint_interp.trim(t_now, t_now)
        return super()._handle_hold_safe(action, t_now)

    @action_handler(ActionType.CANCEL_TRAJECTORY)
    def _handle_cancel_trajectory(self, action: ModuleAction, t_now: float) -> bool:
        """Hold leftover scheduled chunks; do not abort an exclusive go-to.

        The reaper fires this after a robot-side skill dies, so a policy's
        last ``SCHEDULE_*`` chunk is not left running. The controller has
        often already sent ``MOVE_J`` / ``MOVE_L`` by then — that exclusive
        interpolator is the new author and must keep driving to its target.
        """
        if self._exclusive_move:
            return True
        if self._joint_interp is not None:
            self._joint_interp = self._joint_interp.trim(t_now, t_now)
        return super()._handle_cancel_trajectory(action, t_now)

    def _warn_missing_pose(self, kind: str) -> None:
        logger.warning(
            "Module '%s': %s missing pose fields %s; ignoring.",
            self.name,
            kind,
            self._cartesian_pose_names(),
        )

    def _warn_missing_joints(self, kind: str) -> None:
        logger.warning(
            "Module '%s': %s missing joint fields %s; ignoring.",
            self.name,
            kind,
            self._ik_joint_names(),
        )

    @action_handler(ActionType.MOVE_L)
    def _handle_move_l(self, action: ModuleAction, t_now: float) -> bool:
        extracted = self._extract_pose_waypoint(action)
        if extracted is None:
            self._warn_missing_pose("MOVE_L")
            return True
        pose, _ = extracted
        if str(action.params.get("frame") or "world") == "ee":
            try:
                current = self._read_current_pose()
            except Exception:  # noqa: BLE001
                current = np.zeros(6, dtype=np.float64)
            pose = compose_ee_delta(current, pose)
        self.moveL(
            pose,
            speed=_float_param(action, "speed"),
            acceleration=_float_param(action, "accel"),
            rot_speed=_float_param(action, "rot_speed"),
            rot_acceleration=_float_param(action, "rot_accel"),
            t_now=t_now,
        )
        return True

    @action_handler(ActionType.STOP_L)
    def _handle_stop_l(self, action: ModuleAction, t_now: float) -> bool:
        self.stopL(
            acceleration=_float_param(action, "accel"),
            rot_acceleration=_float_param(action, "rot_accel"),
            t_now=t_now,
        )
        return True

    @action_handler(ActionType.STOP_J)
    def _handle_stop_j(self, action: ModuleAction, t_now: float) -> bool:
        if self._joint_interp is not None:
            self._joint_interp = self._joint_interp.trim(t_now, t_now)
        return True

    @action_handler(ActionType.MOVE_J)
    def _handle_move_j(self, action: ModuleAction, t_now: float) -> bool:
        extracted = self._extract_joint_waypoint(action)
        if extracted is None:
            self._warn_missing_joints("MOVE_J")
            return True
        q, _ = extracted
        speed = _float_param(action, "speed")
        self.moveJ(
            q,
            speed=speed if speed is not None else float(self.default_joint_speed),
            acceleration=_float_param(action, "accel"),
            t_now=t_now,
        )
        return True

    @action_handler(ActionType.SERVO_L)
    def _handle_servo_l(self, action: ModuleAction, t_now: float) -> bool:
        extracted = self._extract_pose_waypoint(action)
        if extracted is None:
            self._warn_missing_pose("SERVO_L")
            return True
        self.servoL(
            extracted[0],
            duration=self._adaptive_drive_horizon(t_now),
            t_now=t_now,
        )
        return True

    @action_handler(ActionType.SERVO_J)
    def _handle_servo_j(self, action: ModuleAction, t_now: float) -> bool:
        extracted = self._extract_joint_waypoint(action)
        if extracted is None:
            self._warn_missing_joints("SERVO_J")
            return True
        self.servoJ(
            extracted[0],
            duration=self._adaptive_drive_horizon(t_now),
            t_now=t_now,
        )
        return True

    @action_handler(ActionType.SCHEDULE_L)
    def _handle_schedule_l(self, action: ModuleAction, t_now: float) -> bool:
        extracted = self._extract_pose_waypoint(action)
        if extracted is None:
            self._warn_missing_pose("SCHEDULE_L")
            return True
        pose, target_time_wall = extracted
        self.scheduleL(pose, target_time_wall, t_now=t_now)
        return True

    @action_handler(ActionType.SCHEDULE_J)
    def _handle_schedule_j(self, action: ModuleAction, t_now: float) -> bool:
        extracted = self._extract_joint_waypoint(action)
        if extracted is None:
            self._warn_missing_joints("SCHEDULE_J")
            return True
        q, target_time_wall = extracted
        self.scheduleJ(q, target_time_wall, t_now=t_now)
        return True
