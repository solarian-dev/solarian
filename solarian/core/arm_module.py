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
from solarian.core.data import PosData
from solarian.utils.trajectory_interpolator import JointTrajectoryInterpolator

logger = logging.getLogger("module")

# Default moveJ speed (rad/s), matching UMI's UR5 ``moveJ(..., speed=1.05)``.
_DEFAULT_HOME_SPEED = 1.05


class ArmModule(ActuatorModule):
    """Actuator module with IK, Cartesian interpolators, and UR-style moves."""

    # Optional IK backend spec (``MinkIK(...)``), a ``.build()`` object, or a
    # ready :class:`DifferentialIKBackend`. Required when Cartesian actuators
    # are declared.
    ik_backend = None

    # Home joint configuration in radians, ordered as ``ik_backend`` joints.
    # ``None`` → zeros. A dict keyed by joint name is also accepted.
    home_q: np.ndarray | dict[str, float] | None = None
    home_speed: float = _DEFAULT_HOME_SPEED

    def __init__(self, cfg=None, name=None, observation_buffer=None, action_queue=None):
        super().__init__(cfg, name, observation_buffer, action_queue)
        if cfg is None:
            return
        self._cached_ik_backend = None
        self._joint_interp: JointTrajectoryInterpolator | None = None

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

        cached = getattr(self, "_cached_ik_backend", None)
        if cached is not None:
            return cached

        spec = type(self).ik_backend
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
        spec = type(self).ik_backend
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

    def _home_q_rad(self) -> np.ndarray:
        names = self._ik_joint_names()
        hq = self.home_q
        if hq is None:
            return np.zeros(len(names), dtype=np.float64)
        if isinstance(hq, dict):
            return np.array([float(hq[n]) for n in names], dtype=np.float64)
        return np.asarray(hq, dtype=np.float64).reshape(-1)

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
        missing = [n for n in names if n not in fields]
        if missing:
            logger.warning(
                "Module '%s': SCHEDULE_WAYPOINT missing pose fields %s; ignoring.",
                self.name,
                missing,
            )
            return None
        pose = np.array([float(fields[n].value) for n in names], dtype=np.float64)
        target_time_wall = float(fields[names[0]].timestamp)
        return pose, target_time_wall

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
        drop_timestamps: bool = False,
    ) -> None:
        self._ensure_pose_interpolator(t_now)
        assert self.interp is not None
        pose = np.asarray(pose, dtype=np.float64)
        if drop_timestamps:
            target_time = t_now + self._adaptive_drive_horizon(t_now)
            self.interp = self.interp.drive_to_waypoint(
                pose=pose,
                time=target_time,
                curr_time=t_now,
                **self._pose_speed_kwargs(),
            )
            self.last_waypoint_time = target_time
            return

        target_time = time.monotonic() - synced_time() + float(target_time_wall)
        target_time = self._resolve_waypoint_time(target_time, t_now)
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

    def _min_servo_duration(self, duration: float) -> float:
        period = 1.0 / max(float(self.frequency), 1.0)
        return max(float(duration), period)

    # ========= UR-style motion API (SI units) ============
    def moveJ(self, q, speed: float = _DEFAULT_HOME_SPEED, acceleration: float = 1.4, t_now: float | None = None) -> None:
        """Exclusive joint-space move to ``q`` (radians).

        Stops Cartesian servo (UR ``servoStop`` + ``moveJ``). ``acceleration`` is
        accepted for UR5 overrides; the software interpolator is speed-limited
        linear only.
        """
        del acceleration  # software interpolator has no trapezoid
        t = float(t_now if t_now is not None else time.monotonic())
        self._clear_pose_interpolator()
        q_target = np.asarray(q, dtype=np.float64).reshape(-1)
        q_now = self._read_joints_rad()
        interp = JointTrajectoryInterpolator(
            times=np.array([t]), values=q_now[None, :]
        )
        self._joint_interp = interp.drive_to_waypoint(
            q_target, time=t, curr_time=t, max_joint_speed=float(speed)
        )

    def moveL(self, pose, speed: float = 0.25, acceleration: float = 1.2, t_now: float | None = None) -> None:
        """Exclusive linear Cartesian move to ``pose`` (metres + rotvec)."""
        del acceleration
        t = float(t_now if t_now is not None else time.monotonic())
        self._cancel_joint_motion()
        self._ensure_pose_interpolator(t)
        assert self.interp is not None
        pose = np.asarray(pose, dtype=np.float64)
        self.interp = self.interp.drive_to_waypoint(
            pose=pose,
            time=t,
            curr_time=t,
            **self._pose_speed_kwargs(max_pos_speed=float(speed)),
        )
        self.last_waypoint_time = float(self.interp.times[-1])

    def servoJ(self, q, duration: float = 0.1, t_now: float | None = None) -> None:
        """Streaming joint target (radians) over ``duration`` seconds."""
        t = float(t_now if t_now is not None else time.monotonic())
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
        """Sample joint motion if active, otherwise the Cartesian interpolator."""
        t = time.monotonic()
        joint_interp = getattr(self, "_joint_interp", None)
        if joint_interp is not None:
            self._write_joints_rad(joint_interp(t))
            if t >= float(joint_interp.times[-1]):
                self._joint_interp = None
                self._clear_pose_interpolator()
            return
        if self._is_pose_interpolator() or self._cartesian_pose_names():
            self._ensure_pose_interpolator()
            if self.interp is not None:
                self._write_pose(self.interp(t))

    @action_handler(ActionType.SCHEDULE_WAYPOINT)
    def _handle_schedule_waypoint(self, action: ModuleAction, t_now: float) -> bool:
        extracted = self._extract_pose_waypoint(action)
        if extracted is None:
            return True
        pose, target_time_wall = extracted
        if self._drop_timestamps(action):
            self.servoL(pose, duration=self._adaptive_drive_horizon(t_now), t_now=t_now)
        else:
            self._cancel_joint_motion()
            self._schedule_pose_waypoint(
                pose, target_time_wall, t_now, drop_timestamps=False
            )
        return True

    @action_handler(ActionType.SET_HOME)
    def _handle_set_home(self, action: ModuleAction, t_now: float) -> bool:
        del action
        self.moveJ(self._home_q_rad(), speed=float(self.home_speed), t_now=t_now)
        return True
