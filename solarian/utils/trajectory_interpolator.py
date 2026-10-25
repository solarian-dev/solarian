"""
Trajectory interpolators for different action spaces.

This module provides a hierarchy of interpolators:
- TrajectoryInterpolator: Abstract base class
- ScalarTrajectoryInterpolator: For 1D values (e.g., gripper width)
- JointTrajectoryInterpolator: For N-dimensional joint space
- PoseTrajectoryInterpolator: For 6D Cartesian poses with Slerp rotation

Each interpolator supports:
- Time-based interpolation via __call__(t)
- Waypoint scheduling with speed limits
- Trimming to time windows
"""

from abc import ABC, abstractmethod
from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def _extend(times: np.ndarray, values: np.ndarray, t: float, value) -> tuple:
    """Add *value* at *t*, replacing the final sample when the times collide.

    A waypoint the module is already at -- or one already due -- yields a
    duration that rounds to zero at float64 resolution, which would duplicate an
    instant. Interpolation needs times to be strictly increasing: ``Slerp``
    rejects a repeat outright, and ``interp1d`` silently turns it into an
    infinite slope. Replacing the last sample keeps the trajectory a function of
    time, and holds the pose that was asked for.
    """
    value = np.asarray(value, dtype=np.float64)
    if len(times) and t <= times[-1]:
        return np.asarray(times), np.concatenate([values[:-1], value[None]])
    return np.append(times, [t]), np.concatenate([values, value[None]])


# Knot spacing (s) for an acceleration-limited move. The interpolators are
# piecewise linear, so the trapezoid is only as smooth as this sampling; 10 ms
# is finer than any module tick that samples it.
_PROFILE_DT = 0.01
# Lookback (s) for the finite-difference velocity of a trajectory.
_VELOCITY_DT = 0.01
# A profile shorter than this (s) is an instant: added to a monotonic clock
# days into uptime, it would not even advance the time.
_MIN_PROFILE_S = 1e-6


def _knots(total: float, dt: float) -> np.ndarray:
    """0, dt, 2dt, ... and *total*, never closer than dt/2 to one another.

    The sample times get added to ``time.monotonic()``; at ~5e5 s of uptime
    float64 resolves only ~1e-10 s, so a last sample a hair before *total*
    would land on the same instant and break ``Slerp``.
    """
    inner = np.arange(0.0, max(total - 0.5 * dt, 0.0), dt)
    if inner.size == 0:
        inner = np.array([0.0])
    return np.append(inner, total)


def _trapezoid(dist: float, v_max: float, a_max: float, v0: float = 0.0,
               dt: float = _PROFILE_DT) -> tuple[np.ndarray, np.ndarray]:
    """Knot times from 0 and fractions of *dist* for a move that ends at rest.

    Ramps from *v0* at *a_max*, cruises at *v_max*, and decelerates at *a_max*
    to stop on the target; a move too short to reach *v_max* peaks lower
    (triangular). A *v0* that cannot stop within *dist* at *a_max* decelerates
    at exactly the rate that does, so the target is never overshot.
    """
    if dist <= 0.0:
        return np.array([0.0]), np.array([1.0])
    # An unlimited axis still needs finite arithmetic: inf * 0 is nan.
    a_max = min(float(a_max), 1e9)
    v_max = min(float(v_max), 1e9)
    v0 = min(max(float(v0), 0.0), v_max)
    if v0 * v0 >= 2.0 * a_max * dist:
        a_dec = v0 * v0 / (2.0 * dist)
        total = 2.0 * dist / v0
        if total < _MIN_PROFILE_S:
            return np.array([0.0]), np.array([1.0])
        taus = _knots(total, dt)
        x = v0 * taus - 0.5 * a_dec * taus * taus
    else:
        v_peak = min(v_max, float(np.sqrt(a_max * dist + 0.5 * v0 * v0)))
        t1 = (v_peak - v0) / a_max
        d1 = (v_peak * v_peak - v0 * v0) / (2.0 * a_max)
        d3 = v_peak * v_peak / (2.0 * a_max)
        t2 = max(dist - d1 - d3, 0.0) / v_peak
        total = t1 + t2 + v_peak / a_max
        if total < _MIN_PROFILE_S:
            return np.array([0.0]), np.array([1.0])
        taus = _knots(total, dt)
        ta = np.minimum(taus, t1)
        tc = np.clip(taus - t1, 0.0, t2)
        td = np.clip(taus - t1 - t2, 0.0, None)
        x = (v0 * ta + 0.5 * a_max * ta * ta
             + v_peak * tc
             + v_peak * td - 0.5 * a_max * td * td)
    s = np.clip(x / dist, 0.0, 1.0)
    s[-1] = 1.0
    return taus, s


def _lookback(times: np.ndarray, t: float) -> float:
    """Finite-difference window ending at *t* that stays on the trajectory.

    Reaching back before the first knot reads the clamped start pose, which
    halves the apparent speed right after a retarget.
    """
    return min(_VELOCITY_DT, float(t) - float(times[0]))


def _stretch(taus: np.ndarray, duration: float) -> np.ndarray:
    """Slow a profile uniformly so it lasts at least *duration*."""
    if taus[-1] > 0.0 and duration > taus[-1]:
        return taus * (duration / taus[-1])
    return taus


class TrajectoryInterpolator(ABC):
    """Abstract base class for trajectory interpolators."""
    
    @property
    @abstractmethod
    def times(self) -> np.ndarray:
        """Return the waypoint times."""
        pass
    
    @property
    @abstractmethod
    def values(self) -> np.ndarray:
        """Return the waypoint values."""
        pass
    
    @abstractmethod
    def __call__(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Interpolate value at time t."""
        pass
    
    @abstractmethod
    def schedule_waypoint(self, value, time, curr_time=None, 
                          last_waypoint_time=None, **kwargs) -> "TrajectoryInterpolator":
        """Schedule a new waypoint, respecting speed limits."""
        pass
    
    @abstractmethod
    def trim(self, start_t: float, end_t: float) -> "TrajectoryInterpolator":
        """Trim the interpolator to a time window."""
        pass


class ScalarTrajectoryInterpolator(TrajectoryInterpolator):
    """
    Interpolator for 1D scalar values like gripper width.
    
    Uses simple linear interpolation between waypoints.
    """
    
    def __init__(self, times: np.ndarray, values: np.ndarray):
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        values = np.atleast_1d(np.asarray(values, dtype=np.float64))
        assert len(times) == len(values), f"times and values must have same length: {len(times)} vs {len(values)}"
        assert len(times) >= 1, "Must have at least one waypoint"
        
        self._times = times
        self._values = values
        self.single_step = len(times) == 1
        
        if not self.single_step:
            assert np.all(times[1:] >= times[:-1]), "times must be non-decreasing"
            self._interp = si.interp1d(
                times, values,
                kind='linear',
                fill_value=(values[0], values[-1]),
                bounds_error=False,
                assume_sorted=True
            )
    
    @property
    def times(self) -> np.ndarray:
        return self._times
    
    @property
    def values(self) -> np.ndarray:
        return self._values
    
    def __call__(self, t: Union[float, np.ndarray]) -> np.ndarray:
        is_single = isinstance(t, numbers.Number)
        if is_single:
            t = np.array([t])
        
        if self.single_step:
            result = np.full(len(t), self._values[0])
        else:
            result = self._interp(t)
        
        if is_single:
            return result[0]
        return result
    
    def trim(self, start_t: float, end_t: float) -> "ScalarTrajectoryInterpolator":
        assert start_t <= end_t, f"start_t must be <= end_t: {start_t} vs {end_t}"
        
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        all_times = np.unique(all_times)
        all_values = self(all_times)
        
        return ScalarTrajectoryInterpolator(times=all_times, values=all_values)
    
    def drive_to_waypoint(self, value, time, curr_time,
                          max_speed=np.inf) -> "ScalarTrajectoryInterpolator":
        """Drive to a waypoint, starting from curr_time."""
        assert max_speed > 0, "max_speed must be positive"
        time = max(time, curr_time)
        
        curr_value = self(curr_time)
        distance = abs(value - curr_value)
        min_duration = distance / max_speed
        duration = time - curr_time
        duration = max(duration, min_duration)
        assert duration >= 0
        last_waypoint_time = curr_time + duration
        
        trimmed_interp = self.trim(curr_time, curr_time)
        times, values = _extend(
            trimmed_interp.times, trimmed_interp.values, last_waypoint_time, value
        )

        return ScalarTrajectoryInterpolator(times, values)
    
    def schedule_waypoint(self, value, time, curr_time=None,
                          last_waypoint_time=None, 
                          max_speed=np.inf) -> "ScalarTrajectoryInterpolator":
        """
        Schedule a waypoint at a target time, respecting speed limits.
        
        This implements the same scheduling logic as PoseTrajectoryInterpolator
        for consistency.
        """
        assert max_speed > 0, "max_speed must be positive"
        if last_waypoint_time is not None:
            assert curr_time is not None
        
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time
        
        if curr_time is not None:
            if time <= curr_time:
                return self
            start_time = max(curr_time, start_time)
            
            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time
        
        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        
        assert start_time <= end_time
        assert end_time <= time
        
        trimmed_interp = self.trim(start_time, end_time)
        
        # Determine speed-limited duration
        duration = time - end_time
        end_value = trimmed_interp(end_time)
        distance = abs(value - end_value)
        min_duration = distance / max_speed
        duration = max(duration, min_duration)
        assert duration >= 0
        actual_end_time = end_time + duration
        
        times, values = _extend(
            trimmed_interp.times, trimmed_interp.values, actual_end_time, value
        )
        
        return ScalarTrajectoryInterpolator(times, values)


class JointTrajectoryInterpolator(TrajectoryInterpolator):
    """
    Interpolator for N-dimensional joint space.
    
    Uses linear interpolation per joint. Supports per-joint speed limits.
    """
    
    def __init__(self, times: np.ndarray, values: np.ndarray):
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        assert len(times) == len(values), f"times and values must have same length: {len(times)} vs {len(values)}"
        assert len(times) >= 1, "Must have at least one waypoint"
        
        self._times = times
        self._values = values
        self.n_joints = values.shape[1]
        self.single_step = len(times) == 1
        
        if not self.single_step:
            assert np.all(times[1:] >= times[:-1]), "times must be non-decreasing"
            self._interp = si.interp1d(
                times, values, axis=0,
                kind='linear',
                fill_value=(values[0], values[-1]),
                bounds_error=False,
                assume_sorted=True
            )
    
    @property
    def times(self) -> np.ndarray:
        return self._times
    
    @property
    def values(self) -> np.ndarray:
        return self._values
    
    def __call__(self, t: Union[float, np.ndarray]) -> np.ndarray:
        is_single = isinstance(t, numbers.Number)
        if is_single:
            t = np.array([t])
        
        if self.single_step:
            result = np.tile(self._values[0], (len(t), 1))
        else:
            result = self._interp(t)
        
        if is_single:
            return result[0]
        return result
    
    def trim(self, start_t: float, end_t: float) -> "JointTrajectoryInterpolator":
        assert start_t <= end_t, f"start_t must be <= end_t: {start_t} vs {end_t}"
        
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        all_times = np.unique(all_times)
        all_values = self(all_times)
        
        return JointTrajectoryInterpolator(times=all_times, values=all_values)
    
    def _compute_min_duration(self, start_value: np.ndarray, end_value: np.ndarray,
                               max_joint_speed: Union[float, np.ndarray]) -> float:
        """Compute minimum duration based on per-joint speed limits."""
        distances = np.abs(end_value - start_value)
        max_joint_speed = np.atleast_1d(max_joint_speed)
        if len(max_joint_speed) == 1:
            max_joint_speed = np.full(self.n_joints, max_joint_speed[0])
        return float(np.max(distances / max_joint_speed))
    
    def velocity(self, t: float) -> np.ndarray:
        """Joint velocity just before *t*; zero past the last waypoint."""
        h = _lookback(self.times, t)
        if h <= 0.0:
            return np.zeros(self.n_joints)
        return (self(t) - self(t - h)) / h

    def drive_to_waypoint(self, value, time, curr_time,
                          max_joint_speed=np.inf,
                          max_joint_acc=None) -> "JointTrajectoryInterpolator":
        """Drive to a waypoint, starting from curr_time.

        With *max_joint_acc* the move is a trapezoid that starts at the current
        velocity along the new path and ends at rest; without it, one
        constant-speed segment.
        """
        value = np.atleast_1d(value)
        assert len(value) == self.n_joints, f"value must have {self.n_joints} joints"
        time = max(time, curr_time)
        
        curr_value = self(curr_time)
        if max_joint_acc is not None:
            return self._drive_profiled(
                curr_value, value, time - curr_time, curr_time,
                max_joint_speed, max_joint_acc,
            )
        min_duration = self._compute_min_duration(curr_value, value, max_joint_speed)
        duration = time - curr_time
        duration = max(duration, min_duration)
        assert duration >= 0
        last_waypoint_time = curr_time + duration
        
        trimmed_interp = self.trim(curr_time, curr_time)
        times, values = _extend(
            trimmed_interp.times, trimmed_interp.values, last_waypoint_time, value
        )
        
        return JointTrajectoryInterpolator(times, values)

    def _drive_profiled(self, start, end, duration, curr_time,
                        max_joint_speed, max_joint_acc) -> "JointTrajectoryInterpolator":
        """One shared time scaling; the joint that needs longest sets it."""
        delta = end - start
        dist = np.abs(delta)
        speeds = np.broadcast_to(np.atleast_1d(max_joint_speed), dist.shape)
        accs = np.broadcast_to(np.atleast_1d(max_joint_acc), dist.shape)
        if not np.any(dist > 0.0):
            return JointTrajectoryInterpolator(np.array([curr_time]), end[None, :])
        ratio = dist / speeds
        lead = int(np.argmax(ratio if np.any(ratio > 0.0) else dist))
        rate = max(float(self.velocity(curr_time)[lead] / delta[lead]), 0.0)
        taus, s = max(
            (_trapezoid(dist[i], speeds[i], accs[i], rate * dist[i])
             for i in range(len(dist))),
            key=lambda profile: profile[0][-1],
        )
        taus = _stretch(taus, duration)
        values = start[None, :] + s[:, None] * delta[None, :]
        return JointTrajectoryInterpolator(curr_time + taus, values)
    
    def schedule_waypoint(self, value, time, curr_time=None,
                          last_waypoint_time=None,
                          max_joint_speed=np.inf) -> "JointTrajectoryInterpolator":
        """
        Schedule a waypoint at a target time, respecting per-joint speed limits.
        """
        value = np.atleast_1d(value)
        assert len(value) == self.n_joints, f"value must have {self.n_joints} joints"
        
        if last_waypoint_time is not None:
            assert curr_time is not None
        
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time
        
        if curr_time is not None:
            if time <= curr_time:
                return self
            start_time = max(curr_time, start_time)
            
            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time
        
        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        
        assert start_time <= end_time
        assert end_time <= time
        
        trimmed_interp = self.trim(start_time, end_time)
        
        # Determine speed-limited duration
        duration = time - end_time
        end_value = trimmed_interp(end_time)
        min_duration = self._compute_min_duration(end_value, value, max_joint_speed)
        duration = max(duration, min_duration)
        assert duration >= 0
        actual_end_time = end_time + duration
        
        times, values = _extend(
            trimmed_interp.times, trimmed_interp.values, actual_end_time, value
        )
        
        return JointTrajectoryInterpolator(times, values)


# Helper functions for pose interpolation
def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    """Compute the angular distance between two rotations."""
    return (b * a.inv()).magnitude()


def pose_distance(start_pose, end_pose):
    """Compute position and rotation distances between two poses."""
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist


class PoseTrajectoryInterpolator(TrajectoryInterpolator):
    """
    Interpolator for 6D Cartesian poses (x, y, z, rx, ry, rz).
    
    Uses linear interpolation for position and Slerp for rotation.
    Rotation is represented as axis-angle (rotation vector).
    """
    
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            pos = poses[:, :3]
            rot = st.Rotation.from_rotvec(poses[:, 3:])

            self.pos_interp = si.interp1d(times, pos, axis=0, assume_sorted=True)
            self.rot_interp = st.Slerp(times, rot)
    
    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.pos_interp.x
    
    @property
    def values(self) -> np.ndarray:
        """Alias for poses for consistency with base class."""
        return self.poses
    
    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        else:
            n = len(self.times)
            poses = np.zeros((n, 6))
            poses[:, :3] = self.pos_interp.y
            poses[:, 3:] = self.rot_interp(self.times).as_rotvec()
            return poses

    def trim(self, start_t: float, end_t: float) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        all_times = np.unique(all_times)
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)
    
    def velocity(self, t: float) -> np.ndarray:
        """Linear (m/s) and world-frame angular (rad/s) velocity just before *t*.

        Zero past the last waypoint, where the trajectory holds.
        """
        h = _lookback(self.times, t)
        if h <= 0.0:
            return np.zeros(6)
        before, now = self(t - h), self(t)
        turn = (st.Rotation.from_rotvec(now[3:])
                * st.Rotation.from_rotvec(before[3:]).inv())
        return np.concatenate([(now[:3] - before[:3]) / h, turn.as_rotvec() / h])

    def drive_to_waypoint(self, pose, time, curr_time,
                          max_pos_speed=np.inf, 
                          max_rot_speed=np.inf,
                          max_pos_acc=None,
                          max_rot_acc=None) -> "PoseTrajectoryInterpolator":
        """Drive to *pose*, starting from curr_time.

        With an acceleration limit the move is a trapezoid that starts at the
        current velocity along the new path and ends at rest; without one, a
        single constant-speed segment.
        """
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        time = max(time, curr_time)
        
        curr_pose = self(curr_time)
        if max_pos_acc is not None or max_rot_acc is not None:
            return self._drive_profiled(
                curr_pose, np.asarray(pose, dtype=np.float64),
                time - curr_time, curr_time,
                max_pos_speed, max_rot_speed,
                np.inf if max_pos_acc is None else max_pos_acc,
                np.inf if max_rot_acc is None else max_rot_acc,
            )
        pos_dist, rot_dist = pose_distance(curr_pose, pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        trimmed_interp = self.trim(curr_time, curr_time)
        times, poses = _extend(
            trimmed_interp.times, trimmed_interp.poses, last_waypoint_time, pose
        )

        return PoseTrajectoryInterpolator(times, poses)

    def _drive_profiled(self, start, end, duration, curr_time,
                        max_pos_speed, max_rot_speed,
                        max_pos_acc, max_rot_acc,
                        rate: float | None = None) -> "PoseTrajectoryInterpolator":
        """Straight line plus slerp, on one time scaling set by the slower axis.

        *rate* is the progress (1/s) along the path at *curr_time*; when not
        given it is read off the current motion.
        """
        delta_pos = end[:3] - start[:3]
        start_rot = st.Rotation.from_rotvec(start[3:])
        delta_rot = (st.Rotation.from_rotvec(end[3:]) * start_rot.inv()).as_rotvec()
        pos_dist = float(np.linalg.norm(delta_pos))
        rot_dist = float(np.linalg.norm(delta_rot))
        if pos_dist == 0.0 and rot_dist == 0.0:
            return PoseTrajectoryInterpolator(np.array([curr_time]), end[None, :])

        if rate is None:
            # Read off whichever axis dominates the move.
            velocity = self.velocity(curr_time)
            if pos_dist / max_pos_speed >= rot_dist / max_rot_speed:
                rate = float(velocity[:3] @ delta_pos) / (pos_dist * pos_dist)
            else:
                rate = float(velocity[3:] @ delta_rot) / (rot_dist * rot_dist)
        if rate < 0.0:
            # Heading back the way it came: brake to rest first rather than
            # reversing in one tick.
            braked = self.stop(curr_time, max_pos_acc, max_rot_acc)
            rest_time = float(braked.times[-1])
            rest = braked(rest_time)
            onward = PoseTrajectoryInterpolator(
                np.array([rest_time]), rest[None, :]
            )._drive_profiled(
                rest, end, duration - (rest_time - curr_time),
                rest_time, max_pos_speed, max_rot_speed, max_pos_acc, max_rot_acc,
            )
            return PoseTrajectoryInterpolator(
                np.concatenate([braked.times, onward.times[1:]]),
                np.concatenate([braked.poses, onward.poses[1:]]),
            )
        rate = max(rate, 0.0)

        taus, s = max(
            (_trapezoid(pos_dist, max_pos_speed, max_pos_acc, rate * pos_dist),
             _trapezoid(rot_dist, max_rot_speed, max_rot_acc, rate * rot_dist)),
            key=lambda profile: profile[0][-1],
        )
        taus = _stretch(taus, duration)
        poses = np.empty((len(s), 6))
        poses[:, :3] = start[:3] + s[:, None] * delta_pos
        poses[:, 3:] = (
            st.Rotation.from_rotvec(s[:, None] * delta_rot) * start_rot
        ).as_rotvec()
        return PoseTrajectoryInterpolator(curr_time + taus, poses)

    def stop(self, curr_time, max_pos_acc, max_rot_acc) -> "PoseTrajectoryInterpolator":
        """Decelerate from the current motion to rest at the braking pose."""
        pose = self(curr_time)
        velocity = self.velocity(curr_time)
        speed = float(np.linalg.norm(velocity[:3]))
        rot_speed = float(np.linalg.norm(velocity[3:]))
        if speed == 0.0 and rot_speed == 0.0:
            return PoseTrajectoryInterpolator(np.array([curr_time]), pose[None, :])
        target = pose.copy()
        target[:3] += velocity[:3] * speed / (2.0 * max_pos_acc)
        turn = velocity[3:] * rot_speed / (2.0 * max_rot_acc)
        target[3:] = (
            st.Rotation.from_rotvec(turn) * st.Rotation.from_rotvec(pose[3:])
        ).as_rotvec()
        # Braking distance v^2/2a covered from speed v is progress 2a/v per
        # second; the axis that takes longer to stop sets the shared scaling.
        rates = [
            2.0 * acc / v
            for v, acc in ((speed, max_pos_acc), (rot_speed, max_rot_acc))
            if v > 0.0
        ]
        return self._drive_profiled(
            pose, target, 0.0, curr_time,
            max(speed, 1e-9), max(rot_speed, 1e-9), max_pos_acc, max_rot_acc,
            rate=min(rates),
        )

    def schedule_waypoint(self, pose, time, 
                          max_pos_speed=np.inf, 
                          max_rot_speed=np.inf,
                          curr_time=None,
                          last_waypoint_time=None) -> "PoseTrajectoryInterpolator":
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                return self
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)

        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)

        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        actual_last_waypoint_time = end_time + duration

        times, poses = _extend(
            trimmed_interp.times, trimmed_interp.poses, actual_last_waypoint_time, pose
        )

        return PoseTrajectoryInterpolator(times, poses)

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        pose = np.zeros((len(t), 6))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            pose = np.zeros((len(t), 6))
            pose[:, :3] = self.pos_interp(t)
            pose[:, 3:] = self.rot_interp(t).as_rotvec()

        if is_single:
            pose = pose[0]
        return pose

