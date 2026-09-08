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
    
    def drive_to_waypoint(self, value, time, curr_time,
                          max_joint_speed=np.inf) -> "JointTrajectoryInterpolator":
        """Drive to a waypoint, starting from curr_time."""
        value = np.atleast_1d(value)
        assert len(value) == self.n_joints, f"value must have {self.n_joints} joints"
        time = max(time, curr_time)
        
        curr_value = self(curr_time)
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
    
    def drive_to_waypoint(self, pose, time, curr_time,
                          max_pos_speed=np.inf, 
                          max_rot_speed=np.inf) -> "PoseTrajectoryInterpolator":
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        time = max(time, curr_time)
        
        curr_pose = self(curr_time)
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

