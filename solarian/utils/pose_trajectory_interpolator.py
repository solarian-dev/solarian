"""
Backward compatibility module.

This module re-exports PoseTrajectoryInterpolator and related functions
from the new trajectory_interpolator module for backward compatibility.

New code should import from solarian.utils.trajectory_interpolator directly.
"""

from solarian.utils.trajectory_interpolator import (
    PoseTrajectoryInterpolator,
    rotation_distance,
    pose_distance,
)
