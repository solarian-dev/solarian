"""Hinge-axis marker used by the calibrate sign-check highlight."""

import numpy as np

from solarian.sim.mujoco_world import hinge_arrow_ends, hinge_world_frame


def test_hinge_uses_joint_id_when_xanchor_is_per_joint():
    """MuJoCo 3: xanchor is njnt x 3. dofadr 9 is wrist_flex in a cube+arm+gripper scene."""
    xanchor = np.arange(9 * 3, dtype=float).reshape(9, 3)
    xaxis = np.zeros((9, 3))
    xaxis[4] = [0.0, 0.0, 1.0]
    origin, axis = hinge_world_frame(xanchor, xaxis, joint_id=4, dofadr=9, njnt=9)
    np.testing.assert_allclose(origin, xanchor[4])
    np.testing.assert_allclose(axis, [0.0, 0.0, 1.0])


def test_hinge_uses_dofadr_when_xanchor_is_per_dof():
    xanchor = np.arange(14 * 3, dtype=float).reshape(14, 3)
    xaxis = np.zeros((14, 3))
    xaxis[9] = [1.0, 0.0, 0.0]
    origin, axis = hinge_world_frame(xanchor, xaxis, joint_id=4, dofadr=9, njnt=9)
    np.testing.assert_allclose(origin, xanchor[9])
    np.testing.assert_allclose(axis, [1.0, 0.0, 0.0])


def test_hinge_none_when_both_indices_are_out_of_range():
    xanchor = np.zeros((3, 3))
    xaxis = np.zeros((3, 3))
    assert hinge_world_frame(xanchor, xaxis, joint_id=9, dofadr=9, njnt=3) is None


def test_hinge_arrow_is_centered_on_the_pivot():
    start, tip = hinge_arrow_ends([1.0, 2.0, 3.0], [0.0, 0.0, 2.0], length=0.10)
    np.testing.assert_allclose(start, [1.0, 2.0, 2.95])
    np.testing.assert_allclose(tip, [1.0, 2.0, 3.05])


def test_hinge_arrow_zero_axis_collapses_to_the_pivot():
    start, tip = hinge_arrow_ends([0.1, 0.0, 0.0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(start, [0.1, 0.0, 0.0])
    np.testing.assert_allclose(tip, [0.1, 0.0, 0.0])
