"""EE-frame pose helpers."""

import numpy as np
import pytest

from solarian.core.pose import (
    apply_ee_rotation,
    apply_ee_translation,
    compose_ee_delta,
    quat_to_rotvec,
    rotation_angle_between,
    rotvec_to_quat,
)


def test_rotvec_quat_roundtrip():
    rv = np.array([0.1, -0.2, 0.3])
    np.testing.assert_allclose(quat_to_rotvec(rotvec_to_quat(rv)), rv, atol=1e-9)


def test_rotation_angle_between_two_rotvecs_of_a_half_turn_is_zero():
    # Pointing straight down: the same rotation, rotation vectors 2π apart.
    assert rotation_angle_between([np.pi, 0.0, 0.0], [-np.pi, 0.0, 0.0]) < 1e-6
    assert rotation_angle_between([0.0, 0.0, 0.1], [0.0, 0.0, 0.3]) == pytest.approx(0.2)


def test_apply_ee_translation_along_tool_x():
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])
    out = apply_ee_translation(pose, 0, 0.1)
    # +90° about z maps tool +x onto world +y
    np.testing.assert_allclose(out[:3], [0.0, 0.1, 0.0], atol=1e-9)


def test_compose_ee_delta_matches_helpers():
    pose = np.array([1.0, 2.0, 3.0, 0.1, 0.0, 0.0])
    via_helpers = apply_ee_rotation(
        apply_ee_translation(pose, 2, 0.05), 1, 0.02
    )
    delta = np.array([0.0, 0.0, 0.05, 0.0, 0.02, 0.0])
    np.testing.assert_allclose(compose_ee_delta(pose, delta), via_helpers, atol=1e-9)
