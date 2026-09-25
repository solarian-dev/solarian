"""The observation encoding between PolicyClient and PolicyServer."""

from __future__ import annotations

import json

import numpy as np

from solarian.core.ai.system1.policy_client import _serialize_obs
from solarian.core.ai.system1.policy_server import _to_numpy_obs


def _round_trip(obs):
    """Everything the wire does to an obs dict, and nothing it doesn't."""
    return _to_numpy_obs(json.loads(json.dumps(_serialize_obs(obs))))


def test_an_observation_survives_the_wire_unchanged():
    """A policy reads pixels, so "close enough" is not the bar here."""
    obs = {
        "camera0_rgb": np.random.rand(1, 3, 224, 224).astype(np.float32),
        "robot0_eef_pos": np.random.rand(2, 3).astype(np.float64),
        "robot0_gripper_width": np.array([[0.05]], dtype=np.float32),
    }

    out = _round_trip(obs)

    for key, value in obs.items():
        assert out[key].dtype == value.dtype
        assert out[key].shape == value.shape
        assert np.array_equal(out[key], value)


def test_the_encoded_image_is_far_smaller_than_its_decimal_form():
    """The whole point: decimal text costs both the payload and the CPU.

    Measured on a Pi 5, the list form of this array was 3.05 MB and 142 ms
    to encode, against 176 ms for the policy it was being sent to.
    """
    image = np.random.rand(1, 3, 224, 224).astype(np.float32)

    packed = len(json.dumps(_serialize_obs({"camera0_rgb": image})))
    as_lists = len(json.dumps({"camera0_rgb": image.tolist()}))

    assert packed < as_lists / 3


def test_a_non_contiguous_array_is_encoded_from_what_it_shows():
    """A crop or a transpose is a view, and its buffer is not its contents."""
    view = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]

    out = _round_trip({"robot0_eef_pos": view})

    assert np.array_equal(out["robot0_eef_pos"], view)


def test_plain_lists_are_still_accepted():
    """curl and notebooks post JSON lists, and the endpoint is public."""
    out = _to_numpy_obs({"robot0_eef_pos": [[1.0, 2.0, 3.0]], "task": "cup"})

    assert np.array_equal(out["robot0_eef_pos"], np.array([[1.0, 2.0, 3.0]]))
    assert out["task"] == "cup"


def test_a_decoded_array_can_be_written_to():
    """Policies and torch expect to own their input; a read-only one warns."""
    out = _round_trip({"robot0_eef_pos": np.zeros((2, 3), dtype=np.float32)})

    out["robot0_eef_pos"][0, 0] = 1.0
