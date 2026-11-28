"""ObservationBuffer: one API over a shared-memory ring and an in-process one."""

from __future__ import annotations

import pickle
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import pytest

from solarian.core.data import ImageData, MaskData, PosData
from solarian.core.execution import ExecutionPlace
from solarian.core.observation import CalculatedField, ModuleObs, SensorField
from solarian.utils.shared_memory.observation_buffer import ObservationBuffer


def _shared_memory_works() -> bool:
    """The shm ring needs the ``atomics`` native lib to match the interpreter.

    Probes with the same ``atype`` the ring uses, so this cannot skip over a
    working buffer on an ``atomics`` API it never calls.
    """
    try:
        from atomics import UINT, atomicview

        import ctypes

        buf = ctypes.create_string_buffer(8)
        with atomicview(buffer=buf, atype=UINT):
            pass
        return True
    except Exception:
        return False


needs_shared_memory = pytest.mark.skipif(
    not _shared_memory_works(),
    reason="atomics native library unavailable for this interpreter",
)


@pytest.fixture
def obs_cls():
    """A one-field ModuleObs, restoring the registry the base class keeps."""
    previous = ModuleObs._active_subclass
    src = PosData(path=("mod", "joint", "pos"))

    class _Obs(ModuleObs):
        joint = SensorField(src)

    yield _Obs
    ModuleObs._active_subclass = previous


def _sample(obs_cls, value):
    return obs_cls({"joint": {"value": value, "timestamp": 0.0}})


# --- local mode ------------------------------------------------------------


def test_local_buffer_holds_a_payload_shared_memory_could_not():
    """The reason local mode exists: a frozenset is not a number or an array."""
    buffer = ObservationBuffer.local(get_max_k=4)

    buffer.put({"held_keys": frozenset({"w", "a"})})

    assert buffer.get() == {"held_keys": frozenset({"w", "a"})}
    assert buffer.is_local


def test_local_buffer_returns_the_identical_object_it_was_given():
    buffer = ObservationBuffer.local()
    payload = object()

    buffer.put(payload)

    assert buffer.get() is payload


def test_local_buffer_drops_the_oldest_past_its_capacity():
    buffer = ObservationBuffer.local(get_max_k=3)

    for i in range(5):
        buffer.put(i)

    assert buffer.get_all() == [2, 3, 4]
    assert buffer.count == 5


def test_local_buffer_survives_a_pickle_round_trip():
    buffer = ObservationBuffer.local()
    buffer.put({"held_keys": frozenset({"s"})})

    revived = pickle.loads(pickle.dumps(buffer))

    assert revived.get() == {"held_keys": frozenset({"s"})}
    assert revived.module_obs_class is None


# --- accessors shared by both modes ----------------------------------------


def test_latest_is_none_before_anything_lands():
    assert ObservationBuffer.local().latest() is None


def test_latest_is_the_most_recent_put():
    buffer = ObservationBuffer.local()
    buffer.put("first")
    buffer.put("second")

    assert buffer.latest() == "second"


def test_last_k_clamps_to_what_the_buffer_actually_holds():
    buffer = ObservationBuffer.local(get_max_k=8)
    buffer.put(1)
    buffer.put(2)

    assert buffer.last_k(5) == [1, 2]


def test_last_k_clamps_to_capacity_rather_than_asserting():
    buffer = ObservationBuffer.local(get_max_k=2)
    for i in range(4):
        buffer.put(i)

    assert buffer.last_k(10) == [2, 3]


def test_last_k_of_zero_is_empty():
    buffer = ObservationBuffer.local()
    buffer.put(1)

    assert buffer.last_k(0) == []


def test_get_last_k_still_asserts_on_an_over_long_window():
    buffer = ObservationBuffer.local(get_max_k=4)
    buffer.put(1)

    with pytest.raises(AssertionError):
        buffer.get_last_k(3)


# --- shared-memory mode ----------------------------------------------------


@needs_shared_memory
def test_shared_memory_mode_round_trips_a_typed_observation(obs_cls):
    with SharedMemoryManager() as shm_manager:
        buffer = ObservationBuffer.create_from_example(
            shm_manager=shm_manager,
            example_observation=_sample(obs_cls, 0.0),
            get_max_k=4,
        )

        buffer.put(_sample(obs_cls, 1.5))

        assert not buffer.is_local
        assert np.isclose(float(buffer.get()["joint"].value), 1.5)


@needs_shared_memory
def test_shared_memory_last_k_returns_oldest_first(obs_cls):
    with SharedMemoryManager() as shm_manager:
        buffer = ObservationBuffer.create_from_example(
            shm_manager=shm_manager,
            example_observation=_sample(obs_cls, 0.0),
            get_max_k=4,
        )

        for value in (1.0, 2.0, 3.0):
            buffer.put(_sample(obs_cls, value))

        values = [float(o["joint"].value) for o in buffer.last_k(10)]
        assert np.allclose(values, [1.0, 2.0, 3.0])


@pytest.fixture
def camera_obs_cls():
    """An image and a mask segmented from it, the mask declared without a shape."""
    previous = ModuleObs._active_subclass
    image = ImageData(path=("cam", "camera", "rgb_image"), shape=(6, 8, 3))

    class _CameraObs(ModuleObs):
        rgb = SensorField(image)
        target_mask = CalculatedField(
            inputs=[image],
            processor=object,
            output=MaskData(),
            execution_place=ExecutionPlace.ROBOT,
        )

    yield _CameraObs
    ModuleObs._active_subclass = previous


def _camera_sample(camera_obs_cls, mask):
    return camera_obs_cls(
        {
            "rgb": {"value": np.zeros((6, 8, 3), dtype=np.uint8), "timestamp": 0.0},
            "target_mask": {"value": mask, "timestamp": 0.0},
        }
    )


def test_a_shapeless_mask_is_sized_like_its_image(camera_obs_cls):
    assert camera_obs_cls.zeros()["target_mask"].value.shape == (6, 8)


def _at(data, path):
    data.path = path
    return data


def test_two_cameras_of_one_class_keep_their_own_image_sizes():
    """Both cameras read one class-level descriptor; only the path separates them."""
    from solarian.core.configs import CameraConfig
    from solarian.core.data import bind_shape
    from solarian.core.opencv_camera import OpenCVCamera

    previous = ModuleObs._active_subclass
    shared = ImageData()
    right = ("right_cam", "camera", "rgb_image")
    left = ("left_cam", "camera", "rgb_image")
    for path, (w, h) in ((right, (1014, 760)), (left, (640, 360))):
        cfg = CameraConfig(port="0", fps=30, width=2028, height=1520, resize_width=w, resize_height=h)
        bind_shape(path, OpenCVCamera(cfg).data_shapes()["rgb_image"])

    class _TwoCameraObs(ModuleObs):
        right_rgb = SensorField(_at(shared, right))
        right_mask = CalculatedField(
            inputs=[shared],
            processor=object,
            output=MaskData(),
            execution_place=ExecutionPlace.ROBOT,
        )
        left_rgb = SensorField(_at(shared, left))

    try:
        zeros = _TwoCameraObs.zeros()
        assert zeros["right_rgb"].value.shape == (760, 1014, 3)
        assert zeros["left_rgb"].value.shape == (360, 640, 3)
        assert zeros["right_mask"].value.shape == (760, 1014)

        frame = np.zeros((760, 1014, 3), dtype=np.uint8)
        obs = _TwoCameraObs(
            {
                "right_rgb": {"value": frame, "timestamp": 0.0},
                "left_rgb": {"value": np.zeros((360, 640, 3), dtype=np.uint8), "timestamp": 0.0},
                "right_mask": {"value": np.zeros((760, 1014), dtype=np.uint8), "timestamp": 0.0},
            }
        )
        assert obs["right_rgb"].value is frame
    finally:
        ModuleObs._active_subclass = previous


@needs_shared_memory
def test_a_none_mask_reads_back_empty_not_a_lap_old_mask(camera_obs_cls):
    mask = np.ones((6, 8), dtype=np.uint8)
    with SharedMemoryManager() as shm_manager:
        # Two slots, so the third put lands on the first one's slot.
        buffer = ObservationBuffer.create_from_example(
            shm_manager=shm_manager,
            example_observation=camera_obs_cls.zeros(),
            get_max_k=1,
            get_time_budget=0.01,
            put_desired_frequency=1,
        )

        buffer.put(_camera_sample(camera_obs_cls, mask))
        assert (buffer.get()["target_mask"].value == mask).all()
        buffer.put(_camera_sample(camera_obs_cls, mask))
        buffer.put(_camera_sample(camera_obs_cls, None))

        assert not buffer.get()["target_mask"].value.any()
