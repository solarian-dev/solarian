"""ObservationBuffer: one API over a shared-memory ring and an in-process one."""

from __future__ import annotations

import pickle
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import pytest

from solarian.core.data import PosData
from solarian.core.observation import ModuleObs, SensorField
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
