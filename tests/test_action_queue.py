"""ModuleActionQueue: one API over a shared-memory queue and an in-process one."""

from __future__ import annotations

from multiprocessing.managers import SharedMemoryManager

import pytest

from solarian.core.action import ActionType, ActuatorField, ModuleAction
from solarian.core.data import PosData
from solarian.utils.shared_memory.action_queue import ModuleActionQueue
from solarian.utils.shared_memory.shared_memory_queue import Empty


def _shared_memory_works() -> bool:
    """The shm queue needs the ``atomics`` native lib to match the interpreter.

    Probes with the same ``atype`` ``SharedMemoryQueue`` uses, so this cannot
    skip over a working queue on an ``atomics`` API it never calls.
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
def action_cls():
    """A one-field ModuleAction, restoring the registry the base class keeps."""
    previous = ModuleAction._active_subclass
    src = PosData(path=("mod", "joint", "pos"))

    class _Action(ModuleAction):
        joint = ActuatorField(src)

    yield _Action
    ModuleAction._active_subclass = previous


def _waypoint(action_cls, value):
    return action_cls(
        action_type=ActionType.SCHEDULE_J,
        data={"joint": {"value": value, "timestamp": 0.0}},
    )


def test_a_local_queue_says_so():
    assert ModuleActionQueue.local().is_local


def test_a_local_queue_round_trips_an_action(action_cls):
    queue = ModuleActionQueue.local()

    queue.put(_waypoint(action_cls, 1.5))

    assert queue.get().joint.value == 1.5


def test_a_local_queue_hands_back_the_identical_object(action_cls):
    """No schema, no flattening: there is no fork for a copy to serve."""
    queue = ModuleActionQueue.local()
    action = _waypoint(action_cls, 1.5)

    queue.put(action)

    assert queue.get() is action


def test_get_all_drains_a_chunk_oldest_first(action_cls):
    """What keeps a chunk of waypoints in order on the way to the robot."""
    queue = ModuleActionQueue.local()
    for value in (0.0, 1.0, 2.0):
        queue.put(_waypoint(action_cls, value))

    drained = queue.get_all()

    assert [a.joint.value for a in drained] == [0.0, 1.0, 2.0]
    assert queue.empty()


def test_get_k_takes_only_what_was_asked_for(action_cls):
    queue = ModuleActionQueue.local()
    for value in (0.0, 1.0, 2.0):
        queue.put(_waypoint(action_cls, value))

    assert [a.joint.value for a in queue.get_k(2)] == [0.0, 1.0]
    assert queue.qsize() == 1


def test_draining_an_empty_queue_raises_like_the_shared_memory_one():
    """Both backings raise ``Empty``, so a consumer needs no branch."""
    queue = ModuleActionQueue.local()

    with pytest.raises(Empty):
        queue.get_all()


def test_clear_discards_what_was_queued(action_cls):
    queue = ModuleActionQueue.local()
    queue.put(_waypoint(action_cls, 1.0))

    queue.clear()

    assert queue.empty()


# --- shared-memory mode ----------------------------------------------------


def _shm_queue(shm_manager, action_cls, **kwargs):
    return ModuleActionQueue.create_from_example(
        shm_manager=shm_manager,
        example_action=action_cls.zeros(),
        buffer_size=8,
        **kwargs,
    )


@needs_shared_memory
def test_shared_memory_mode_round_trips_actuator_data(action_cls):
    with SharedMemoryManager() as shm_manager:
        queue = _shm_queue(shm_manager, action_cls)

        queue.put(_waypoint(action_cls, 1.5))

        assert not queue.is_local
        assert queue.get().joint.value == pytest.approx(1.5)


@needs_shared_memory
def test_params_survive_the_crossing_to_a_module(action_cls):
    """A module reads its control vocabulary off ``params``, so losing it on
    the way through shared memory silently changes what the move means."""
    with SharedMemoryManager() as shm_manager:
        queue = _shm_queue(shm_manager, action_cls)
        action = _waypoint(action_cls, 1.5)
        action.params = {"frame": "ee", "speed": 0.25}

        queue.put(action)

        assert queue.get().params == {"frame": "ee", "speed": 0.25}


@needs_shared_memory
def test_a_string_param_crosses_rather_than_raising(action_cls):
    """``frame`` is a name, not a number; the numeric flatten cannot take it."""
    with SharedMemoryManager() as shm_manager:
        queue = _shm_queue(shm_manager, action_cls)
        action = _waypoint(action_cls, 1.5)
        action.params = {"frame": "world", "speed": None}

        queue.put(action)

        restored = queue.get().params
        assert restored["frame"] == "world"
        # ``None`` means "use the module default" and is not the same as absent.
        assert restored["speed"] is None
        assert "speed" in restored


@needs_shared_memory
def test_an_action_without_params_arrives_with_none(action_cls):
    with SharedMemoryManager() as shm_manager:
        queue = _shm_queue(shm_manager, action_cls)

        queue.put(_waypoint(action_cls, 1.5))

        assert queue.get().params == {}


@needs_shared_memory
def test_params_too_big_for_the_slot_say_so(action_cls):
    with SharedMemoryManager() as shm_manager:
        queue = _shm_queue(shm_manager, action_cls, max_params_bytes=32)
        action = _waypoint(action_cls, 1.5)
        action.params = {"frame": "x" * 64}

        with pytest.raises(ValueError, match="max_params_bytes"):
            queue.put(action)


# --- the recorder's tap ----------------------------------------------------


@needs_shared_memory
def test_the_tap_sees_a_command_the_robot_process_never_handled(action_cls):
    """A skill on the robot puts straight in here from its own process.

    Recording where the message arrived instead would store a policy
    rollout with no actions at all.
    """
    with SharedMemoryManager() as shm_manager:
        queue = _shm_queue(shm_manager, action_cls)
        queue.open_tap(shm_manager)

        queue.put(_waypoint(action_cls, 1.5))

        tapped = queue.drain_tap()
        assert [a.joint.value for a in tapped] == [pytest.approx(1.5)]
        # And the module still gets it: a tap is a copy, not a diversion.
        assert queue.get().joint.value == pytest.approx(1.5)


@needs_shared_memory
def test_draining_the_tap_twice_does_not_record_the_same_command_twice(action_cls):
    with SharedMemoryManager() as shm_manager:
        queue = _shm_queue(shm_manager, action_cls)
        queue.open_tap(shm_manager)
        queue.put(_waypoint(action_cls, 1.5))

        assert len(queue.drain_tap()) == 1
        assert queue.drain_tap() == []


@needs_shared_memory
def test_a_queue_with_no_tap_records_nothing_and_still_works(action_cls):
    """Recording disabled is the default; it must cost the action path nothing."""
    with SharedMemoryManager() as shm_manager:
        queue = _shm_queue(shm_manager, action_cls)

        queue.put(_waypoint(action_cls, 1.5))

        assert queue.drain_tap() == []
        assert queue.get().joint.value == pytest.approx(1.5)


@needs_shared_memory
def test_a_full_tap_never_blocks_the_module(action_cls):
    """A gap in the dataset is recoverable; a stalled control stream is not."""
    with SharedMemoryManager() as shm_manager:
        queue = _shm_queue(shm_manager, action_cls)
        queue.open_tap(shm_manager)

        for value in range(16):
            queue.put(_waypoint(action_cls, float(value)))
            # The module keeps up; the recorder is what stopped draining.
            queue.get()

        # The tap holds what it can and drops the rest rather than refusing
        # the put that would have reached the module.
        assert len(queue.drain_tap()) == 8
        assert queue.empty()
