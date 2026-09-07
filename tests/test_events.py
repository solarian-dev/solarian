"""Event core: emission through a runner sink, drain, and handler dispatch."""

import queue
from types import SimpleNamespace

import pytest

from solarian.core.action import Action
from solarian.core.ai.system1.skill import Skill
from solarian.core.controller import Controller
from solarian.core.event import (
    Event,
    EventHost,
    EventType,
    collect_event_handlers,
    event_handler,
    event_inbox_for,
)
from solarian.core.execution import ExecutionMode
from solarian.core.runner import InlineModuleRunner, ProcessModuleRunner


@pytest.fixture(autouse=True)
def schema_free_actions():
    """Exercise the schema-free ``Action`` path (see test_skill_controller)."""
    previous = Action._active_subclass
    Action._active_subclass = None
    yield
    Action._active_subclass = previous


class FakeModule:
    """Stands in for a Module: owns a runner and emits through its sink."""

    execution_mode = ExecutionMode.INLINE
    events = (EventType.KEY_DOWN, EventType.KEY_UP)

    def __init__(self, name="keyboard"):
        self.name = name
        self.frequency = 60.0
        self._runner = InlineModuleRunner(self)

    def emit(self, event_type, **data):
        self._runner.event_sink.append(
            Event(type=int(event_type), source=self.name, data=dict(data))
        )

    def drain_events(self):
        return self._runner.drain_events()


class TestEventType:
    def test_names_core_values(self):
        assert EventType.name_of(EventType.KEY_DOWN) == "KEY_DOWN"

    def test_projects_extend_above_custom_start(self):
        class MyEventType(EventType):
            LIMIT_SWITCH = 100

        assert EventType.name_of(100) == "MyEventType.LIMIT_SWITCH"

    def test_reserved_range_is_rejected(self):
        with pytest.raises(ValueError):
            class Clashing(EventType):
                OOPS = 1


class TestEvent:
    def test_is_immutable(self):
        event = Event(type=EventType.KEY_DOWN, source="keyboard", data={"key": "w"})
        with pytest.raises(Exception):
            event.type = EventType.KEY_UP

    def test_carries_source_and_payload(self):
        event = Event(type=EventType.KEY_DOWN, source="keyboard", data={"key": "w"})
        assert event.source == "keyboard"
        assert event.data["key"] == "w"
        assert event.timestamp > 0


class TestHandlerCollection:
    def test_subclass_handler_adds_to_the_inherited_one(self):
        class Base:
            @event_handler(EventType.KEY_DOWN)
            def base_down(self, event):
                pass

            @event_handler(EventType.KEY_UP)
            def base_up(self, event):
                pass

        class Derived(Base):
            @event_handler(EventType.KEY_DOWN)
            def derived_down(self, event):
                pass

        handlers = collect_event_handlers(Derived())
        assert handlers[int(EventType.KEY_DOWN)] == ["base_down", "derived_down"]
        assert handlers[int(EventType.KEY_UP)] == ["base_up"]

    def test_same_named_override_replaces_rather_than_adds(self):
        class Base:
            @event_handler(EventType.KEY_DOWN)
            def on_down(self, event):
                pass

        class Derived(Base):
            @event_handler(EventType.KEY_DOWN)
            def on_down(self, event):
                pass

        assert collect_event_handlers(Derived())[int(EventType.KEY_DOWN)] == ["on_down"]


class TestRunnerSinks:
    def test_inline_sink_round_trips(self):
        module = FakeModule()
        module.emit(EventType.KEY_DOWN, key="w")
        module.emit(EventType.KEY_UP, key="w")

        drained = module.drain_events()
        assert [int(e.type) for e in drained] == [
            int(EventType.KEY_DOWN),
            int(EventType.KEY_UP),
        ]
        assert module.drain_events() == []

    def test_process_sink_round_trips(self):
        runner = ProcessModuleRunner(FakeModule())
        runner.event_sink.put(Event(type=EventType.KEY_DOWN, data={"key": "q"}))

        # The mp queue has a feeder thread, so the item may not be readable
        # on the very first poll.
        drained = []
        for _ in range(100):
            drained = runner.drain_events()
            if drained:
                break
        assert [e.data["key"] for e in drained] == ["q"]


class EventCtrl(Controller):
    """Controller with only event plumbing wired up (no zmq / shared memory)."""

    def __init__(self):
        self._init_skill_orchestration()
        self.cfg = SimpleNamespace(fps=30, waypoint_lookahead_s=0.25)
        self.skill_command_queue = SimpleNamespace(empty=lambda: True)
        self.skill_status_buffer = None
        self.modules = {"keyboard": FakeModule()}
        self.observation_buffers = {}
        self.seen: list[Event] = []

    @event_handler(EventType.KEY_DOWN)
    def on_key_down(self, event):
        self.seen.append(event)


class TestControllerDrain:
    def test_dispatches_to_the_declared_handler(self):
        ctrl = EventCtrl()
        ctrl.modules["keyboard"].emit(EventType.KEY_DOWN, key="w")

        events = ctrl._drain_events()

        assert [e.data["key"] for e in ctrl.seen] == ["w"]
        assert len(events) == 1

    def test_unhandled_events_are_still_returned(self):
        ctrl = EventCtrl()
        ctrl.modules["keyboard"].emit(EventType.KEY_UP, key="w")

        events = ctrl._drain_events()

        assert ctrl.seen == []
        assert [int(e.type) for e in events] == [int(EventType.KEY_UP)]

    def test_events_are_ordered_by_when_they_happened(self):
        ctrl = EventCtrl()
        module = ctrl.modules["keyboard"]
        sink = module._runner.event_sink
        sink.append(Event(type=EventType.KEY_DOWN, data={"key": "late"}, timestamp=2.0))
        sink.append(Event(type=EventType.KEY_DOWN, data={"key": "early"}, timestamp=1.0))

        ctrl._drain_events()

        assert [e.data["key"] for e in ctrl.seen] == ["early", "late"]

    def test_a_failing_handler_does_not_break_the_tick(self):
        class BoomCtrl(EventCtrl):
            @event_handler(EventType.KEY_DOWN)
            def on_key_down(self, event):
                raise RuntimeError("handler exploded")

        ctrl = BoomCtrl()
        ctrl.modules["keyboard"].emit(EventType.KEY_DOWN, key="w")

        events = ctrl._drain_events()
        assert len(events) == 1

    def test_the_tick_sees_what_the_drain_dispatched(self):
        ctrl = EventCtrl()
        ctrl.modules["keyboard"].emit(EventType.KEY_DOWN, key="w")

        ctrl._drain_events()
        ctrl.control_step()

        assert [e.data["key"] for e in ctrl.seen] == ["w"]
        assert [e.data["key"] for e in ctrl._events] == ["w"]


class EventRobot(EventHost):
    """A robot's share of the event plumbing, without its shared memory."""

    def __init__(self):
        self.modules = {"limit_switch": FakeModule("limit_switch")}
        self._local_skills = {}
        self.seen: list[Event] = []
        self._init_events()

    @event_handler(EventType.BUTTON_DOWN)
    def on_button_down(self, event):
        self.seen.append(event)


class TestRobotDrain:
    """A robot owning modules is the same situation as a controller owning them."""

    def test_a_robot_dispatches_its_modules_events(self):
        robot = EventRobot()
        robot.modules["limit_switch"].emit(EventType.BUTTON_DOWN, button=1)

        events = robot._drain_events()

        assert [e.data["button"] for e in robot.seen] == [1]
        assert len(events) == 1

    def test_the_real_robot_class_is_an_event_host(self):
        from solarian.core.robot import Robot

        assert issubclass(Robot, EventHost)


class InboxSkill:
    """Stands in for a running skill: reachable only through its inbox."""

    def __init__(self, inbox, name="policy"):
        self.skill_name = name
        self.event_inbox = inbox


class TestFanOut:
    def test_a_running_skill_gets_the_batch_in_its_inbox(self):
        inbox = queue.Queue()
        robot = EventRobot()
        robot._local_skills["policy"] = InboxSkill(inbox)
        robot.modules["limit_switch"].emit(EventType.BUTTON_DOWN, button=1)

        robot._fan_out_events(robot._drain_events())

        assert inbox.get_nowait().data["button"] == 1

    def test_every_skill_gets_its_own_copy(self):
        inboxes = [queue.Queue(), queue.Queue()]
        robot = EventRobot()
        for i, inbox in enumerate(inboxes):
            robot._local_skills[f"s{i}"] = InboxSkill(inbox, name=f"s{i}")
        robot.modules["limit_switch"].emit(EventType.BUTTON_DOWN, button=1)

        robot._fan_out_events(robot._drain_events())

        assert [i.get_nowait().data["button"] for i in inboxes] == [1, 1]

    def test_a_skill_with_no_inbox_is_skipped(self):
        robot = EventRobot()
        robot._local_skills["bare"] = InboxSkill(None)
        robot.modules["limit_switch"].emit(EventType.BUTTON_DOWN, button=1)

        robot._fan_out_events(robot._drain_events())  # must not raise

    def test_a_tick_with_no_events_posts_nothing(self):
        inbox = queue.Queue()
        robot = EventRobot()
        robot._local_skills["policy"] = InboxSkill(inbox)

        robot._fan_out_events(robot._drain_events())

        assert inbox.empty()


class TestInboxKind:
    """A queue only has to cross a process boundary when the skill does."""

    def test_a_process_skill_gets_one_that_survives_the_crossing(self):
        skill = _RecordingSkill(execution_mode=ExecutionMode.PROCESS)
        assert type(event_inbox_for(skill)).__module__.startswith("multiprocessing")

    @pytest.mark.parametrize("mode", [ExecutionMode.INLINE, ExecutionMode.THREAD])
    def test_a_same_process_skill_gets_a_plain_queue(self, mode):
        skill = _RecordingSkill(execution_mode=mode)
        assert isinstance(event_inbox_for(skill), queue.Queue)


class TestSkillInbox:
    def test_a_step_reads_what_the_host_posted(self):
        inbox = queue.Queue()
        skill = _RecordingSkill(event_inbox=inbox)
        inbox.put(Event(type=EventType.KEY_DOWN, data={"key": "w"}))

        skill.worker_setup(reconnect=False)
        skill.worker_tick()

        assert [e.data["key"] for e in skill.seen[-1]] == ["w"]

    def test_an_inbox_is_emptied_by_the_step_that_reads_it(self):
        inbox = queue.Queue()
        skill = _RecordingSkill(event_inbox=inbox)
        inbox.put(Event(type=EventType.KEY_DOWN, data={"key": "w"}))

        skill.worker_setup(reconnect=False)
        skill.worker_tick()
        skill.worker_tick()

        assert skill.seen[-1] == []

    def test_no_inbox_means_no_events_rather_than_a_crash(self):
        skill = _RecordingSkill()

        skill.worker_setup(reconnect=False)
        skill.worker_tick()

        assert skill.seen[-1] == []


class _FakeStatusBuffer:
    def put(self, skill_id, status, timestamp):
        pass


class _RecordingSkill(Skill):
    execution_mode = ExecutionMode.INLINE

    def __init__(self, **kwargs):
        super().__init__(
            name="recorder", skills_status_buffer=_FakeStatusBuffer(), **kwargs
        )
        # Whatever the host posted for the step that is running.
        self.seen: list[list] = []

    def control_step(self, dt):
        self.seen.append(list(self.events))
        return None
