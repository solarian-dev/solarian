"""Events: something that happened, as opposed to state or intent.

Solarian carries three data types between its four concepts:

============  =========================================
Observation   state — what the world currently looks like
Event         something that happened, at a point in time
Action        something we want to happen
============  =========================================

Modules produce observations and events; skills and controllers consume them
and produce actions. Continuous signals (joint positions, held keys, a
joystick axis) stay observations — flooding the bus with one event per sample
would make the event path useless for the discrete things it is good at.
Events are for transitions: a key went down, a button was clicked, a limit
switch tripped.

Events are queued by the emitting module and drained by the host on its own
control tick, so handlers run on the control thread rather than on whatever
device callback thread produced them. That makes the ordering deterministic
and removes a whole class of races between input threads and the control loop.

Declaring what a module emits and reacting to it mirrors the action system::

    class KeyboardModule(Module):
        events = (EventType.KEY_DOWN, EventType.KEY_UP)

    class MyController(Controller):
        @event_handler(EventType.KEY_DOWN)
        def on_key_down(self, event: Event) -> None:
            ...
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass, field
from multiprocessing import Queue as MpQueue
from typing import Any, Callable, Mapping

_logger = logging.getLogger("event")


class EventType:
    """Open registry of event types, mirroring :class:`~solarian.core.action.ActionType`.

    Core reserves values below :attr:`CUSTOM_START`; projects and plugins
    subclass this to add their own::

        class MyEventType(EventType):
            LIMIT_SWITCH = 100
    """

    _registry: dict[int, str] = {}

    # Keyboard
    KEY_DOWN = 0
    KEY_UP = 1

    # Pointing devices
    BUTTON_DOWN = 2
    BUTTON_UP = 3

    CUSTOM_START = 100

    def __init__(self, value: int = 0):
        self._value = int(value)

    def __int__(self):
        return self._value

    def __eq__(self, other):
        if isinstance(other, EventType):
            return self._value == other._value
        if isinstance(other, int):
            return self._value == other
        return NotImplemented

    def __hash__(self):
        return hash(self._value)

    def __repr__(self):
        name = self._registry.get(self._value)
        if name:
            return f"EventType({name}={self._value})"
        return f"EventType({self._value})"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, val in vars(cls).items():
            if isinstance(val, int) and not name.startswith("_"):
                if val < EventType.CUSTOM_START and name not in vars(EventType):
                    raise ValueError(f"{name}={val} conflicts with reserved range")
                EventType._registry[val] = f"{cls.__name__}.{name}"

    @classmethod
    def name_of(cls, value: int) -> str:
        return cls._registry.get(value, f"UNKNOWN({value})")


# ``CUSTOM_START`` is the boundary marker for project-defined types, not an
# event type itself, so it is deliberately left out of the registry.
EventType._registry.update(
    {
        value: name
        for name, value in vars(EventType).items()
        if isinstance(value, int)
        and not name.startswith("_")
        and name != "CUSTOM_START"
    }
)


@dataclass(frozen=True)
class Event:
    """Something that happened, stamped and attributed to its source.

    Immutable because an event is a record of the past: the same instance is
    handed to every handler and to every running skill on a tick, and none of
    them should be able to change what the others see.

    Attributes:
        type: An :class:`EventType` value.
        source: Name of the module that emitted it.
        data: Payload, e.g. ``{"key": "w"}``. Keep it small and picklable —
            events from a process-backed module cross a queue.
        timestamp: Wall-clock time the event occurred.
    """

    type: int
    source: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"Event({EventType.name_of(int(self.type))}, source={self.source!r}, "
            f"data={dict(self.data)!r})"
        )


def event_handler(event_type: int):
    """Register a method as a handler for *event_type*.

    Several methods may handle the same event type and all of them run, so a
    subclass can react to a key press without displacing the base class's own
    reaction to it. Redeclaring a handler *by the same method name* does
    override it, which is how you replace inherited behavior deliberately.

    Usage::

        class MyController(Controller):
            @event_handler(EventType.KEY_DOWN)
            def on_key_down(self, event: Event) -> None:
                if event.data["key"] == "q":
                    self.stop_all_skills()
    """

    def decorator(method: Callable) -> Callable:
        setattr(method, "_handles_event", int(event_type))
        return method

    return decorator


def collect_event_handlers(obj) -> dict[int, list[str]]:
    """Map event type -> handler method names for every ``@event_handler`` on *obj*.

    Walked root → leaf, so base handlers run before derived ones and a
    same-named override replaces rather than adds.
    """
    by_name: dict[str, int] = {}
    for klass in reversed(type(obj).__mro__):
        for attr_name, value in klass.__dict__.items():
            if callable(value) and hasattr(value, "_handles_event"):
                by_name[attr_name] = int(getattr(value, "_handles_event"))

    handlers: dict[int, list[str]] = {}
    for attr_name, event_type in by_name.items():
        handlers.setdefault(event_type, []).append(attr_name)
    return handlers


def event_inbox_for(skill) -> "queue.Queue | MpQueue":
    """The right queue for posting events to *skill*.

    A skill in its own process needs one that survives the crossing; anything
    on the host's own loop or thread does not, and a plain queue is cheaper.
    Either way the skill drains it the same way, so nothing downstream has to
    know which it got.
    """
    if skill.execution_mode.crosses_process_boundary:
        return MpQueue()
    return queue.Queue()


class EventHost:
    """Drains its modules' events once a tick and dispatches them.

    Mixed into both :class:`~solarian.core.controller.Controller` and
    :class:`~solarian.core.robot.Robot`: a robot owning modules is the same
    situation as a controller owning modules, and a skill placed on the robot
    should see events for the same reason a controller-side one does.
    """

    # Supplied by the host. Declared so this mixin can read them directly.
    modules: dict
    _local_skills: dict

    def _init_events(self) -> None:
        self._event_handlers = collect_event_handlers(self)
        #: What the most recent drain produced, for the tick to hand to skills.
        self._events: list[Event] = []

    def _drain_events(self) -> list[Event]:
        """Collect what this host's modules emitted and run the handlers.

        Runs at the top of the control tick, so handlers execute on the
        control thread regardless of which device callback thread or child
        process produced the event. Events are ordered by when they happened,
        not by which module reported them.
        """
        events: list[Event] = []
        for name, module in self.modules.items():
            try:
                events.extend(module.drain_events())
            except Exception:
                _logger.exception("Failed to drain events from module '%s'", name)
        events.sort(key=lambda event: event.timestamp)
        for event in events:
            self.handle_event(event)
        self._events = events
        return events

    def handle_event(self, event: Event) -> None:
        """Run every ``@event_handler`` registered for this event's type.

        One misbehaving handler must not cost the others their turn, nor stall
        the control tick, so failures are logged and swallowed.
        """
        for handler_name in self._event_handlers.get(int(event.type), ()):
            try:
                getattr(self, handler_name)(event)
            except Exception:
                _logger.exception(
                    "Event handler '%s' failed for %r", handler_name, event
                )

    def _fan_out_events(self, events: list[Event]) -> None:
        """Post this tick's batch to every local skill's inbox.

        Every skill gets its own copy through the same mechanism, whether it
        runs in a process, on a thread, or inline on this loop, so a skill
        never has to know which it is to read its events. A skill the host
        skips this tick keeps its events for the step that does run, rather
        than losing them.
        """
        if not events:
            return
        for name, skill in list(self._local_skills.items()):
            inbox = skill.event_inbox
            if inbox is None:
                continue
            for event in events:
                try:
                    inbox.put_nowait(event)
                except Exception:  # noqa: BLE001 - a full inbox must not stall the tick
                    _logger.debug("Skill '%s': dropped event %r", name, event)
