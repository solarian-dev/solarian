"""The contract between a controller and whatever draws it.

A controller produces one :class:`UISnapshot` per frame and consumes the
:class:`UICommand` list its UI hands back. Neither side knows anything else
about the other, which is what lets the console, a pygame window and a browser
on the far side of a WebSocket all drive the same controller: a snapshot
crosses a link, a controller cannot.

Everything in a snapshot is either a primitive, a tuple of primitives or an
image array, so :mod:`solarian.core.ui.external` can put one on the wire
without the receiver needing the project's Python classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np


class UICommandType(str, Enum):
    """What a UI may ask of a controller."""

    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    #: Click on a displayed image. ``data``: ``label`` (the UIImage's
    #: ``module/field``), ``point`` (``(x, y)`` in image pixels) and
    #: ``button``.
    CLICK = "click"
    START_SKILL = "start_skill"
    STOP_SKILL = "stop_skill"
    STOP_ALL = "stop_all"
    RECORD_START = "record_start"
    RECORD_STOP = "record_stop"
    #: Seal and delete the episode in progress, then start a fresh one.
    RECORD_REDO = "record_redo"
    #: One reading of a teleop device sampled wherever that device is plugged in.
    INPUT_SAMPLE = "input_sample"
    QUIT = "quit"


@dataclass(frozen=True)
class UICommand:
    type: UICommandType
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type.value, "data": dict(self.data)}

    @classmethod
    def from_dict(cls, payload: dict) -> "UICommand":
        # Deserialized off the wire, so a peer can name a command this build
        # does not have; ValueError from the enum is the caller's signal.
        return cls(
            type=UICommandType(payload["type"]),
            data=dict(payload.get("data") or {}),
        )


@dataclass
class UIImage:
    """One camera frame and any boxes and masks drawn over it.

    ``data`` is ``HxWx3`` uint8 RGB. ``jpeg``, when set, is that same frame
    already encoded — the bytes that arrived from the robot — and a front end
    that sends a picture sends those rather than encoding ``data``. ``masks``
    are ``HxW`` 0/1 arrays the size of ``data``.
    """

    label: str
    data: np.ndarray
    bboxes: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    jpeg: bytes | None = None


@dataclass
class UISnapshot:
    """Everything needed to draw one frame of a controller.

    Built once per tick by :meth:`~solarian.core.controller.Controller.ui_snapshot`
    and handed to exactly one UI, so a field nobody renders costs one attribute
    read rather than a round trip.
    """

    #: Synced-clock stamp, so a remote UI can age the frame against its own.
    t_ns: int = 0
    fps: float | None = None
    #: ``(title, [(field, value) | (field, value, rgb)])`` dashboard panels.
    sections: list[tuple[str, list[tuple]]] = field(default_factory=list)
    images: list[UIImage] = field(default_factory=list)
    #: ``(keys, description)`` rows for the bindings panel.
    key_help: list[tuple[str, str]] = field(default_factory=list)
    #: ``{"available": [...], "running": [...]}`` skill names.
    skills: dict[str, list[str]] = field(default_factory=dict)
    #: Pre-formatted rows (latencies, clock offset, link age).
    telemetry: dict[str, str] = field(default_factory=dict)
    #: Recording state as the robot reports it: ``enabled``, ``active``,
    #: ``episode`` (sealed so far) and ``request``. Echoed back rather than
    #: tracked locally, because the robot is what actually records and a
    #: command the controller sent is not evidence that it arrived.
    recording: dict[str, Any] = field(default_factory=dict)
    status: str = ""


def recording_status(state: dict[str, Any]) -> str:
    """One line every front end shows the same way.

    ``episode`` counts what is sealed, so the episode recording right now is
    the one after it — the operator is told the number they will find on disk,
    not an index they have to adjust.
    """
    if not state.get("enabled"):
        return "not recording"
    if state.get("active"):
        return f"REC episode {state.get('episode', 0)}"
    return f"paused, {state.get('episode', 0)} recorded"


@runtime_checkable
class ControllerUI(Protocol):
    """A front end for a controller.

    The controller drives this from its main thread: ``poll`` for input,
    ``render`` the frame, ``tick`` to pace the loop, ``close`` on the way out.
    """

    #: False when this UI has no window or terminal of its own to read keys
    #: from, so the controller must start the global OS key listener for it.
    captures_keys: bool

    def poll(self) -> list[UICommand]: ...

    def render(self, snapshot: UISnapshot) -> None: ...

    def tick(self, fps: int) -> None: ...

    def close(self) -> None: ...
