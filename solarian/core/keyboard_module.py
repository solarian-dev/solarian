"""Keyboard as a module: key transitions are events, held keys are state.

A keyboard is a device connected to the runtime, so it is a
:class:`~solarian.core.module.Module` like any other — but it produces events
rather than consuming actions, and it costs almost nothing to service. That is
why it defaults to :attr:`~solarian.core.execution.ExecutionMode.INLINE`:
a dedicated process for something that does nothing between keystrokes would
buy isolation nobody needs and add a queue hop to every keypress.

Two input paths feed the same event stream:

* a global pynput listener, for OS-level key events when the controller has no
  window of its own;
* :meth:`on_press` / :meth:`on_release`, called directly by the GUI when it
  owns the window and therefore the key events.

Both only enqueue. Handlers run later, on the control thread, when the
controller drains — which is what makes key handling deterministic and keeps
the listener thread from mutating state the control loop is reading.
"""

from __future__ import annotations

import logging

from solarian.core.configs import ModuleConfig
from solarian.core.event import EventType
from solarian.core.execution import ExecutionMode
from solarian.core.module import Module

logger = logging.getLogger("module")

_pynput_keyboard = None
_PYNPUT_TRIED = False


def _get_pynput_keyboard():
    """Return ``pynput.keyboard`` (cached), or ``None`` if unavailable.

    Deferred because importing pynput initializes Cocoa on macOS, which makes
    any later ``fork()`` in this process unsafe. Modules are forked, so this
    must not happen at import time.
    """
    global _pynput_keyboard, _PYNPUT_TRIED
    if _PYNPUT_TRIED:
        return _pynput_keyboard
    _PYNPUT_TRIED = True
    try:
        from pynput import keyboard as _kb

        _pynput_keyboard = _kb
    except Exception:  # pragma: no cover - pynput is optional on some platforms
        _pynput_keyboard = None
    return _pynput_keyboard


class KeyboardModule(Module):
    """Emits KEY_DOWN / KEY_UP and publishes the set of currently held keys."""

    execution_mode = ExecutionMode.INLINE

    events = (EventType.KEY_DOWN, EventType.KEY_UP)

    #: A keyboard has nothing to describe in YAML, so projects don't have to
    #: ship a ``conf/keyboard.yaml`` that says nothing.
    default_config = {"frequency": 60.0}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._listener = None
        # Touched only by the input thread, to drop OS auto-repeat.
        self._pressed: set[str] = set()
        # Touched only by the control thread, recomputed as events drain.
        self._held: set[str] = set()
        #: Optional ``(key, is_press) -> bool`` hook consulted with the raw
        #: device key before anything else. Returning ``True`` means the key
        #: was consumed and no event is emitted, which is how a host puts the
        #: keyboard into a modal mode — a text prompt, say, must swallow the
        #: very keys it is collecting rather than also steering the robot.
        self.interceptor = None

    # ========= observations ============
    @property
    def held_keys(self) -> frozenset[str]:
        """Keys currently held down.

        Continuous state, so it is an observation rather than a flood of
        events: a skill jogging an arm wants to ask "is w down right now",
        not to replay every repeat the OS generated.
        """
        return frozenset(self._held)

    def update_observations(self):
        """Publish the held-key set.

        The base implementation reads sensor fields declared in the robot's
        ``Observation`` schema, which does not describe a controller-side
        keyboard — and could not hold a set of strings if it did. This module
        gets a local buffer instead, so it publishes the plain state.
        """
        if self.observation_buffer is None:
            return
        self.observation_buffer.put({"held_keys": self.held_keys})

    def control_step(self):
        """Input-only: nothing to drive."""

    # ========= events ============
    def drain_events(self):
        """Hand over queued events, folding them into the held-key set first.

        Recomputing here rather than in the input callback is what keeps
        :attr:`held_keys` consistent: it only ever changes on the control
        thread, between ticks.
        """
        events = super().drain_events()
        for event in events:
            key = event.data.get("key")
            if key is None:
                continue
            if int(event.type) == int(EventType.KEY_DOWN):
                self._held.add(key)
            elif int(event.type) == int(EventType.KEY_UP):
                self._held.discard(key)
        return events

    def on_press(self, char: str) -> None:
        """Report a key going down. Safe to call from any thread."""
        self.emit(EventType.KEY_DOWN, key=char)

    def on_release(self, char: str) -> None:
        """Report a key coming up. Safe to call from any thread."""
        self.emit(EventType.KEY_UP, key=char)

    # ========= global OS listener ============
    @staticmethod
    def _pynput_char(key) -> str | None:
        """Lowercase character for a pynput key, or ``None`` for non-character
        keys (arrows, modifiers, function keys, ...)."""
        try:
            char = key.char
        except AttributeError:
            return None
        if not char:
            return None
        return char.lower()

    def _on_pynput_press(self, key) -> None:
        if self.interceptor is not None and self.interceptor(key, True):
            return
        char = self._pynput_char(key)
        if char is None:
            return
        # Filter OS auto-repeat: only fire on the rising edge.
        if char in self._pressed:
            return
        self._pressed.add(char)
        self.on_press(char)

    def _on_pynput_release(self, key) -> None:
        if self.interceptor is not None and self.interceptor(key, False):
            return
        char = self._pynput_char(key)
        if char is None:
            return
        # Deliberately not gated on having seen a matching press: pynput on
        # macOS occasionally reports different ``key.char`` values for the
        # same physical key across press and release (IME / dead-key / layout
        # transitions), and swallowing those releases is worse than letting a
        # stray one through. ``_pressed`` still suppresses auto-repeat.
        self._pressed.discard(char)
        self.on_release(char)

    def start_listener(self) -> None:
        """Start a daemon pynput listener feeding global OS key events.

        Only needed when the host has no window of its own; a GUI calls
        :meth:`on_press` / :meth:`on_release` directly instead.
        """
        if self._listener is not None:
            return
        pynput_keyboard = _get_pynput_keyboard()
        if pynput_keyboard is None:
            logger.warning(
                "pynput is not available; keyboard events will not be "
                "emitted by module '%s'.",
                self.name,
            )
            return
        try:
            listener = pynput_keyboard.Listener(
                on_press=self._on_pynput_press,
                on_release=self._on_pynput_release,
            )
            listener.daemon = True
            listener.start()
            self._listener = listener
            logger.info("pynput keyboard listener started")
        except Exception:
            logger.exception("Failed to start pynput keyboard listener")
            self._listener = None

    def stop_listener(self) -> None:
        listener = self._listener
        if listener is None:
            return
        self._listener = None
        try:
            listener.stop()
        except Exception:
            logger.debug("pynput listener.stop() raised", exc_info=True)
        self._pressed.clear()
        self._held.clear()

    def worker_teardown(self, exit_reason) -> None:
        self.stop_listener()
        super().worker_teardown(exit_reason)
