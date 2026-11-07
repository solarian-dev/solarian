"""Layout helpers and key routing for the pygame dashboard. No window."""

from types import SimpleNamespace

from solarian.core.controller_gui import (
    ControllerGUI,
    _recording_lines,
    clip_text,
    selected_text,
    split_temperature_rows,
    wrap_text,
)
from solarian.core.ui.base import UICommandType
from solarian.core.ui.pygame_ui import PygameUI, chat_command


def _width(text: str) -> tuple[int, int]:
    return (len(text) * 10, 16)


def test_clip_text_keeps_a_line_that_fits():
    assert clip_text(_width, "home", 40) == "home"


def test_clip_text_ends_a_long_line_with_an_ellipsis():
    # "place object at the TAPNext point" is 34 chars, 340px. Budget is 200.
    clipped = clip_text(_width, "place object at the TAPNext point", 200)
    assert clipped.endswith("…")
    assert _width(clipped)[0] <= 200
    assert clipped.startswith("place object")


def test_clip_text_gives_up_when_even_the_ellipsis_does_not_fit():
    assert clip_text(_width, "home", 0) == ""
    assert clip_text(_width, "home", 5) == ""


def test_temperature_fields_leave_the_position_list_in_order():
    rows = [
        ("shoulder_pan", "-0.143"),
        ("shoulder_pan_temp", "29", (240, 80, 80)),
        ("elbow_flex", "45.528"),
        ("elbow_flex_temp", "37"),
    ]
    main, temps = split_temperature_rows(rows)
    assert main == [("shoulder_pan", "-0.143"), ("elbow_flex", "45.528")]
    assert temps == [
        ("shoulder_pan_temp", "29", (240, 80, 80)),
        ("elbow_flex_temp", "37"),
    ]


def test_wrap_text_breaks_at_spaces_and_keeps_newlines():
    assert wrap_text(_width, "pick the battery now", 100) == ["pick the", "battery", "now"]
    assert wrap_text(_width, "a\nb", 100) == ["a", "b"]
    assert wrap_text(_width, "unbreakableword", 60)[0].endswith("…")


def test_chat_lines_become_agent_commands():
    assert chat_command("put the batteries away").data == {"text": "put the batteries away"}
    assert chat_command("/cancel").type == UICommandType.AGENT_CANCEL
    assert chat_command(" /RESET ").type == UICommandType.AGENT_RESET


_KEYS = {13: "return", 27: "escape", 8: "backspace", 119: "w", 99: "c", 118: "v", 120: "x", 97: "a"}
CMD, ALT, SHIFT = 0x0400, 0x0100, 0x0001
BACKSPACE, LEFT = 8, 1073741904


class _FakeClipboard:
    def __init__(self, text=""):
        self.text = text

    def get(self):
        return self.text

    def set(self, text):
        self.text = text


class _FakePygame:
    QUIT, KEYDOWN, KEYUP, TEXTINPUT, MOUSEWHEEL, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION = range(8)
    K_RETURN, K_KP_ENTER, K_ESCAPE, K_BACKSPACE = 13, 271, 27, 8
    K_c, K_v, K_x, K_a, K_DELETE = 99, 118, 120, 97, 127
    K_LEFT, K_RIGHT, K_HOME, K_END = LEFT, 1073741903, 1073741898, 1073741901
    KMOD_CTRL, KMOD_META, KMOD_ALT, KMOD_SHIFT = 0x0040, CMD, ALT, SHIFT

    def __init__(self):
        self.queue = []
        self.repeat = (0, 0)
        self.event = SimpleNamespace(get=self._drain)
        self.key = SimpleNamespace(
            name=lambda key: _KEYS.get(key, "unknown"),
            start_text_input=lambda: None,
            stop_text_input=lambda: None,
            set_repeat=self._set_repeat,
        )

    def _set_repeat(self, delay, interval):
        self.repeat = (delay, interval)

    def _drain(self):
        events, self.queue = self.queue, []
        return events

    def down(self, key, mod=0):
        self.queue.append(SimpleNamespace(type=self.KEYDOWN, key=key, mod=mod))

    def up(self, key):
        self.queue.append(SimpleNamespace(type=self.KEYUP, key=key, mod=0))

    def text(self, text):
        self.queue.append(SimpleNamespace(type=self.TEXTINPUT, text=text))

    def click(self, pos, button=1):
        self.queue.append(SimpleNamespace(type=self.MOUSEBUTTONDOWN, pos=pos, button=button))


def _gui():
    gui = ControllerGUI.__new__(ControllerGUI)
    gui._pygame = _FakePygame()
    gui._held = set()
    gui._input = None
    gui._image_rects = []
    gui._toggle_rects = []
    gui._chat_lines = []
    gui._selection = None
    gui._clipboard = _FakeClipboard()
    gui._scale = 1.0
    gui._input_rect = None
    gui._chat_rect = None
    return gui


def _poll(gui):
    log = []
    quit_requested = gui.poll(
        on_press=lambda k: log.append(("down", k)),
        on_release=lambda k: log.append(("up", k)),
        on_submit=lambda t: log.append(("say", t)),
    )
    return log, quit_requested


def test_typing_a_message_never_drives_the_robot():
    gui = _gui()
    pg = gui._pygame
    pg.down(119)
    assert _poll(gui) == ([("down", "w")], False)

    # Enter releases the held key, then keys only type.
    pg.down(13)
    pg.text("w")
    pg.down(119)
    pg.up(119)
    pg.text("ow")
    pg.down(8)
    pg.down(13)
    assert _poll(gui) == ([("up", "w"), ("say", "wo")], False)
    assert gui._input is None


def test_escape_drops_the_message_rather_than_quitting():
    gui = _gui()
    pg = gui._pygame
    pg.down(13)
    pg.text("hello")
    pg.down(27)
    assert _poll(gui) == ([], False)
    assert gui._input is None

    pg.down(27)
    assert _poll(gui) == ([], True)


def test_a_selection_copies_as_the_text_it_was_wrapped_from():
    lines = [("you: put the", "\n"), ("battery away", " "), ("Done.", "\n")]

    assert selected_text(lines, (0, 5), (1, 7)) == "put the battery"
    assert selected_text(lines, (1, 8), (2, 5)) == "away\nDone."


def test_cmd_v_opens_the_chat_with_the_clipboard_and_cmd_c_never_drives_the_robot():
    gui = _gui()
    pg = gui._pygame
    gui._clipboard.text = "put the\nbatteries away"
    pg.down(118, CMD)
    assert _poll(gui) == ([], False)
    assert gui._input.text == "put the batteries away"

    pg.down(120, CMD)
    assert _poll(gui) == ([], False)
    assert (gui._input.text, gui._clipboard.text) == ("", "put the batteries away")

    pg.down(27)
    gui._chat_lines = [("Done.", "\n")]
    gui._selection = ((0, 0), (0, 4))
    pg.down(99, CMD)
    assert _poll(gui) == ([], False)
    assert gui._clipboard.text == "Done"


def test_the_chat_line_edits_like_a_text_field():
    gui = _gui()
    pg = gui._pygame
    pg.down(13)
    pg.text("put the battery away")
    _poll(gui)
    edit = gui._input
    assert pg.repeat != (0, 0)

    pg.down(BACKSPACE, ALT)
    _poll(gui)
    assert edit.text == "put the battery "

    pg.down(LEFT, ALT)
    pg.down(LEFT, ALT | SHIFT)
    _poll(gui)
    assert (edit.selected(), edit.cursor) == ("the ", 4)

    pg.text("a ")
    _poll(gui)
    assert edit.text == "put a battery "

    pg.down(97, CMD)
    pg.down(99, CMD)
    _poll(gui)
    assert (edit.selected(), gui._clipboard.text) == ("put a battery ", "put a battery ")

    pg.down(BACKSPACE)
    pg.down(13)
    assert _poll(gui) == ([], False)
    assert gui._input is None and pg.repeat == (0, 0)


def _recording(*, controller, robot, active=False):
    return {
        "targets": {"controller": controller, "robot": robot},
        "enabled": controller or robot,
        "active": active,
        "episode": 0,
        "session": None,
        "robot": {"enabled": True, "active": active and robot, "episode": 0, "session": None},
        "controller": {"active": active and controller, "episode": 0, "session": None},
    }


def test_clicking_a_recording_switch_sends_its_flipped_value():
    """The flipped value, not a toggle: a click between two snapshots stays one flip."""
    ui = PygameUI.__new__(PygameUI)
    ui._gui = gui = _gui()
    ui._chat = False
    ui._targets = {"controller": False, "robot": True}
    gui._toggle_rects = [("robot", (100, 10, 30, 14)), ("controller", (100, 30, 30, 14))]

    gui._pygame.click((110, 35))
    gui._pygame.click((10, 35))
    commands = ui.poll()

    assert [(c.type, c.data) for c in commands] == [
        (UICommandType.RECORD_TARGETS, {"controller": True})
    ]


def test_the_recording_panel_says_when_a_switch_waits_for_the_next_episode():
    idle = _recording_lines(_recording(controller=True, robot=False))
    assert [(kind, side, on) for kind, side, on, _, _ in idle] == [
        ("switch", "robot", False),
        ("switch", "controller", True),
    ]

    live = _recording_lines(_recording(controller=True, robot=True, active=True))
    assert live[-1][0] == "hint" and all(line[3] for line in live[:2])
