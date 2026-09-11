"""On-screen prompt key mapping for the sim viewer."""

from solarian.sim.viewer import _poll_prompt, _wrap_prompt, format_joint_hud_lines, prompt_reply


def test_enter_and_space_confirm_enter_prompt():
    prompt = {"kind": "enter", "text": "go"}
    assert prompt_reply(13, prompt) == ""
    assert prompt_reply(10, prompt) == ""
    assert prompt_reply(32, prompt) == ""
    assert prompt_reply(ord("y"), prompt) is None


def test_yn_keys_answer_yesno_prompt():
    prompt = {"kind": "yesno", "text": "same?"}
    assert prompt_reply(ord("y"), prompt) == "y"
    assert prompt_reply(ord("N"), prompt) == "n"
    assert prompt_reply(13, prompt) is None
    assert prompt_reply(ord("w"), prompt) is None


def test_no_prompt_never_consumes_keys():
    assert prompt_reply(13, None) is None
    assert prompt_reply(ord("y"), None) is None


def test_wrap_prompt_keeps_paragraphs():
    lines = _wrap_prompt("Move the arm.\n\nPress ENTER.", 20)
    assert "Move the arm." in lines
    assert "" in lines
    assert "Press ENTER." in lines


def test_format_joint_hud_lines_marks_active_and_missing():
    lines = format_joint_hud_lines(
        {
            "joints": ["shoulder_pan", "wrist_roll"],
            "real": {"shoulder_pan": -2.6},
            "sim": {"shoulder_pan": 0.012, "wrist_roll": -0.4},
            "active": "wrist_roll",
        }
    )
    assert lines[0].startswith("joint")
    assert "nrm" in lines[0] and "rad" in lines[0]
    assert any(line.startswith(" shoulder_pan") and "-2.6" in line and "0.012" in line for line in lines)
    assert any(line.startswith("*wrist_roll") and "--" in line and "-0.400" in line for line in lines)


class _FakeConn:
    def __init__(self, messages):
        self.messages = list(messages)

    def poll(self):
        return bool(self.messages)

    def recv(self):
        return self.messages.pop(0)


def test_poll_prompt_stores_hud_and_ask():
    state = {"prompt": None, "hud": None}
    conn = _FakeConn(
        [
            ("hud", {"joints": ["elbow_flex"], "real": {}, "sim": {}}),
            ("ask", {"text": "same?", "kind": "yesno"}),
        ]
    )
    _poll_prompt(conn, state)
    assert state["hud"]["joints"] == ["elbow_flex"]
    assert state["prompt"]["kind"] == "yesno"
    assert state["prompt"]["text"] == "same?"
