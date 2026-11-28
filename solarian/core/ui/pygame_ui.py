"""Single-window front end: the pygame dashboard behind the UI protocol."""

from __future__ import annotations

from solarian.core.controller_gui import ControllerGUI
from solarian.core.ui.base import UICommand, UICommandType, UISnapshot

#: Chat lines that are commands to the agent rather than messages to it.
_CHAT_COMMANDS = {
    "/cancel": UICommandType.AGENT_CANCEL,
    "/reset": UICommandType.AGENT_RESET,
}


def chat_command(text: str) -> UICommand:
    """What a submitted chat line asks of the controller."""
    command = _CHAT_COMMANDS.get(text.strip().lower())
    if command is not None:
        return UICommand(command)
    return UICommand(UICommandType.AGENT_MESSAGE, {"text": text})


class PygameUI:
    """Dashboard, camera grid and key capture in one resizable window.

    Real key-down / key-up transitions and multiple simultaneously held keys,
    captured through the window's own event queue — so no OS accessibility
    permission, unlike the global listener the OpenCV UI needs. With an agent,
    a chat column too.
    """

    captures_keys = True

    def __init__(self, title: str = "Solarian Controller", chat: bool = False):
        self._gui = ControllerGUI(title=title)
        self._chat = chat
        #: The switches as last rendered, so a click sends the flipped value
        #: rather than a toggle a lagging snapshot would apply twice.
        self._targets: dict[str, bool] = {}

    def poll(self) -> list[UICommand]:
        commands: list[UICommand] = []

        def toggle(side: str) -> None:
            on = not self._targets.get(side, False)
            self._targets[side] = on
            commands.append(UICommand(UICommandType.RECORD_TARGETS, {side: on}))

        quit_requested = self._gui.poll(
            on_press=lambda key: commands.append(
                UICommand(UICommandType.KEY_DOWN, {"key": key})
            ),
            on_release=lambda key: commands.append(
                UICommand(UICommandType.KEY_UP, {"key": key})
            ),
            on_click=lambda label, point, button: commands.append(
                UICommand(
                    UICommandType.CLICK,
                    {"label": label, "point": point, "button": button},
                )
            ),
            on_submit=(lambda text: commands.append(chat_command(text))) if self._chat else None,
            on_toggle=toggle,
        )
        if quit_requested:
            commands.append(UICommand(UICommandType.QUIT))
        return commands

    def render(self, snapshot: UISnapshot) -> None:
        recording = snapshot.recording or None
        if recording is not None:
            self._targets = dict(recording["targets"])
        sections = snapshot.sections
        if snapshot.telemetry:
            sections = [("status", list(snapshot.telemetry.items())), *sections]
        self._gui.render(
            sections=sections,
            images=[
                (image.label, image.data, image.bboxes, image.masks, image.bbox_texts)
                for image in snapshot.images
            ],
            key_help=snapshot.key_help,
            status=snapshot.status,
            fps=snapshot.fps,
            agent=snapshot.agent,
            recording=recording,
        )

    def tick(self, fps: int) -> None:
        self._gui.tick(fps)

    def close(self) -> None:
        self._gui.close()
