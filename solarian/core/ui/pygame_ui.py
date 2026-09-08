"""Single-window front end: the pygame dashboard behind the UI protocol."""

from __future__ import annotations

from solarian.core.controller_gui import ControllerGUI
from solarian.core.ui.base import UICommand, UICommandType, UISnapshot


class PygameUI:
    """Dashboard, camera grid and key capture in one resizable window.

    Real key-down / key-up transitions and multiple simultaneously held keys,
    captured through the window's own event queue — so no OS accessibility
    permission, unlike the global listener the OpenCV UI needs.
    """

    captures_keys = True

    def __init__(self, title: str = "Solarian"):
        self._gui = ControllerGUI(title=title)

    def poll(self) -> list[UICommand]:
        commands: list[UICommand] = []
        quit_requested = self._gui.poll(
            on_press=lambda key: commands.append(
                UICommand(UICommandType.KEY_DOWN, {"key": key})
            ),
            on_release=lambda key: commands.append(
                UICommand(UICommandType.KEY_UP, {"key": key})
            ),
        )
        if quit_requested:
            commands.append(UICommand(UICommandType.QUIT))
        return commands

    def render(self, snapshot: UISnapshot) -> None:
        self._gui.render(
            sections=snapshot.sections,
            images=[(image.label, image.data) for image in snapshot.images],
            key_help=snapshot.key_help,
            status=snapshot.status,
            fps=snapshot.fps,
        )

    def tick(self, fps: int) -> None:
        self._gui.tick(fps)

    def close(self) -> None:
        self._gui.close()
