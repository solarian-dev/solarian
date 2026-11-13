"""Pluggable controller front ends.

``conf/controller.yaml`` names one with ``ui:``; :func:`build_ui` constructs it.
Adding a front end means adding a module here and a name to :data:`UI_NAMES` —
the controller itself has no idea which one it is driving.
"""

from __future__ import annotations

import logging

from solarian.core.ui.base import (
    ControllerUI,
    UICommand,
    UICommandType,
    UIImage,
    UISnapshot,
)

logger = logging.getLogger(__name__)

#: Accepted values of ``ControllerConfig.ui``.
UI_NAMES = ("console", "gui", "opencv", "external", "null")

__all__ = [
    "ControllerUI",
    "UICommand",
    "UICommandType",
    "UIImage",
    "UISnapshot",
    "UI_NAMES",
    "build_ui",
]


def build_ui(controller) -> ControllerUI:
    """Construct the front end this controller's config asks for.

    Takes the controller rather than the config alone because some front ends
    need more than config: ``gui`` shows a chat box when there is an agent,
    and ``external`` serves the controller's own ``hello`` payload. The UI
    objects themselves get plain data, never the controller, so a front end
    cannot reach past the snapshot/command protocol.
    """
    cfg = controller.cfg
    name = _resolve_name(cfg)

    if name == "gui":
        from solarian.core.ui.pygame_ui import PygameUI

        try:
            return PygameUI(chat=controller.agent is not None)
        except Exception:
            # No display, no SDL, or a remote shell: the cameras are still
            # worth showing, so fall back rather than refuse to run.
            logger.exception("pygame UI unavailable; falling back to ui=opencv")
            name = "opencv"

    if name == "opencv":
        from solarian.core.ui.opencv import OpenCVUI

        return OpenCVUI()

    if name == "external":
        from solarian.core.ui.external import ExternalUI

        return ExternalUI(
            bind=cfg.ui_bind,
            token=cfg.ui_token,
            image_fps=cfg.ui_image_fps,
            hello=controller.ui_hello(),
        )

    if name == "null":
        from solarian.core.ui.null import NullUI

        return NullUI()

    from solarian.core.ui.console import ConsoleUI

    return ConsoleUI()


def _resolve_name(cfg) -> str:
    name = cfg.ui.lower()
    if name not in UI_NAMES:
        raise ValueError(
            f"Unknown ui '{cfg.ui}'; expected one of {', '.join(UI_NAMES)}"
        )
    return name
