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
UI_NAMES = ("console", "pygame", "opencv", "external", "null")

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

    Takes the controller rather than the config alone because two of the front
    ends need more than config: ``pygame`` honours the deprecated ``use_gui``
    class attribute, and ``external`` serves the controller's own ``hello``
    payload. The UI objects themselves get plain data, never the controller, so
    a front end cannot reach past the snapshot/command protocol.
    """
    cfg = controller.cfg
    # More than one class in the MRO declaring it means a project subclass did,
    # on top of the base default — the difference between "opted in" and "never
    # touched it", which is what decides whether a warning is useful.
    declared = [k for k in type(controller).__mro__ if "use_gui" in k.__dict__]
    name = _resolve_name(
        cfg, use_gui=type(controller).use_gui, use_gui_declared=len(declared) > 1
    )

    if name == "pygame":
        from solarian.core.ui.pygame_ui import PygameUI

        try:
            return PygameUI()
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


def _resolve_name(cfg, use_gui: bool, use_gui_declared: bool) -> str:
    """Resolve ``cfg.ui``, honouring the flags it replaced.

    An explicit ``ui`` always wins; the deprecated flags only steer the default,
    so a project that has migrated is never second-guessed by a stale flag.
    """
    name = (cfg.ui or "console").lower()
    if name not in UI_NAMES:
        raise ValueError(
            f"Unknown ui '{cfg.ui}'; expected one of {', '.join(UI_NAMES)}"
        )
    if name != "console":
        return name

    if cfg.headless:
        # The old headless loop was the status board plus cbreak stdin, which is
        # exactly ConsoleUI — so this maps to the default rather than to `null`,
        # and a headless deployment keeps its key input.
        logger.warning(
            "controller.headless is deprecated; ui=console is the default and "
            "behaves the same. Use ui=null for a controller with no UI at all."
        )
        return "console"

    if not use_gui:
        logger.warning(
            "Controller.use_gui is deprecated; set ui=opencv in "
            "conf/controller.yaml instead."
        )
        return "opencv"

    if use_gui_declared:
        # Silently swapping the front end under a project that asked for the
        # window would look like the window broke.
        logger.warning(
            "Controller.use_gui is deprecated and no longer opens the pygame "
            "window: ui=console is the default. Set ui=pygame in "
            "conf/controller.yaml to keep the window."
        )

    return name
