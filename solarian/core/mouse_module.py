"""Mouse as a module: clicks on displayed images are events.

The front end owns the window, so it is the one that sees the click. It
reports it here in image pixels, already mapped back from however the frame
was scaled for display, together with which observation the image was. A
handler then needs nothing from the UI to act on it — it can, say, point a
tracker at what was clicked.
"""

from __future__ import annotations

from solarian.core.event import EventType
from solarian.core.execution import ExecutionMode
from solarian.core.module import Module


class MouseModule(Module):
    """Emits CLICK events for clicks on displayed observation images."""

    kind: str = "input"

    execution_mode = ExecutionMode.INLINE

    events = (EventType.CLICK,)

    default_config = {"frequency": 60.0}

    def update_observations(self):
        """Nothing continuous to publish: a click is a transition."""

    def control_step(self):
        """Input-only: nothing to drive."""

    def on_click(
        self, point: tuple[int, int], module: str, field: str, button: int = 1
    ) -> None:
        """Report a click at *point* on the image of ``module/field``.

        Safe to call from any thread.
        """
        self.emit(
            EventType.CLICK,
            point=(int(point[0]), int(point[1])),
            module=module,
            field=field,
            button=int(button),
        )
