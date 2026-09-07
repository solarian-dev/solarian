"""How a runtime component is scheduled, independent of what it is.

A :class:`~solarian.core.module.Module` describes a logical device and a
:class:`~solarian.core.ai.system1.skill.Skill` describes control logic; neither
should encode whether it happens to live in its own OS process. That is an
operational decision, so it is a property (:class:`ExecutionMode`) resolved at
launch time by :func:`~solarian.core.runner.make_runner`.

Keeping it a property rather than a base class means the same class can be
deployed differently per robot, per project, or per test::

    camera = OpenCVCameraModule(cfg, execution_mode=ExecutionMode.INLINE)

and new modes (remote workers, shared processes, asyncio) can be added without
the class hierarchy growing to encode deployment topology.
"""

from __future__ import annotations

from enum import Enum


class ExecutionMode(str, Enum):
    """Scheduling strategy for a module or skill.

    * ``PROCESS`` — own OS process. Isolates timing from the host loop; the
      right default for cameras, arms, grippers, and anything else with
      hard real-time or expensive work.
    * ``THREAD`` — own thread in the host process. For blocking device APIs
      (HID readers, serial event sources) that need isolation from the host
      loop but not from the GIL-bound host process.
    * ``INLINE`` — no loop of its own; the host ticks it. For cheap,
      event-driven components such as a keyboard, where a process would cost
      more than the work it does.
    """

    PROCESS = "process"
    THREAD = "thread"
    INLINE = "inline"

    @property
    def crosses_process_boundary(self) -> bool:
        """True when the component's code runs in a different process.

        Callers use this to decide whether per-process state (hardware
        handles, log routing, processor graphs) has to be rebuilt.
        """
        return self is ExecutionMode.PROCESS


class ExecutionPlace(str, Enum):
    """Which runtime owns a component, independent of how it is scheduled.

    :class:`ExecutionPlace` answers "on whose machine does this live",
    :class:`ExecutionMode` answers "how is it scheduled there". They are
    orthogonal: a skill placed on the robot may run in its own process or
    inline on the robot's loop.

    * ``CONTROLLER`` — the controller process (or a child of it).
    * ``ROBOT`` — the robot runtime. The controller does not launch it
      locally; it enqueues an ``Action`` the robot's handler picks up.
    * ``SERVER`` — the relay server. Dispatched like ``ROBOT``; which side
      picks the action up is a transport-layer concern.
    """

    CONTROLLER = "controller"
    ROBOT = "robot"
    SERVER = "server"

    @property
    def is_remote(self) -> bool:
        """True when starting it means sending an action rather than launching."""
        return self in (ExecutionPlace.ROBOT, ExecutionPlace.SERVER)


class WorkerExit(str, Enum):
    """Why a worker loop ended, passed to ``worker_teardown``."""

    COMPLETED = "completed"  # the component reported it was finished
    STOPPED = "stopped"  # an external stop was requested
    FAILED = "failed"  # an exception escaped the worker
