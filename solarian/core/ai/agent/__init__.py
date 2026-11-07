"""The supervising agent built into :class:`~solarian.core.controller.Controller`.

A planner model plans and calls the controller's tools (skills declared as
:class:`~solarian.core.skill.Tool`); a monitor model watches the actions they
start. See :mod:`solarian.core.ai.agent.harness`.
"""

from solarian.core.ai.agent.harness import AgentHarness, RunStatus

__all__ = ["AgentHarness", "RunStatus"]
