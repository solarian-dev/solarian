"""
System 1 — reactive skill execution.

This package defines the public API that skill plugins build on:

* :class:`Skill` — base class for a skill (control logic).
* :class:`PolicyTrainer` — base class for training a policy in a plugin.

A *skill plugin* implements a :class:`Skill` subclass and talks to a policy
via :class:`~solarian.services.policy.PolicyClient`.  A *policy plugin* (keyed
by model family / library) ships a
:class:`~solarian.services.policy.PolicyServer` (inference, a service like
any other) and a :class:`PolicyTrainer` (training) for that policy —
deep-learning or classical.
"""

from solarian.core.execution import ExecutionMode, ExecutionPlace
from solarian.core.ai.system1.skill import Skill, SkillStatus
from solarian.core.ai.system1.policy_trainer import PolicyTrainer, ModalSpec

__all__ = [
    "Skill",
    "SkillStatus",
    "ExecutionMode",
    "ExecutionPlace",
    "PolicyTrainer",
    "ModalSpec",
]
