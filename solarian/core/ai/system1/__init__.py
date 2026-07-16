"""
System 1 — reactive skill execution.

This package defines the public API that skill plugins build on:

* :class:`Skill` — base class for a skill (runs as its own process).
* :class:`PolicyClient` — generic HTTP client to a policy server.
* :class:`PolicyServer` — base class for hosting a policy over HTTP.
* :class:`PolicyTrainer` — base class for training a policy in a plugin.

A *skill plugin* implements a :class:`Skill` subclass and talks to a policy
via :class:`PolicyClient`.  A *policy plugin* (keyed by model family /
library) ships a :class:`PolicyServer` (inference) and a
:class:`PolicyTrainer` (training) for that policy — deep-learning or
classical.
"""

from solarian.core.ai.system1.skill import Skill, SkillStatus, SkillRunner
from solarian.core.ai.system1.policy_client import PolicyClient
from solarian.core.ai.system1.policy_server import PolicyServer
from solarian.core.ai.system1.policy_trainer import PolicyTrainer, ModalSpec

__all__ = [
    "Skill",
    "SkillStatus",
    "SkillRunner",
    "PolicyClient",
    "PolicyServer",
    "PolicyTrainer",
    "ModalSpec",
]
