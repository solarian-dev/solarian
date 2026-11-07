"""
System 1 — training the policies that skills run.

* :class:`PolicyTrainer` — base class for training a policy in a plugin.

Skills themselves live in :mod:`solarian.core.skill`: a controller runs them
whether or not it has an agent. A *policy plugin* (keyed by model family /
library) ships a :class:`~solarian.services.policy.PolicyServer` (inference, a
service like any other) and a :class:`PolicyTrainer` (training) for that
policy — deep-learning or classical.
"""

from solarian.core.ai.system1.policy_trainer import PolicyTrainer, ModalSpec

__all__ = [
    "PolicyTrainer",
    "ModalSpec",
]
