"""
In-container entry point for Modal training runs.

The generic :class:`~solarian.core.training.modal_launcher.ModalLauncher`
writes the resolved config to the Modal volume and then runs this module
under ``accelerate launch`` inside the GPU container::

    accelerate launch -m solarian.core.training.modal_entry --config <path>

It is intentionally tiny and library-agnostic: it loads the JSON config,
resolves the trainer via the ``solarian.trainers`` registry, and calls
:meth:`~solarian.core.ai.system1.policy_trainer.PolicyTrainer.train`.  All
processes spawned by ``accelerate`` run this same code; the trainer's own
loop (e.g. ``accelerate.Accelerator``) handles rank coordination.
"""

import argparse
import json
import logging

from solarian.core.ai.system1.trainer_registry import get_trainer_class

logger = logging.getLogger(__name__)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Solarian Modal training entry point")
    parser.add_argument("--config", "-c", required=True, help="Path to resolved config JSON.")
    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        cfg_dict = json.load(f)

    policy = cfg_dict.get("policy")
    if not policy:
        raise ValueError("Config is missing the 'policy' key needed to resolve a trainer.")

    trainer_cls = get_trainer_class(policy)
    cfg = trainer_cls.config_class().model_validate(cfg_dict)
    trainer = trainer_cls(cfg, output_dir=getattr(cfg, "output_dir", None))
    trainer.train()


if __name__ == "__main__":
    main()
