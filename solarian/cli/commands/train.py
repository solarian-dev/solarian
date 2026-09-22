"""
``solarian train`` — library-agnostic training dispatcher.

The framework does not know how any particular policy is trained.  It reads
the ``policy`` key from the composed config, resolves the matching
:class:`~solarian.core.ai.system1.policy_trainer.PolicyTrainer` subclass via
the ``solarian.trainers`` entry-point group, validates the config with that
trainer's :meth:`config_class`, and runs it — either locally or, when
``modal.enabled`` is set, on Modal via the generic launcher.

Concrete training lives entirely in policy plugins (e.g.
``solarian-policy-diffusion``, ``solarian-policy-lerobot``).
"""

import os
import sys
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.ai.system1.trainer_registry import (
    available_trainer_names,
    get_trainer_class,
)


@CommandRegistry.register('train', 'train')
class TrainCommand(BaseCommand):
    help = 'Train a policy via an installed policy trainer plugin.'
    description = """
    Reads the policy key from conf/train.yaml, resolves the matching trainer plugin
    (solarian-policy-diffusion, solarian-policy-lerobot, ...), validates the whole
    config against that trainer's schema and runs it.

    Checkpoints land in output_dir, defaulting to models/<job_name>. Set
    modal.enabled to run the same config on Modal's GPUs instead of this machine.
    The framework knows nothing about the training itself: everything specific
    lives in the plugin, so the keys the config accepts are the ones that trainer
    declares.
    """
    examples = (
        "solarian train",
        "solarian train policy=diffusion job_name=cup_arrange_v3",
        "solarian train modal.enabled=true",
    )

    def get_config_class(self):
        # Validation is deferred to the resolved trainer's config_class(),
        # so the command receives the raw composed config.
        return None

    def handle(self, cfg) -> None:
        try:
            if isinstance(cfg, DictConfig):
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            else:
                cfg_dict = dict(cfg)

            policy = cfg_dict.get("policy")
            if not policy:
                available = ", ".join(available_trainer_names()) or "(none installed)"
                raise ValueError(
                    "Training config must set 'policy' to select a trainer. "
                    f"Available trainers: {available}."
                )

            # Resolve the trainer plugin and validate the config with its schema.
            trainer_cls = get_trainer_class(policy)
            train_cfg = trainer_cls.config_class().model_validate(cfg_dict)

            # Cloud path: hand off to the generic Modal launcher, which sets
            # up the image / volume / GPUs and runs the trainer remotely.
            modal_cfg = getattr(train_cfg, "modal", None)
            if modal_cfg is not None and getattr(modal_cfg, "enabled", False):
                from solarian.core.training.modal_launcher import ModalLauncher

                ModalLauncher(trainer_cls, train_cfg).run()
                return

            # Local path: resolve an output directory and run in-process.
            if train_cfg.output_dir:
                output_dir = Path(train_cfg.output_dir)
            else:
                project_path = os.getenv("SOLARIAN_PROJECT_PATH", os.getcwd())
                output_dir = Path(project_path) / "models" / train_cfg.job_name
            output_dir.mkdir(parents=True, exist_ok=True)
            train_cfg.output_dir = str(output_dir)

            trainer = trainer_cls(train_cfg, output_dir=str(output_dir))
            trainer.train()

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            import traceback
            print(traceback.format_exc())
            sys.exit(1)
