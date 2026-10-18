"""
``solarian train`` — library-agnostic training dispatcher.

The framework does not know how any particular policy is trained.  It reads
the ``trainer`` key from the training file, resolves the matching
:class:`~solarian.core.ai.system1.policy_trainer.PolicyTrainer` subclass via
the ``solarian.trainers`` entry-point group, validates the config with that
trainer's :meth:`config_class`, and runs it on this machine. The
``solarian-modal`` plugin's ``solarian modal-train`` runs the same config on
Modal.

Concrete training lives entirely in policy plugins (e.g.
``solarian-umi-skill``, ``solarian-policy-lerobot``).
"""

import os
import sys
from pathlib import Path

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.ai.system1.trainer_registry import load_train_config


@CommandRegistry.register('train')
class TrainCommand(BaseCommand):
    category = "Training & Data"
    help = 'Train a policy via an installed policy trainer plugin.'
    description = """
    Runs one training file, conf/training/<name>.yaml. Its trainer key selects
    the installed trainer plugin (solarian-umi-skill, solarian-policy-lerobot,
    ...), which validates the whole file against its schema and runs it.

    Checkpoints land in output_dir, defaulting to models/<job_name>. To run the
    same file on Modal's GPUs instead, install solarian-modal and use
    solarian modal-train. The framework knows nothing about the training itself: everything specific
    lives in the plugin, so the keys the file accepts are the ones that trainer
    declares. Any key is overridden as key=value.
    """
    examples = (
        "solarian train --conf conf/training/umi_pick",
        "solarian train --conf conf/training/umi_pick training.num_epochs=50",
    )
    # The trainer's own resolvers have to be registered first.
    resolve_config = False

    def get_config_class(self):
        # Validation is deferred to the resolved trainer's config_class().
        return None

    def handle(self, cfg) -> None:
        try:
            trainer_cls, _cfg_dict, train_cfg = load_train_config(cfg)

            if train_cfg.output_dir:
                output_dir = Path(train_cfg.output_dir).expanduser()
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
