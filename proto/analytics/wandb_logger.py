import re
from pathlib import Path
from glob import glob
from typing import Dict, Any

import wandb

from proto.common.configs import WandBLoggerConfig, TrainConfig
from proto.analytics.train_logger import TrainLogger, TrainDataLog


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    """Get the WandB run ID from filesystem."""
    # Get the WandB run ID.
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    return wandb_run_id



class WandBLogger(TrainLogger):
    def __init__(self, logger_cfg: WandBLoggerConfig, train_cfg: TrainConfig):
        super().__init__(logger_cfg, train_cfg)
        
        # The validation should already have happened in the Pydantic models
        # Determine run_id based on config
        if self.train_cfg.resume:
            # If we're resuming, either use provided run_id or get from filesystem
            if self.logger_cfg.run_id:
                wandb_run_id = self.logger_cfg.run_id
            else:
                # For resuming, we need output_dir (this has been validated by Pydantic)
                try:
                    wandb_run_id = get_wandb_run_id_from_filesystem(Path(self.train_cfg.output_dir))
                except Exception as e:
                    raise ValueError(f"Failed to get WandB run ID from filesystem: {str(e)}")
        else:
            # Not resuming, no run_id needed
            wandb_run_id = None
            
        # Initialize wandb
        try:
            self.run = wandb.init(
                project=self.logger_cfg.project,  # Required field validated by Pydantic
                entity=self.logger_cfg.entity,
                id=wandb_run_id,
                name=self.train_cfg.job_name,  # Required field validated by Pydantic
                resume="must" if self.train_cfg.resume else None,
                save_code=False,
                notes=self.logger_cfg.notes,
                mode=self.logger_cfg.mode,
                dir=self.train_cfg.output_dir,  # Required when resuming, validated by Pydantic
                config=self.train_cfg.model_dump(),
                job_type="train_eval",
                tags=self.train_cfg.to_tags(),
            )
        except Exception as e:
            # Provide a clean error message for wandb initialization issues
            raise ValueError(f"Error initializing WandB logger: {str(e)}")


    def log(self, data: TrainDataLog):
        data_log = data.to_dict()
        step = data_log.pop("step")
        self.run.log(data_log, step=step)
