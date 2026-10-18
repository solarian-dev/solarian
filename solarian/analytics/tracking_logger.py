from solarian import tracking
from solarian.analytics.train_logger import TrainDataLog, TrainLogger
from solarian.core.configs import TrainConfig, WandBLoggerConfig


class TrackingLogger(TrainLogger):
    """A trainer's ``logger:`` config, logged through solarian.tracking.

    ``logger.type`` names the backend, so the same config logs to wandb or to
    Solarian Base.
    """

    def __init__(self, logger_cfg: WandBLoggerConfig, train_cfg: TrainConfig):
        super().__init__(logger_cfg, train_cfg)
        try:
            self.run = tracking.init(
                backend=logger_cfg.type,
                project=logger_cfg.project,
                entity=getattr(logger_cfg, "entity", None),
                id=getattr(logger_cfg, "run_id", None),
                name=train_cfg.job_name,
                resume="must" if train_cfg.resume else None,
                notes=logger_cfg.notes,
                mode=getattr(logger_cfg, "mode", None),
                dir=train_cfg.output_dir,
                config=train_cfg.model_dump(),
                job_type="train_eval",
                tags=train_cfg.to_tags(),
            )
        except Exception as e:
            raise ValueError(f"Error initializing the {logger_cfg.type} logger: {e}") from e

    def log(self, data: TrainDataLog):
        data_log = data.to_dict()
        step = data_log.pop("step")
        self.run.log(data_log, step=step)
