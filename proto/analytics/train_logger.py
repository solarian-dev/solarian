from abc import abstractmethod
from proto.common.configs import TrainLoggerConfig
from proto.analytics.logger import Logger
from proto.analytics.logger import LogData
from proto.common.configs import TrainConfig
from typing import Dict, Any


class LogTrainData(LogData):
    step: int | None = None
    data: Dict[str, Any]
    


class TrainLogger(Logger):
    def __init__(self, logger_cfg: TrainLoggerConfig, train_cfg: TrainConfig):
        self.logger_cfg = logger_cfg
        self.train_cfg = train_cfg
        
        # Validation should already have been handled by Pydantic models

    @abstractmethod
    def log(self, data: LogTrainData):
        pass