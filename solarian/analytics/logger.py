from abc import ABC, abstractmethod
from solarian.core.configs import TrainLoggerConfig
from pydantic import BaseModel


class DataLog(BaseModel):
    pass


class Logger(ABC):
    def __init__(self, cfg: TrainLoggerConfig):
        self.cfg = cfg


    @abstractmethod
    def log(self, data: DataLog):
        pass


