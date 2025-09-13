from abc import ABC, abstractmethod
from proto.common.configs import LoggerConfig
from pydantic import BaseModel


class DataLog(BaseModel):
    pass


class Logger(ABC):
    def __init__(self, cfg: LoggerConfig):
        self.cfg = cfg


    @abstractmethod
    def log(self, data: DataLog):
        pass


