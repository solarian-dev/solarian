from abc import abstractmethod
from solarian.core.configs import TrainLoggerConfig
from solarian.analytics.logger import Logger
from solarian.analytics.logger import DataLog
from solarian.core.configs import TrainConfig
from typing import Dict, Any


class TrainDataLog(DataLog):
    metrics: Dict[str, Any] = {}
    averaged_metrics: list[str] = ["loss", "grad"]

    def update(self, metric: str, value: float):
        if metric not in self.metrics:
            self.metrics[metric] = {
                "count": 0,
                "sum": 0,
            }

        self.metrics[metric]["count"] += 1
        self.metrics[metric]["sum"] += value
        self.metrics[metric]["last"] = value
        self.metrics[metric]["avg"] = self.metrics[metric]["sum"] / self.metrics[metric]["count"]

    def update_metrics(self, metrics: Dict[str, float]):
        for metric, value in metrics.items():
            self.update(metric, value)

    def get(self, metric: str) -> float:
        if metric not in self.metrics:
            return 0
        if any(substring in metric for substring in self.averaged_metrics):
            return self.metrics[metric]["sum"] / self.metrics[metric]["count"]
        return self.metrics[metric]["last"]


    def to_dict(self):
        data = {}
        for metric in self.metrics:
            data[metric] = self.get(metric)
        return data

    def reset(self):
        self.metrics = {}


    def reset_pattern(self, pattern: str):
        for metric in self.metrics:
            if pattern in metric:
                self.reset_metric(metric)

    def reset_metric(self, metric: str):
        if metric not in self.metrics:    
            self.metrics[metric] = {
                "count": 0,
                "sum": 0,
            }
        else:
            self.metrics[metric]["count"] = 0
            self.metrics[metric]["sum"] = 0
    


class TrainLogger(Logger):
    def __init__(self, logger_cfg: TrainLoggerConfig, train_cfg: TrainConfig):
        self.logger_cfg = logger_cfg
        self.train_cfg = train_cfg
        
        # Validation should already have been handled by Pydantic models

    @abstractmethod
    def log(self, data: TrainDataLog):
        pass