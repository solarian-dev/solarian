from solarian.core.configs import LoggerConfig
from solarian.analytics.logger import Logger


class LoggerFactory():

    @staticmethod
    def create_logger(logger_cfg: LoggerConfig, **kwargs) -> Logger:
        """A logger for ``logger_cfg``, whose ``type`` names a solarian.tracking backend.

        Raises:
            ValueError: If the backend is unknown or fails to start.
        """
        from solarian.analytics.tracking_logger import TrackingLogger

        return TrackingLogger(logger_cfg, **kwargs)
