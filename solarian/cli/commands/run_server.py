import logging
import time
import hydra
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.configs import ServerConfig
from solarian.utils import get_class, StatusBoard, setup_process_logging


@CommandRegistry.register('run-server', 'server')
class RunServerCommand(BaseCommand):
    help = "Run the relay server the robot and the controller connect through"
    description = """
    Runs the relay the robot and the controller both dial into: it forwards
    observations and actions between them and serves the clock-sync port they
    measure their shared timebase against. Runs until Ctrl-C.

    Start it first, on whichever machine both peers can reach; conf/server.yaml
    holds the ports and, when CurveZMQ is on, the allowlist of peer public keys
    that `solarian keygen` prints.
    """
    examples = (
        "solarian run-server",
        "solarian run-server -d pi_relay_mac",
    )

    def get_config_class(self):
        from solarian.conf import settings

        """Return the config class for this command."""
        return ServerConfig

    def handle(self, cfg: DictConfig) -> None:
        from solarian.conf import settings

        log_level = getattr(logging, getattr(settings, 'LOG_LEVEL', 'INFO'), logging.INFO)
        setup_process_logging(level=log_level, logger_name=None)
        ServerClass = get_class(settings.SERVER_CLASS)
        server = ServerClass(cfg)
        try:
            server.start()
        except KeyboardInterrupt:
            StatusBoard.stop()
        except Exception:
            StatusBoard.print_exception()
