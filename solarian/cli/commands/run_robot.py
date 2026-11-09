import argparse
import logging
import signal

from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.configs import RobotConfig
from solarian.utils import get_class, StatusBoard, setup_process_logging

# run-robot is the subcommand, robot is the config name
@CommandRegistry.register('run-robot', 'robot')
class RunRobotCommand(BaseCommand):
    category = "Robot"
    help = "Run the robot: connect its modules and join the relay a controller drives it through"
    description = """
    Builds the project's Robot class, connects every module it declares, and joins
    the relay so a controller or a policy can drive it. Runs until Ctrl-C.

    This is the robot-side process: run it on the machine the hardware is wired
    to, with the deployment profile that names that machine's device ports. It
    holds the actuator buses and cameras open, so calibration and camera tools
    cannot run at the same time.

    With --sim the same robot runs in MuJoCo instead of on hardware, in the
    scene its class declares.
    """
    usage = "[--sim] [key=value ...]"
    options = {
        "--sim": "Run in MuJoCo instead of on hardware; the robot must declare a scene.",
    }
    examples = (
        "solarian run-robot",
        "solarian run-robot --sim",
        "solarian run-robot -d pi",
        "solarian run-robot fps=60",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian run-robot")
        parser.add_argument("--sim", action="store_true", help="Run in MuJoCo instead of on hardware.")
        ns, overrides = parser.parse_known_args(argv)
        self.sim = ns.sim
        return overrides

    def get_config_class(self):
        """Return the config class for this command."""
        return RobotConfig

    def handle(self, cfg: RobotConfig) -> None:
        from solarian.conf import settings

        log_level = getattr(logging, getattr(settings, 'LOG_LEVEL', 'INFO'), logging.INFO)
        setup_process_logging(level=log_level, logger_name=None)
        if self.sim:
            cfg.sim = True
        RobotClass = get_class(settings.ROBOT_CLASS)
        robot = RobotClass(cfg)

        # The first SIGTERM takes the same path as Ctrl-C, so start()'s
        # finally still disconnects every module. A second one, or one that
        # lands once disconnect() has begun, must not abort that shutdown.
        robot._shutdown_started = False

        def _on_sigterm(signum, frame):
            if robot._shutdown_started:
                return
            robot._shutdown_started = True
            raise KeyboardInterrupt

        previous_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
        try:
            try:
                robot.connect()
                robot.initialize()
                robot.start()
            except KeyboardInterrupt:
                StatusBoard.stop()
            except Exception:
                StatusBoard.print_exception()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
