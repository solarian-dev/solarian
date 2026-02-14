"""
Management command to open an interactive Python shell with project settings loaded.
"""
import code
import sys
import os

from omegaconf import DictConfig, OmegaConf
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register('shell', None)
class ShellCommand(BaseCommand):
    help = 'Open an interactive Python shell with project settings loaded.'

    def get_config_class(self):
        """This command doesn't require configuration."""
        return None

    def _create_robot(self, settings):
        """Load the robot config and create the robot instance for the project."""
        from solarian.utils import get_class
        from solarian.utils.configs import parse_with_friendly_errors
        from solarian.core.configs import RobotConfig

        conf_path = os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))

        # Clear any existing Hydra state (manage.py already initializes one)
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Use Hydra to resolve the robot config (handles defaults like connection)
        with initialize_config_dir(version_base=None, config_dir=conf_path, job_name="solarian_shell"):
            hydra_cfg = compose(config_name="robot")
            cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)

        robot_cfg = parse_with_friendly_errors(RobotConfig, cfg_dict)

        RobotClass = get_class(settings.ROBOT_CLASS)
        robot = RobotClass(robot_cfg)
        return robot

    def handle(self, cfg: DictConfig = None) -> None:
        from solarian.conf import settings

        # Build a namespace with useful objects pre-loaded
        namespace = {
            'settings': settings,
            'PROJECT': getattr(settings, 'PROJECT', None),
        }

        # Try to import commonly useful modules into the namespace
        try:
            import solarian
            namespace['solarian'] = solarian
        except ImportError:
            pass

        # Create the project's robot instance
        robot = None
        try:
            robot = self._create_robot(settings)
            namespace['robot'] = robot
        except Exception as e:
            print(f"Warning: Could not create robot instance: {e}")
            print("The 'robot' variable will not be available in the shell.\n")

        robot_line = "  robot     - Robot instance\n" if robot is not None else ""
        banner = (
            "\nSolarian Interactive Shell\n"
            "─────────────────────────\n"
            "Available objects:\n"
            "  settings  - Project settings module\n"
            "  PROJECT   - Project instance\n"
            f"{robot_line}"
            "  solarian  - The solarian package\n"
            "\n"
            "Type exit() or Ctrl-D to quit.\n"
        )

        # Try IPython first for a richer experience
        try:
            from IPython import start_ipython
            sys.argv = ['solarian-shell']
            start_ipython(argv=[], user_ns=namespace, display_banner=False)
            return
        except ImportError:
            pass

        # Fall back to the standard interactive console with tab completion
        try:
            import readline
            import rlcompleter
            readline.set_completer(rlcompleter.Completer(namespace).complete)
            readline.parse_and_bind("tab: complete")
        except ImportError:
            pass

        code.interact(banner=banner, local=namespace)
