"""
Management command to open an interactive Python shell with project settings loaded.
"""
import code
import sys
import os

from omegaconf import DictConfig
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register('shell', None)
class ShellCommand(BaseCommand):
    category = "Utilities"
    help = 'Open an interactive Python shell with project settings loaded.'
    description = """
    Starts IPython (or the stdlib console when IPython is not installed) with the
    project already wired up: `settings`, a `robot` built from conf/robot.yaml and
    a `controller` built from conf/controller.yaml, plus the `solarian` package.

    The instances are constructed but not connected, so hardware stays untouched
    until you call robot.connect() yourself.
    """
    usage = ""
    examples = (
        "solarian shell",
        "solarian shell -d pi",
    )

    def get_config_class(self):
        """This command doesn't require configuration."""
        return None

    def _load_config(self, config_name, config_class):
        """Load a Hydra config and parse it into the given config class."""
        from solarian.utils.configs import parse_with_friendly_errors, resolve_with_deployment

        conf_path = os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(version_base=None, config_dir=conf_path, job_name="solarian_shell"):
            hydra_cfg = compose(config_name=config_name)
            cfg_dict = resolve_with_deployment(hydra_cfg)

        return parse_with_friendly_errors(config_class, cfg_dict)

    def _create_robot(self, settings):
        """Load the robot config and create the robot instance for the project."""
        from solarian.utils import get_class
        from solarian.core.configs import RobotConfig

        robot_cfg = self._load_config("robot", RobotConfig)
        RobotClass = get_class(settings.ROBOT_CLASS)
        return RobotClass(robot_cfg)

    def _create_controller(self, settings):
        """Load the controller config and create the controller instance for the project."""
        from solarian.utils import get_class
        from solarian.core.configs import ControllerConfig

        ControllerClass = get_class(settings.CONTROLLER_CLASS)
        controller_cfg = self._load_config("controller", ControllerConfig)
        return ControllerClass(controller_cfg)

    def _create_server(self, settings):
        """Load the server config and create the server instance for the project."""
        from solarian.utils import get_class
        from solarian.core.configs import ServerConfig

        ServerClass = get_class(settings.SERVER_CLASS)
        server_cfg = self._load_config("server", ServerConfig)
        return ServerClass(server_cfg)

    def handle(self, cfg: DictConfig = None) -> None:
        from solarian.conf import settings

        # Build a namespace with useful objects pre-loaded
        namespace = {
            'settings': settings,
        }

        # Try to import commonly useful modules into the namespace
        try:
            import solarian
            namespace['solarian'] = solarian
        except ImportError:
            pass

        # Create the project's robot instance
        robot = self._create_robot(settings)
        namespace['robot'] = robot

        # Create the project's controller instance
        controller = self._create_controller(settings)
        # controller = None
        namespace['controller'] = controller

        # Create the project's server instance
        # server = self._create_server(settings)
        # namespace['server'] = server

        robot_line = "  robot      - Robot instance\n" if robot is not None else ""
        controller_line = "  controller - Controller instance\n" if controller is not None else ""
        banner = (
            "\nSolarian Interactive Shell\n"
            "─────────────────────────\n"
            "Available objects:\n"
            "  settings   - Project settings module\n"
            f"{robot_line}"
            f"{controller_line}"
            "  solarian   - The solarian package\n"
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
