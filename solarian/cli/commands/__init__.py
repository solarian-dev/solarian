"""
Register all built-in commands lazily — only metadata is loaded here.
The actual command modules are imported on-demand when a command is invoked.
"""
from solarian.cli.base import CommandRegistry

_pkg = "solarian.cli.commands"

# Project commands
CommandRegistry.register_lazy(
    "create-project",
    module_path=f"{_pkg}.create_project",
    help="Create a new Solarian project with the specified name",
)
CommandRegistry.register_lazy(
    "add-module",
    module_path=f"{_pkg}.add_module",
    help="Install a module plugin and add its configuration to the project",
)
CommandRegistry.register_lazy(
    "list-plugins",
    module_path=f"{_pkg}.list_plugins",
    help="List all installed Solarian plugins (modules, actuators, and buses).",
    config_name=None,
)

# Robot commands
CommandRegistry.register_lazy(
    "run-robot",
    module_path=f"{_pkg}.run_robot",
    help="Run robot with the specified configuration",
    config_name="robot",
)
CommandRegistry.register_lazy(
    "calibrate-module",
    module_path=f"{_pkg}.calibrate_module",
    help="Calibrate module device.",
    config_name="robot",
)
CommandRegistry.register_lazy(
    "setup-module",
    module_path=f"{_pkg}.setup_module",
    help="Setup module device.",
    config_name="robot",
)
CommandRegistry.register_lazy(
    "write-robot-calibration",
    module_path=f"{_pkg}.write_robot_calibration",
    help="Write robot calibration from configuration.",
    config_name="robot",
)

# Controller commands
CommandRegistry.register_lazy(
    "run-controller",
    module_path=f"{_pkg}.run_controller",
    help="Teleop client that connects to a relay server",
    config_name="controller",
)
CommandRegistry.register_lazy(
    "calibrate-controller",
    module_path=f"{_pkg}.calibrate_controller",
    help="Calibrates controller",
    config_name="controller",
)
CommandRegistry.register_lazy(
    "write-controller-calibration",
    module_path=f"{_pkg}.write_controller_calibration",
    help="Writes controller calibration to the controller",
    config_name="controller",
)
CommandRegistry.register_lazy(
    "run-policy",
    module_path=f"{_pkg}.run_policy",
    help="Policy client that connects to a relay server",
    config_name="controller",
)
CommandRegistry.register_lazy(
    "run-server",
    module_path=f"{_pkg}.run_server",
    help="Run the relay server",
    config_name="server",
)

# Recording & Replay commands
CommandRegistry.register_lazy(
    "record",
    module_path=f"{_pkg}.record",
    help="Record sessions",
    config_name="record",
)
CommandRegistry.register_lazy(
    "replay",
    module_path=f"{_pkg}.replay",
    help="Replay sessions",
    config_name="replay",
)

# Training & Data commands
CommandRegistry.register_lazy(
    "train",
    module_path=f"{_pkg}.train",
    help="Train a model.",
    config_name="train",
)
CommandRegistry.register_lazy(
    "dataset2lerobot",
    module_path=f"{_pkg}.dataset2lerobot",
    help="Convert dataset to lerobot dataset.",
    config_name="dataset2lerobot",
)

# Utility commands
CommandRegistry.register_lazy(
    "detect-device",
    module_path=f"{_pkg}.detect_device",
    help="Detects devices connected to the computer (port and name).",
)
CommandRegistry.register_lazy(
    "shell",
    module_path=f"{_pkg}.shell",
    help="Open an interactive Python shell with project settings loaded.",
)
