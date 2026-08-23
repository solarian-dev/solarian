"""
Register all built-in commands lazily — only metadata is loaded here.
The actual command modules are imported on-demand when a command is invoked.

Each command declares ``setup_modules`` — the project-level modules that
``setup()`` should import.  This avoids pulling in hardware-specific
dependencies on machines that don't need them (e.g. a Raspberry Pi running
the robot should never import the controller module, which may require
packages only installed on the controller machine).

Convention:
  - Robot-side commands  → ("robot", "observation", "action")
  - Controller-side cmds → ("controller",)
  - Server-side cmds     → ("server",)
  - Utility / data cmds  → ()   (no project modules needed)
  - None                 → all modules (backward-compatible default)
"""
from solarian.cli.base import CommandRegistry

_pkg = "solarian.cli.commands"

# ── Project commands ─────────────────────────────────────────────────
CommandRegistry.register_lazy(
    "create-project",
    module_path=f"{_pkg}.create_project",
    help="Create a new Solarian project with the specified name",
)
CommandRegistry.register_lazy(
    "add-module",
    module_path=f"{_pkg}.add_module",
    help="Install a module plugin and copy its default configuration to the project",
)
CommandRegistry.register_lazy(
    "create-plugin",
    module_path=f"{_pkg}.create_plugin",
    help="Scaffold a new Solarian plugin (or add a component type to an existing one)",
)
CommandRegistry.register_lazy(
    "onshape-export",
    module_path=f"{_pkg}.onshape_export",
    help="Export an Onshape assembly into an attachable MuJoCo sim model",
    setup_modules=(),
)

# ── Robot commands ───────────────────────────────────────────────────
CommandRegistry.register_lazy(
    "run-robot",
    module_path=f"{_pkg}.run_robot",
    help="Run robot with the specified configuration",
    config_name="robot",
    setup_modules=("robot", "observation", "action"),
)
CommandRegistry.register_lazy(
    "calibrate-module",
    module_path=f"{_pkg}.calibrate_module",
    help="Calibrate module device.",
    config_name="robot",
    setup_modules=("robot",),
)
CommandRegistry.register_lazy(
    "calibrate-gripper",
    module_path=f"{_pkg}.calibrate_gripper",
    help="Calibrate gripper latency (cross-correlation) or ranges (ruler check).",
    config_name="robot",
    setup_modules=("robot",),
)
CommandRegistry.register_lazy(
    "setup-module",
    module_path=f"{_pkg}.setup_module",
    help="Setup module device.",
    config_name="robot",
    setup_modules=("robot",),
)
CommandRegistry.register_lazy(
    "write-robot-calibration",
    module_path=f"{_pkg}.write_robot_calibration",
    help="Write robot calibration from configuration.",
    config_name="robot",
    setup_modules=("robot",),
)
CommandRegistry.register_lazy(
    "calibrate-ik",
    module_path=f"{_pkg}.align_ik",
    help="Guided sim<->real alignment for an IK arm (live mirror + sign + Cartesian checks).",
    config_name="robot",
    # Instantiates the full Robot (for the scene + live mirror), which validates
    # the Observation schema, so the observation/action modules must be imported.
    setup_modules=("robot", "observation", "action"),
)
CommandRegistry.register_lazy(
    "verify-ik-calibration",
    module_path=f"{_pkg}.verify_ik_calibration",
    help="Verify a module's calibration matches its IK model joint limits.",
    config_name="robot",
    setup_modules=("robot",),
)
CommandRegistry.register_lazy(
    "derive-model-limits",
    module_path=f"{_pkg}.derive_model_limits",
    help="Derive MJCF joint ranges (radians) from the real arm's mechanical limits.",
    config_name="robot",
    setup_modules=("robot",),
)

# ── Controller commands ──────────────────────────────────────────────
CommandRegistry.register_lazy(
    "run-controller",
    module_path=f"{_pkg}.run_controller",
    help="Teleop client that connects to a relay server",
    config_name="controller",
    setup_modules=("controller", "observation", "action"),
)
CommandRegistry.register_lazy(
    "calibrate-controller",
    module_path=f"{_pkg}.calibrate_controller",
    help="Calibrates controller",
    config_name="controller",
    setup_modules=("controller",),
)
CommandRegistry.register_lazy(
    "calibrate-umi-camera",
    module_path=f"{_pkg}.calibrate_umi_camera",
    help="Live gold vs wrist-camera overlay for UMI policy-image alignment.",
    config_name="controller",
    setup_modules=("robot", "observation"),
)
CommandRegistry.register_lazy(
    "write-controller-calibration",
    module_path=f"{_pkg}.write_controller_calibration",
    help="Writes controller calibration to the controller",
    config_name="controller",
    setup_modules=("controller",),
)
CommandRegistry.register_lazy(
    "run-policy",
    module_path=f"{_pkg}.run_policy",
    help="Run the policy server associated with a skill",
    config_name=None,
    setup_modules=(),
)
CommandRegistry.register_lazy(
    "run-server",
    module_path=f"{_pkg}.run_server",
    help="Run the relay server",
    config_name="server",
    setup_modules=("server",),
)

# ── Recording & Replay commands ──────────────────────────────────────
CommandRegistry.register_lazy(
    "record",
    module_path=f"{_pkg}.record",
    help="Record sessions",
    config_name="record",
    setup_modules=("robot", "observation", "action", "controller"),
)
CommandRegistry.register_lazy(
    "replay",
    module_path=f"{_pkg}.replay",
    help="Replay sessions",
    config_name="replay",
    setup_modules=("robot", "observation", "action", "controller"),
)

# ── Training & Data commands ─────────────────────────────────────────
CommandRegistry.register_lazy(
    "train",
    module_path=f"{_pkg}.train",
    help="Train a model.",
    config_name="train",
    setup_modules=(),
)
CommandRegistry.register_lazy(
    "dataset2lerobot",
    module_path=f"{_pkg}.dataset2lerobot",
    help="Convert dataset to lerobot dataset.",
    config_name="dataset2lerobot",
    setup_modules=(),
)

# ── Utility commands ─────────────────────────────────────────────────
CommandRegistry.register_lazy(
    "sync-clock",
    module_path=f"{_pkg}.sync_clock",
    help="Sync Solarian wall-clock offset to the relay server (does not change OS time).",
    config_name="server",
    setup_modules=(),
)
CommandRegistry.register_lazy(
    "detect-device",
    module_path=f"{_pkg}.detect_device",
    help="Detects devices connected to the computer (port and name).",
)
CommandRegistry.register_lazy(
    "list-cameras",
    module_path=f"{_pkg}.list_cameras",
    help="Detect connected cameras and optionally open a live preview.",
)
CommandRegistry.register_lazy(
    "shell",
    module_path=f"{_pkg}.shell",
    help="Open an interactive Python shell with project settings loaded.",
)
