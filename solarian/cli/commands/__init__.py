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
  - Server-side cmds     → ("robot", "observation", "action", "server")
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
    category="Project",
)
CommandRegistry.register_lazy(
    "install",
    module_path=f"{_pkg}.install",
    help="Install a plugin into this environment and project",
    setup_modules=(),
    category="Project",
)
CommandRegistry.register_lazy(
    "create-plugin",
    module_path=f"{_pkg}.create_plugin",
    help="Scaffold a new Solarian plugin (or add a component type to an existing one)",
    category="Project",
)
CommandRegistry.register_lazy(
    "add-op",
    module_path=f"{_pkg}.add_op",
    help="Write a dataset op stub into this project's <package>/datasets/",
    setup_modules=(),
    category="Project",
)
CommandRegistry.register_lazy(
    "onshape-export",
    module_path=f"{_pkg}.onshape_export",
    help="Export an Onshape assembly into an attachable MuJoCo sim model",
    setup_modules=(),
    category="Utilities",
)

# ── Robot commands ───────────────────────────────────────────────────
CommandRegistry.register_lazy(
    "run-robot",
    module_path=f"{_pkg}.run_robot",
    help="Run the robot: connect its modules and join the relay a controller drives it through",
    config_name="robot",
    setup_modules=("robot", "observation", "action"),
    category="Robot",
)
CommandRegistry.register_lazy(
    "calibrate",
    module_path=f"{_pkg}.calibrate",
    help="Calibrate a module: hard-stop sweep, then IK alignment for arms.",
    config_name="robot",
    setup_modules=("robot", "observation", "action"),
    category="Robot",
)
CommandRegistry.register_lazy(
    "calibrate-arm",
    module_path=f"{_pkg}.calibrate_arm",
    help="Measure the arm's max_speed and its two latencies, into the module yaml.",
    config_name="robot",
    setup_modules=("robot",),
    category="Robot",
)
CommandRegistry.register_lazy(
    "calibrate-gripper",
    module_path=f"{_pkg}.calibrate_gripper",
    help="Measure the gripper's max_speed and its two latencies, into the module yaml.",
    config_name="robot",
    setup_modules=("robot",),
    category="Robot",
)
CommandRegistry.register_lazy(
    "calibrate-gripper-ranges",
    module_path=f"{_pkg}.calibrate_gripper_ranges",
    help="Sweep the gripper and record the aperture curve against a ruler.",
    config_name="robot",
    setup_modules=("robot",),
    category="Robot",
)
CommandRegistry.register_lazy(
    "calibrate-camera-latency",
    module_path=f"{_pkg}.calibrate_camera_latency",
    help="Measure the camera's receive_latency by filming a timestamp QR code (UMI method).",
    config_name="robot",
    setup_modules=("robot",),
    category="Robot",
)
CommandRegistry.register_lazy(
    "calibrate-hand-eye",
    module_path=f"{_pkg}.calibrate_hand_eye",
    help="Measure camera-to-fingertip (robot-world hand-eye) against what a UMI policy assumes.",
    config_name="robot",
    setup_modules=("robot",),
    category="Robot",
)
CommandRegistry.register_lazy(
    "setup-module",
    module_path=f"{_pkg}.setup_module",
    help="Run a module's one-off device setup (actuator IDs, baud rates, modes)",
    config_name="robot",
    setup_modules=("robot",),
    category="Robot",
)
CommandRegistry.register_lazy(
    "calibrate-servo",
    module_path=f"{_pkg}.calibrate_servo",
    help="Prepare a servo for use: set its ID, reset registers, and centre its range.",
    config_name=None,
    setup_modules=(),
    category="Robot",
)
CommandRegistry.register_lazy(
    "write-robot-calibration",
    module_path=f"{_pkg}.write_robot_calibration",
    help="Write the calibration in the robot's yamls onto its hardware",
    config_name="robot",
    setup_modules=("robot",),
    category="Robot",
)
CommandRegistry.register_lazy(
    "verify-calibration",
    module_path=f"{_pkg}.verify_calibration",
    help="Verify calibration spans against the IK model, then live-mirror the arm.",
    config_name="robot",
    setup_modules=("robot", "observation", "action"),
    category="Robot",
)
CommandRegistry.register_lazy(
    "derive-model-limits",
    module_path=f"{_pkg}.derive_model_limits",
    help="Derive MJCF joint ranges (radians) from the real arm's mechanical limits.",
    config_name="robot",
    setup_modules=("robot",),
    category="Robot",
)

# ── Controller commands ──────────────────────────────────────────────
CommandRegistry.register_lazy(
    "run-controller",
    module_path=f"{_pkg}.run_controller",
    help="Teleop client: drive the robot through the relay",
    config_name="controller",
    setup_modules=("controller", "observation", "action"),
    category="Controller",
)
CommandRegistry.register_lazy(
    "calibrate-controller",
    module_path=f"{_pkg}.calibrate_controller",
    help="Calibrate the operator-side hardware (leader arm, pedals, ...)",
    config_name="controller",
    setup_modules=("controller",),
    category="Controller",
)
CommandRegistry.register_lazy(
    "calibrate-umi-camera",
    module_path=f"{_pkg}.calibrate_umi_camera",
    help="Calibrate a connected camera onto GoPro Hero 10 + Max Lens Mod optics and overlay vs UMI gold.",
    config_name="controller",
    setup_modules=(),
    category="Controller",
)
CommandRegistry.register_lazy(
    "write-controller-calibration",
    module_path=f"{_pkg}.write_controller_calibration",
    help="Write the calibration in the controller's yamls onto its hardware",
    config_name="controller",
    setup_modules=("controller",),
    category="Controller",
)
CommandRegistry.register_lazy(
    "run-service",
    module_path=f"{_pkg}.run_service",
    help="Run a service: a project conf's server (tracker, policy) or an installed service",
    config_name=None,
    setup_modules=(),
    category="Controller",
)
CommandRegistry.register_lazy(
    "run-server",
    module_path=f"{_pkg}.run_server",
    help="Run the relay server the robot and the controller connect through",
    config_name="server",
    # The schema modules too: server-computed fields and server-hosted skills
    # are declared there.
    setup_modules=("robot", "observation", "action", "server"),
    category="Controller",
)

# ── Recording commands ───────────────────────────────────────────────
CommandRegistry.register_lazy(
    "record",
    module_path=f"{_pkg}.record",
    help="Teleoperate and record a training dataset, episode by episode",
    config_name="commands/record",
    setup_modules=("robot", "observation", "action", "controller"),
    category="Recording",
)
# ── Training & Data commands ─────────────────────────────────────────
CommandRegistry.register_lazy(
    "dataset",
    module_path=f"{_pkg}.dataset",
    help="Curate and enrich recorded sessions: delete, split, track, mirror, merge",
    setup_modules=(),
    category="Training & Data",
)
CommandRegistry.register_lazy(
    "export-dataset",
    module_path=f"{_pkg}.export_dataset",
    help="Convert a recorded session into a LeRobot dataset",
    setup_modules=(),
    category="Training & Data",
)
CommandRegistry.register_lazy(
    "train",
    module_path=f"{_pkg}.train",
    help="Train a policy via an installed policy trainer plugin.",
    config_name=None,
    setup_modules=(),
    category="Training & Data",
)
# ── Utility commands ─────────────────────────────────────────────────
CommandRegistry.register_lazy(
    "sync-clock",
    module_path=f"{_pkg}.sync_clock",
    help="Sync Solarian wall-clock offset to the relay server (does not change OS time).",
    config_name="server",
    setup_modules=(),
    category="Utilities",
)
CommandRegistry.register_lazy(
    "keygen",
    module_path=f"{_pkg}.keygen",
    help="Generate CurveZMQ keypairs for the relay, the robot and the controller.",
    config_name=None,
    setup_modules=(),
    category="Utilities",
)
CommandRegistry.register_lazy(
    "detect-device",
    module_path=f"{_pkg}.detect_device",
    help="Detect a serial device by plugging it in (prints port and name)",
    category="Utilities",
)
CommandRegistry.register_lazy(
    "list-cameras",
    module_path=f"{_pkg}.list_cameras",
    help="Detect connected cameras and optionally open a live preview.",
    category="Utilities",
)
CommandRegistry.register_lazy(
    "shell",
    module_path=f"{_pkg}.shell",
    help="Open an interactive Python shell with project settings loaded.",
    category="Utilities",
)
