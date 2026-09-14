from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError, ConfigDict
from enum import Enum
from typing import Tuple, Optional, List, Union, Literal, Any, Dict, Set
import os
from pathlib import Path
import numpy as np

from solarian.core.execution import ExecutionMode


def _convert_numpy_to_float(v):
    """Convert numpy types to Python float."""
    if isinstance(v, np.ndarray):
        return float(v.item()) if v.size == 1 else float(v)
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v

def _convert_numpy_to_int(v):
    """Convert numpy types to Python int (for enum coercion)."""
    if isinstance(v, np.ndarray):
        return int(v.item()) if v.size == 1 else int(v)
    if isinstance(v, (np.floating, np.integer)):
        return int(v)
    return v

class ActionValue(str, Enum):
    STOP = "stop"
    RESET = "reset"
    REWIND = "rewind"


class SafeState(str, Enum):
    """What a module does when the link to whatever commands it goes quiet.

    Which one is safe is a property of the machine, not of the framework: an arm
    holding a payload must not drop it, and a light arm is safest limp.
    """
    #: Stop where you are: abandon any future waypoint, keep holding position.
    HOLD = "hold"
    #: Release torque, so the arm can be moved by hand.
    RELAX = "relax"

    def __str__(self):
        return self.value


class InterpolatorType(str, Enum):
    """Types of trajectory interpolators for modules."""
    POSE = "pose"       # 6D Cartesian with Slerp for rotation
    JOINT = "joint"     # N-dim joint space, linear interpolation
    SCALAR = "scalar"   # 1D value, linear interpolation
    
    def __str__(self):
        return self.value


class FisheyeDistortionConfig(BaseModel):
    """Calibrated OpenCV fisheye distortion for a simulated camera.

    MuJoCo renders a rectilinear (pinhole) image; these are the real camera's
    OpenCV fisheye intrinsics (``K``) and distortion coefficients (``D``) used to
    warp the pinhole render into an image that matches the real lens (e.g. a
    GoPro with a fisheye mod). ``K``/``D`` are the values produced by
    ``cv2.fisheye.calibrate`` at the ``calib_width`` x ``calib_height``
    resolution; they are rescaled to the render resolution automatically.
    """
    #: 3x3 real-camera intrinsics matrix (row-major).
    K: list[list[float]]
    #: OpenCV fisheye distortion coefficients (k1, k2, k3, k4).
    D: list[float]
    #: Resolution (px) the ``K``/``D`` were calibrated at.
    calib_width: int
    calib_height: int
    #: Supersample factor for the pinhole source render (>1 reduces the heavy
    #: edge-stretch aliasing before the distortion remap downsamples it).
    render_scale: float = 1.0

class SimSensorConfig(BaseModel):
    """Simulation mapping for a sensor (e.g. the MuJoCo camera to render)."""
    camera: str | None = None
    #: Optional calibrated fisheye distortion applied to the rendered frames.
    distortion: FisheyeDistortionConfig | None = None

class ComponentConfig(BaseModel):
    id: int = 0
    port: str
    configs: dict[str, Any] = {}

class SensorConfig(ComponentConfig):
    fps: int
    width: int
    height: int
    rotation: int | None
    channels: int
    color_mode: str
    video_format: str
    resize_width: int | None
    resize_height: int | None
    #: Path to a GoPro optical-remap JSON from ``solarian calibrate-umi-camera``.
    #: Applied after color/rotation and before ``resize_*``. Relative paths are
    #: resolved against the project / conf directory.
    optical_remap: str | None = None
    #: Optional simulation mapping; used only when the robot runs with is_sim.
    sim: SimSensorConfig | None = None

class CameraConfig(SensorConfig):
    fps: int
    width: int
    height: int
    rotation: int | None
    channels: int
    color_mode: str
    video_format: str
    resize_width: int | None
    resize_height: int | None
    optical_remap: str | None = None


class PiCameraSensorConfig(CameraConfig):
    """libcamera / Picamera2 controls for a Raspberry Pi CSI camera.

    Unset fields leave the driver default (auto-exposure and default ISP).
    """

    #: EV stops for AEC/AGC (``rpicam-hello --ev``).
    exposure_value: float | None = None
    #: Pixel offset after capture, -1.0 (black) to 1.0 (white).
    brightness: float | None = Field(default=None, ge=-1.0, le=1.0)
    #: 1.0 is the default amount of contrast.
    contrast: float | None = Field(default=None, gt=0)
    #: Exposure time in microseconds. Gain can still vary unless ``gain`` is also set.
    shutter: int | None = Field(default=None, gt=0)
    #: Analogue gain. Exposure can still vary unless ``shutter`` is also set.
    gain: float | None = Field(default=None, gt=0)

class ActuatorConfig(ComponentConfig):
    """Per-actuator yaml: deployment + measured values.

    The actuator class, default servo model, and continuous-vs-limited live on
    the Module declaration. Yaml ``model:`` / ``mode:`` override those defaults
    when set. ``type:`` in yaml is ignored (bus comes from ``bus_class``). A
    leftover nested ``sim:`` block is flattened on load.
    """

    model_config = ConfigDict(extra="ignore")

    port: str | None = None
    model: str | None = None
    mode: str | None = None
    sim_joint: str | None = None
    sim_range_min: float = -3.141592653589793
    sim_range_max: float = 3.141592653589793
    sim_drive_mode: int = 0
    real_range_min: int | None = None
    real_range_max: int | None = None
    center_offset: int | None = None
    drive_mode: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _flatten_legacy_sim(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data.pop("type", None)
        if "center_offset" not in data and "homing_offset" in data:
            data["center_offset"] = data["homing_offset"]
        data.pop("homing_offset", None)
        sim = data.pop("sim", None)
        if isinstance(sim, dict):
            if data.get("sim_joint") is None and sim.get("joint") is not None:
                data["sim_joint"] = sim["joint"]
            if "sim_range_min" not in data and "range_min" in sim:
                data["sim_range_min"] = sim["range_min"]
            if "sim_range_max" not in data and "range_max" in sim:
                data["sim_range_max"] = sim["range_max"]
            if "sim_drive_mode" not in data and "drive_mode" in sim:
                data["sim_drive_mode"] = sim["drive_mode"]
        return data


class ServoConfig(ActuatorConfig, SensorConfig):
    pass

# A module is composed by joints and bodies
class ModuleConfig(BaseModel):
    # Process/control loop settings
    frequency: float # control loop frequency in Hz
    get_max_k: int = 128  # max number of states to keep in ring buffer
    commands_per_cycle: int = 1  # max commands to process per control loop iteration
    command_queue_size: int = 1024  # size of the command input queue
    launch_timeout: float = 3.0  # timeout for process launch in seconds
    #: Overrides the module class's ``execution_mode``. Lets the same module
    #: run in a process in production and inline in a test or on a small host.
    execution: ExecutionMode | None = None

    actuators: dict[str, ActuatorConfig] = {}
    sensors: dict[str, SensorConfig] = {}

    receive_latency: float = 0.0  # latency in seconds for receiving data from sensors

    #: What this module does when the robot's action-link watchdog fires. See
    #: :class:`SafeState`.
    link_lost: SafeState = SafeState.HOLD

    #: Operator-chosen home (normalized joint values), written by ``solarian calibrate``.
    #: Skill poses like ``move.yaml`` ``home`` are a separate vocabulary.
    home: dict[str, float] = {}

    #: Deprecated sidecar JSON. Motor calibration now lives on each actuator
    #: (``real_range_*`` / ``center_offset`` / ``drive_mode``). Still read as a
    #: fallback when those fields are empty.
    calibration_file: str | None = None
    
    # # Interpolator settings
    interpolator_type: InterpolatorType = InterpolatorType.JOINT
    max_speed: float | list[float] = 1.0  # max speed for interpolation (single or per-joint)
    #: Adaptive drive horizon bounds (seconds). Used by ``SERVO_*``
    #: (``servoL`` / ``servoJ`` / ``drive_to_waypoint``).
    drive_horizon_min_s: float = 1.0 / 30.0
    drive_horizon_max_s: float = 0.12
    drive_horizon_ema_alpha: float = 0.2
    drive_horizon_multiplier: float = 1.5

class RemoteInputConfig(ModuleConfig):
    """Config for a teleop device sampled off-process.

    Reuses the device's own yaml, so substituting a remote sampler for a local
    module needs no second config file — only the hardware fields stop meaning
    anything, because this process has no bus to reach them with.
    """
    #: Publish nothing once the newest sample is older than this. A dropped input
    #: link must stop the follower where it is, not latch the last pose.
    stale_after_s: float = 0.25

    @field_validator("calibration_file", mode="before")
    @classmethod
    def _calibration_belongs_to_the_sampler(cls, _value):
        """Drop any calibration file the local module's yaml names.

        The sampler owns calibration for a device it is the only one holding, and
        sends normalized values; a controller with no bus cannot read a
        calibration back to compare, so keeping the path would only make
        ``Controller.connect`` try to calibrate hardware that is not there.
        """
        return None


class ConnectionConfig(BaseModel):
    ip: str
    action_sender_port: int
    action_receiver_port: int
    state_sender_port: int
    state_receiver_port: int
    #: ZMQ REP port for application-level clock sync (Cristian / NTP-style).
    #: Controller and robot estimate an offset so ``SCHEDULE_*`` stamps share
    #: the relay server's wall clock without requiring OS NTP/root.
    clock_sync_port: int = 5559
    #: CurveZMQ (see :mod:`solarian.core.curve`). Unset on all three leaves the
    #: link plaintext, which is fine on a LAN and not fine for a relay reachable
    #: from the internet. Generate with ``solarian keygen``.
    #: The relay's public key. Set on the robot and the controller, so each knows
    #: which relay it is willing to talk to.
    curve_server_key: str | None = None
    #: This peer's own keypair. On the relay, the pair whose public half the
    #: clients carry as ``curve_server_key``.
    curve_public_key: str | None = None
    curve_secret_key: str | None = None
    #: Seconds of controller silence after which the robot stops its skills and
    #: puts every module in its ``link_lost`` state. ``None`` disables the
    #: watchdog, which is defensible on a LAN and is not once the command path
    #: crosses the internet: without it a dropped link leaves the follower
    #: driving toward the last waypoint it was given with nobody left to change
    #: its mind. Lives on the link rather than on the robot so that switching
    #: connection profile switches the safety behaviour with it.
    link_timeout_s: float | None = None
    #: How often an idle controller tells the robot it is still there. Has to
    #: stay well under ``link_timeout_s``; a third of it is a reasonable rule, so
    #: one lost heartbeat does not trip the watchdog.
    heartbeat_s: float = 0.5

# A robot is composed by multiple modules, eg: arms, legs, base, etc.
class RobotConfig(BaseModel):
    img_compression_quality: int
    connection: ConnectionConfig
    fps: int
    #: Run in MuJoCo simulation. ``None`` (default) defers to the robot class's
    #: ``is_sim`` attribute; set it here (or via ``run-robot sim=true``) to
    #: override, so the same robot class can run on hardware or in sim.
    sim: bool | None = None

class ControllerConfig(BaseModel):
    fps: int
    tps: int
    system2_model: str
    system2_prompt: str
    system2_base_url: str
    system2_api_key: str
    policy_path: str
    device: str
    #: Front end that drives this controller: ``console`` (default), ``pygame``,
    #: ``opencv``, ``external`` or ``null``. See :mod:`solarian.core.ui`.
    ui: str = "console"
    #: ``host:port`` the ``external`` UI serves its WebSocket on. Loopback by
    #: default: the intended peer is a bridge beside it, not the internet.
    ui_bind: str = "127.0.0.1:8700"
    #: Shared secret an ``external`` UI client must present. Unset accepts any
    #: connection that can reach ``ui_bind``.
    ui_token: str | None = None
    #: Cap on external-UI frames that carry camera images. Dashboard frames are
    #: not throttled; JPEG is the expensive part.
    ui_image_fps: int = 15
    #: Downscale camera frames wider than this before encoding, to bound the
    #: external UI's bandwidth.
    ui_image_max_width: int = 640
    #: Names of declared modules whose device is sampled by the UI instead of by
    #: this process, replaced at construction by a
    #: :class:`~solarian.core.remote_input_module.RemoteInputModule`. The way a
    #: cloud-hosted controller uses a leader arm plugged into the operator's
    #: laptop, without the project's code changing.
    remote_input_modules: list[str] = []
    #: Deprecated, superseded by ``ui``. ``true`` still selects the console UI,
    #: which is what the old headless loop was; use ``ui: null`` for no UI.
    headless: bool = False
    #: Unused by ``MOVE_*`` / ``SERVO_*`` skills. Kept so existing controller
    #: yaml still loads.
    waypoint_lookahead_s: float = 0.25
    connection: ConnectionConfig
    modules: dict[str, ModuleConfig]

class RecordConfig(ControllerConfig):
    repo_id: str
    num_episodes: int
    episode_time_s: float
    reset_time_s: float
    
    reset_period: int
    play_sounds: bool
    display_cameras: bool
    tasks: list[str] | None
    tags: str | None
    private: bool
    num_image_writer_processes: int
    num_image_writer_threads_per_camera: int
    start_episode: int
    video: bool
    
    video_res: list[int]

class ReplayConfig(ControllerConfig):
    repo_id: str
    episodes: list[int]
    reset_time_s: float
    reset_period: int
    play_sounds: bool
    display_cameras: bool
    batch_size: int
    
class ServerConfig(BaseModel):
    connection: ConnectionConfig
    img_compression_quality: int
    #: CURVE public keys allowed to connect, i.e. this deployment's robot and
    #: controller. Empty means no allowlist: encryption but no authorization, so
    #: anyone holding the relay's public key gets in. A relay on a public address
    #: must set this.
    curve_allowed_client_keys: list[str] = []
    # image_processing: list[str]
    # tracking_model: str | None
    # depth_model: str | None
    # ref_x: int
    # ref_y: int
    

class ConfigBaseModel(BaseModel):
    """Base model with enhanced error handling for configs"""
    
    def parse_with_friendly_errors(self, data):
        """Parse config data with user-friendly error messages"""
        try:
            return self.model_validate(data)
        except ValidationError as e:
            # Get the config filename if possible
            config_path = os.environ.get("SOLARIAN_CURRENT_CONFIG", "")
            
            # Format a user-friendly error message
            error_msg = f"Configuration Error in {self.__class__.__name__}:\n"
            error_msg += f"  {str(e)}\n"
            
            if config_path:
                error_msg += f"  Please check your configuration file: {config_path}\n"
                
            # Re-raise with the friendly message
            raise ValueError(error_msg) from e


class LoggerConfig(ConfigBaseModel):
    type: Literal['wandb'] | None = None
    
class TrainLoggerConfig(ConfigBaseModel):
    type: Literal['none', 'wandb']
    project: str
    notes: str | None = None
    log_artifact: bool = False

class WandBLoggerConfig(TrainLoggerConfig):
    entity: str | None = None
    run_id: str | None = None
    mode: str | None = None
    project: str = Field(..., description="The WandB project name to log to")
    
    @field_validator('project')
    @classmethod
    def validate_project(cls, v):
        if not v:
            raise ValueError("WandB project name cannot be empty")
        return v

class SchedulerConfig(ConfigBaseModel):
    type: Literal['cosine'] | None = None
    
class PrecisionConfig(ConfigBaseModel):
    amp_enabled: bool = False # Use mixed precision training and enable grad scaler
    init_scale: float = 2 ** 16
    growth_interval: int = 2000
    growth_factor: float = 2.0

class TrainOptimizerConfig(ConfigBaseModel):
    pass

class AdamOptimizerConfig(TrainOptimizerConfig):
    type: Literal['adam'] = 'adam'
    lr: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0001
    
class AdamWOptimizerConfig(TrainOptimizerConfig):
    type: Literal['adamw'] = 'adamw'
    lr: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0001
    
class ModalConfig(ConfigBaseModel):
    """Cloud-GPU (Modal) orchestration for training.

    Consumed by :class:`solarian.core.training.modal_launcher.ModalLauncher`.
    The framework builds the container image from the trainer's
    :meth:`~solarian.core.ai.system1.policy_trainer.PolicyTrainer.modal_spec`
    and runs the trainer remotely; only orchestration knobs live here.
    """
    enabled: bool = False
    gpu_type: str = "A100-80GB"
    gpus: int = 1
    #: Modal volume mounted at ``/data`` for datasets + outputs.
    volume: str = "solarian-data"
    #: Names of ``modal.Secret`` objects to attach (e.g. ``["wandb"]``).
    secrets: list[str] = []
    timeout: int = 24 * 60 * 60
    mixed_precision: str = "bf16"
    app_name: str = "solarian-train"


class TrainConfig(ConfigBaseModel):
    """Thin, library-agnostic training config.

    The framework only knows these common fields; everything policy- or
    library-specific (model architecture, optimizer, etc.) is carried
    through untouched (``extra='allow'``) and validated by the plugin
    trainer's own :meth:`config_class`.  The ``policy`` field selects which
    installed trainer (``solarian.trainers`` entry point) runs.
    """
    model_config = ConfigDict(extra="allow")

    #: Name of the trainer to run (resolved via the ``solarian.trainers``
    #: entry-point group), e.g. ``"diffusion"`` or ``"lerobot_act"``.
    policy: str

    train_dataset: str
    val_dataset: str | None = None
    output_dir: str | None = None
    val_percentage: float | None = None

    job_name: str
    resume: bool = False
    batch_size: int = 64
    steps: int = 100000
    num_workers: int = 4
    log_freq: int = 100
    val_freq: int = 1000
    save_freq: int = 10000

    logger: Optional[TrainLoggerConfig] = None
    modal: ModalConfig = Field(default_factory=ModalConfig)

    def to_tags(self) -> list[str]:
        """Generic run tags for logging. Plugins may override to add more."""
        return [
            f"policy:{self.policy}",
            f"batch_size:{self.batch_size}",
            f"train_dataset:{self.train_dataset}",
            f"val_dataset:{self.val_dataset if self.val_dataset else self.train_dataset}",
        ]



class SetupModuleConfig(ConfigBaseModel):
    module_name: str