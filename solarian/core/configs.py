from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError, ConfigDict
from enum import Enum
from typing import Tuple, Optional, List, Union, Literal, Any, Dict, Set
import os
from pathlib import Path
import numpy as np


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


class InterpolatorType(str, Enum):
    """Types of trajectory interpolators for modules."""
    POSE = "pose"       # 6D Cartesian with Slerp for rotation
    JOINT = "joint"     # N-dim joint space, linear interpolation
    SCALAR = "scalar"   # 1D value, linear interpolation
    
    def __str__(self):
        return self.value


class SimActuatorConfig(BaseModel):
    """Simulation mapping for an actuator.

    Added to the *same* module yaml as the real hardware fields; it is ignored
    when the module runs on real hardware and only consumed when the robot runs
    with ``is_sim = True``. It says which MuJoCo joint/actuator the servo drives
    and the radian range the normalized command space maps onto. Leave
    ``joint``/``actuator`` unset for a virtual joint (one with no MuJoCo
    counterpart) that just echoes the commanded value.
    """
    joint: str | None = None
    actuator: str | None = None
    range_min: float = -3.141592653589793
    range_max: float = 3.141592653589793
    drive_mode: int = 0

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

class ActuatorConfig(ComponentConfig):
    model: str
    # TODO validate mode string value
    mode: str
    #: Optional simulation mapping; used only when the robot runs with is_sim.
    sim: SimActuatorConfig | None = None
    

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

    actuators: dict[str, ActuatorConfig] = {}
    sensors: dict[str, SensorConfig] = {}

    receive_latency: float = 0.0  # latency in seconds for receiving data from sensors
    
    calibration_file: str | None = None
    # gripper_open_degree: Optional[float] = 30
    
    # # Interpolator settings
    interpolator_type: InterpolatorType = InterpolatorType.JOINT
    max_speed: float | list[float] = 1.0  # max speed for interpolation (single or per-joint)
    #: If True, SCHEDULE_WAYPOINT ignores client timestamps and uses adaptive
    #: ``drive_to_waypoint`` from the robot clock (best for WiFi teleop). If
    #: False, keep UMI-style timestamped ``schedule_waypoint`` (policy chunks).
    drive_waypoints: bool = False
    #: Adaptive drive horizon bounds (seconds). Only used when drive_waypoints.
    drive_horizon_min_s: float = 1.0 / 30.0
    drive_horizon_max_s: float = 0.12
    drive_horizon_ema_alpha: float = 0.2
    drive_horizon_multiplier: float = 1.5

class ConnectionConfig(BaseModel):
    connection_id: str = "connection_1"
    ip: str
    action_sender_port: int
    action_receiver_port: int
    state_sender_port: int
    state_receiver_port: int
    #: ZMQ REP port for application-level clock sync (Cristian / NTP-style).
    #: Controller and robot estimate an offset so SCHEDULE_WAYPOINT stamps share
    #: the relay server's wall clock without requiring OS NTP/root.
    clock_sync_port: int = 5559

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
    headless: bool = False
    #: Seconds ahead of ``time.time()`` for keyboard / teleop SCHEDULE_WAYPOINT
    #: stamps. Must exceed Mac→robot action latency or waypoints are dropped.
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
    # image_processing: list[str]
    # tracking_model: str | None
    # depth_model: str | None
    # ref_x: int
    # ref_y: int
    

# A project is composed by a robot, a server, a controller and a config
# In its top level config it also points to the classes that implement the robot, server, controller and config
class ProjectConfig(BaseModel):
    name: str
    controller: ControllerConfig
    server: ServerConfig
    robot: RobotConfig

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