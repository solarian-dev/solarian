from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError, ConfigDict
from enum import Enum
from typing import Tuple, Optional, List, Union, Literal, Any, Dict, Set
import os
from pathlib import Path
import numpy as np







class DataComputer(str, Enum):
    SERVER = "server"
    ROBOT = "robot"
    CONTROLLER = "controller"

    def __str__(self):
        return self.value

class DataSpec:
    def __init__(self, shape: Tuple[int], computer: DataComputer, lerobot_group: str, lerobot_names: List[str]):
        self.shape = shape
        self.computer = computer
        self.lerobot_group = lerobot_group
        self.lerobot_names = lerobot_names
    

# class ValueName(str, Enum):
#     POS = "pos"
#     SPEED = "speed"
#     IMAGE = "image"
#     TORQUE_ENABLE = "torque_enable"
#     BOUNDING_BOX = "bounding_box"
#     DEPTH_IMAGE = "depth_image"
#     TIMESTAMP = "timestamp"

#     def __str__(self):
#         return self.value

# VALUE_NAME_TO_STRING = {
#     ValueName.POS: "pos",
#     ValueName.SPEED: "speed",
#     ValueName.IMAGE: "image",
#     ValueName.TORQUE_ENABLE: "torque_enable",
#     ValueName.BOUNDING_BOX: "bounding_box",
#     ValueName.DEPTH_IMAGE: "depth_image",
#     ValueName.TIMESTAMP: "timestamp",
# }


class ValueName(int, Enum):
    POS = 0
    SPEED = 1
    IMAGE = 2
    TORQUE_ENABLE = 3
    BOUNDING_BOX = 4
    DEPTH_IMAGE = 5
    TIMESTAMP = 6

    def __str__(self):
        return str(self.value)


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

class PosValue(BaseModel):
    value_type: Literal[ValueName.POS] = ValueName.POS
    value: float = 0

    @field_validator('value_type', mode='before')
    @classmethod
    def convert_value_type(cls, v):
        return _convert_numpy_to_int(v)

    @field_validator('value', mode='before')
    @classmethod
    def convert_value(cls, v):
        return _convert_numpy_to_float(v)

class SpeedValue(BaseModel):
    value_type: Literal[ValueName.SPEED] = ValueName.SPEED
    value: float = 0

    @field_validator('value_type', mode='before')
    @classmethod
    def convert_value_type(cls, v):
        return _convert_numpy_to_int(v)

    @field_validator('value', mode='before')
    @classmethod
    def convert_value(cls, v):
        return _convert_numpy_to_float(v)

class ImageValue(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value_type: Literal[ValueName.IMAGE] = ValueName.IMAGE
    value: np.ndarray = Field(default_factory=lambda: np.zeros((0, 0, 0), dtype=np.uint8))
    shape: Tuple[int, int, int] = Field(default_factory=lambda: (0, 0, 0))

    @field_validator('value_type', mode='before')
    @classmethod
    def convert_value_type(cls, v):
        return _convert_numpy_to_int(v)

    @model_validator(mode='after')
    def value_from_shape(self):
        """When shape is declared and non-empty, set value to zeros of that shape."""
        if self.shape != (0, 0, 0) and (
            self.value.size == 0 or self.value.shape != self.shape
        ):
            self.value = np.zeros(self.shape, dtype=np.uint8)
        return self

    @classmethod
    def create_empty_value(cls, shape: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """Create an empty ndarray for use as default or with a given shape."""
        return np.zeros(shape, dtype=np.uint8)


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



class ComponentConfig(BaseModel):
    port: str | int
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
    id: int
    model: str
    # TODO validate mode string value
    mode: str
    

class ServoConfig(ActuatorConfig, SensorConfig):
    pass

# # A body has mass and inertia and it might have sensors attached to it
# class BodyConfig(BaseModel):
#     mass: float
#     inertia: float
#     sensors: dict[str, SensorConfig] = {}
#     actuators: dict[str, ActuatorConfig] = {}

# # A joint is composed by a parent body, a child body, an axis and an actuator
# class JointConfig(BaseModel):
#     parent_body: str
#     child_body: str
#     sensors: dict[str, SensorConfig] = {}
#     actuators: dict[str, ActuatorConfig] = {}

# A module is composed by joints and bodies
class ModuleConfig(BaseModel):
    # Process/control loop settings
    frequency: float = 50.0  # control loop frequency in Hz
    get_max_k: int = 128  # max number of states to keep in ring buffer
    commands_per_cycle: int = 1  # max commands to process per control loop iteration
    command_queue_size: int = 1024  # size of the command input queue
    launch_timeout: float = 3.0  # timeout for process launch in seconds

    actuators: dict[str, ActuatorConfig] = {}
    sensors: dict[str, SensorConfig] = {}
    
    # calibration_file: str | None = None
    # gripper_open_degree: Optional[float] = 30
    # # Interpolator settings
    # interpolator_type: InterpolatorType = InterpolatorType.JOINT
    # max_speed: float | list[float] = 1.0  # max speed for interpolation (single or per-joint)
    

class ConnectionConfig(BaseModel):
    connection_id: str = "connection_1"
    ip: str
    action_sender_port: int
    action_receiver_port: int
    state_sender_port: int
    state_receiver_port: int

# A robot is composed by multiple modules, eg: arms, legs, base, etc.
class RobotConfig(BaseModel):
    img_compression_quality: int
    connection: ConnectionConfig
    fps: int

class ControllerConfig(BaseModel):
    fps: int
    tps: int
    system2_model: str
    system2_prompt: str
    system2_base_url: str
    system2_api_key: str
    policy_path: str
    device: str
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
    image_processing: list[str]
    tracking_model: str | None
    depth_model: str | None
    ref_x: int
    ref_y: int
    img_compression_quality: int

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
    type: Literal['adam', 'adamw'] = 'adamw'
    

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
    
    
    


class ModelConfig(ConfigBaseModel):
    type: Literal['act', 'diffusion'] = 'act'

class ACTConfig(ModelConfig):
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100
    use_vae: bool = True
    temporal_ensemble_coeff: float | None = None


class TrainConfig(ConfigBaseModel):
    train_dataset: str
    val_dataset: str | None = None
    output_dir: str | None = None
    val_percentage: float | None = None

    policy_path: str | None = None
    job_name: str
    resume: bool
    batch_size: int
    image_channels: int
    steps: int
    num_workers: int
    log_freq: int
    val_freq: int
    save_freq: int
    
    optimizer: Union[AdamOptimizerConfig, AdamWOptimizerConfig] = Field(discriminator="type")
    scheduler: SchedulerConfig
    precision: PrecisionConfig
    logger: Union[WandBLoggerConfig] = Field(discriminator="type")
    model: Union[ACTConfig] = Field(discriminator="type")
    
    @model_validator(mode='after')
    def validate_policy_path_when_resuming(cls, values) -> 'TrainConfig':
        """Validate that policy_path is provided when resuming"""
        if values.resume and not values.policy_path:
            raise ValueError("policy_path must be specified when resume is true")
        return values
    
    @field_validator('logger')
    @classmethod
    def validate_logger_config(cls, v):
        """Validate the logger configuration"""
        if v.type == "wandb":
            if not hasattr(v, 'project') or not v.project:
                raise ValueError("logger.project must be specified when logger is enabled with type 'wandb'")
        return v
    
    def to_tags(self) -> list[str]:
        tags = [
            f"lr:{self.optimizer.lr}",
            f"batch_size:{self.batch_size}",
            f"train_dataset:{self.train_dataset}",
            f"val_dataset:{self.val_dataset if self.val_dataset else self.train_dataset}",
            f"policy:{'act' if type(self.model) == ACTConfig else 'unknown'}",
        ]
        return tags



class SetupModuleConfig(ConfigBaseModel):
    module_name: str