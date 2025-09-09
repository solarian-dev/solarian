from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from enum import Enum
from typing import Tuple, Optional, List
import os
from pathlib import Path


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
    

class ValueName(str, Enum):
    POS = "pos"
    SPEED = "speed"
    IMAGE = "image"
    TORQUE_ENABLE = "torque_enable"
    BOUNDING_BOX = "bounding_box"

    def __str__(self):
        return self.value


class ActionValue(str, Enum):
    STOP = "stop"
    RESET = "reset"
    REWIND = "rewind"

class ComponentType(str, Enum):
    FEETECH = "feetech"
    DYNAMIXEL = "dynamixel"
    OPENCV = "opencv"
    UR5_CARTESIAN = "ur5_cartesian"
    UR5_JOINT = "ur5_joint"

    def __str__(self):
        return self.value
        
class ComponentConfig(BaseModel):
    type: ComponentType
    port: str | int

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
    mode: str

class ServoConfig(ActuatorConfig, SensorConfig):
    pass

# A body has mass and inertia and it might have sensors attached to it
class BodyConfig(BaseModel):
    mass: float
    inertia: float
    sensors: dict[str, SensorConfig] = {}
    actuators: dict[str, ActuatorConfig] = {}

# A joint is composed by a parent body, a child body, an axis and an actuator
class JointConfig(BaseModel):
    parent_body: str
    child_body: str
    sensors: dict[str, SensorConfig] = {}
    actuators: dict[str, ActuatorConfig] = {}

# A module is composed by joints and bodies
class ModuleConfig(BaseModel):
    type: str
    calibration_file: str
    gripper_open_degree: Optional[float] = 30
    bodies: dict[str, BodyConfig] = {}
    joints: dict[str, JointConfig] = {}

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
    modules: dict[str, ModuleConfig]

class ControllerConfig(BaseModel):
    fps: int
    tps: int
    system2_model: str
    system2_prompt: str
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

# A project is composed by a robot, a server, a controller and a config
# In its top level config it also points to the classes that implement the robot, server, controller and config
class ProjectConfig(BaseModel):
    name: str
    controller: ControllerConfig
    server: ServerConfig
    robot: RobotConfig


class ConfigBaseModel(BaseModel):
    """Base model with enhanced error handling for configs"""
    
    @classmethod
    def parse_with_friendly_errors(cls, data):
        """Parse config data with user-friendly error messages"""
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            # Get the config filename if possible
            config_path = os.environ.get("PROTO_CURRENT_CONFIG", "")
            
            # Format a user-friendly error message
            error_msg = f"Configuration Error in {cls.__name__}:\n"
            error_msg += f"  {str(e)}\n"
            
            if config_path:
                error_msg += f"  Please check your configuration file: {config_path}\n"
                
            # Re-raise with the friendly message
            raise ValueError(error_msg) from e


class LoggerConfig(ConfigBaseModel):
    type: str
    

class TrainLoggerConfig(LoggerConfig):
    enable: bool = False
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


class ACTConfig(ConfigBaseModel):
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
    learning_rate: float
    batch_size: int
    steps: int
    num_workers: int
    log_freq: int
    save_freq: int
    
    logger: TrainLoggerConfig
    model: ACTConfig
    
    @model_validator(mode='after')
    def validate_output_dir_when_resuming(cls, values) -> 'TrainConfig':
        """Validate that output_dir is provided when resuming"""
        if values.resume and not values.output_dir:
            raise ValueError("output_dir must be specified when resume is true")
        return values
    
    @field_validator('logger')
    @classmethod
    def validate_logger_config(cls, v):
        """Validate the logger configuration"""
        if v.enable and v.type == "wandb":
            if not hasattr(v, 'project') or not v.project:
                raise ValueError("logger.project must be specified when logger is enabled with type 'wandb'")
        return v
    
    def to_tags(self) -> list[str]:
        tags = [
            f"lr:{self.learning_rate}",
            f"batch_size:{self.batch_size}",
            f"train_dataset:{self.train_dataset}",
            f"val_dataset:{self.val_dataset if self.val_dataset else self.train_dataset}",
            f"policy:{'act' if type(self.model) == ACTConfig else 'unknown'}",
        ]
        return tags