from pydantic import BaseModel
from enum import Enum
from typing import Tuple, Optional


class ValueName(str, Enum):
    POS = "pos"
    SPEED = "speed"
    IMAGE = "image"
    TORQUE_ENABLE = "torque_enable"
    BOUNDING_BOX = "bounding_box"

    def __str__(self):
        return self.value

class ComponentType(str, Enum):
    FEETECH = "feetech"
    DYNAMIXEL = "dynamixel"
    OPENCV = "opencv"

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

class CameraConfig(SensorConfig):
    fps: int
    width: int
    height: int
    rotation: int | None
    channels: int
    color_mode: str

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

# A joint is composed by a parent body, a child body, an axis and an actuator
class JointConfig(BaseModel):
    parent_body: str
    child_body: str
    actuators: dict[str, ActuatorConfig] = {}

# A module is composed by joints and bodies
class ModuleConfig(BaseModel):
    type: str
    calibration_file: str
    gripper_open_degree: Optional[float] = 30
    bodies: dict[str, BodyConfig]
    joints: dict[str, JointConfig]

class ConnectionConfig(BaseModel):
    ip: str
    action_sender_port: int
    action_receiver_port: int
    state_sender_port: int
    state_receiver_port: int

# A robot is composed by multiple modules, eg: arms, legs, base, etc.
class RobotConfig(BaseModel):
    img_compression_quality: int
    connection: ConnectionConfig
    modules: dict[str, ModuleConfig]

class ControllerConfig(BaseModel):
    fps: int
    tps: int
    system2_model: str
    system2_prompt: str
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
    video: bool
    video_res: list[int]

class ServerConfig(BaseModel):
    connection: ConnectionConfig
    image_processing: list[str]
    image_processor_model: str | None

# A project is composed by a robot, a server, a controller and a config
# In its top level config it also points to the classes that implement the robot, server, controller and config
class ProjectConfig(BaseModel):
    name: str
    controller: ControllerConfig
    server: ServerConfig
    robot: RobotConfig
