import abc

from dataclasses import dataclass, field

from pathlib import Path

from lerobot.common.robot_devices.motors.configs import (
    DynamixelMotorsBusConfig,
    FeetechMotorsBusConfig,
    MotorsBusConfig,
)

from lerobot.common.robot_devices.cameras.configs import (
    CameraConfig,
    OpenCVCameraConfig,
)

@dataclass
class RobotConfig(abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@dataclass
class So100WithKochRobotConfig(RobotConfig):
    calibration_dir: str = ".cache/calibration/so100_with_koch"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem58FD0167771",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58FD0167571",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False


@dataclass
class LeKiwiWithKochRobotConfig(RobotConfig):
    
    calibration_dir: str = ".cache/calibration/lekiwi_with_koch"

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": OpenCVCameraConfig(
                # camera_index=1, fps=30, width=640, height=480, rotation=90
                camera_index=1, fps=30, width=320, height=240, rotation=90
                # camera_index="/dev/video2", fps=30, width=640, height=480, rotation=180
            ),
            "wrist": OpenCVCameraConfig(
                camera_index=0, fps=30, width=640, height=480, rotation=None
                # camera_index="/dev/video0", fps=30, width=640, height=480, rotation=None
            ),
        }
    )

    base_motor_ids = ["left_wheel", "back_wheel", "right_wheel"]
    arm_motor_ids = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper"
    ]

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58FD0167571",
                # port="/dev/ttyACM0",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                    "left_wheel": (7, "sts3215"),
                    "back_wheel": (8, "sts3215"),
                    "right_wheel": (9, "sts3215"),
                },
            ),
        }
    )

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem58FD0167771",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False

@dataclass
class DummyRobotConfig(RobotConfig):
    
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            # "front": OpenCVCameraConfig(
            #     # camera_index=1, fps=30, width=640, height=480, rotation=90
            #     camera_index=1, fps=30, width=320, height=240, rotation=90
            #     # camera_index="/dev/video2", fps=30, width=640, height=480, rotation=180
            # ),
            "wrist": OpenCVCameraConfig(
                camera_index=0, fps=30, width=640, height=480, rotation=None
                # camera_index="/dev/video0", fps=30, width=640, height=480, rotation=None
            ),
        }
    )

    base_motor_ids = []
    arm_motor_ids = []
    
    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    image_processing = ['wrist']
    # image_processing = []

    base_motor_ids = []
    arm_motor_ids = []

    