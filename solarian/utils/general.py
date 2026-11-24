import time
import importlib
import logging
import numpy as np
from functools import cache
import platform
import traceback
import subprocess
import importlib
import select
import sys





def get_class(class_path):
    """Dynamically import and return a class from a string path"""
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def name_process(name: str) -> None:
    """Show this process as ``name`` in ``top``, ``htop`` and ``ps -o comm`` on Linux.

    The kernel keeps the first 15 bytes. Threads started afterwards inherit
    it. Does nothing where there is no ``/proc``.
    """
    try:
        with open("/proc/self/comm", "w") as comm:
            comm.write(name[:15])
    except OSError:
        pass


# TODO reimplement
def busy_wait(seconds):
    if platform.system() == "Darwin":
        # On Mac, `time.sleep` is not accurate but we can combine it with busy wait
        end_time = time.perf_counter() + seconds
        if seconds > 0.005:
            time.sleep(seconds-0.005)
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)

# TODO reimplement
def degps_to_raw(degps: float) -> int:
    steps_per_deg = 4096.0 / 360.0
    speed_in_steps = degps * steps_per_deg
    speed_int = int(round(speed_in_steps))
    # Cap the value to fit within signed 16-bit range (-32768 to 32767)
    if speed_int > 0x7FFF:
        speed_int = 0x7FFF  # 32767 -> maximum positive value
    elif speed_int < -0x8000:
        speed_int = -0x8000  # -32768 -> minimum negative value
    return speed_int

# TODO reimplement
def raw_to_degps(raw_speed: int) -> float:
    steps_per_deg = 4096.0 / 360.0
    magnitude = raw_speed & 0x7FFF
    degps = magnitude / steps_per_deg
    if raw_speed & 0x8000:
        degps = -degps
    return degps

# TODO reimplement
def body_to_wheel_raw(
    x_cmd: float,
    y_cmd: float,
    theta_cmd: float,
    wheel_radius: float = 0.05,
    base_radius: float = 0.125,
    max_raw: int = 3000,
) -> dict:
    """
    Convert desired body-frame velocities into wheel raw actions.

    Parameters:
        x_cmd      : Linear velocity in x (m/s).
        y_cmd      : Linear velocity in y (m/s).
        theta_cmd  : Rotational velocity (deg/s).
        wheel_radius: Radius of each wheel (meters).
        base_radius : Distance from the center of rotation to each wheel (meters).
        max_raw    : Maximum allowed raw action (ticks) per wheel.

    Returns:
        A dictionary with wheel raw actions:
            {"left_wheel": value, "back_wheel": value, "right_wheel": value}.

    Notes:
        - Internally, the method converts theta_cmd to rad/s for the kinematics.
        - The raw action is computed from the wheels angular speed in deg/s
        using degps_to_raw(). If any action exceeds max_raw, all actions
        are scaled down proportionally.
    """
    # Convert rotational velocity from deg/s to rad/s.
    theta_rad = theta_cmd * (np.pi / 180.0)
    # Create the body velocity vector [x, y, theta_rad].
    velocity_vector = np.array([x_cmd, y_cmd, theta_rad])

    # # Define the wheel mounting angles (defined from y axis cw)
    # angles = np.radians(np.array([300, 180, 60]))
    # Define the wheel mounting angles with a -90° offset.
    angles = np.radians(np.array([240, 0, 120]) - 90)

    # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
    # The third column (base_radius) accounts for the effect of rotation.
    m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

    # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
    wheel_linear_speeds = m.dot(velocity_vector)
    wheel_angular_speeds = wheel_linear_speeds / wheel_radius

    # Convert wheel angular speeds from rad/s to deg/s.
    wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

    # Scaling
    steps_per_deg = 4096.0 / 360.0
    raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
    max_raw_computed = max(raw_floats)
    if max_raw_computed > max_raw:
        scale = max_raw / max_raw_computed
        wheel_degps = wheel_degps * scale

    # Convert each wheel’s angular speed (deg/s) to a raw integer.
    wheel_raw = [degps_to_raw(deg) for deg in wheel_degps]

    return {"left_wheel": wheel_raw[0], "back_wheel": wheel_raw[1], "right_wheel": wheel_raw[2]}

# TODO reimplement
def wheel_raw_to_body(
    wheel_raw: dict, wheel_radius: float = 0.05, base_radius: float = 0.125
) -> tuple:
    """
    Convert wheel raw action feedback back into body-frame velocities.

    Parameters:
        wheel_raw   : Dictionary with raw wheel actions (keys: "left_wheel", "back_wheel", "right_wheel").
        wheel_radius: Radius of each wheel (meters).
        base_radius : Distance from the robot center to each wheel (meters).

    Returns:
        A tuple (x_cmd, y_cmd, theta_cmd) where:
            x_cmd      : Linear velocity in x (m/s).
            y_cmd      : Linear velocity in y (m/s).
            theta_cmd  : Rotational velocity in deg/s.
    """
    # Extract the raw values in order.
    raw_list = [
        int(wheel_raw.get("left_wheel", 0)),
        int(wheel_raw.get("back_wheel", 0)),
        int(wheel_raw.get("right_wheel", 0)),
    ]

    # Convert each raw action back to an angular speed in deg/s.
    wheel_degps = np.array([raw_to_degps(r) for r in raw_list])
    # Convert from deg/s to rad/s.
    wheel_radps = wheel_degps * (np.pi / 180.0)
    # Compute each wheel’s linear speed (m/s) from its angular speed.
    wheel_linear_speeds = wheel_radps * wheel_radius

    # Define the wheel mounting angles (defined from y axis cw)
    angles = np.radians(np.array([300, 180, 60]))
    m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

    # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
    m_inv = np.linalg.inv(m)
    velocity_vector = m_inv.dot(wheel_linear_speeds)
    x_cmd, y_cmd, theta_rad = velocity_vector
    theta_cmd = theta_rad * (180.0 / np.pi)
    return (x_cmd, y_cmd, theta_cmd)

# TODO reimplement
def say(text, blocking=False):
    system = platform.system()

    if system == "Darwin":
        cmd = ["say", text]

    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")

    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]

    else:
        raise RuntimeError("Unsupported operating system for text-to-speech.")

    if blocking:
        subprocess.run(cmd, check=True)
    else:
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if system == "Windows" else 0)

# TODO reimplement
def log_say(text, play_sounds, blocking=False):
    logging.info(text)

    if play_sounds:
        say(text, blocking)


# @cache
# def is_headless():
#     """Detects if python is running without a monitor."""
#     try:
#         import pynput  # noqa

#         return False
#     except Exception:
#         print(
#             "Error trying to import pynput. Switching to headless mode. "
#             "As a result, the video stream from the cameras won't be shown, "
#             "and you won't be able to change the control flow with keyboards. "
#             "For more info, see traceback below.\n"
#         )
#         traceback.print_exc()
#         print()
#         return True


def enter_pressed() -> bool:
    if platform.system() == "Windows":
        import msvcrt

        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key in (b"\r", b"\n")  # enter key
        return False
    else:
        return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline().strip() == ""


