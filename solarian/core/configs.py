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
    #: Relative paths are resolved against the project / conf directory. The
    #: warp is *not* applied on the capture path: a policy needs GoPro optics,
    #: and nothing else does, so whoever needs it applies it to the frames it
    #: reads. See :func:`solarian.core.optical_remap.load_policy_remap`.
    optical_remap: str | None = None
    #: Size of the remapped GoPro plane, for the consumers that build it. Unset
    #: uses the plane the calibration was solved on, which is four times the
    #: pixels a 224 policy crop can use. Only the aspect ratio reaches the
    #: policy, so this is purely how much detail to carry into the crop.
    remap_width: int | None = None
    remap_height: int | None = None
    #: Resolution the recorded archive stream is captured at, before the
    #: optical remap and before the ``resize_*`` the observation path applies.
    #: Unset archives at the sensor's own ``width`` / ``height``. Lowering it
    #: is the knob for encoder CPU on a Raspberry Pi; it does not touch the
    #: sensor mode, so the field of view and the calibration stay valid.
    archive_width: int | None = None
    archive_height: int | None = None
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

    #: Seconds between a reading being physically true and this module
    #: stamping it, subtracted at the point of stamping in
    #: ``update_observations`` so every consumer sees the corrected instant.
    #: Measured by ``solarian calibrate-arm`` / ``calibrate-gripper``.
    receive_latency: float = 0.0

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
    #: Acceleration limit for ``moveL`` / ``moveJ`` / ``stopL``, shaped like
    #: ``max_speed`` (``[pos m/s^2, rot rad/s^2]`` on a pose interpolator).
    #: ``None`` keeps go-to moves at constant speed with no ramp.
    max_accel: float | list[float] | None = None
    #: Joint acceleration limit (rad/s^2) for an arm's ``moveJ``; ``max_accel``
    #: is Cartesian there. ``None`` keeps ``moveJ`` at constant speed.
    max_joint_accel: float | None = None
    #: Seconds between commanding a motion and the hardware performing it.
    #: The mirror of :attr:`receive_latency`: that one moves a timestamp
    #: backwards, this one moves a target time earlier, so a waypoint has to be
    #: sent ``action_latency`` before it is due. Read by whoever schedules
    #: waypoints rather than applied here, because the module cannot tell a
    #: deadline it should anticipate from one it should hit exactly.
    action_latency: float = 0.0
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

class ArchiveConfig(BaseModel):
    """Where a recording side writes its sessions, and how it encodes video.

    Shared by the robot's :class:`RecordingConfig` and the controller's
    :class:`ControllerRecordingConfig`: both write the same session layout.
    """

    #: Where sessions are written. Relative paths resolve against the project,
    #: which is only the right default while the project lives on local disk —
    #: point this elsewhere rather than mux an mp4 over a network mount.
    root: str = "recordings/robot"
    #: Grid the archive video is encoded onto. Defaults to the side's ``fps``.
    fps: float | None = None
    codec: str = "libx264"
    #: Constant Rate Factor. 18 is near-transparent, 23 is the x264 default;
    #: 20 keeps an archive that survives a later re-encode at policy resolution.
    crf: int = 20
    #: Keyframe interval, in frames. Not, despite appearances, about random
    #: access — nothing seeks an archive, the export decodes each one start to
    #: finish. What a short GOP actually buys is quality, because every
    #: keyframe is a fresh anchor the following frames drift from: measured on
    #: wrist-camera footage, 30 holds 4 dB more luma PSNR than 120 at the same
    #: crf. It also bounds what a killed episode loses, since ``frag_keyframe``
    #: starts a fragment per keyframe. It is paid for in size and a little CPU,
    #: keyframes being the bulky, expensive ones.
    keyint: int = 30
    preset: str = "veryfast"
    #: Raw libx264 knobs, applied over whatever ``preset`` chose. The presets
    #: move a dozen settings together, which is a blunt way to buy CPU; naming
    #: the one that matters is cheaper. Motion estimation is that one, and
    #: ``{"x264-params": "subme=0:me=dia"}`` takes an eighth off the encode for
    #: 0.1 dB and no extra bits, where dropping the preset to ``superfast``
    #: saves the same CPU and doubles the file.
    x264_options: dict[str, str] = {}
    #: Encoder threads per camera. Frame threading trades total CPU for
    #: wall-clock, which is the wrong way round here: on a Pi 5 at 1014x760 a
    #: single thread costs 0.50 of a core against 0.55 for two and 0.63 for
    #: four, and one thread still finishes a frame in half its 33 ms slot.
    #: Raise it only if the encoder starts dropping frames.
    threads: int = 1
    #: Captures buffered per camera before frames are dropped rather than
    #: stalling the camera's loop. This is the encoder's licence to fall
    #: behind: scene motion drives its cost, so a burst of it puts the encoder
    #: over budget for a few seconds, and a queue deep enough to cover the
    #: burst turns dropped captures into a backlog it clears afterwards. Costs
    #: ``width * height * 3`` bytes a slot, so the useful depth is a question
    #: of how much RAM the board has and how long a take runs.
    queue_size: int = 64
    #: Stop recording once the session reaches this size.
    max_session_gb: float | None = 64.0


# A robot is composed by multiple modules, eg: arms, legs, base, etc.
class RecordingConfig(ArchiveConfig):
    """Episode recording on the robot, at the fidelity only the robot has.

    On by default: a session that was not recorded cannot be recovered, and
    the cost of an mp4 nobody wanted is a deletion. ``max_session_gb`` is the
    counterweight — on-by-default without a cap fills an SD card.
    """

    #: Whether the robot starts encoders at all. A controller can only ask a
    #: robot with this on to record.
    enabled: bool = True


class ControllerRecordingConfig(ArchiveConfig):
    """Which sides record an episode, and how the controller stores its own.

    The controller's dataset has the robot's layout, built from what reached
    the controller: the observations as received and the actions it sent.
    """

    #: Initial positions of the GUI's two recording switches. Read when an
    #: episode starts, so flipping one mid-episode applies to the next.
    controller: bool = True
    robot: bool = True
    root: str = "recordings/controller"


class RobotConfig(BaseModel):
    img_compression_quality: int
    connection: ConnectionConfig
    fps: int
    #: Rate camera frames are put on the wire for the preview and for skills
    #: and processors hosted off the robot. Robot-side skills read every
    #: capture from shared memory regardless.
    preview_image_hz: float = 30.0
    recording: RecordingConfig = RecordingConfig()
    #: Run in MuJoCo simulation. ``None`` (default) defers to the robot class's
    #: ``is_sim`` attribute; set it here (or via ``run-robot sim=true``) to
    #: override, so the same robot class can run on hardware or in sim.
    sim: bool | None = None

class ModelEndpointConfig(BaseModel):
    """One model behind one API.

    ``provider`` picks the wire format, not the vendor: an OpenAI-compatible
    server (vLLM, a gateway) is ``openai`` with its ``base_url``.
    """

    #: ``anthropic``, ``openai`` (chat completions), ``openai_responses``
    #: (``/v1/responses``: OpenAI reasoning models with tools) or, for the
    #: monitor only, ``openai_decisions`` (``/v1/decisions``).
    provider: str = "openai"
    model: str
    #: Environment variable holding the API key, so the key never sits in a
    #: yaml that gets committed.
    api_key_env: str | None = None
    base_url: str | None = None
    #: Output token cap per call.
    max_tokens: int = 4096
    #: Anthropic extended thinking budget in tokens; ``None`` disables it.
    thinking_budget: int | None = None
    #: OpenAI ``reasoning_effort`` for reasoning models; ``None`` leaves the default.
    reasoning_effort: str | None = None
    timeout_s: float = 120.0


class MonitorConfig(ModelEndpointConfig):
    #: Seconds between assessments while an agent action runs. A tick whose
    #: previous call is still in flight is skipped, never queued.
    period_s: float = 1.0
    #: Longest side of each frame sent to the monitor, in pixels.
    image_max_side: int = 512
    #: Frames per camera in each assessment, one per tick, newest last: two
    #: or more let the monitor judge motion rather than a still.
    history: int = 2
    #: What the monitor should know about this robot and its cameras, appended
    #: to the generic instructions: which parts are always in view, what a held
    #: object looks like.
    instructions: str = ""
    #: Per camera (``module/field``), a region ``[x0, y0, x1, y1]`` as fractions
    #: of the capture. The monitor sees only that region of the camera, at
    #: capture resolution: where the detail that decides success is small and
    #: always in the same place, such as what the gripper holds. Monitor frames
    #: never carry overlays; a tracker mask that slips would mislead it.
    close_ups: dict[str, tuple[float, float, float, float]] = {}
    #: Also send the frame from the action's start, the planner's point marked
    #: on it, so the monitor knows which object is the target. Off when the
    #: close-ups alone decide, as for a gripper's fingertips.
    reference: bool = True
    #: Mark where the tracker has the target on every monitor frame: a 1 px
    #: box just outside its mask's bounds, not a mask, so the object stays visible.
    #: Nothing is marked on a frame whose camera has no tracker output.
    mark_target: bool = False
    timeout_s: float = 10.0


class AgentBudgets(BaseModel):
    #: Planner calls per run.
    max_turns: int = 80
    max_minutes: float = 20.0
    #: Failed or timed-out runs of one tool in a row before the planner is
    #: told to change approach or ask the user.
    retries_per_step: int = 3


class AgentConfig(BaseModel):
    """The supervising agent: a slow planner and a fast monitor.

    The planner reads the task, plans and calls the controller's tools; the
    monitor watches running actions and finishes them or wakes the planner.
    Different models because they are paid for differently: the planner is
    called a few times a minute, the monitor every ``period_s``.
    """

    planner: ModelEndpointConfig
    monitor: MonitorConfig | None = None
    #: Seconds between planner wakes while actions run and nothing else
    #: happened; ``None`` never wakes it on time alone. Actions ending and
    #: the monitor are the fast path; this is a backstop.
    planner_heartbeat_s: float | None = 20.0
    budgets: AgentBudgets = AgentBudgets()
    #: Project instructions appended to the framework's system prompt: the
    #: robot, the workspace, what the tools are good and bad at.
    system_prompt: str = ""
    #: Camera images the agent sees, as ``module/field`` labels. Empty: all.
    cameras: list[str] = []
    #: Longest side of each image sent to the planner, in pixels.
    image_max_side: int = 768
    #: Draw a labelled pixel grid on planner images, to help it point.
    image_grid: bool = True
    #: Module readings the agent sees, ``{module: [field, ...]}``. Empty: all.
    state_fields: dict[str, list[str]] = {}
    #: Images kept in the planner's context; older ones become a text stub.
    max_images_in_context: int = 4
    #: A point chosen on an image older than this is refused as stale.
    max_point_age_s: float = 15.0
    #: Open a recording episode per run when the robot can record.
    record_runs: bool = True
    #: Plan and point, but never start a skill.
    dry_run: bool = False


class ControllerConfig(BaseModel):
    fps: int
    tps: int
    #: Deprecated: read only by the legacy ``solarian.core.ai.old`` AIController.
    system2_model: str = ""
    system2_prompt: str = ""
    system2_base_url: str = ""
    system2_api_key: str = ""
    #: The supervising agent (see :class:`AgentConfig`). ``None``: no agent.
    agent: AgentConfig | None = None
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
    #: not throttled; JPEG is the expensive part. The picture itself is the
    #: JPEG the robot already sent, at that JPEG's size and quality.
    ui_image_fps: int = 15
    #: Names of declared modules whose device is sampled by the UI instead of by
    #: this process, replaced at construction by a
    #: :class:`~solarian.core.remote_input_module.RemoteInputModule`. The way a
    #: cloud-hosted controller uses a leader arm plugged into the operator's
    #: laptop, without the project's code changing.
    remote_input_modules: list[str] = []
    #: Start an episode as soon as the session is up, on whichever sides
    #: ``recording`` selects. This only decides whether an operator has to
    #: press a key first. A teleop session that was not recorded cannot be
    #: recovered.
    record: bool = True
    recording: ControllerRecordingConfig = ControllerRecordingConfig()
    #: Task string stamped on the episodes this session records.
    task: str | None = None
    #: Deprecated, superseded by ``ui``. ``true`` still selects the console UI,
    #: which is what the old headless loop was; use ``ui: null`` for no UI.
    headless: bool = False
    #: Unused by ``MOVE_*`` / ``SERVO_*`` skills. Kept so existing controller
    #: yaml still loads.
    waypoint_lookahead_s: float = 0.25
    connection: ConnectionConfig
    modules: dict[str, ModuleConfig]

class RecordConfig(ControllerConfig):
    """The ``solarian record`` run: an operator collecting a training set.

    Only the shape of the session lives here — how many episodes, how long
    each one runs, how long to reset between them. Everything about *how* an
    episode is stored belongs to the recording sides' configs: the robot's
    :class:`RecordingConfig` and the controller's ``recording``.
    """

    #: Name of the session these episodes are filed under. A number and a
    #: hash are prepended, so a rerun gets its own directory rather than
    #: mixing with episodes an older build wrote.
    name: str = "session"
    num_episodes: int
    #: Cut the episode here. ``null`` records until the operator says stop.
    episode_time_s: float | None
    #: Teleop time between episodes, for putting the scene back.
    reset_time_s: float
    #: Reset every N episodes rather than after each one.
    reset_period: int = 1
    play_sounds: bool = True
    #: Tasks to cycle through, one episode each. Empty records the
    #: controller's own ``task``.
    tasks: list[str] = []

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
    #: Run skills whose ``execution_place`` is ``server`` here. Needs the
    #: project's robot module configs, to size the observation buffers.
    host_skills: bool = True
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
    type: str | list[str] | None = None
    
class TrainLoggerConfig(ConfigBaseModel):
    #: A solarian.tracking backend (wandb, solarian_base, none), or a list of
    #: them to log to several. $SOLARIAN_TRACKING overrides it.
    type: str | list[str]
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
    
class TrainConfig(ConfigBaseModel):
    """Thin, library-agnostic training config.

    The framework only knows these common fields; everything policy- or
    library-specific (model architecture, optimizer, etc.) is carried
    through untouched (``extra='allow'``) and validated by the plugin
    trainer's own :meth:`config_class`.  The ``trainer`` field selects which
    installed trainer (``solarian.trainers`` entry point) runs.
    """
    model_config = ConfigDict(extra="allow")

    #: Name of the trainer to run (resolved via the ``solarian.trainers``
    #: entry-point group), e.g. ``"diffusion"`` or ``"lerobot_act"``.
    trainer: str

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

    def to_tags(self) -> list[str]:
        """Generic run tags for logging. Plugins may override to add more."""
        return [
            f"trainer:{self.trainer}",
            f"batch_size:{self.batch_size}",
            f"train_dataset:{self.train_dataset}",
            f"val_dataset:{self.val_dataset if self.val_dataset else self.train_dataset}",
        ]


class SetupModuleConfig(ConfigBaseModel):
    module_name: str