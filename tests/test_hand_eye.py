"""Robot-world hand-eye: frame conventions, PnP path and the derived fixes.

A hand-eye solve has no self-evident answer to eyeball, and every quantity it
produces is one inversion away from a number that looks just as plausible and
is wrong. So these build a rig whose transforms are known, push it through the
same code the command runs on hardware, and require the ground truth back.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.spatial.transform as st

from solarian.cli.commands.calibrate_hand_eye import (
    UMI_TIP_IN_CAMERA,
    _board_in_camera,
    _board_points,
    _derive,
    _fit_error,
    _offset_to_mat,
    _pose_to_mat,
    _solve,
)
from solarian.core.ik.spec import FrameOffset
from solarian.core.optical_remap import load_shipped_gopro_intrinsics

FRAME_WH = (640, 360)
SQUARE_M = 0.025
BOARD = (9, 6)

# The camera sits back and to the side of the flange, yawed 3 deg off it: the
# umi-slim CAD mount plus exactly the kind of build error this has to recover.
CAMERA_IN_FLANGE = np.eye(4)
CAMERA_IN_FLANGE[:3, :3] = st.Rotation.from_rotvec(
    np.deg2rad([1.0, 3.0, -0.5])
).as_matrix()
CAMERA_IN_FLANGE[:3, 3] = [-0.0002, -0.0847, 0.0217]

# Declared TCP: the umi-slim CAD tip with identity rotation, as so106 wires it.
IK_TCP = FrameOffset(pos=(0.0, -0.0157, 0.2057), quat=(1.0, 0.0, 0.0, 0.0))

BOARD_IN_BASE = np.eye(4)
BOARD_IN_BASE[:3, :3] = st.Rotation.from_rotvec([0.0, 0.0, 0.3]).as_matrix()
BOARD_IN_BASE[:3, 3] = [0.28, -0.05, 0.002]


def _mat_to_pose(mat: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [mat[:3, 3], st.Rotation.from_matrix(mat[:3, :3]).as_rotvec()]
    )


def _simulate(ik_tcp, n=14, pixel_noise=0.0, seed=0):
    """The pose/board pairs ``_collect`` would produce on a known rig.

    Corners go out through the real fisheye model and come back through
    ``_board_in_camera``, so the undistort + planar-PnP path is exercised
    rather than assumed.
    """
    import cv2

    rng = np.random.default_rng(seed)
    obj_pts = _board_points(BOARD, SQUARE_M)
    intr = load_shipped_gopro_intrinsics(FRAME_WH)
    tcp_in_flange = _offset_to_mat(ik_tcp)

    samples = []
    while len(samples) < n:
        flange_in_base = np.eye(4)
        flange_in_base[:3, :3] = st.Rotation.from_rotvec(
            rng.uniform(-0.45, 0.45, 3) + np.array([np.pi, 0.0, 0.0])
        ).as_matrix()
        flange_in_base[:3, 3] = [0.25, -0.1, 0.30] + rng.uniform(-0.06, 0.06, 3)
        camera_in_base = flange_in_base @ CAMERA_IN_FLANGE
        tx_world2cam = np.linalg.inv(camera_in_base) @ BOARD_IN_BASE
        if tx_world2cam[2, 3] < 0.05:
            continue

        px, _ = cv2.fisheye.projectPoints(
            obj_pts.reshape(-1, 1, 3),
            np.ascontiguousarray(cv2.Rodrigues(tx_world2cam[:3, :3])[0]),
            np.ascontiguousarray(tx_world2cam[:3, 3].reshape(3, 1)),
            intr["K"],
            intr["D"],
        )
        px = px.reshape(-1, 2) + rng.normal(0.0, pixel_noise, (len(obj_pts), 2))
        if px.min() < 0 or px[:, 0].max() > FRAME_WH[0] or px[:, 1].max() > FRAME_WH[1]:
            continue

        rvec, tvec, _rms = _board_in_camera(px.astype(np.float32), obj_pts, intr)
        recovered = np.eye(4)
        recovered[:3, :3] = cv2.Rodrigues(rvec)[0]
        recovered[:3, 3] = np.asarray(tvec).squeeze()
        samples.append(
            {
                "tx_base2tcp": np.linalg.inv(
                    _pose_to_mat(_mat_to_pose(flange_in_base @ tcp_in_flange))
                ),
                "tx_world2cam": recovered,
            }
        )
    return samples


def test_board_in_camera_recovers_a_known_board_pose():
    """Fisheye pixels -> board pose, the input every other number rests on."""
    import cv2

    obj_pts = _board_points(BOARD, SQUARE_M)
    intr = load_shipped_gopro_intrinsics(FRAME_WH)
    truth = np.eye(4)
    truth[:3, :3] = st.Rotation.from_rotvec([0.15, -0.2, 0.05]).as_matrix()
    truth[:3, 3] = [-0.09, -0.05, 0.32]

    px, _ = cv2.fisheye.projectPoints(
        obj_pts.reshape(-1, 1, 3),
        np.ascontiguousarray(cv2.Rodrigues(truth[:3, :3])[0]),
        np.ascontiguousarray(truth[:3, 3].reshape(3, 1)),
        intr["K"],
        intr["D"],
    )
    rvec, tvec, rms_px = _board_in_camera(
        px.reshape(-1, 2).astype(np.float32), obj_pts, intr
    )
    assert rms_px < 0.05
    assert np.allclose(np.asarray(tvec).squeeze(), truth[:3, 3], atol=1e-4)
    assert np.allclose(cv2.Rodrigues(rvec)[0], truth[:3, :3], atol=1e-3)


def test_board_points_scale_with_square_size():
    pts = _board_points((9, 6), 0.025)
    assert pts.shape == (54, 3)
    assert np.allclose(pts[:, 2], 0.0)
    # Corner-to-corner along each axis is (count - 1) squares.
    assert pts[:, 0].max() == pytest.approx(8 * 0.025)
    assert pts[:, 1].max() == pytest.approx(5 * 0.025)


def test_solve_recovers_the_rig_exactly():
    """Noise-free: any inverted or transposed frame shows up here immediately."""
    samples = _simulate(IK_TCP)
    tx_base2world, tx_tcp2camera = _solve(samples)

    assert np.allclose(tx_base2world, np.linalg.inv(BOARD_IN_BASE), atol=1e-6)
    # tcp -> camera is constant across poses and equals the two fixed mounts.
    expected = np.linalg.inv(CAMERA_IN_FLANGE) @ _offset_to_mat(IK_TCP)
    assert np.allclose(tx_tcp2camera, expected, atol=1e-6)

    trans_mm, rot_deg = _fit_error(samples, tx_base2world, tx_tcp2camera)
    assert trans_mm.max() < 1e-3
    assert rot_deg.max() < 1e-4


def test_derive_separates_the_hardware_offset_from_the_config_rotation():
    _, tx_tcp2camera = _solve(_simulate(IK_TCP))
    derived = _derive(tx_tcp2camera, IK_TCP)

    # The camera mount comes back whatever ik_tcp happens to claim, which is
    # what makes it usable as a cross-check against CAD.
    assert np.allclose(derived["camera_in_flange"], CAMERA_IN_FLANGE[:3, 3], atol=1e-6)

    # The suggested quaternion is the camera's orientation in the flange, which
    # is the orientation UMI training gives the fingertip frame.
    suggested = st.Rotation.from_quat(np.roll(derived["quat"], -1))
    truth = st.Rotation.from_matrix(CAMERA_IN_FLANGE[:3, :3])
    assert np.degrees(np.linalg.norm((truth.inv() * suggested).as_rotvec())) < 1e-4
    assert derived["misalign_deg"] == pytest.approx(
        np.degrees(np.linalg.norm(truth.as_rotvec())), abs=1e-4
    )

    # The fingertip offset is the declared TCP origin read in camera axes, and
    # the reported difference is what the mount has to make up.
    tip = (np.linalg.inv(CAMERA_IN_FLANGE) @ _offset_to_mat(IK_TCP))[:3, 3]
    assert np.allclose(derived["tip_in_camera"], tip, atol=1e-6)
    assert np.allclose(derived["tip_delta"], UMI_TIP_IN_CAMERA - tip, atol=1e-9)


def test_suggested_rotation_is_idempotent():
    """Applying the printed quaternion has to leave nothing left to correct."""
    _, first = _solve(_simulate(IK_TCP))
    suggested = _derive(first, IK_TCP)

    fixed = FrameOffset(
        pos=IK_TCP.pos, quat=tuple(float(v) for v in suggested["quat"])
    )
    _, second = _solve(_simulate(fixed))
    after = _derive(second, fixed)

    assert after["misalign_deg"] < 1e-4
    assert np.allclose(after["quat"], suggested["quat"], atol=1e-6)
    # Moving the declared frame cannot move the physical mount.
    assert np.allclose(after["camera_in_flange"], CAMERA_IN_FLANGE[:3, 3], atol=1e-6)


def test_corner_noise_stays_well_under_the_offset_being_measured():
    """Sets the noise floor the command's report thresholds are chosen against."""
    _, tx_tcp2camera = _solve(_simulate(IK_TCP, pixel_noise=0.3, seed=3))
    derived = _derive(tx_tcp2camera, IK_TCP)

    tip = (np.linalg.inv(CAMERA_IN_FLANGE) @ _offset_to_mat(IK_TCP))[:3, 3]
    assert np.linalg.norm(derived["tip_in_camera"] - tip) < 0.005

    suggested = st.Rotation.from_quat(np.roll(derived["quat"], -1))
    truth = st.Rotation.from_matrix(CAMERA_IN_FLANGE[:3, :3])
    assert np.degrees(np.linalg.norm((truth.inv() * suggested).as_rotvec())) < 1.5


def test_rot_between_deg_measures_the_angle_between_orientations():
    from solarian.cli.commands.calibrate_hand_eye import _rot_between_deg

    flat = np.zeros(6)
    assert _rot_between_deg(flat, flat) == pytest.approx(0.0)
    turned = np.concatenate([np.zeros(3), np.deg2rad([0.0, 0.0, 12.0])])
    assert _rot_between_deg(flat, turned) == pytest.approx(12.0)
    # Position is irrelevant: only the rotation part counts.
    moved = np.concatenate([[1.0, 2.0, 3.0], turned[3:]])
    assert _rot_between_deg(flat, moved) == pytest.approx(12.0)


class _FakeArm:
    """Records the torque calls _collect makes, and replays scripted poses."""

    def __init__(self, poses):
        self.poses = poses
        self.tick = -1
        self.torque = []

    def read_pose(self):
        return self.poses[min(self.tick, len(self.poses) - 1)]

    def start_calibration_mode(self):
        self.torque.append("off")

    def hold_measured_pose(self):
        self.torque.append("hold")

    def end_calibration_mode(self):
        self.torque.append("end")

    def enable_torque(self):
        self.torque.append("on")


class _FakeCam:
    """Steps the fake arm once per read, then stops the loop from outside."""

    frequency = 30.0

    def __init__(self, arm, stop_after):
        self.arm = arm
        self.stop_after = stop_after
        self.reads = 0

    def read_sensors(self, _cls, names):
        self.reads += 1
        if self.reads > self.stop_after:
            raise KeyboardInterrupt
        self.arm.tick += 1
        return {names[0]: {"value": np.zeros((360, 640, 3), dtype=np.uint8)}}


def _run_collect(monkeypatch, poses, stop_after, n_samples=99):
    """Drive _collect with the board detector stubbed out and no display."""
    import cv2

    import solarian.cli.commands.calibrate_hand_eye as mod

    monkeypatch.setattr(mod, "find_chessboard", lambda *_a: np.zeros((54, 1, 2), np.float32))
    monkeypatch.setattr(
        mod, "_board_in_camera", lambda *_a: (np.zeros((3, 1)), np.array([0.0, 0.0, 0.3]), 0.2)
    )
    # No preview window in a test run; this is also the headless code path.
    monkeypatch.setattr(cv2, "namedWindow", lambda *_a, **_k: (_ for _ in ()).throw(cv2.error("no display")))
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "")

    ticks = iter(range(10_000))
    monkeypatch.setattr(
        mod.time, "monotonic", lambda: next(ticks) * 0.25, raising=False
    )
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None, raising=False)

    arm = _FakeArm(poses)
    cam = _FakeCam(arm, stop_after)
    samples = mod._collect(
        arm,
        cam,
        "camera",
        BOARD,
        _board_points(BOARD, SQUARE_M),
        load_shipped_gopro_intrinsics(FRAME_WH),
        "bgr",
        n_samples,
    )
    return arm, cam, samples


def _pose(x, yaw_deg):
    return np.concatenate([[x, 0.0, 0.2], np.deg2rad([0.0, 0.0, yaw_deg])])


def test_collect_never_energises_the_arm_even_when_interrupted(monkeypatch):
    """Torque off once and never back on, so the arm stays in the user's hands.

    Re-energising would either fight her grip or snap to the goal position the
    servos still hold from before free-drive, and she is holding the arm
    precisely so she can put it down herself. _FakeArm keeps enable_torque and
    hold_measured_pose so a regression that calls either shows up here, and it
    has to hold even when the run is interrupted part-way.
    """
    poses = [_pose(0.0, 30.0 * i) for i in range(6)]
    arm, _cam, samples = _run_collect(monkeypatch, poses, stop_after=20)

    assert arm.torque == ["off", "end"]
    assert isinstance(samples, list)


def test_detect_measures_on_the_full_frame_not_the_preview(monkeypatch):
    """The cheap gate runs on a small copy; the corners come from the full one.

    Returning the gate's corners would throw away the resolution the frame was
    read at, which is the whole reason the bus resize is dropped for the run.
    """
    import solarian.cli.commands.calibrate_hand_eye as mod

    seen: list[tuple[int, int]] = []

    def fake_find(gray, _board):
        seen.append(gray.shape[:2])
        # A single corner dead centre, so its coordinates name the resolution.
        return np.array(
            [[[gray.shape[1] / 2.0, gray.shape[0] / 2.0]]], dtype=np.float32
        )

    monkeypatch.setattr(mod, "find_chessboard", fake_find)
    corners = mod._detect(np.zeros((1080, 1920), np.uint8), BOARD, (640, 360))

    assert seen == [(360, 640), (1080, 1920)]
    assert corners[0, 0].tolist() == [960.0, 540.0]


def test_detect_skips_the_full_frame_when_the_gate_finds_nothing(monkeypatch):
    """Misses are the slow case at full resolution, so they stop at the gate."""
    import solarian.cli.commands.calibrate_hand_eye as mod

    seen: list[tuple[int, int]] = []

    def fake_find(gray, _board):
        seen.append(gray.shape[:2])
        return None

    monkeypatch.setattr(mod, "find_chessboard", fake_find)
    assert mod._detect(np.zeros((1080, 1920), np.uint8), BOARD, (640, 360)) is None
    assert seen == [(360, 640)]


def test_collect_takes_one_sample_per_pose_once_the_arm_settles(monkeypatch):
    """Three still reads per pose is enough to trip the stillness gate."""
    poses = []
    for i in range(5):
        poses.extend([_pose(0.0, 40.0 * i)] * 4)
    _arm, _cam, samples = _run_collect(monkeypatch, poses, stop_after=len(poses))

    # Five distinct orientations, each held still, so five samples and no
    # duplicates: the rotation gate suppresses re-firing on the same pose.
    assert len(samples) == 5


def test_collect_rejects_poses_that_only_translate(monkeypatch):
    """A translation-only sweep leaves the lever arm unobservable, so it gates."""
    poses = [_pose(0.02 * i, 0.0) for i in range(8) for _ in range(4)]
    _arm, _cam, samples = _run_collect(monkeypatch, poses, stop_after=len(poses))

    # Only the very first sample has no predecessor to compare against; every
    # later pose is pure translation and must be refused.
    assert len(samples) == 1


def test_umi_training_tip_matches_the_dataset_plan_constants():
    """The constant the whole command compares against, from UMI's own terms."""
    cam_to_center_height = 0.086
    cam_to_mount_offset = 0.01465
    tcp_offset = 0.205
    assert np.allclose(
        UMI_TIP_IN_CAMERA,
        [0.0, cam_to_center_height, cam_to_mount_offset + tcp_offset],
    )
