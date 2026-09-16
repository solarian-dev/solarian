"""Measure where this rig's fingertips sit in its own camera frame.

A UMI policy is trained against one number it can never observe: the rigid
transform from the wrist camera to the fingertips. UMI's dataset plan bakes it
into every episode as a constant, so at eval time the fingertips have to sit
where the checkpoint believes they are, in the camera's own axes, or every
command it issues lands displaced by the difference. Nothing in an image
reveals the error -- a frame that looks perfectly aimed is still aimed at the
wrong distance.

Ported from UMI's ``record_robot_world_hand_eye.py`` +
``calibrate_robot_world_hand_eye.py`` pair, with a checkerboard in place of the
ArUco tag (Solarian already ships one, and the fisheye code that reads it) and
with recording and solving in one command, because a Solarian module owns its
hardware for the lifetime of a process.

The result splits into two halves that are fixed in different places:

* A **rotation**, which config can absorb. The arm's ``ik_tcp`` declares the
  TCP pose in the flange, and its orientation decides which way a policy's
  translation commands point, because an action is a delta in tool axes. Paste
  the printed quaternion onto that ``FrameOffset``.
* A **translation**, which config cannot. Under a relative pose
  representation, ``X . [I|p] . X^-1 == [I|p]``: a translation action moves the
  fingertips by the same vector wherever the declared TCP origin sits, so no
  ``ik_tcp`` edit corrects a camera mounted at the wrong distance. The printed
  offset is an instruction for the mount.

    solarian calibrate-hand-eye square=25.0

Overrides: ``samples=16 board=9x6 arm=right_arm``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st
from omegaconf import OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.cli.commands._calibration import (
    free_drive_pose,
    module_running,
    pick_module,
    returns_to_start,
)
from solarian.cli.commands._ik_common import load_module, select_ik_module
from solarian.core.camera import Camera
from solarian.core.data import ImageData
from solarian.core.fisheye import find_chessboard, load_intrinsics, parse_board
from solarian.core.optical_remap import camera_output_hw, resolve_remap_path
from solarian.core.pose import rotvec_to_quat

# UMI's ``scripts_slam_pipeline/06_generate_dataset_plan.py`` freezes the
# camera-to-fingertip transform into every training episode:
#   cam_to_center_height = 0.086             (fixed for the UMI gripper)
#   cam_to_tip_offset    = 0.01465 + 0.205   (GoPro mount + finger length)
# with zero rotation, so a released checkpoint expects its fingertips here.
UMI_TIP_IN_CAMERA = np.array([0.0, 0.086, 0.21965])

_SAMPLES = 12
# cv2.calibrateRobotWorldHandEye needs 3. Below roughly 8 poses the translation
# is dominated by corner noise instead of by the geometry being solved for.
_MIN_SAMPLES = 8
# The camera thread can still be holding a frame shot while the arm was being
# moved by hand, which would be paired with the pose it ended up at.
_FLUSH_FRAMES = 3
# Corner noise on a downscaled fisheye frame puts a floor under the rotation
# this can resolve; under it a "correction" would just be that noise.
_ROT_NOISE_DEG = 1.0
# A pose the two constants cannot explain to better than this is not a noisy
# sample, it is a wrong one.
_OUTLIER_MM = 8.0


def _pose_to_mat(pose) -> np.ndarray:
    """``(x, y, z, rx, ry, rz)`` metres + rotvec -> 4x4."""
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    mat = np.eye(4)
    mat[:3, :3] = st.Rotation.from_rotvec(pose[3:6]).as_matrix()
    mat[:3, 3] = pose[:3]
    return mat


def _offset_to_mat(offset) -> np.ndarray:
    """A :class:`~solarian.core.ik.spec.FrameOffset` -> 4x4 child-in-parent."""
    mat = np.eye(4)
    quat = np.asarray(offset.quat, dtype=np.float64)
    mat[:3, :3] = st.Rotation.from_quat(np.roll(quat, -1)).as_matrix()
    mat[:3, 3] = np.asarray(offset.pos, dtype=np.float64)
    return mat


def _board_points(board: tuple[int, int], square_m: float) -> np.ndarray:
    """Inner-corner grid on z=0, ordered as ``find_chessboard`` returns them."""
    cols, rows = board
    pts = np.zeros((cols * rows, 3), dtype=np.float64)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return pts


def _frame_intrinsics(sensor_cfg, frame_wh: tuple[int, int], conf_dir: Path):
    """The fisheye model describing the frames this camera emits, at their size.

    Frames leave the bus already warped into GoPro optics by
    ``finish_camera_frame``, so what describes them is the GoPro model out of
    the remap JSON, rescaled to whatever ``resize_*`` left behind. Without a
    remap there is nothing to measure against: the point of the exercise is the
    pose in the frame the UMI checkpoint consumes.
    """
    if not sensor_cfg.optical_remap:
        print(
            "This camera has no optical_remap, so its frames are not in UMI "
            "policy optics and a pose measured in them says nothing about what "
            "the checkpoint expects. Run `solarian calibrate-umi-camera` first."
        )
        return None
    path = resolve_remap_path(str(sensor_cfg.optical_remap), conf_dir)
    with open(path, "r") as fp:
        remap = json.load(fp)
    return load_intrinsics(remap["gopro_intrinsics"], frame_wh)


def _board_in_camera(corners, obj_pts, intr):
    """Board pose in camera axes, as ``(rvec, tvec, reprojection_rms_px)``.

    Undistorting the corners to normalized rays first is what lets ``solvePnP``
    run with an identity camera matrix, and that is the only way to use it on a
    fisheye frame whose distortion its pinhole model cannot express.
    """
    import cv2

    norm = cv2.fisheye.undistortPoints(
        corners.reshape(-1, 1, 2).astype(np.float64), intr["K"], intr["D"]
    )
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, norm, np.eye(3), np.zeros(4), flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, np.eye(3), np.zeros(4))
    # Scaled by fx so the residual reads in the pixels the corners came from.
    err = np.linalg.norm((proj - norm).reshape(-1, 2), axis=1) * float(intr["K"][0, 0])
    return rvec, tvec, float(np.sqrt(np.mean(err**2)))


def _collect(arm, camera, sensor_name, board, obj_pts, intr, gray_code, n_samples):
    """Pair each hand-placed arm pose with a board detection from that pose."""
    import cv2

    samples: list[dict] = []
    while len(samples) < n_samples:
        pose = free_drive_pose(
            arm,
            f"Sample {len(samples) + 1} of {n_samples}: aim the camera at the "
            "board from a new angle. Vary tilt and roll, not just distance — "
            "poses that differ only by a translation leave the rotation "
            "unobservable and the solve degenerate.",
        )
        for _ in range(_FLUSH_FRAMES):
            frame = camera.read_sensors(ImageData, [sensor_name])[sensor_name]["value"]
            time.sleep(1.0 / max(float(camera.frequency), 1.0))

        corners = find_chessboard(cv2.cvtColor(frame, gray_code), board)
        if corners is None:
            print("  board not found in this frame; reposition and retry.")
            continue
        solved = _board_in_camera(corners, obj_pts, intr)
        if solved is None:
            print("  PnP failed on this detection; reposition and retry.")
            continue

        rvec, tvec, rms_px = solved
        tx_world2cam = np.eye(4)
        tx_world2cam[:3, :3] = cv2.Rodrigues(rvec)[0]
        tx_world2cam[:3, 3] = np.asarray(tvec).squeeze()
        samples.append(
            {
                # OpenCV's base2gripper: the base expressed in TCP coordinates.
                "tx_base2tcp": np.linalg.inv(_pose_to_mat(pose)),
                "tx_world2cam": tx_world2cam,
            }
        )
        print(
            f"  board at {np.linalg.norm(tvec):.3f} m, reprojection "
            f"{rms_px:.2f} px, TCP z {pose[2]:+.4f} m"
        )
    return samples


def _solve(samples) -> tuple[np.ndarray, np.ndarray]:
    """``(tx_base2world, tx_tcp2camera)`` from the pose/board pairs."""
    import cv2

    r_b2w, t_b2w, r_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
        R_world2cam=[s["tx_world2cam"][:3, :3] for s in samples],
        t_world2cam=[s["tx_world2cam"][:3, 3] for s in samples],
        R_base2gripper=[s["tx_base2tcp"][:3, :3] for s in samples],
        t_base2gripper=[s["tx_base2tcp"][:3, 3] for s in samples],
        # UMI's note, kept: the LI method returns wrong results on this problem.
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    )
    tx_base2world = np.eye(4)
    tx_base2world[:3, :3] = r_b2w
    tx_base2world[:3, 3] = np.asarray(t_b2w).squeeze()
    tx_tcp2camera = np.eye(4)
    tx_tcp2camera[:3, :3] = r_g2c
    tx_tcp2camera[:3, 3] = np.asarray(t_g2c).squeeze()
    return tx_base2world, tx_tcp2camera


def _fit_error(samples, tx_base2world, tx_tcp2camera):
    """How far each pose is from ``world->cam == tcp->cam . base->tcp . world->base``.

    The only check on a hand-eye result that needs no second measurement: two
    constant unknowns have to explain every pose pair at once, so a misdetected
    board or a servo that sagged under its own weight shows up as one outlier.
    """
    world2base = np.linalg.inv(tx_base2world)
    trans_mm, rot_deg = [], []
    for s in samples:
        predicted = tx_tcp2camera @ s["tx_base2tcp"] @ world2base
        err = np.linalg.inv(s["tx_world2cam"]) @ predicted
        trans_mm.append(float(np.linalg.norm(err[:3, 3]) * 1000.0))
        rotvec = st.Rotation.from_matrix(err[:3, :3]).as_rotvec()
        rot_deg.append(float(np.degrees(np.linalg.norm(rotvec))))
    return np.asarray(trans_mm), np.asarray(rot_deg)


def _derive(tx_tcp2camera: np.ndarray, ik_tcp) -> dict:
    """Turn the solved transform into the numbers that get acted on."""
    # ik_tcp is written in the flange frame, so the rotation to put on it is
    # the camera's orientation there. UMI trains with the fingertip frame and
    # the camera frame sharing one orientation, which is what makes the
    # camera's rotation the right thing to declare for the TCP.
    tx_camera2flange = _offset_to_mat(ik_tcp) @ np.linalg.inv(tx_tcp2camera)
    tip_in_camera = tx_tcp2camera[:3, 3]
    misalign = st.Rotation.from_matrix(tx_tcp2camera[:3, :3]).as_rotvec()
    return {
        "tip_in_camera": tip_in_camera,
        "tip_delta": UMI_TIP_IN_CAMERA - tip_in_camera,
        "camera_in_flange": tx_camera2flange[:3, 3],
        "quat": rotvec_to_quat(
            st.Rotation.from_matrix(tx_camera2flange[:3, :3]).as_rotvec()
        ),
        "misalign_deg": float(np.degrees(np.linalg.norm(misalign))),
    }


def _report(derived, trans_mm, rot_deg, arm_name, ik_tcp, frame_wh) -> None:
    res = f"{frame_wh[0]}x{frame_wh[1]}"
    print(f"\n=== fit over {trans_mm.size} poses ===")
    print(
        f"  residual {np.sqrt(np.mean(trans_mm**2)):.1f} mm rms "
        f"({trans_mm.max():.1f} mm worst), "
        f"{np.sqrt(np.mean(rot_deg**2)):.2f} deg rms"
    )
    if trans_mm.max() > _OUTLIER_MM:
        print(
            f"  The worst pose is {trans_mm.max():.1f} mm out, which is the "
            "size of the thing being measured. Re-run before acting on any of "
            "this: one bad detection poisons the translation."
        )

    tip, delta = derived["tip_in_camera"], derived["tip_delta"] * 1000.0
    print("\n=== fingertips in camera axes ===")
    print(f"  measured   [{tip[0]:+.4f}, {tip[1]:+.4f}, {tip[2]:+.4f}] m  (from {res} frames)")
    print(
        f"  UMI train  [{UMI_TIP_IN_CAMERA[0]:+.4f}, {UMI_TIP_IN_CAMERA[1]:+.4f}, "
        f"{UMI_TIP_IN_CAMERA[2]:+.4f}] m  (06_generate_dataset_plan.py)"
    )
    print(f"  difference [{delta[0]:+.1f}, {delta[1]:+.1f}, {delta[2]:+.1f}] mm")
    print(
        "\n  This half is hardware. The policy sends translation deltas in tool\n"
        "  axes, and a delta moves the fingertips by the same vector wherever\n"
        "  ik_tcp puts its origin, so no config value absorbs it. Close the gap\n"
        "  by moving one of the two, along the camera's own axes (+Z out of the\n"
        "  lens):\n"
        f"    camera     [{-delta[0]:+.1f}, {-delta[1]:+.1f}, {-delta[2]:+.1f}] mm\n"
        f"    fingertips [{delta[0]:+.1f}, {delta[1]:+.1f}, {delta[2]:+.1f}] mm\n"
        "  Moving the camera is the better of the two: it also makes the fingers\n"
        "  look the right size in frame, and the policy uses its own fingertips\n"
        "  as a metric reference (training and eval both mask with\n"
        "  gripper=True, finger=False, so it can see them)."
    )

    print(f"\n=== {arm_name} ik_tcp rotation ===")
    print(
        f"  the declared TCP axes are {derived['misalign_deg']:.2f} deg off the "
        "camera's"
    )
    if derived["misalign_deg"] < _ROT_NOISE_DEG:
        print(
            f"  Under the {_ROT_NOISE_DEG:g} deg this procedure resolves on {res} "
            "frames. Leave ik_tcp as it is."
        )
    else:
        quat = derived["quat"]
        print(
            "  A policy translation is a delta in these axes, so this angle "
            "skews\n  every command it issues. Set the arm's ik_tcp to:\n"
        )
        pos = ", ".join(f"{float(v):.6f}" for v in ik_tcp.pos)
        print(f"    pos=({pos}),")
        print(
            f"    quat=({quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}),\n"
        )
        print(
            "  pos is unchanged — only the orientation was measured here. "
            "Re-running\n  after the edit should report an angle under the "
            "noise floor."
        )

    cam = derived["camera_in_flange"]
    print(
        f"\n  Cross-check: the camera solved to [{cam[0]:+.4f}, {cam[1]:+.4f}, "
        f"{cam[2]:+.4f}] m in the flange\n  frame. Compare it against the "
        "gripper CAD's camera mount; a large\n  disagreement means the board, "
        "the square size or a pose is wrong, not\n  that the mount is."
    )


@CommandRegistry.register("calibrate-hand-eye", "robot")
class CalibrateHandEyeCommand(BaseCommand):
    help = "Measure camera-to-fingertip (robot-world hand-eye) against what a UMI policy assumes."
    params = ("arm", "board", "square", "samples")

    def handle(self, cfg) -> None:
        import cv2

        square_mm = OmegaConf.select(cfg, "square")
        if square_mm is None:
            print(
                "square=<mm> is required: the side of one printed checkerboard "
                "square, off a ruler. It is the only thing that sets the scale "
                "of the answer, so a 4% printing error is a 4% error in the "
                "measured offset."
            )
            return
        board = parse_board(str(OmegaConf.select(cfg, "board") or "9x6"))
        n_samples = max(int(OmegaConf.select(cfg, "samples") or _SAMPLES), _MIN_SAMPLES)

        arm_name, template = select_ik_module(OmegaConf.select(cfg, "arm"))
        if template.ik_tcp is None:
            print(
                f"{arm_name} declares no ik_tcp, so read_pose() reports the "
                "flange and there is no fingertip frame to measure. Give the "
                "module its tool offset first."
            )
            return
        arm = load_module(arm_name, template)

        camera = pick_module("camera", sensor_cls=Camera)
        if camera is None:
            return
        sensor_name = next(
            name for name, s in camera.sensors.items() if isinstance(s, Camera)
        )
        sensor_cfg = camera.sensors[sensor_name].cfg
        gray_code = (
            cv2.COLOR_RGB2GRAY
            if str(sensor_cfg.color_mode) == "rgb"
            else cv2.COLOR_BGR2GRAY
        )
        obj_pts = _board_points(board, float(square_mm) / 1000.0)
        # Resolved before any hardware opens: a missing remap should fail now,
        # not after the board is taped down and the arm is energised.
        conf_dir = Path(
            os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))
        )
        height, width = camera_output_hw(sensor_cfg, conf_dir)
        frame_wh = (width, height)
        intr = _frame_intrinsics(sensor_cfg, frame_wh, conf_dir)
        if intr is None:
            return

        with module_running(
            camera,
            f"The camera will be opened. Tape the {board[0]}x{board[1]} "
            "checkerboard flat where the arm can see it from several angles, "
            "and do not move it again: it is the fixed world frame of the "
            "solve.",
        ):
            with module_running(
                arm,
                "Torque will be released so you can aim the camera by hand at "
                f"{n_samples} poses, and the arm will drive back to where it "
                "started at the end.\n"
                "Keep the volume clear and keep a hand on the power.",
            ):
                max_speed = arm.cfg.max_speed
                speed = float(
                    max_speed
                    if isinstance(max_speed, (int, float))
                    else max_speed[0]
                )
                with returns_to_start(arm, speed):
                    samples = _collect(
                        arm,
                        camera,
                        sensor_name,
                        board,
                        obj_pts,
                        intr,
                        gray_code,
                        n_samples,
                    )
                    input(
                        "\nTake the board out of the arm's path, then press "
                        "Enter to drive back to the start pose..."
                    )

        tx_base2world, tx_tcp2camera = _solve(samples)
        trans_mm, rot_deg = _fit_error(samples, tx_base2world, tx_tcp2camera)
        derived = _derive(tx_tcp2camera, template.ik_tcp)
        _report(derived, trans_mm, rot_deg, arm_name, template.ik_tcp, frame_wh)

        out_path = conf_dir / "calibrations" / f"{camera.name}_hand_eye.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(
                {
                    # UMI's key and convention (OpenCV's R_g2c / t_g2c) with the
                    # gripper frame being the flange, which is what its own
                    # recorder zeroes the TCP to. This file therefore drops
                    # straight into a UMI eval config's hand_eye_calibration.
                    "tx_gripper2camera": (
                        tx_tcp2camera @ np.linalg.inv(_offset_to_mat(template.ik_tcp))
                    ).tolist(),
                    "tx_base2world": tx_base2world.tolist(),
                    # The same transform from the frame Solarian publishes and
                    # the policy consumes: the flange composed with ik_tcp.
                    "tx_tcp2camera": tx_tcp2camera.tolist(),
                    "ik_tcp": {
                        "pos": [float(v) for v in template.ik_tcp.pos],
                        "quat": [float(v) for v in template.ik_tcp.quat],
                    },
                    "arm": arm_name,
                    "board": {
                        "cols": board[0],
                        "rows": board[1],
                        "square_mm": float(square_mm),
                    },
                    "frame_size": list(frame_wh),
                    "n_samples": len(samples),
                    "fit": {
                        "trans_rms_mm": float(np.sqrt(np.mean(trans_mm**2))),
                        "trans_max_mm": float(trans_mm.max()),
                        "rot_rms_deg": float(np.sqrt(np.mean(rot_deg**2))),
                    },
                    "umi_tip_in_camera": UMI_TIP_IN_CAMERA.tolist(),
                    "tip_in_camera": derived["tip_in_camera"].tolist(),
                    "tip_delta_mm": (derived["tip_delta"] * 1000.0).tolist(),
                    "ik_tcp_quat_measured": derived["quat"].tolist(),
                },
                fp,
                indent=2,
            )
        print(f"\nWrote {out_path}")
