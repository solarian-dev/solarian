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

Torque is released once and left off for good, so the arm walks from pose to
pose in one continuous motion and stays in the user's hands at the end. That
leaves no hand free for a keypress, so samples are taken when the arm settles.
A preview window draws the detected corners and the board axes solved from
them, which is the check worth watching: corners only prove the detector found
a grid, while axes standing square on the board prove the pose behind them —
the input every number below rests on — is right.

The camera's ``resize_*`` is dropped for the run so the board is measured on
the remap's full 1920x1080 rather than the 640x360 the policy consumes. Corner
precision scales with focal length in pixels, and the downscale throws away a
factor of three of it for nothing.

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
    _opencv_bgr,
    module_running,
    pick_module,
)
from solarian.cli.commands._ik_common import load_module, select_ik_module
from solarian.core.camera import Camera
from solarian.core.data import ImageData
from solarian.core.fisheye import find_chessboard, load_intrinsics, parse_board
from solarian.core.optical_remap import resolve_remap_path
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
_CAM_WINDOW = "calibrate-hand-eye"
# Width the cheap detection gate runs at; see _detect. Nothing to do with what
# the window shows, which is the full frame.
_GATE_W = 640
# Width the window opens at. The image in it is full resolution -- this only
# stops a 1920-wide frame opening fullscreen on a Pi screen, and the window is
# WINDOW_NORMAL so it can be dragged larger to inspect detail.
_WINDOW_W = 960
# Overlay geometry is authored against a 640-wide view and scaled from there,
# so text and axes stay the same apparent size whatever the frame is.
_OVERLAY_REF_W = 640
# Torque stays off for the whole session, so the arm has to be held the whole
# time and a keypress per sample would mean letting go of it. Samples trigger
# on the arm sitting inside this much jitter for this long instead.
_STILL_S = 0.6
_STILL_POS_M = 0.005
_STILL_ROT_DEG = 2.5
# Consecutive samples must differ in rotation: a translation-only pair leaves
# the lever arm unobservable, and the lever arm is the measurement.
_MIN_ROT_STEP_DEG = 8.0
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


def _remap_gopro_schema(sensor_cfg, conf_dir: Path):
    """The fisheye schema describing the frames this camera emits.

    Frames leave the bus already warped into GoPro optics by
    ``finish_camera_frame``, so the GoPro model out of the remap JSON is what
    describes them. Without a remap there is nothing to measure against: the
    point of the exercise is the pose in the frame the checkpoint consumes.

    Returned as an unscaled schema rather than as intrinsics, because the
    resolution it has to be rescaled to is whatever the camera turns out to
    emit, and that is not known until a frame has been read.
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
        return json.load(fp)["gopro_intrinsics"]


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


def _rot_between_deg(pose_a, pose_b) -> float:
    """Angle between the orientations of two 6D poses, in degrees."""
    rot_a = st.Rotation.from_rotvec(np.asarray(pose_a, dtype=np.float64)[3:6])
    rot_b = st.Rotation.from_rotvec(np.asarray(pose_b, dtype=np.float64)[3:6])
    return float(np.degrees(np.linalg.norm((rot_a.inv() * rot_b).as_rotvec())))


def _draw_board(view, corners, solved, obj_pts, intr, k: float) -> None:
    """Draw the corner path, then the solved board frame on top of it.

    Everything is already in full-frame pixels; *k* only fattens the strokes so
    they stay visible at whatever resolution the frame is. That is also why the
    corner path is drawn here rather than by ``cv2.drawChessboardCorners``,
    whose marker size is hardcoded and vanishes on a 1920-wide frame.

    The axes are the check that matters. The path only says the detector found
    something grid-shaped and walked it in order; red X and green Y lying along
    the rows and columns, with blue Z square to them, say the pose behind them
    is right -- and that pose is what every number below is built on.
    """
    import cv2

    if corners is None:
        return
    path = corners.reshape(-1, 2).astype(np.int32)
    cv2.polylines(view, [path], False, (0, 0, 0), round(4 * k))
    cv2.polylines(view, [path], False, (0, 255, 255), max(1, round(1.5 * k)))
    for point in path:
        cv2.circle(view, tuple(point), round(3 * k), (0, 0, 255), -1)
    if solved is None:
        return
    rvec, tvec, _rms = solved
    length = float(obj_pts[:, 0].max()) / 2.0
    tips = np.array(
        [[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]],
        dtype=np.float64,
    )
    pts, _ = cv2.fisheye.projectPoints(
        tips.reshape(-1, 1, 3),
        np.ascontiguousarray(np.asarray(rvec, dtype=np.float64).reshape(3, 1)),
        np.ascontiguousarray(np.asarray(tvec, dtype=np.float64).reshape(3, 1)),
        intr["K"],
        intr["D"],
    )
    pts = pts.reshape(-1, 2).astype(int)
    for tip, color in zip(pts[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(view, tuple(pts[0]), tuple(tip), (0, 0, 0), round(5 * k))
        cv2.line(view, tuple(pts[0]), tuple(tip), color, round(3 * k))
    cv2.circle(view, tuple(pts[0]), round(5 * k), (255, 255, 255), -1)


def _detect(gray, board, gate_wh):
    """Board corners at full resolution, gated by a cheap look at a small copy.

    ``findChessboardCorners`` is fast when it finds a board and slow when it
    does not, because CALIB_CB_FAST_CHECK scans the whole image before giving
    up -- at 1920x1080 that is ~50 ms against ~6 ms for the same miss at
    640x360. Most frames while the arm is being aimed are misses, so paying for
    them at full resolution is what would make the preview crawl.
    """
    import cv2

    small = cv2.resize(gray, gate_wh, interpolation=cv2.INTER_AREA)
    if find_chessboard(small, board) is None:
        return None
    return find_chessboard(gray, board)


def _draw_status(view, lines, k: float) -> None:
    """Stack ``(text, colour)`` lines top-left, each over a dark outline.

    *k* scales the type with the frame, so the same overlay reads the same on a
    640-wide view and on a 1920-wide one.
    """
    import cv2

    for row, (text, color) in enumerate(lines):
        origin = (round(10 * k), round((24 + row * 22) * k))
        cv2.putText(
            view, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k, (0, 0, 0),
            round(3 * k),
        )
        cv2.putText(
            view, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k, color,
            max(1, round(k)),
        )


def _collect(arm, camera, sensor_name, board, obj_pts, intr, color_mode, n_samples):
    """Pair hand-placed arm poses with board detections, torque off throughout.

    Torque comes off once and never goes back on, so the arm can be walked from
    pose to pose in one continuous motion and is left in the user's hands at the
    end. That leaves no hand free to press anything, so a sample is taken when
    the arm settles rather than on a keypress, and the preview window shows what
    the detector and PnP are making of the board while it is being aimed.

    The window shows the full frame; only the cheap detection gate works on a
    downscaled copy, and that is for latency, not for display.
    """
    import cv2

    # Off the intrinsics rather than off a frame, so the overlay can never end
    # up scaled against a different resolution than the one being measured.
    frame_w, frame_h = int(intr["DIM"][0]), int(intr["DIM"][1])
    gate_scale = min(1.0, _GATE_W / frame_w)
    gate_wh = (round(frame_w * gate_scale), round(frame_h * gate_scale))
    overlay_k = frame_w / _OVERLAY_REF_W

    print(
        f"\nAim the camera at the board and hold still: a sample is taken once "
        f"the arm settles and the axes lock onto the board. Rotate at least "
        f"{_MIN_ROT_STEP_DEG:g} deg between samples — tilt and roll, not just "
        "distance, or the solve cannot see the lever arm it is measuring."
    )
    input("Take hold of the arm, then press Enter to release torque...")
    arm.start_calibration_mode()

    preview = True
    window_wh = (
        round(frame_w * min(1.0, _WINDOW_W / frame_w)),
        round(frame_h * min(1.0, _WINDOW_W / frame_w)),
    )
    try:
        cv2.namedWindow(_CAM_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_CAM_WINDOW, *window_wh)
    except cv2.error:
        preview = False
    print(
        "Torque is OFF and stays off. Keep hold of the arm: it is yours to put "
        "down at the end, nothing will energise or drive it home."
        + (
            f"\nThe window shows the full {frame_w}x{frame_h} frame, opened at "
            f"{window_wh[0]}x{window_wh[1]} -- drag it larger to inspect "
            "focus.\n"
            "SPACE takes a sample regardless of the gates, q finishes early."
            if preview
            else "\nNo display, so no preview window: samples are taken on "
            "stillness alone. Ctrl+C to stop."
        )
    )

    res_label = f"frame {frame_w}x{frame_h}, fx {intr['K'][0, 0]:.0f} px"
    samples: list[dict] = []
    anchor = arm.read_pose()
    anchor_mono = time.monotonic()
    flash_mono = 0.0
    report_mono = 0.0
    try:
        while len(samples) < n_samples:
            frame = camera.read_sensors(ImageData, [sensor_name])[sensor_name]["value"]
            pose = arm.read_pose()
            full = _opencv_bgr(frame, color_mode)
            corners = _detect(
                cv2.cvtColor(full, cv2.COLOR_BGR2GRAY), board, gate_wh
            )
            solved = (
                _board_in_camera(corners, obj_pts, intr)
                if corners is not None
                else None
            )

            now = time.monotonic()
            if (
                np.linalg.norm(pose[:3] - anchor[:3]) > _STILL_POS_M
                or _rot_between_deg(anchor, pose) > _STILL_ROT_DEG
            ):
                anchor, anchor_mono = pose, now
            still_for = now - anchor_mono
            rot_step = (
                _rot_between_deg(samples[-1]["pose"], pose) if samples else np.inf
            )

            key = -1
            if preview:
                # Copied, not drawn on in place: for a BGR camera _opencv_bgr
                # hands back the bus's own frame buffer, and the overlay must
                # not end up in whatever the bus serves next.
                view = full.copy()
                _draw_board(view, corners, solved, obj_pts, intr, overlay_k)
                if solved is None:
                    detection = ("board not found", (0, 0, 255))
                else:
                    detection = (
                        f"board {np.linalg.norm(solved[1]):.2f} m, "
                        f"{solved[2]:.2f} px reprojection",
                        (0, 255, 0),
                    )
                if now - flash_mono < 0.6:
                    gate = ("SAMPLE TAKEN", (0, 255, 255))
                elif solved is None:
                    gate = ("waiting for the board", (0, 165, 255))
                elif rot_step < _MIN_ROT_STEP_DEG:
                    gate = (
                        f"rotate more: {rot_step:.0f} of "
                        f"{_MIN_ROT_STEP_DEG:g} deg",
                        (0, 165, 255),
                    )
                else:
                    gate = (
                        f"hold still {min(still_for, _STILL_S):.1f}/"
                        f"{_STILL_S:g} s",
                        (0, 255, 255),
                    )
                _draw_status(
                    view,
                    [
                        (f"sample {len(samples)} / {n_samples}", (255, 255, 255)),
                        detection,
                        gate,
                        ("SPACE force   q finish", (200, 200, 200)),
                        # Hershey fonts are ASCII only, so no typographic dots.
                        (res_label, (200, 200, 200)),
                    ],
                    overlay_k,
                )
                try:
                    cv2.imshow(_CAM_WINDOW, view)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error:
                    preview = False
            elif now - report_mono > 1.0:
                report_mono = now
                found = "board found" if solved is not None else "no board"
                turned = "first" if not samples else f"rotated {rot_step:.0f} deg"
                print(
                    f"  {len(samples)}/{n_samples} samples, {found}, "
                    f"still {still_for:.1f} s, {turned}"
                )

            if key == ord("q"):
                break
            # SPACE skips the stillness and rotation gates but never the
            # detection: there is nothing to record without a board pose.
            if solved is None or not (
                key == ord(" ")
                or (still_for >= _STILL_S and rot_step >= _MIN_ROT_STEP_DEG)
            ):
                continue

            rvec, tvec, rms_px = solved
            tx_world2cam = np.eye(4)
            tx_world2cam[:3, :3] = cv2.Rodrigues(rvec)[0]
            tx_world2cam[:3, 3] = np.asarray(tvec).squeeze()
            samples.append(
                {
                    # OpenCV's base2gripper: the base in TCP coordinates.
                    "tx_base2tcp": np.linalg.inv(_pose_to_mat(pose)),
                    "tx_world2cam": tx_world2cam,
                    "pose": pose,
                }
            )
            flash_mono = now
            print(
                f"  sample {len(samples)}/{n_samples}: board at "
                f"{np.linalg.norm(tvec):.3f} m, reprojection {rms_px:.2f} px, "
                f"TCP z {pose[2]:+.4f} m"
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if preview:
            try:
                cv2.destroyWindow(_CAM_WINDOW)
            except cv2.error:
                pass
        # Deliberately no enable_torque: the user is holding the arm and will
        # put it down herself. Energising here would either fight her grip or
        # snap to the pre-freedrive goal position still latched in the servos.
        arm.end_calibration_mode()
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
        obj_pts = _board_points(board, float(square_mm) / 1000.0)
        # Resolved before any hardware opens: a missing remap should fail now,
        # not after the board is taped down and the arm is in the user's hands.
        conf_dir = Path(
            os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))
        )
        gopro_schema = _remap_gopro_schema(sensor_cfg, conf_dir)
        if gopro_schema is None:
            return

        # Corner precision scales with focal length in pixels, and resize_* is
        # set to the 640x360 the policy wants, where the GoPro model is fx~141
        # and a quarter-pixel of corner noise is already 0.1 deg. Dropping the
        # resize leaves the remap's own 1920x1080, three times that. The bus
        # reads these two fields per frame, so clearing them is enough.
        sensor_cfg.resize_width = None
        sensor_cfg.resize_height = None

        with module_running(
            camera,
            f"The camera will be opened. Tape the {board[0]}x{board[1]} "
            "checkerboard flat where the arm can see it from several angles, "
            "and do not move it again: it is the fixed world frame of the "
            "solve.",
        ):
            # Intrinsics come from the frame that actually arrived, never from
            # what the config predicted: intrinsics describing a different
            # resolution than the image they are used on would not fail, it
            # would quietly return a plausible and wrong answer.
            frame = camera.read_sensors(ImageData, [sensor_name])[sensor_name]["value"]
            frame_wh = (frame.shape[1], frame.shape[0])
            intr = load_intrinsics(gopro_schema, frame_wh)
            print(
                f"\nMeasuring on {frame_wh[0]}x{frame_wh[1]} frames "
                f"(fx {intr['K'][0, 0]:.0f} px)."
            )

            with module_running(
                arm,
                "Torque will be released so you can aim the camera by hand at "
                f"{n_samples} poses, and it stays released: the arm is yours to "
                "hold throughout and to put down at the end.\n"
                "Keep the volume clear and keep a hand on the power.",
            ):
                samples = _collect(
                    arm,
                    camera,
                    sensor_name,
                    board,
                    obj_pts,
                    intr,
                    str(sensor_cfg.color_mode),
                    n_samples,
                )

        if len(samples) < _MIN_SAMPLES:
            print(
                f"\nOnly {len(samples)} samples; the translation needs at least "
                f"{_MIN_SAMPLES} to come out of the geometry rather than out of "
                "the corner noise. Nothing solved."
            )
            return

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
