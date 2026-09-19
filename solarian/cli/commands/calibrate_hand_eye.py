"""Measure where this rig's fingertips sit in its own camera frame.

A UMI policy is trained against one number it can never observe: the rigid
transform from the wrist camera to the fingertips. UMI's dataset plan bakes it
into every episode as a constant, so at eval time the fingertips have to sit
where the checkpoint believes they are, in the camera's own axes, or every
command it issues lands displaced by the difference. Nothing in an image
reveals the error -- a frame that looks perfectly aimed is still aimed at the
wrong distance.

Ported from UMI's ``record_robot_world_hand_eye.py`` +
``calibrate_robot_world_hand_eye.py`` pair, keeping their target: the 160 mm
tag on the table marker page they publish for download. Recording and solving
are one command here, because a Solarian module owns its hardware for the
lifetime of a process.

Each pose is sampled with the arm held by its own servos rather than by hand.
Torque comes off to walk the arm into position, then on to hold it while the
hands come away, because the pose the servos hold is the one forward
kinematics describes -- a hand-held arm sags and flexes by degrees, and
degrees at the tag's distance are tens of mm of error. Confirm three times
per pose: once positioned, once let go, once holding the arm again so torque
can come off without dropping it.

A preview window draws the tag outline and the axes solved from it, which is
the check worth watching: an outline only proves a quad with the right ID was
found, while axes lying along the printed edges prove the pose behind them --
the input every number below rests on -- is right.

Nothing is printed unless the fit earns it. Two constants have to explain
every pose pair at once, so the residual is a real verdict, and a run that
cannot resolve the offset it is measuring says so and stops rather than
handing over numbers that look like measurements.

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

    solarian calibrate-hand-eye

Overrides: ``samples=16 tag=13 marker=160.0 arm=right_arm``.
"""

from __future__ import annotations

import json
import os
import textwrap
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
from solarian.core.arm_module import _LATCH_JUMP_TICKS
from solarian.core.camera import Camera
from solarian.core.data import ImageData
from solarian.core.fisheye import load_intrinsics
from solarian.core.optical_remap import resolve_remap_path
from solarian.core.pose import rotvec_to_quat

# UMI's ``scripts_slam_pipeline/06_generate_dataset_plan.py`` freezes the
# camera-to-fingertip transform into every training episode:
#   cam_to_center_height = 0.086             (fixed for the UMI gripper)
#   cam_to_tip_offset    = 0.205 - 0.01465   (finger length minus GoPro mount)
# Optical center is in front of the mount screw, so the mount offset
# shortens the tip distance. Zero rotation, so a released checkpoint
# expects its fingertips here.
UMI_TIP_IN_CAMERA = np.array([0.0, 0.086, 0.19035])

# UMI's published table marker, umi-gripper.github.io/share/table_marker.pdf:
# one 220x220 mm page carrying DICT_4X4_50 tag 13 at 160.0 mm, measured off
# the file itself. Their scripts/gen_aruco_tag_pdf.py emits a different
# artifact -- an eight-page aruco_cubes_letter.pdf of ids 12-19 -- and the
# `--tag_id 12` in their hand-eye script's usage line refers to that one, not
# to the PDF they put behind the download link. The download is what people
# actually print, so it is the default here.
_ARUCO_DICT = "DICT_4X4_50"
_TAG_ID = 13
_MARKER_MM = 160.0
# Four coplanar points admit two poses. The runner-up has to reproject this
# much worse than the winner or the tag has not actually determined which way
# it leans, and picking wrong is an error of tens of degrees, silently. From
# 300 noise trials at 0.3 px through this exact fisheye path: beyond
# _MIN_TILT_DEG the ratio's 5th percentile is 25 and the wrong pose never won,
# while at 5 deg the median is 6-10 and it won up to 42% of the time.
_MIN_AMBIGUITY_RATIO = 10.0
_MIN_TILT_DEG = 20.0

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
# Enter arrives as CR or LF depending on the HighGUI backend; space is there
# because it is the reflex key for "now" in a video window.
_CONFIRM_KEYS = (13, 10, 32)
# Advisory, not enforced: consecutive poses that differ only by a translation
# leave the lever arm unobservable, and the lever arm is the measurement. The
# user is the one looking at the rig, so this warns rather than refuses.
_MIN_ROT_STEP_DEG = 8.0
# Let the arm stop ringing after the hand comes off it, before reading.
_SETTLE_S = 0.4
# The two pose reads bracketing a frame grab should agree to this, or the arm
# was knocked and the frame is paired with a pose it has already left.
_MOVE_POS_M = 0.003
_MOVE_ROT_DEG = 1.0
# Energising onto a correctly latched pose is silent. Anything past this is
# the arm being driven somewhere, and it gets cut short rather than watched.
# Pass marks for the fit. See _grade: these are what it takes to resolve an
# offset of a few tens of mm, not aspirational tolerances.
_MAX_RESIDUAL_MM = 8.0
_MAX_ROT_RMS_DEG = 1.0
# A pose this far outside the median is a wrong detection, not a noisy one.
_OUTLIER_FACTOR = 3.0


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


def _marker_points(marker_m: float) -> np.ndarray:
    """The tag's four corners on z=0, ordered as ArUco reports them.

    Centred on the tag, both because that is cv2.aruco's own convention and
    because SOLVEPNP_IPPE_SQUARE requires exactly this order and centring.
    """
    half = marker_m / 2.0
    return np.array(
        [
            [-half, +half, 0.0],
            [+half, +half, 0.0],
            [+half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _aruco_detector():
    import cv2

    params = cv2.aruco.DetectorParameters()
    # The whole pose rests on four corners, so they are worth refining, and
    # UMI's own detector does the same.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, _ARUCO_DICT)),
        params,
    )


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


def _tag_in_camera(corners, obj_pts, intr):
    """Tag pose in camera axes: ``(rvec, tvec, reprojection_rms_px, ratio)``.

    Undistorting the corners to normalized rays first is what lets the solver
    run with an identity camera matrix, and that is the only way to use it on a
    fisheye frame whose distortion a pinhole model cannot express.

    ``solvePnPGeneric`` is used over ``solvePnP`` for the second solution, not
    the first. A square's four corners fit two poses that mirror through the
    tag plane, and near fronto-parallel both reproject so well that the wrong
    one wins on noise alone. Returning how much worse the runner-up is lets
    the caller refuse a pose the geometry never pinned down, instead of
    averaging a flipped one into the answer.
    """
    import cv2

    norm = cv2.fisheye.undistortPoints(
        corners.reshape(-1, 1, 2).astype(np.float64), intr["K"], intr["D"]
    )
    found, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj_pts, norm, np.eye(3), np.zeros(4), flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not found:
        return None
    errs = np.asarray(errs, dtype=np.float64).ravel()
    order = np.argsort(errs)
    best = int(order[0])
    ratio = (
        float(errs[order[1]] / max(errs[best], 1e-12)) if found > 1 else float("inf")
    )
    rvec = np.ascontiguousarray(np.asarray(rvecs[best], dtype=np.float64).reshape(3, 1))
    tvec = np.ascontiguousarray(np.asarray(tvecs[best], dtype=np.float64).reshape(3, 1))
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, np.eye(3), np.zeros(4))
    # Scaled by fx so the residual reads in the pixels the corners came from.
    err = np.linalg.norm((proj - norm).reshape(-1, 2), axis=1) * float(intr["K"][0, 0])
    return rvec, tvec, float(np.sqrt(np.mean(err**2))), ratio


def _rot_between_deg(pose_a, pose_b) -> float:
    """Angle between the orientations of two 6D poses, in degrees."""
    rot_a = st.Rotation.from_rotvec(np.asarray(pose_a, dtype=np.float64)[3:6])
    rot_b = st.Rotation.from_rotvec(np.asarray(pose_b, dtype=np.float64)[3:6])
    return float(np.degrees(np.linalg.norm((rot_a.inv() * rot_b).as_rotvec())))


def _draw_tag(view, corners, solved, obj_pts, intr, k: float) -> None:
    """Draw the tag outline, then the solved tag frame on top of it.

    Everything is already in full-frame pixels; *k* only fattens the strokes so
    they stay visible at whatever resolution the frame is, which is why this
    draws its own outline rather than calling ``drawDetectedMarkers``, whose
    stroke width vanishes on a 1920-wide frame.

    The axes are the check that matters. The outline only says a quad carrying
    the right ID was found; red X and green Y lying along the printed edges,
    with blue Z standing out of the paper, say the pose behind them is right --
    and that pose is what every number below is built on.
    """
    import cv2

    if corners is None:
        return
    path = corners.reshape(-1, 2).astype(np.int32)
    cv2.polylines(view, [path], True, (0, 0, 0), round(4 * k))
    cv2.polylines(view, [path], True, (0, 255, 255), max(1, round(1.5 * k)))
    for point in path:
        cv2.circle(view, tuple(point), round(3 * k), (0, 0, 255), -1)
    if solved is None:
        return
    rvec, tvec, _rms, _ratio = solved
    length = float(obj_pts[:, 0].max())
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


def _detect(gray, detector, tag_id: int, gate_wh):
    """``(corners, ids_in_view)`` -- the wanted tag, and whatever else is there.

    Marker detection thresholds and walks every contour in the image, and on a
    frame with no tag in it that whole sweep is spent proving the absence.
    Most frames while the arm is being aimed are exactly that, so the small
    copy answers "is it there" cheaply and only a hit pays full resolution for
    the corners the measurement needs.

    The ids come back too because there is more than one 160 mm table tag in
    circulation -- the published page and the generated eight-page sheet do
    not carry the same one. "Not found" and "you are pointing at a different
    tag" are the same frame to the detector and completely different problems
    to the user, and the gate has already paid for the distinction.
    """
    import cv2

    small = cv2.resize(gray, gate_wh, interpolation=cv2.INTER_AREA)
    _corners, ids, _rejected = detector.detectMarkers(small)
    seen = [] if ids is None else sorted({int(i) for i in ids.ravel()})
    if tag_id not in seen:
        return None, seen
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is not None:
        for quad, found in zip(corners, ids.ravel()):
            if int(found) == tag_id:
                return quad.reshape(4, 2).astype(np.float64), seen
    return None, seen


def _missing_tag_reason(tag_id: int, seen) -> str:
    """Why there is no pose, phrased so the next action is obvious."""
    others = [i for i in seen if i != tag_id]
    if not others:
        return f"tag {tag_id} is not in this frame"
    names = ", ".join(str(i) for i in others)
    return (
        f"tag {tag_id} is not in this frame, but tag {names} is -- same "
        f"160 mm table marker, different id, so re-run with tag={others[0]}"
    )


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


def _shell_line() -> str | None:
    """The line waiting on stdin, stripped, or None when none is waiting.

    The preview window is the obvious place to press Enter, but the shell is
    where the instruction was read, so both are accepted. This has to poll
    rather than block: an input() call would freeze the window on a stale
    frame, which is the one thing the preview must not do.
    """
    import select
    import sys

    try:
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        raw = sys.stdin.readline()
    except (OSError, ValueError):  # not a selectable stream
        return None
    # A closed stdin selects as ready forever, and an empty read is the only
    # thing that distinguishes it from a bare Enter.
    return raw.strip() if raw else None


def _measure(
    arm, camera, sensor_name, detector, tag_id, obj_pts, intr, color_mode, gate_wh
):
    """One sample, on dedicated reads, or ``(None, reason)`` saying why not.

    The pose is read on both sides of the frame grab. With the servos holding
    and the hands away the two should agree; if they do not, the arm was
    knocked or is still creeping, and a frame paired with a pose the arm has
    already left is exactly the corrupt sample the fit cannot distinguish from
    a bad rig.
    """
    import cv2

    pose_before = arm.read_pose()
    frame = camera.read_sensors(ImageData, [sensor_name])[sensor_name]["value"]
    pose_after = arm.read_pose()

    drift_m = float(np.linalg.norm(pose_after[:3] - pose_before[:3]))
    drift_deg = _rot_between_deg(pose_before, pose_after)
    if drift_m > _MOVE_POS_M or drift_deg > _MOVE_ROT_DEG:
        return None, (
            f"the arm moved {drift_m * 1000:.0f} mm / {drift_deg:.1f} deg while "
            "measuring"
        )

    full = _opencv_bgr(frame, color_mode)
    corners, seen = _detect(
        cv2.cvtColor(full, cv2.COLOR_BGR2GRAY), detector, tag_id, gate_wh
    )
    if corners is None:
        return None, _missing_tag_reason(tag_id, seen)
    solved = _tag_in_camera(corners, obj_pts, intr)
    if solved is None:
        return None, "the tag corners did not solve to a pose"
    rvec, tvec, rms_px, ratio = solved
    if ratio < _MIN_AMBIGUITY_RATIO:
        return None, (
            f"the tag is too flat-on to tell which way it leans (ambiguity "
            f"{ratio:.1f}, {_MIN_AMBIGUITY_RATIO:g} needed) -- tilt it past "
            f"{_MIN_TILT_DEG:g} deg"
        )

    tx_world2cam = np.eye(4)
    tx_world2cam[:3, :3] = cv2.Rodrigues(rvec)[0]
    tx_world2cam[:3, 3] = np.asarray(tvec).squeeze()
    return {
        # OpenCV's base2gripper: the base in TCP coordinates.
        "tx_base2tcp": np.linalg.inv(_pose_to_mat(pose_after)),
        "tx_world2cam": tx_world2cam,
        "pose": pose_after,
        "rms_px": rms_px,
        "ratio": ratio,
        "tag_m": float(np.linalg.norm(tvec)),
    }, None


def _collect(
    arm,
    camera,
    sensor_name,
    detector,
    tag_id,
    obj_pts,
    intr,
    color_mode,
    n_samples,
    recapture=None,
):
    """Pair arm poses with tag detections, three confirmations per pose.

    The cycle is walk, hold, let go, measure, take the arm, release. Torque
    stays on after the sample so the arm does not drop before a hand is on
    it. It comes on while the hand is still there so a jerk is caught, and
    the sample is taken only after the hand is away so the pose belongs to
    the servos rather than to the grip.

    *recapture* is the 1-based pose numbers being replaced after a failed
    fit; the prompts name those instead of counting from one. The window
    shows the full frame; only the detection gate works on a downscaled
    copy, and that is for latency, not for display.
    """
    import cv2

    # Off the intrinsics rather than off a frame, so the overlay can never end
    # up scaled against a different resolution than the one being measured.
    frame_w, frame_h = int(intr["DIM"][0]), int(intr["DIM"][1])
    gate_scale = min(1.0, _GATE_W / frame_w)
    gate_wh = (round(frame_w * gate_scale), round(frame_h * gate_scale))
    overlay_k = frame_w / _OVERLAY_REF_W

    # initialize() energised the arm on the way in, so this is what makes it
    # movable again -- it is not redundant with whatever state it was left in.
    arm.start_calibration_mode()
    holding = False
    releasing = False

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
    window_help = (
        f"\n\nThe window shows the full {frame_w}x{frame_h} frame, opened "
        f"at {window_wh[0]}x{window_wh[1]} -- drag it\nlarger to inspect "
        "focus. ENTER confirms and q finishes early, from\neither the "
        "window or this shell."
        if preview
        else "\n\nNo display, so no preview window."
    )
    if recapture:
        names = ", ".join(str(i) for i in recapture)
        print(
            f"\nRecapturing pose{'s' if len(recapture) != 1 else ''} {names}. "
            "Same three confirmations:\n  hold, let go, take the arm."
            + window_help
        )
    else:
        print(
            f"\nThree confirmations per pose, {n_samples} poses.\n"
            "  1. Torque is off: walk the arm until the tag is in view, tilted at "
            f"least\n     {_MIN_TILT_DEG:g} deg off flat-on, then confirm. The arm "
            "energises and holds.\n"
            "  2. Let go of the arm, then confirm again. The sample is taken with "
            "the\n     servos holding it, not your hand.\n"
            "  3. Take hold of the arm, then confirm. Torque comes off so you can "
            "walk\n     it to the next pose.\n"
            f"Vary tilt and roll by {_MIN_ROT_STEP_DEG:g} deg or more between "
            "poses: poses that differ\nonly by a translation leave the lever arm "
            "being measured unobservable."
            + window_help
        )

    res_label = f"frame {frame_w}x{frame_h}, fx {intr['K'][0, 0]:.0f} px"
    samples: list[dict] = []
    flash_mono = 0.0
    try:
        while len(samples) < n_samples or releasing:
            # A blocking prompt has to be answered before the frame is grabbed,
            # or the status would describe a pose the user has already left.
            # The preview is already grabbing continuously, so there the
            # keypress marks a live frame and the order is the other way round.
            if not preview:
                if releasing:
                    prompt = (
                        "  hold the arm, then Enter to release it "
                        "(q to finish): "
                    )
                elif holding:
                    prompt = (
                        "  let go of the arm, then Enter to measure "
                        "(q to finish): "
                    )
                else:
                    which = (
                        recapture[len(samples)]
                        if recapture is not None
                        else f"{len(samples) + 1}/{n_samples}"
                    )
                    prompt = (
                        f"Recapture pose {which}: position the arm,"
                        if recapture is not None
                        else f"Pose {which}: position the arm,"
                    ) + " then Enter to hold it (q to finish): "
                if input(prompt).strip().lower() == "q":
                    break

            confirmed = not preview
            if preview:
                frame = camera.read_sensors(ImageData, [sensor_name])[sensor_name][
                    "value"
                ]
                full = _opencv_bgr(frame, color_mode)
                corners, seen = _detect(
                    cv2.cvtColor(full, cv2.COLOR_BGR2GRAY), detector, tag_id, gate_wh
                )
                solved = (
                    _tag_in_camera(corners, obj_pts, intr)
                    if corners is not None
                    else None
                )
                pose = arm.read_pose()
                rot_step = (
                    _rot_between_deg(samples[-1]["pose"], pose) if samples else None
                )
                now = time.monotonic()

                # Copied, not drawn on in place: for a BGR camera _opencv_bgr
                # hands back the bus's own frame buffer, and the overlay must
                # not end up in whatever the bus serves next.
                view = full.copy()
                _draw_tag(view, corners, solved, obj_pts, intr, overlay_k)
                if solved is None and [i for i in seen if i != tag_id]:
                    other = [i for i in seen if i != tag_id][0]
                    detection = (
                        f"that is tag {other}, not {tag_id} -- restart with "
                        f"tag={other}",
                        (0, 165, 255),
                    )
                elif solved is None:
                    detection = (f"tag {tag_id} not found", (0, 0, 255))
                elif solved[3] < _MIN_AMBIGUITY_RATIO:
                    detection = (
                        f"too flat-on: ambiguity {solved[3]:.1f} of "
                        f"{_MIN_AMBIGUITY_RATIO:g} -- tilt it more",
                        (0, 165, 255),
                    )
                else:
                    detection = (
                        f"tag {np.linalg.norm(solved[1]):.2f} m, "
                        f"{solved[2]:.2f} px, ambiguity {solved[3]:.0f}",
                        (0, 255, 0),
                    )
                if now - flash_mono < 0.6:
                    turn = ("SAMPLE TAKEN", (0, 255, 255))
                elif rot_step is None:
                    turn = ("first pose, any angle", (200, 200, 200))
                elif rot_step < _MIN_ROT_STEP_DEG:
                    turn = (
                        f"only {rot_step:.0f} deg from the last, "
                        f"{_MIN_ROT_STEP_DEG:g} wanted",
                        (0, 165, 255),
                    )
                else:
                    turn = (f"{rot_step:.0f} deg from the last", (0, 255, 0))
                _draw_status(
                    view,
                    [
                        (f"pose {len(samples)} / {n_samples}", (255, 255, 255)),
                        detection,
                        turn,
                        (
                            (
                                "HOLD THE ARM, then ENTER to release",
                                (0, 255, 255),
                            )
                            if releasing
                            else (
                                "HOLDING - let go, then ENTER to measure",
                                (0, 255, 255),
                            )
                            if holding
                            else (
                                "ENTER to hold this pose   q finish",
                                (200, 200, 200),
                            )
                        ),
                        # Hershey fonts are ASCII only, so no typographic dots.
                        (res_label, (200, 200, 200)),
                    ],
                    overlay_k,
                )
                key = -1
                try:
                    cv2.imshow(_CAM_WINDOW, view)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error:
                    preview = False
                line = _shell_line()
                if key == ord("q") or (line is not None and line.lower() == "q"):
                    break
                confirmed = key in _CONFIRM_KEYS or line is not None

            if not confirmed:
                continue

            if releasing:
                arm.disable_torque()
                holding = False
                releasing = False
                if len(samples) < n_samples:
                    print("  limp. Walk it to the next pose.")
                continue

            if not holding:
                # Latch before energising, and energise while the hand is still
                # on the arm: both halves of not being surprised by this.
                try:
                    latched = arm.enable_torque_holding()
                except RuntimeError as exc:
                    print(f"  {exc}")
                    continue
                holding = True
                time.sleep(_SETTLE_S)
                # wrist_roll spins about the tool axis, so a full turn leaves
                # TCP xyz where it started and a Cartesian jump check cannot
                # see it. The joints can.
                if arm.joint_command_delta(latched) > _LATCH_JUMP_TICKS:
                    arm.disable_torque()
                    holding = False
                    print(
                        "  a joint moved on energising, so torque is back off "
                        "and nothing was recorded. The latched goal did not "
                        "match the arm -- often wrist_roll taking the long "
                        "way around its wrap. Re-seat this pose and try "
                        "again; if it repeats, the joint reads are unreliable."
                    )
                    continue
                print("  holding. Let go of the arm, then confirm to measure.")
                continue

            time.sleep(_SETTLE_S)
            sample, reason = _measure(
                arm,
                camera,
                sensor_name,
                detector,
                tag_id,
                obj_pts,
                intr,
                color_mode,
                gate_wh,
            )
            if sample is None:
                print(f"  nothing recorded: {reason}.")
            else:
                samples.append(sample)
                flash_mono = time.monotonic()
                print(
                    f"  sample {len(samples)}/{n_samples}: tag at "
                    f"{sample['tag_m']:.3f} m, reprojection {sample['rms_px']:.2f} px, "
                    f"ambiguity {sample['ratio']:.0f}, TCP z {sample['pose'][2]:+.4f} m"
                )
            # Still energised: the arm must not go limp until a hand is on it.
            releasing = True
            print("  hold the arm, then confirm to release it.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if preview:
            try:
                cv2.destroyWindow(_CAM_WINDOW)
            except cv2.error:
                pass
        # end_calibration_mode does not touch torque, so leaving mid-hold would
        # hand back an energised arm. The user is holding it and will put it
        # down herself, so it has to come back limp however this exits.
        arm.disable_torque()
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


def _grade(trans_mm, rot_deg) -> dict:
    """Whether the fit resolved anything, and the numbers behind that call.

    The limits come from what the answer is for. The offset being measured is
    a few tens of mm, and the solve averages the poses, so a residual at
    ``_MAX_RESIDUAL_MM`` over ``_MIN_SAMPLES`` poses still lands the offset
    inside a few mm -- tight enough to act on. Looser than that and the
    correction is smaller than its own error bar.
    """
    n = int(trans_mm.size)
    trans_rms = float(np.sqrt(np.mean(trans_mm**2)))
    rot_rms = float(np.sqrt(np.mean(rot_deg**2)))
    checks = (
        (
            "translation residual",
            f"{trans_rms:.1f} mm rms",
            f"{_MAX_RESIDUAL_MM:.1f} mm",
            trans_rms <= _MAX_RESIDUAL_MM,
        ),
        (
            "rotation residual",
            f"{rot_rms:.2f} deg rms",
            f"{_MAX_ROT_RMS_DEG:.2f} deg",
            rot_rms <= _MAX_ROT_RMS_DEG,
        ),
        ("poses", f"{n}", f"{_MIN_SAMPLES} min", n >= _MIN_SAMPLES),
    )
    median = float(np.median(trans_mm))
    return {
        "n": n,
        "trans_rms": trans_rms,
        "trans_max": float(trans_mm.max()),
        "rot_rms": rot_rms,
        # What the poses disagree by, divided down by how many agreed: the
        # error bar on the offsets printed below.
        "stderr_mm": trans_rms / np.sqrt(max(n, 1)),
        "rot_floor_deg": 3.0 * rot_rms / np.sqrt(max(n, 1)),
        "checks": checks,
        "passed": all(ok for *_, ok in checks),
        "outliers": [
            i
            for i, mm in enumerate(trans_mm)
            if mm > _OUTLIER_FACTOR * median and mm > _MAX_RESIDUAL_MM
        ],
    }


def _banner(text: str) -> None:
    rule = "=" * 68
    print(f"\n{rule}\n{text}\n{rule}")


def _para(text: str, indent: str = "  ") -> None:
    """Wrap a paragraph that carries interpolated numbers.

    The fixed prose below is hand-wrapped, but a paragraph whose width depends
    on how many digits a residual came out to cannot be, and an unwrapped one
    is unreadable in the terminal this is read in.
    """
    print(
        textwrap.fill(
            text, width=76, initial_indent=indent, subsequent_indent=indent
        )
    )


def _print_checks(grade) -> None:
    for label, value, limit, ok in grade["checks"]:
        print(f"  {label:<22}{value:<16}{limit:<12}{'ok' if ok else 'FAIL'}")


def _diagnose(samples, trans_mm, grade) -> None:
    """Name which of the two failure shapes this is; they need opposite fixes.

    A couple of poses far outside the rest is a misdetection, and re-running
    fixes it. Error spread evenly with the rotation residual over its limit is
    the arm, and re-running the same way reproduces it exactly.
    """
    if grade["outliers"]:
        print(
            f"\n  Suspect poses, residual over {_OUTLIER_FACTOR:g}x the median:"
        )
        for i in grade["outliers"]:
            print(
                f"    pose {i + 1}: {trans_mm[i]:.1f} mm, "
                f"{samples[i]['rms_px']:.2f} px reprojection, "
                f"tag at {samples[i]['tag_m']:.2f} m"
            )
        keep = [s for i, s in enumerate(samples) if i not in grade["outliers"]]
        if len(keep) >= _MIN_SAMPLES:
            regrade = _grade(*_fit_error(keep, *_solve(keep)))
            if regrade["passed"]:
                _para(
                    f"Without them the remaining {len(keep)} poses do fit, at "
                    f"{regrade['trans_rms']:.1f} mm rms. So the rig is fine and "
                    "these are bad detections: recapture them, and watch the "
                    "axes sit square on the tag before confirming each sample."
                )
                return
            _para(
                f"Dropping them still leaves {regrade['trans_rms']:.1f} mm rms, "
                "so they are not the whole story. Recapture them, and name any "
                "other poses whose residual looks just as wrong."
            )

    if grade["rot_rms"] > _MAX_ROT_RMS_DEG:
        tag_m = float(np.mean([s["tag_m"] for s in samples]))
        implied = 1000.0 * tag_m * np.sin(np.radians(grade["rot_rms"]))
        print()
        _para(
            "The error is spread across the poses, and the rotation residual "
            f"is {grade['rot_rms']:.2f} deg. Only the rotation chain feeds "
            f"that number, and a {_MARKER_MM:.0f} mm tag gives its own "
            "orientation to a fraction of a degree at this range, so what is "
            "left is the arm: its forward kinematics does not know the tool "
            "orientation this well."
        )
        print()
        _para(
            f"At the tag's {tag_m:.2f} m, {grade['rot_rms']:.2f} deg is "
            f"{implied:.0f} mm, which is the translation residual above -- one "
            "cause, whole budget. Recapturing the same way will reproduce it; "
            "what changes it is how the arm is held while sampling."
        )


def _default_replacements(trans_mm, grade) -> list[int]:
    """Poses to recapture if the user just confirms: outliers, else the worst."""
    if grade["outliers"]:
        return list(grade["outliers"])
    return [int(np.argmax(trans_mm))]


def _parse_replacements(line: str, n: int, default: list[int]) -> list[int] | None:
    """Enter keeps *default*, pose numbers pick a set, q gives up.

    Returns 0-based indices, or None when the user wants to stop. Raises
    ValueError on tokens that are not pose numbers in ``1..n``.
    """
    text = line.strip().lower()
    if text in ("q", "quit", "n", "no"):
        return None
    if text in ("", "y", "yes"):
        return list(default)
    indices: list[int] = []
    for tok in text.replace(",", " ").split():
        if not tok.isdigit():
            raise ValueError(f"{tok!r} is not a pose number")
        i = int(tok)
        if i < 1 or i > n:
            raise ValueError(f"pose {i} is not in 1..{n}")
        if i - 1 not in indices:
            indices.append(i - 1)
    if not indices:
        raise ValueError("name at least one pose")
    return indices


def _ask_replacements(samples, trans_mm, grade) -> list[int] | None:
    """Offer to substitute the poison poses instead of starting over."""
    default = _default_replacements(trans_mm, grade)
    named = ", ".join(
        f"{i + 1} ({trans_mm[i]:.1f} mm)" for i in default
    )
    kind = "poisoning the fit" if grade["outliers"] else "the worst residual"
    noun = "Pose" if len(default) == 1 else "Poses"
    verb = "it" if len(default) == 1 else "them"
    print(
        f"\n  {noun} {named} {'is' if len(default) == 1 else 'are'} {kind}."
    )
    print(
        f"  Enter to recapture {verb}, pose numbers (e.g. 1 9 12) to pick "
        "others, or q to stop."
    )
    while True:
        try:
            return _parse_replacements(
                input("  recapture: "), len(samples), default
            )
        except ValueError as exc:
            print(
                f"  {exc}. Pose numbers, Enter for the suspect ones, or q "
                "to stop."
            )


def _report_fail(samples, trans_mm, grade) -> None:
    _banner("CALIBRATION FAILED -- DO NOT USE THESE NUMBERS.")
    _print_checks(grade)
    print(
        "\n  No offsets are printed, deliberately. A fit this loose cannot "
        "resolve\n  the tens of mm it exists to measure, so any number here "
        "would be noise\n  wearing the costume of a measurement."
    )
    _diagnose(samples, trans_mm, grade)


def _report_pass(derived, grade, arm_name, ik_tcp, frame_wh) -> None:
    res = f"{frame_wh[0]}x{frame_wh[1]}"
    _banner("CALIBRATION PASSED")
    _print_checks(grade)
    print()
    _para(
        f"Offsets below carry about {grade['stderr_mm']:.1f} mm of error, from "
        f"{grade['trans_rms']:.1f} mm rms over {grade['n']} poses."
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
        "  ik_tcp puts its origin, so no config value absorbs it."
    )

    rotate = derived["misalign_deg"] > grade["rot_floor_deg"]
    print(f"\n=== {arm_name} ik_tcp rotation ===")
    _para(
        f"the declared TCP axes are {derived['misalign_deg']:.2f} deg off the "
        f"camera's, against a {grade['rot_floor_deg']:.2f} deg floor for this "
        "fit"
    )
    if not rotate:
        print(
            "  That is inside what this run can resolve, so it is not a "
            "measurement of\n  anything. Leave ik_tcp alone."
        )

    print("\n=== what to do with this ===")
    step = 1
    print(
        f"  {step}. Move the camera "
        f"[{-delta[0]:+.1f}, {-delta[1]:+.1f}, {-delta[2]:+.1f}] mm along its own\n"
        "     axes (+Z out of the lens). Moving the camera beats moving the\n"
        "     fingertips: it also makes the fingers look the right size in\n"
        "     frame, and the policy uses its own fingertips as a metric\n"
        "     reference (training and eval both mask with gripper=True,\n"
        "     finger=False, so it can see them)."
    )
    if rotate:
        step += 1
        quat = derived["quat"]
        pos = ", ".join(f"{float(v):.6f}" for v in ik_tcp.pos)
        print(
            f"  {step}. Set {arm_name}'s ik_tcp to the orientation below. A policy\n"
            "     translation is a delta in these axes, so this angle skews every\n"
            "     command it issues. pos is unchanged; only orientation was\n"
            "     measured here.\n\n"
            f"       pos=({pos}),\n"
            f"       quat=({quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}),"
        )
    step += 1
    print(
        f"  {step}. Re-run this command. The edit landed if the difference above\n"
        f"     drops under about {max(grade['stderr_mm'] * 3.0, 5.0):.0f} mm and the "
        "rotation under its floor. If it\n"
        "     does not move, the change did not reach the running config."
    )

    cam = derived["camera_in_flange"]
    print(
        f"\n  Cross-check before cutting anything: the camera solved to\n"
        f"  [{cam[0]:+.4f}, {cam[1]:+.4f}, {cam[2]:+.4f}] m in the flange frame. "
        "Compare that against the\n  gripper CAD's camera mount. A disagreement "
        "much larger than the error\n  above means the tag, its printed size or "
        "a pose is wrong, not the mount."
    )


def _write_failed_record(
    path, arm_name, tag_id, marker_mm, frame_wh, samples, trans_mm, rot_deg, grade
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(
            {
                "passed": False,
                "arm": arm_name,
                "tag": {"id": tag_id, "marker_mm": marker_mm},
                "frame_size": list(frame_wh),
                "fit": {
                    "trans_rms_mm": grade["trans_rms"],
                    "trans_max_mm": grade["trans_max"],
                    "rot_rms_deg": grade["rot_rms"],
                    "limits": {
                        "trans_rms_mm": _MAX_RESIDUAL_MM,
                        "rot_rms_deg": _MAX_ROT_RMS_DEG,
                    },
                },
                "poses": [
                    {
                        "pose": [float(v) for v in s["pose"]],
                        "tx_world2cam": s["tx_world2cam"].tolist(),
                        "residual_mm": float(trans_mm[i]),
                        "residual_deg": float(rot_deg[i]),
                        "rms_px": s["rms_px"],
                        "ambiguity": s["ratio"],
                        "tag_m": s["tag_m"],
                    }
                    for i, s in enumerate(samples)
                ],
            },
            fp,
            indent=2,
        )
    print(f"\nWrote the per-pose record to {path}")


@CommandRegistry.register("calibrate-hand-eye", "robot")
class CalibrateHandEyeCommand(BaseCommand):
    help = "Measure camera-to-fingertip (robot-world hand-eye) against what a UMI policy assumes."
    params = ("arm", "tag", "marker", "samples")

    def handle(self, cfg) -> None:
        tag_id = int(OmegaConf.select(cfg, "tag") or _TAG_ID)
        marker_mm = float(OmegaConf.select(cfg, "marker") or _MARKER_MM)
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
        obj_pts = _marker_points(marker_mm / 1000.0)
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
            f"Tape UMI's table marker tag {tag_id} flat where the arm can "
            "see it from\nseveral angles, and do not move it again: it is the "
            "fixed world frame of\nthe solve.\n"
            "  umi-gripper.github.io/share/table_marker.pdf\n"
            "Print it at 100 percent, never fit-to-page: the sheet is 220 mm "
            "wide so it\noverflows A4, but the outer margin is blank and clips "
            "harmlessly. Then\nmeasure the black square edge to edge -- it has "
            f"to be {marker_mm:.0f} mm, or pass\nmarker=<mm> with what it "
            "actually came out as. That number alone sets the\nscale of the "
            "answer, so a 4 percent printing error is a 4 percent error in "
            "the\nmeasured offset.\n"
            "Connecting the arm energises it, so keep the volume clear and a "
            "hand on\nthe power. It goes limp as soon as the run starts.",
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

            # The prompt above already covered both modules; a second one here
            # would just be a keystroke between the user and the same decision.
            detector = _aruco_detector()
            color_mode = str(sensor_cfg.color_mode)
            fail_path = (
                conf_dir / "calibrations" / f"{camera.name}_hand_eye_FAILED.json"
            )
            with module_running(arm, "Connecting the arm...", confirm=False):
                samples = _collect(
                    arm,
                    camera,
                    sensor_name,
                    detector,
                    tag_id,
                    obj_pts,
                    intr,
                    color_mode,
                    n_samples,
                )
                while True:
                    if len(samples) < _MIN_SAMPLES:
                        print(
                            f"\nOnly {len(samples)} samples; the translation "
                            f"needs at least {_MIN_SAMPLES} to come out of the "
                            "geometry rather than out of the corner noise. "
                            "Nothing solved."
                        )
                        return

                    tx_base2world, tx_tcp2camera = _solve(samples)
                    trans_mm, rot_deg = _fit_error(
                        samples, tx_base2world, tx_tcp2camera
                    )
                    grade = _grade(trans_mm, rot_deg)
                    if grade["passed"]:
                        break

                    _report_fail(samples, trans_mm, grade)
                    replace = _ask_replacements(samples, trans_mm, grade)
                    if replace is None:
                        # Under a name nothing loads: the numbers in it are
                        # the ones just refused.
                        _write_failed_record(
                            fail_path,
                            arm_name,
                            tag_id,
                            marker_mm,
                            frame_wh,
                            samples,
                            trans_mm,
                            rot_deg,
                            grade,
                        )
                        return

                    extra = _collect(
                        arm,
                        camera,
                        sensor_name,
                        detector,
                        tag_id,
                        obj_pts,
                        intr,
                        color_mode,
                        len(replace),
                        recapture=[i + 1 for i in replace],
                    )
                    if not extra:
                        _write_failed_record(
                            fail_path,
                            arm_name,
                            tag_id,
                            marker_mm,
                            frame_wh,
                            samples,
                            trans_mm,
                            rot_deg,
                            grade,
                        )
                        return
                    for src, dst in zip(extra, replace):
                        samples[dst] = src
                    print(
                        "  substituted pose "
                        + ", ".join(str(i + 1) for i in replace[: len(extra)])
                        + ". Re-solving."
                    )

                derived = _derive(tx_tcp2camera, template.ik_tcp)
                _report_pass(derived, grade, arm_name, template.ik_tcp, frame_wh)

        out_path = conf_dir / "calibrations" / f"{camera.name}_hand_eye.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(
                {
                    "passed": True,
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
                    "tag": {"id": tag_id, "marker_mm": marker_mm},
                    "frame_size": list(frame_wh),
                    "n_samples": len(samples),
                    "fit": {
                        "trans_rms_mm": grade["trans_rms"],
                        "trans_max_mm": grade["trans_max"],
                        "rot_rms_deg": grade["rot_rms"],
                        "stderr_mm": grade["stderr_mm"],
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
