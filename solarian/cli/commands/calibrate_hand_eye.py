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

Nothing is printed unless the fit earns it, and what it has to earn is an
error bar: the answer is resolved n times over, each time with one pose left
out, and it passes when dropping any single pose barely moves it. The
per-pose residual is reported but does not decide, because it measures how
much the poses disagree rather than how well they agree on the answer -- it
sits wherever the arm and the tag put it however many poses are taken, so
grading on it can ask for better poses forever without ever getting them.

The camera's ``resize_*`` is dropped for the run, and where the remap is a
pure rotation the warp is dropped too and the tag is measured on the raw
sensor frame, then rotated into policy axes afterwards. Corner precision in
angle scales with focal length in pixels: the 640x360 the policy consumes has
fx~141, the warped 1080p has 424, and the sensor's own 2028x1520 has 719.
Each pose averages several frames on top of that, because corner jitter is
independent between frames and the arm is standing still.

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
from solarian.core.optical_remap import load_policy_remap, resolve_remap_path
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
# Bursts to try before giving a pose back to the user. A wrist still ringing
# or still creeping onto its holding position clears in well under a second,
# and waiting it out costs a fraction of what re-walking the arm does.
_SETTLE_ATTEMPTS = 4
# The two pose reads bracketing a frame grab should agree to this, or the arm
# was knocked and the frame is paired with a pose it has already left.
_MOVE_POS_M = 0.003
_MOVE_ROT_DEG = 1.0
# Frames averaged into one sample. Corner jitter is independent frame to
# frame, so this divides it by three, and at 30 fps it costs a third of a
# second against a pose that already took three confirmations to set up.
_FRAMES_PER_SAMPLE = 9
# Corner wander across those frames that means something is moving rather
# than something is noisy. Sensor noise on a subpixel-refined corner is
# tenths of a pixel; whole pixels are the rig being knocked or the exposure
# being too long to freeze what it is looking at.
_MAX_JITTER_PX = 2.0
# The verdict. The offsets this command prints are acted on by moving a
# camera mount, so what has to be small is how far the answer would move if
# the poses had been slightly different ones -- see _uncertainty. A few mm of
# error bar against a correction of tens of mm is a correction worth making.
_MAX_TIP_STDERR_MM = 3.0
# Ceilings, not pass marks. The per-pose residual is dominated by how well the
# arm knows its own tool orientation, which is a property of the rig and not
# of how carefully a run is done: on an SO-106 the tag alone spends half of a
# 1 deg budget before the arm contributes anything. These are set where a
# residual stops meaning "this rig is noisy" and starts meaning "something is
# wrong" -- a mismeasured marker, a moved board, the wrong tag.
_MAX_RESIDUAL_MM = 25.0
_MAX_ROT_RMS_DEG = 3.0
# A pose this far outside the median is a wrong detection, not a noisy one.
# The floor keeps the multiple from firing on a tight fit, where three times
# the median can still be a perfectly ordinary pose.
_OUTLIER_FACTOR = 3.0
_OUTLIER_FLOOR_MM = 8.0
# Rounds of "add more poses" before the command stops asking. A backstop
# only: what actually ends the loop is the error bar failing to respond, see
# _bar_is_responding. A round cap on its own would cut off a run that is
# converging perfectly well, just from further out than the cap assumed.
_MAX_TOPUP_ROUNDS = 8
# Poses to add per round: the shortfall the projection asks for, held
# between these. The floor stops a pointless one-pose round, and the ceiling
# keeps re-solving often enough that the projection is re-made on the way --
# it is only an extrapolation, and it is usually pessimistic.
_TOPUP_SAMPLES = 4
_MAX_TOPUP_BATCH = 12
# How far above the 1/sqrt(n) projection the error bar may sit before more
# poses are declared useless. Slack, because the bar is itself an estimate
# and wanders a few tenths between rounds.
_BAR_PROGRESS_SLACK = 1.5
# Translation residual as a multiple of the arc the rotation residual
# subtends at the tag, past which the translation is not orientation error
# seen from a distance. At 1.0 the translation error equals the whole of
# that arc, which cannot happen by coincidence: the fit absorbs part of the
# rotation, so simulating this rig across 0-4 percent of marker scale error
# and 0.3-1.0 deg of arm error never puts the ratio above 0.76. Scale is
# then the first suspect, because after orientation the marker's printed
# size is the only thing in the problem that sets a distance.
_SCALE_SUSPECT_FACTOR = 1.0


def _pose_to_mat(pose) -> np.ndarray:
    """``(x, y, z, rx, ry, rz)`` metres + rotvec -> 4x4."""
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    mat = np.eye(4)
    mat[:3, :3] = st.Rotation.from_rotvec(pose[3:6]).as_matrix()
    mat[:3, 3] = pose[:3]
    return mat


def _mat_to_pose(mat: np.ndarray) -> np.ndarray:
    """4x4 -> ``(x, y, z, rx, ry, rz)``, the inverse of :func:`_pose_to_mat`.

    Nine rotation numbers constrained to six degrees of freedom cannot be
    handed to a least-squares solver, which would walk them off the manifold.
    A rotvec is the minimal parameterisation the solver can step in freely.
    """
    return np.concatenate(
        [mat[:3, 3], st.Rotation.from_matrix(mat[:3, :3]).as_rotvec()]
    )


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


def _measurement_optics(sensor_cfg, conf_dir: Path):
    """Where to measure the tag, as ``(schema, tx_cam2policy, on_raw)``.

    The answer has to end up in the axes the checkpoint thinks in, but it
    does not have to be measured there. A pose is a rigid transform, and
    carrying one between two camera frames costs exactly the rotation
    between them -- which the remap already knows, exactly, because it is
    the remap.

    So when the remap is a pure rotation, the sensor's own frame is the
    better place to measure and the result is rotated home afterwards. It is
    sharper, by the ratio of the focal lengths: an SO-106 wrist camera reads
    719 px of focal length across its native 2028x1520, against 424 once
    warped onto the GoPro's much wider field at 1080p, and corner precision
    in angle scales with exactly that. It also skips a resampling pass, and
    with it the part of the warp that the fitted lens model got slightly
    wrong -- a bias that moves as the tag moves around the frame, which is
    indistinguishable from the rig disagreeing with itself.

    A homography remap is not a rotation: it does not preserve the origin,
    so no rigid transform carries a pose across it and the measurement has
    to happen in the warped frame after all.

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
        data = json.load(fp)
    camera_schema = data.get("camera_intrinsics")
    if data.get("model") != "rotation" or camera_schema is None:
        return data["gopro_intrinsics"], np.eye(4), False
    # The remap sends a ray of the sensor to R @ ray in policy axes, so the
    # policy frame is the sensor frame turned by R and a pose crosses over
    # by left-multiplication.
    tx_cam2policy = np.eye(4)
    tx_cam2policy[:3, :3] = np.asarray(data["transform"], dtype=np.float64)
    return camera_schema, tx_cam2policy, True


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


def _read_frame(camera, sensor_name, warp):
    """One frame, in the optics this measurement is being made in.

    The capture path does not warp any more. A homography remap is not a
    rotation, so it cannot be carried in ``tx_cam2policy`` and the tag has
    to be found in the warped frame -- which means warping it here.
    """
    frame = camera.read_sensors(ImageData, [sensor_name])[sensor_name]["value"]
    return frame if warp is None else warp.apply(frame)


def _measure(
    arm,
    camera,
    sensor_name,
    detector,
    tag_id,
    obj_pts,
    intr,
    color_mode,
    gate_wh,
    tx_cam2policy,
    warp,
):
    """One sample, waiting out an arm that has not finished settling.

    Two of the ways :func:`_measure_once` refuses are not verdicts on the
    pose, only on the moment: a wrist still ringing from being let go, or
    still creeping as the servos take up the load. Both pass on their own in
    under a second. Returning them to the caller costs the user the whole
    three-confirmation cycle and a walk back to the same pose, for something
    that waiting fixes -- so the wait happens here.

    Everything else -- no tag, too flat-on, corners that will not solve --
    is about where the arm is pointed, and comes straight back.
    """
    for attempt in range(_SETTLE_ATTEMPTS):
        sample, reason, settling = _measure_once(
            arm,
            camera,
            sensor_name,
            detector,
            tag_id,
            obj_pts,
            intr,
            color_mode,
            gate_wh,
            tx_cam2policy,
            warp,
        )
        if sample is not None or not settling:
            return sample, reason
        if attempt < _SETTLE_ATTEMPTS - 1:
            print(f"  {reason}; waiting for it to settle.")
            time.sleep(_SETTLE_S)
    return None, reason


def _measure_once(
    arm,
    camera,
    sensor_name,
    detector,
    tag_id,
    obj_pts,
    intr,
    color_mode,
    gate_wh,
    tx_cam2policy,
    warp,
):
    """One attempt, as ``(sample, reason, settling)``.

    *settling* marks the refusals that are worth simply waiting out, which
    is the only reason this is split from :func:`_measure`.

    The whole sample rests on four corner positions, and where a corner lands
    moves from frame to frame with sensor noise alone. That jitter is
    independent between frames while the corner underneath is not, so
    averaging the detections of several frames divides it down by the root of
    how many were taken -- the cheapest precision in the command, bought with
    a third of a second per pose. The arm is being held still by its servos
    and the user's hands are away, so there is nothing the extra frames can
    smear.

    The spread across those frames is kept as ``jitter_px`` and reported, so
    the assumption is visible rather than trusted: static error does not
    average away, and a jitter of zero would say the averaging is buying
    nothing.

    The pose is read on both sides of the frame grabs, and averaged. With the
    servos holding and the hands away the two reads should agree; if they do
    not, the arm was knocked or is still creeping, and a frame paired with a
    pose the arm has already left is exactly the corrupt sample the fit
    cannot distinguish from a bad rig.
    """
    import cv2

    pose_before = arm.read_pose()
    detections, seen = [], []
    for _ in range(_FRAMES_PER_SAMPLE):
        frame = _read_frame(camera, sensor_name, warp)
        corners, seen = _detect(
            cv2.cvtColor(_opencv_bgr(frame, color_mode), cv2.COLOR_BGR2GRAY),
            detector,
            tag_id,
            gate_wh,
        )
        if corners is not None:
            detections.append(corners)
    pose_after = arm.read_pose()

    drift_m = float(np.linalg.norm(pose_after[:3] - pose_before[:3]))
    drift_deg = _rot_between_deg(pose_before, pose_after)
    if drift_m > _MOVE_POS_M or drift_deg > _MOVE_ROT_DEG:
        return None, (
            f"the arm moved {drift_m * 1000:.0f} mm / {drift_deg:.1f} deg while "
            "measuring"
        ), True

    if not detections:
        return None, _missing_tag_reason(tag_id, seen), False
    # A tag that only shows up in some frames is a tag on the edge of being
    # detectable, and the frames that did land are the ones flattered by
    # noise. That is not a pose worth averaging.
    if len(detections) < _FRAMES_PER_SAMPLE:
        return None, (
            f"the tag was only found in {len(detections)} of "
            f"{_FRAMES_PER_SAMPLE} frames, so it is barely detectable here -- "
            "move closer, or light the tag better"
        ), False

    stack = np.stack(detections)
    corners = stack.mean(axis=0)
    # Distance of the worst corner from where it sat on average: one number
    # for how much the four of them wandered while the arm stood still.
    jitter_px = float(np.linalg.norm(stack - corners, axis=2).max())
    if jitter_px > _MAX_JITTER_PX:
        return None, (
            f"the corners moved {jitter_px:.1f} px between frames "
            f"({_MAX_JITTER_PX:g} allowed)"
        ), True

    solved = _tag_in_camera(corners, obj_pts, intr)
    if solved is None:
        return None, "the tag corners did not solve to a pose", False
    rvec, tvec, rms_px, ratio = solved
    if ratio < _MIN_AMBIGUITY_RATIO:
        return None, (
            f"the tag is too flat-on to tell which way it leans (ambiguity "
            f"{ratio:.1f}, {_MIN_AMBIGUITY_RATIO:g} needed) -- tilt it past "
            f"{_MIN_TILT_DEG:g} deg"
        ), False

    # Both reads describe an arm that has not moved, so averaging them halves
    # what the servo reads themselves contribute. Through scipy rather than
    # componentwise: two rotvecs for the same orientation can sit at opposite
    # ends of the pi wrap, where their arithmetic mean is a right angle away
    # from both.
    pose = np.concatenate(
        [
            0.5 * (pose_before[:3] + pose_after[:3]),
            st.Rotation.from_rotvec(
                np.stack([pose_before[3:6], pose_after[3:6]])
            )
            .mean()
            .as_rotvec(),
        ]
    )
    tx_world2cam = np.eye(4)
    tx_world2cam[:3, :3] = cv2.Rodrigues(rvec)[0]
    tx_world2cam[:3, 3] = np.asarray(tvec).squeeze()
    return {
        # OpenCV's base2gripper: the base in TCP coordinates.
        "tx_base2tcp": np.linalg.inv(_pose_to_mat(pose)),
        # Measured in whatever frame the sensor emits, reported in the one
        # the policy consumes. Identity whenever those are the same frame.
        "tx_world2cam": tx_cam2policy @ tx_world2cam,
        "pose": pose,
        "rms_px": rms_px,
        "ratio": ratio,
        "jitter_px": jitter_px,
        "tag_m": float(np.linalg.norm(tvec)),
    }, None, False


def _collect(
    arm,
    camera,
    sensor_name,
    detector,
    tag_id,
    obj_pts,
    intr,
    color_mode,
    tx_cam2policy,
    n_samples,
    warp,
    first_pose=1,
):
    """Pair arm poses with tag detections, three confirmations per pose.

    The cycle is walk, hold, let go, measure, take the arm, release. Torque
    stays on after the sample so the arm does not drop before a hand is on
    it. It comes on while the hand is still there so a jerk is caught, and
    the sample is taken only after the hand is away so the pose belongs to
    the servos rather than to the grip.

    *first_pose* is the 1-based number to call the first pose of this batch,
    so a top-up round counts on from the poses already taken rather than
    restarting at one. The window shows the full frame; only the detection
    gate works on a downscaled copy, and that is for latency, not display.
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
    if first_pose > 1:
        last = first_pose + n_samples - 1
        print(
            f"\n{n_samples} more pose{'s' if n_samples != 1 else ''}, "
            f"numbered {first_pose} to {last}. Same three confirmations:\n"
            "  hold, let go, take the arm. Angles you have not used yet are "
            "worth the\n  most here -- a pose repeating one already taken "
            "barely moves the answer."
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
                    prompt = (
                        f"Pose {first_pose + len(samples)} "
                        f"({len(samples) + 1}/{n_samples} this round): "
                        "position the arm, then Enter to hold it "
                        "(q to finish): "
                    )
                if input(prompt).strip().lower() == "q":
                    break

            confirmed = not preview
            if preview:
                frame = _read_frame(camera, sensor_name, warp)
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
                        (
                            f"pose {first_pose + len(samples)}  "
                            f"({len(samples)} / {n_samples} this round)",
                            (255, 255, 255),
                        ),
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
                tx_cam2policy,
                warp,
            )
            if sample is None:
                print(f"  nothing recorded: {reason}.")
            else:
                samples.append(sample)
                flash_mono = time.monotonic()
                print(
                    f"  pose {first_pose + len(samples) - 1} "
                    f"({len(samples)}/{n_samples} this round): tag at "
                    f"{sample['tag_m']:.3f} m, reprojection "
                    f"{sample['rms_px']:.2f} px, ambiguity "
                    f"{sample['ratio']:.0f}, corners +/-{sample['jitter_px']:.2f} "
                    f"px, TCP z {sample['pose'][2]:+.4f} m"
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
    """``(tx_base2world, tx_tcp2camera)`` from the pose/board pairs.

    Closed form for a starting point, then :func:`_refine` for the answer.
    The two are not separable at the call sites -- nothing wants SHAH's
    unrefined output -- so they are one step here.
    """
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
    return _refine(samples, tx_base2world, tx_tcp2camera)


def _refine(samples, tx_base2world, tx_tcp2camera):
    """Descend the residual both transforms are graded on, from SHAH's answer.

    ``calibrateRobotWorldHandEye`` is closed form: SHAH recovers the two
    rotations from the rotation parts alone, then the translations follow
    linearly once those are fixed. Neither step ever looks at the quantity
    :func:`_fit_error` reports, and an error in the rotations is not allowed
    to be paid for by the translations, so the pair it returns sits near the
    minimum rather than at it. On this rig that gap is most of the answer's
    error -- descending it halves the distance to the true fingertip offset.

    Levenberg-Marquardt is monotone from a starting point this close, so the
    result cannot be worse than what went in.
    """
    from scipy.optimize import least_squares

    # Radians and metres only become comparable through a lever arm, and the
    # tag's own distance is the honest one: it is how far an orientation error
    # actually displaces the thing the camera is looking at.
    lever = float(np.mean([s["tag_m"] for s in samples]))

    def residual(x):
        world2base = np.linalg.inv(_pose_to_mat(x[:6]))
        tcp2camera = _pose_to_mat(x[6:])
        out = []
        for s in samples:
            err = np.linalg.inv(s["tx_world2cam"]) @ (
                tcp2camera @ s["tx_base2tcp"] @ world2base
            )
            rotvec = st.Rotation.from_matrix(err[:3, :3]).as_rotvec()
            out.append(np.concatenate([err[:3, 3], rotvec * lever]))
        return np.concatenate(out)

    fit = least_squares(
        residual,
        np.concatenate(
            [_mat_to_pose(tx_base2world), _mat_to_pose(tx_tcp2camera)]
        ),
        method="lm",
        xtol=1e-14,
        ftol=1e-14,
    )
    return _pose_to_mat(fit.x[:6]), _pose_to_mat(fit.x[6:])


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


def _uncertainty(samples) -> dict:
    """Error bars on the answer itself, by leaving each pose out in turn.

    The per-pose residual is a measure of scatter, and scatter does not shrink
    because more poses were taken -- a rig that disagrees with itself by a
    degree keeps doing so at any number of poses. What does shrink is the
    error on the average, and that is the number the offsets below are acted
    on with, so it is the number worth gating.

    The jackknife measures it without assuming anything about where the
    disagreement comes from: resolve the whole rig n times, each time without
    one pose, and see how far the answer moves. If dropping any single pose
    barely moves it, the poses agree about the answer even when they disagree
    about the residual.

    This measures precision, not accuracy. Everything the poses share -- a
    marker printed at the wrong size above all -- moves every leave-one-out
    solve the same way and cancels out of the spread entirely. Hence the
    residual ceilings in :func:`_grade` and the print-scale check there.
    """
    tips, rotvecs = [], []
    for i in range(len(samples)):
        keep = [s for j, s in enumerate(samples) if j != i]
        _tx_base2world, tx_tcp2camera = _solve(keep)
        tips.append(tx_tcp2camera[:3, 3])
        rotvecs.append(
            st.Rotation.from_matrix(tx_tcp2camera[:3, :3]).as_rotvec()
        )

    def stderr(values):
        # The jackknife's (n-1)/n, not 1/(n-1): leave-one-out estimates differ
        # from each other far less than independent samples would, and this is
        # the factor that undoes that.
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        return np.sqrt(
            (n - 1) / n * np.sum((values - values.mean(axis=0)) ** 2, axis=0)
        )

    tip_mm = stderr(tips) * 1000.0
    rot_deg = np.degrees(stderr(rotvecs))
    return {
        # Worst axis rather than the norm: the offset is acted on one axis at
        # a time, by moving a mount.
        "tip_stderr_mm": float(tip_mm.max()),
        "tip_stderr_axes_mm": tip_mm,
        # Three sigma, so an orientation clearing it is one this fit actually
        # resolved rather than one it happened to land on.
        "rot_floor_deg": float(3.0 * rot_deg.max()),
    }


def _grade(trans_mm, rot_deg, uncertainty) -> dict:
    """Whether the fit resolved anything, and the numbers behind that call.

    The verdict is the error bar on the answer, because that is what decides
    whether the printed offset is worth acting on: the correction is tens of
    mm and ``_MAX_TIP_STDERR_MM`` is a few, so a passing fit knows the mount
    is wrong by much more than it knows itself to be uncertain.

    The residual limits are still here but only as ceilings on a gross
    problem. They cannot be the verdict, because the residual measures how
    much the poses disagree per pose, and no amount of care or extra poses
    drives that below whatever the arm and the tag together produce. Grading
    on it is what made a failed run ask for replacement poses forever.
    """
    n = int(trans_mm.size)
    trans_rms = float(np.sqrt(np.mean(trans_mm**2)))
    rot_rms = float(np.sqrt(np.mean(rot_deg**2)))
    checks = (
        (
            "offset error bar",
            f"{uncertainty['tip_stderr_mm']:.1f} mm",
            f"{_MAX_TIP_STDERR_MM:.1f} mm",
            uncertainty["tip_stderr_mm"] <= _MAX_TIP_STDERR_MM,
        ),
        ("poses", f"{n}", f"{_MIN_SAMPLES} min", n >= _MIN_SAMPLES),
        (
            "translation residual",
            f"{trans_rms:.1f} mm rms",
            f"{_MAX_RESIDUAL_MM:.0f} mm max",
            trans_rms <= _MAX_RESIDUAL_MM,
        ),
        (
            "rotation residual",
            f"{rot_rms:.2f} deg rms",
            f"{_MAX_ROT_RMS_DEG:.1f} deg max",
            rot_rms <= _MAX_ROT_RMS_DEG,
        ),
    )
    median = float(np.median(trans_mm))
    return {
        "n": n,
        "trans_rms": trans_rms,
        "trans_max": float(trans_mm.max()),
        "rot_rms": rot_rms,
        "stderr_mm": uncertainty["tip_stderr_mm"],
        "stderr_axes_mm": uncertainty["tip_stderr_axes_mm"],
        "rot_floor_deg": uncertainty["rot_floor_deg"],
        "checks": checks,
        "passed": all(ok for *_, ok in checks),
        "outliers": [
            i
            for i, mm in enumerate(trans_mm)
            if mm > _OUTLIER_FACTOR * median and mm > _OUTLIER_FLOOR_MM
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


def _scale_excess(samples, grade) -> float:
    """Translation residual over what the rotation residual can explain.

    Orientation error is the expensive term in this fit: at the tag's
    distance a degree is centimetres, so almost all of a healthy translation
    residual is the rotation residual seen from far away. What that lever
    cannot account for came in some other way, and only one other thing in
    the problem carries metric scale -- the printed size of the marker.
    """
    tag_m = float(np.mean([s["tag_m"] for s in samples]))
    explained = 1000.0 * tag_m * np.sin(np.radians(grade["rot_rms"]))
    return grade["trans_rms"] / max(explained, 1e-9)


def _diagnose(samples, trans_mm, grade) -> None:
    """Name which failure shape this is; they need different fixes.

    A couple of poses far outside the rest is a misdetection, and dropping or
    retaking them fixes it. Error spread evenly is the rig, and retaking the
    same way reproduces it exactly -- what helps there is more poses, because
    the error bar falls with them even though the residual does not. And a
    translation residual the rotation cannot explain is neither: it is a
    scale error, which more poses would only measure more precisely.
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
        if len(keep) < _MIN_SAMPLES:
            _para(
                f"Dropping them would leave {len(keep)} poses, under the "
                f"{_MIN_SAMPLES} the solve needs, so they have been kept and "
                "are still in the numbers above. More poses would make them "
                "droppable as well as narrowing the error bar."
            )
        else:
            regrade = _grade(
                *_fit_error(keep, *_solve(keep)), _uncertainty(keep)
            )
            if regrade["passed"]:
                _para(
                    f"Without them the remaining {len(keep)} poses do fit, at "
                    f"an error bar of {regrade['stderr_mm']:.1f} mm. So the rig "
                    "is fine and these are bad detections."
                )
                return
            _para(
                f"Dropping them still leaves {regrade['stderr_mm']:.1f} mm of "
                "error bar, so they are not the whole story."
            )

    if grade["rot_rms"] > _MAX_ROT_RMS_DEG:
        print()
        _para(
            f"The error is spread across the poses at {grade['rot_rms']:.2f} "
            "deg of rotation residual, which is past what this command treats "
            "as a working rig. Only the rotation chain feeds that number, and "
            f"a {_MARKER_MM:.0f} mm tag gives its own orientation to a "
            "fraction of a degree at this range, so what is left is the arm: "
            "its forward kinematics does not know the tool orientation this "
            "well. Retaking poses the same way reproduces it exactly."
        )
        print()
        _para(
            "What changes it is the arm, not the capture: check that the "
            "servo calibration is current, that nothing in the wrist has play "
            "in it, and that the arm is fully settled and untouched when each "
            "sample is confirmed."
        )

    # Said last and said always, including alongside the causes above: it is
    # the one thing here that more or better poses cannot touch.
    excess = _scale_excess(samples, grade)
    if excess > _SCALE_SUSPECT_FACTOR:
        tag_m = float(np.mean([s["tag_m"] for s in samples]))
        print()
        _para(
            f"Separately, the {grade['trans_rms']:.1f} mm translation "
            f"residual is {excess:.1f}x the arc {grade['rot_rms']:.2f} deg "
            f"subtends at the tag's {tag_m:.2f} m, where a healthy rig comes "
            "in under 0.8x. Orientation does not account for this much, and "
            "after orientation the only thing in the problem that sets a "
            "distance is the marker."
        )
        print()
        _para(
            f"Measure the black square edge to edge and pass marker=<mm> with "
            f"what it actually is, rather than the {_MARKER_MM:.0f} mm assumed "
            "here. Printer scaling of a few percent is normal and it lands "
            "directly on the offset: the answer comes out wrong by the same "
            "percentage, in the same direction, however many poses are taken, "
            "and the error bar above stays tight while it does. Check too "
            "that the board has not shifted since the first pose -- it is the "
            "fixed world frame of the solve."
        )


def _more_poses_would_help(grade) -> bool:
    """Whether this failure is the kind that extra poses fix.

    A wide error bar on a rig inside its ceilings is poses disagreeing at
    random, and the error on their average comes down as more are taken. A
    residual past a ceiling is the rig itself, which every further pose
    reproduces exactly -- asking for poses there is what a loop that never
    terminates is made of.

    A suspected scale error is deliberately not consulted. It biases every
    pose the same way, so it adds no scatter and is not why the bar is wide;
    the two are separate problems and the poses still fix the one they fix.
    Refusing to continue on a suspicion would be the old non-termination in
    a new costume.
    """
    return (
        grade["trans_rms"] <= _MAX_RESIDUAL_MM
        and grade["rot_rms"] <= _MAX_ROT_RMS_DEG
    )


def _projected_poses(grade) -> int:
    """Poses that should bring the error bar inside the limit.

    The jackknife falls as one over the root of the pose count, so the target
    follows from where the fit already is. Reported rather than imposed: it
    tells the user whether this is four more poses or forty, which is the
    difference between finishing the run and rebuilding the rig.
    """
    needed = grade["n"] * (grade["stderr_mm"] / _MAX_TIP_STDERR_MM) ** 2
    return int(np.ceil(needed))


def _bar_is_responding(first, grade) -> bool:
    """Whether the error bar is still coming down as poses are added.

    This is what ends the loop, rather than a count of rounds. A count
    cannot tell the two cases apart: a rig that needs forty poses and is
    walking towards them, and a rig with a systematic floor that would not
    improve at four hundred. Both look like "still failing after three
    rounds", and cutting the first one off is the old non-termination
    complaint turned inside out -- the user is told what to do and then
    stopped from doing it.

    The test is against the fit's own starting point: the bar should track
    one over the root of the pose count from where it began, and a bar
    sitting well above that line is not being held up by a shortage of
    poses.
    """
    expected = first["stderr_mm"] * np.sqrt(first["n"] / grade["n"])
    return grade["stderr_mm"] <= _BAR_PROGRESS_SLACK * expected


def _parse_topup(line: str, default: int) -> int | None:
    """Enter takes *default* more poses, a number takes that many, q stops.

    Returns the count to add, or None to stop. Raises ValueError on anything
    that is not a positive count.
    """
    text = line.strip().lower()
    if text in ("q", "quit", "n", "no"):
        return None
    if text in ("", "y", "yes"):
        return default
    if not text.isdigit():
        raise ValueError(f"{text!r} is not a number of poses")
    count = int(text)
    if count < 1:
        raise ValueError("add at least one pose")
    return count


def _ask_topup(grade) -> int | None:
    """Offer more poses, which is the thing that actually moves the verdict.

    Not a recapture. Every pose here is already as good as this rig makes
    them, and swapping one for another draws a fresh residual from the same
    distribution -- the old prompt did that, and on a rig whose residual sits
    at its own noise floor it never terminated. Adding poses is different:
    the error bar being graded is the error on their average.
    """
    target = _projected_poses(grade)
    shortfall = max(target - grade["n"], 1)
    # Offer what the projection asks for, not a token handful: a default
    # smaller than the stated shortfall would be the command disagreeing
    # with the number it just printed.
    default = int(np.clip(shortfall, _TOPUP_SAMPLES, _MAX_TOPUP_BATCH))
    print()
    _para(
        f"The offsets are not yet known well enough to act on: "
        f"{grade['stderr_mm']:.1f} mm of error bar against a "
        f"{_MAX_TIP_STDERR_MM:.1f} mm limit."
    )
    _para(
        f"More poses is what narrows it, at one over the root of the count. "
        f"From {grade['n']} poses at {grade['stderr_mm']:.1f} mm, roughly "
        f"{target} poses in total should get there -- about {shortfall} "
        "more. That is an extrapolation and usually a pessimistic one, so it "
        "is re-made after each round. Vary tilt and roll between them; poses "
        "that repeat an angle already taken add the least."
    )
    print(
        f"  Enter to add {default}, a number to add that many, or q to stop."
    )
    while True:
        try:
            return _parse_topup(input("  add poses: "), default)
        except ValueError as exc:
            print(f"  {exc}. A count, Enter for {default}, or q to stop.")


def _report_fail(samples, trans_mm, grade) -> None:
    _banner("CALIBRATION FAILED -- DO NOT USE THESE NUMBERS.")
    _print_checks(grade)
    print()
    if grade["stderr_mm"] > _MAX_TIP_STDERR_MM:
        _para(
            "No offsets are printed, deliberately. A fit this loose cannot "
            "resolve the tens of mm it exists to measure, so any number here "
            "would be noise wearing the costume of a measurement."
        )
    else:
        # Precise and still refused: the checks below caught something every
        # pose shares, which is the one thing the error bar cannot see.
        _para(
            "No offsets are printed, deliberately. This fit is repeatable -- "
            "the poses agree with each other -- but agreeing is not the same "
            "as being right, and what they agree on looks wrong for the "
            "reason below. A confident wrong number is worse than none."
        )
    _diagnose(samples, trans_mm, grade)


def _report_pass(samples, derived, grade, arm_name, ik_tcp, frame_wh) -> None:
    res = f"{frame_wh[0]}x{frame_wh[1]}"
    _banner("CALIBRATION PASSED")
    _print_checks(grade)
    print()
    axes = grade["stderr_axes_mm"]
    _para(
        f"Offsets below carry about {grade['stderr_mm']:.1f} mm of error "
        f"([{axes[0]:.1f}, {axes[1]:.1f}, {axes[2]:.1f}] mm per axis), from "
        f"how far the answer moves when any one of the {grade['n']} poses is "
        "left out."
    )
    _para(
        "That is a bound on precision, not on truth. It cannot see anything "
        f"every pose shares: if the marker's black square is not "
        f"{_MARKER_MM:.0f} mm edge to edge, the offsets below are wrong by "
        "that same percentage and this error bar stays just as tight."
    )
    # The one failure a passing fit can be hiding, so it is said here rather
    # than only on the failure path.
    excess = _scale_excess(samples, grade)
    if excess > _SCALE_SUSPECT_FACTOR:
        print()
        _para(
            f"WARNING: and there is a hint of exactly that. The "
            f"{grade['trans_rms']:.1f} mm translation residual is "
            f"{excess:.1f}x the arc {grade['rot_rms']:.2f} deg subtends at "
            "the tag, where a healthy rig comes in under 0.8x. Measure the "
            "marker before moving anything, and re-run with marker=<mm> if "
            "it is not what was assumed."
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
                    "tip_stderr_mm": grade["stderr_mm"],
                    "tip_stderr_axes_mm": [
                        float(v) for v in grade["stderr_axes_mm"]
                    ],
                    "trans_rms_mm": grade["trans_rms"],
                    "trans_max_mm": grade["trans_max"],
                    "rot_rms_deg": grade["rot_rms"],
                    "scale_excess": _scale_excess(samples, grade),
                    "limits": {
                        "tip_stderr_mm": _MAX_TIP_STDERR_MM,
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
                        "jitter_px": s["jitter_px"],
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
    description = """
    Solves the robot-world hand-eye problem from poses of UMI's table marker: where
    the fingertips sit in the wrist camera's own frame. A UMI checkpoint bakes that
    transform into every episode as a constant it can never observe, so the command
    also reports the difference against what the policy assumes.

    Tape the marker flat where the arm can see it from several angles and do not
    move it again: it is the fixed world frame of the solve. Print it at 100
    percent, never fit-to-page, then measure the black square edge to edge and pass
    marker=<mm> if it is not 160 mm. That number sets the scale of the answer.

    The arm energises on connect and then goes limp, so you pose it by hand for
    each sample. Needs an arm with an ik_tcp and a camera whose yaml points at a
    GoPro optical remap (`solarian calibrate-umi-camera`).
    """
    params = {
        "arm": "Name of the IK arm module to measure; prompts when omitted.",
        "tag": "ArUco id of the table marker, DICT_4X4_50 (default: 13).",
        "marker": "Measured side of the marker's black square in mm (default: 160).",
        "samples": "Poses to collect; 8 minimum, more is better (default: 12).",
    }
    examples = (
        "solarian calibrate-hand-eye",
        "solarian calibrate-hand-eye arm=right_arm marker=158.5 samples=16",
    )

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
        optics = _measurement_optics(sensor_cfg, conf_dir)
        if optics is None:
            return
        schema, tx_cam2policy, on_raw = optics

        # Corner precision scales with focal length in pixels, and resize_* is
        # set to the size the policy wants, where the GoPro model is fx~141
        # and a quarter-pixel of corner noise is already 0.1 deg. Dropping the
        # resize leaves the full frame, three times that. The bus reads these
        # two fields per frame, so clearing them is enough.
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
            # A rotation remap is carried in tx_cam2policy instead, where it
            # costs no resolution. Anything else has to be measured warped.
            warp = (
                None
                if on_raw
                else load_policy_remap(
                    sensor_cfg.optical_remap,
                    (frame.shape[1], frame.shape[0]),
                    conf_dir=conf_dir,
                )
            )
            if warp is not None:
                frame = warp.apply(frame)
            frame_wh = (frame.shape[1], frame.shape[0])
            intr = load_intrinsics(schema, frame_wh)
            print(
                f"\nMeasuring on {frame_wh[0]}x{frame_wh[1]} "
                f"{'raw sensor' if on_raw else 'remapped'} frames "
                f"(fx {intr['K'][0, 0]:.0f} px), {_FRAMES_PER_SAMPLE} frames "
                "averaged per pose."
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
                    tx_cam2policy,
                    n_samples,
                    warp,
                )
                rounds = 0
                first = None
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
                    grade = _grade(trans_mm, rot_deg, _uncertainty(samples))

                    # A misdetection is dropped, not retaken. What is left
                    # already answers the question, so sending the user back
                    # to the arm only spends their time arriving where this
                    # lands immediately -- and only when it genuinely lands
                    # there, which is what the refit without them checks.
                    keep = [
                        s
                        for i, s in enumerate(samples)
                        if i not in grade["outliers"]
                    ]
                    if grade["outliers"] and len(keep) >= _MIN_SAMPLES:
                        if _grade(
                            *_fit_error(keep, *_solve(keep)), _uncertainty(keep)
                        )["passed"]:
                            named = ", ".join(
                                str(i + 1) for i in grade["outliers"]
                            )
                            print(
                                f"\n  Dropped pose {named}: residual over "
                                f"{_OUTLIER_FACTOR:g}x the median, and the "
                                f"remaining {len(keep)} poses fit without "
                                "them. Re-solving."
                            )
                            samples = keep
                            continue

                    if grade["passed"]:
                        break

                    _report_fail(samples, trans_mm, grade)
                    # The yardstick for "are extra poses working" is where
                    # this run started, so it is taken once.
                    first = first or grade

                    add = None
                    if _more_poses_would_help(grade):
                        rounds += 1
                        if not _bar_is_responding(first, grade):
                            print()
                            _para(
                                f"The error bar has not come down the way "
                                f"more poses should bring it: {first['n']} "
                                f"poses gave {first['stderr_mm']:.1f} mm and "
                                f"{grade['n']} should therefore be near "
                                f"{first['stderr_mm'] * np.sqrt(first['n'] / grade['n']):.1f} "
                                f"mm, not {grade['stderr_mm']:.1f}. Something "
                                "is holding it up that counting poses cannot "
                                "fix. Stopping rather than asking again."
                            )
                        elif rounds > _MAX_TOPUP_ROUNDS:
                            print()
                            _para(
                                f"That is {_MAX_TOPUP_ROUNDS} rounds. The bar "
                                "is still coming in, so this is winnable, but "
                                "it wants more poses than one sitting should "
                                "ask for. Re-run when you have the time for "
                                "them."
                            )
                        else:
                            add = _ask_topup(grade)

                    extra = []
                    if add is not None:
                        extra = _collect(
                            arm,
                            camera,
                            sensor_name,
                            detector,
                            tag_id,
                            obj_pts,
                            intr,
                            color_mode,
                            tx_cam2policy,
                            add,
                            warp,
                            first_pose=len(samples) + 1,
                        )
                    if not extra:
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
                    samples = samples + extra
                    print(
                        f"  added {len(extra)} pose"
                        f"{'s' if len(extra) != 1 else ''}, "
                        f"{len(samples)} in total. Re-solving."
                    )

                derived = _derive(tx_tcp2camera, template.ik_tcp)
                _report_pass(
                    samples, derived, grade, arm_name, template.ik_tcp, frame_wh
                )

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
                    # The frame the tag was measured in, which is not
                    # necessarily the one the policy consumes: the transforms
                    # above are in policy axes either way.
                    "measured_on": "raw_sensor" if on_raw else "remapped",
                    "frame_size": list(frame_wh),
                    "frames_per_sample": _FRAMES_PER_SAMPLE,
                    "n_samples": len(samples),
                    "fit": {
                        # The verdict: how far the answer moves when any one
                        # pose is left out. Precision only -- a mismeasured
                        # marker biases every pose alike and never shows here.
                        "stderr_mm": grade["stderr_mm"],
                        "stderr_axes_mm": [
                            float(v) for v in grade["stderr_axes_mm"]
                        ],
                        "trans_rms_mm": grade["trans_rms"],
                        "trans_max_mm": grade["trans_max"],
                        "rot_rms_deg": grade["rot_rms"],
                        "scale_excess": _scale_excess(samples, grade),
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
