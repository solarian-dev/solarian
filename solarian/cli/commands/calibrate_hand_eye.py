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

Each pose is sampled with the arm held by its own servos rather than by hand,
because the pose the servos hold is the one forward kinematics describes -- a
hand-held arm sags and flexes by degrees, and degrees at the tag's distance
are tens of mm of error.

By default only one pose is set by hand: an anchor looking at the tag. The arm
then orbits the tag by itself, rotating the whole tool about the tag centre
through a fixed pattern of tilts and rolls, and samples at each stop. That is
a few seconds a pose instead of a walk and three confirmations, and the
pattern spreads orientation far more evenly than hands do -- which matters,
because the lever arm being measured is resolved by rotation between poses.
``mode=manual`` keeps the hand-walked capture: torque off to walk the arm into
position, on to hold it while the hands come away, and off again once a hand
is back on it.

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

Overrides: ``mode=auto sweep=36 tilt=30 tag=13 marker=160.0 arm=right_arm``,
``samples=12`` for ``mode=manual``, and ``joints=fit`` to measure the arm's
joint zeros in the same sweep (see :func:`_fit_joint_offsets`).
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st
from omegaconf import OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.cli.commands._calibration import (
    _opencv_bgr,
    module_running,
    pick_module,
    returns_to_start,
    run_control_loop,
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
# of how carefully a run is done: on a Solarian Arm 1 the tag alone spends half of a
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
# Soft-L1 scale for _refine, in metres of residual (rotation enters through the
# tag lever, so the same scale applies to it). Below it the loss is least
# squares; a misdetection tens of mm out is down-weighted instead of steering
# the answer, inside every jackknife solve as well as the main one.
_ROBUST_SCALE_M = 0.01

# Automatic sweep. Poses per sweep, counting the taught anchor.
_SWEEP_POSES = 36
# Largest orbit tilt off the anchor's line of sight. The lever arm is resolved
# by rotation between poses, so spread is what narrows the error bar; past
# about 30 deg the tag foreshortens and a wrist this size runs out of joint.
_SWEEP_TILT_DEG = 30.0
# Spin about the line of sight, cycled across targets. Roll keeps the tag in
# the middle of the frame and is the cheapest rotation the arm has to give.
_SWEEP_ROLLS_DEG = (-40.0, -20.0, 0.0, 20.0, 40.0)
# Metres nearer (-) and further (+) from the tag than the anchor, cycled
# across targets; see _orbit_targets for why one distance is not enough.
_SWEEP_RADIAL_M = (-0.08, 0.0, 0.08)
# joints=fit: a joint zero is resolved by how differently the joints are bent
# between poses, so the sweep is longer and tilts further than the hand-eye
# solve alone needs.
_JOINT_SWEEP_POSES = 48
_JOINT_SWEEP_TILT_DEG = 40.0
# A fitted joint zero is written only when its standard error is this small.
# Not tighter: shoulder_lift, elbow_flex and wrist_flex share one pitch axis,
# so their sum is pinned far better than each alone (simulating 48 planned
# poses at 0.3 deg of tag noise gives 0.3-0.7 deg apiece while the camera
# comes out within 1 mm; a real Solarian Arm 1 run gave 0.4-0.6 deg against offsets
# of up to 5 deg), and an eyeballed home is off by degrees.
_OFFSET_STDERR_MAX_DEG = 1.0
# Per-pose residual past which the arm's kinematics, not the tag, is the likely
# cause: a real Solarian Arm 1 fits to 2.6 mm rms with fitted zeros, 6-7 without.
_KINEMATIC_RESIDUAL_MM = 5.0
# |scale - 1| past this many standard errors, and past _SCALE_FLOOR, is a
# distance scale the fit has actually resolved rather than noise.
_SCALE_SIGMAS = 3.0
_SCALE_FLOOR = 0.02
# A tag seen this obliquely has foreshortened corners whose error grows fast.
_MAX_VIEW_DEG = 60.0
# Fingertips above the tag plane, which is the table.
_TCP_CLEARANCE_M = 0.05
# How close differential IK has to land to a target for it to be visited. Loose
# on purpose: the sample pairs the frame with the pose actually read back, so
# a target only has to be near enough to keep the tag in view.
_REACH_POS_M = 0.01
_REACH_ROT_DEG = 3.0
# Closest any two parts may come, by the robot scene's collision meshes. Those
# are measured as convex hulls, already larger than the parts, so this is
# margin for the arm not tracking its path exactly and for joint zeros a few
# degrees off, not for the meshes. 2 cm drops twice the Solarian Arm 1 orbit poses
# 1.5 cm does (simulated), and the sweep is then too thin for joints=fit.
_COLLISION_MARGIN_M = 0.015
# Joint-space step at which the move between two targets is checked.
_PATH_STEP_DEG = 4.0
# Orbit candidates planned per pose wanted. From an anchor near the edge of
# reach only 1 in 16 survives reach, view angle and the collision check (a real
# Solarian Arm 1 run kept 6 of 96), so fewer leaves a sweep, and every top-up, short.
# Each reach check is ~25 ms of IK on a Pi 5, and half fail cheaper tests first.
_CANDIDATES_PER_TARGET = 16
# After a move the gravity joints keep creeping for a second or two while each
# servo's integral term winds up against the load. A sample waits until no
# joint has moved more than _STILL_DEG (two encoder ticks) over _STILL_WINDOW_S,
# up to _STILL_TIMEOUT_S, after which _measure judges the pose as it is.
_STILL_DEG = 0.18
_STILL_WINDOW_S = 0.5
_STILL_TIMEOUT_S = 5.0
_STILL_POLL_S = 0.05
# Sweep speeds, well under the arm's limits: a calibration sample is worth
# more than the second a faster move saves, and a slow stop rings less.
_SWEEP_SPEED_M_S = 0.08
_SWEEP_ROT_SPEED = 0.6
# Azimuth step between sweeps. Irrational in turns, so no later sweep lands on
# the azimuths of an earlier one.
_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


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
    sharper, by the ratio of the focal lengths: a Solarian Arm 1 wrist camera reads
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
    q_before = arm._read_joints_rad()
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
    q_after = arm._read_joints_rad()
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
        # Joint angles the pose was computed from, for the joint-zero fit. The
        # half-difference is wrapped so a wrist read across +-pi averages to
        # where it was, not to the far side of the circle.
        "q": q_before + 0.5 * np.angle(np.exp(1j * (q_after - q_before))),
        "rms_px": rms_px,
        "ratio": ratio,
        "jitter_px": jitter_px,
        "tag_m": float(np.linalg.norm(tvec)),
    }, None, False


def _gate_wh(intr) -> tuple[int, int]:
    """Size of the cheap detection gate's copy of a frame; see :func:`_detect`."""
    frame_w, frame_h = int(intr["DIM"][0]), int(intr["DIM"][1])
    scale = min(1.0, _GATE_W / frame_w)
    return round(frame_w * scale), round(frame_h * scale)


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
    gate_wh = _gate_wh(intr)
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


def _prior_tcp2camera(path: Path, ik_tcp) -> np.ndarray:
    """Best guess at ``tx_tcp2camera`` before this run has measured it.

    Only aims the sweep, so it needs to be roughly right, not right: every
    target rotates about where this guess puts the tag, and a few cm of error
    there moves the tag across the frame rather than out of it. The last
    passing run on this camera is the best guess, taken through its flange
    transform so an ik_tcp edited since still composes correctly; UMI's
    nominal mount, with camera and fingertip axes aligned, is the fallback.
    """
    try:
        with open(path, "r") as fp:
            data = json.load(fp)
        if data.get("passed"):
            gripper2camera = np.asarray(data["tx_gripper2camera"], dtype=np.float64)
            return gripper2camera @ _offset_to_mat(ik_tcp)
    except (OSError, ValueError, KeyError):
        pass
    prior = np.eye(4)
    prior[:3, 3] = UMI_TIP_IN_CAMERA
    return prior


def _orbit_targets(anchor_pose, center, line_of_sight, n, max_tilt_deg, phase=0):
    """TCP poses that orbit the whole tool rigidly about *center*.

    Each target is ``Trans(P) . R . Trans(-P) . T_anchor``, then slid
    ``radial_m`` along the rotated line of sight, so the camera keeps looking
    at the tag. ``R`` rolls about the line of sight first -- which leaves the
    camera where it is, spinning -- then tilts about an axis perpendicular to
    it.

    The slide cycles between nearer, level with and further than the anchor. At one
    distance, a tag distance read a few percent off (marker size, lens
    intrinsics) is indistinguishable from the camera sitting further along
    its own axis, and the fit moves the camera with zero residual to show for
    it. Several distances make the same error disagree with itself.

    Azimuths are evenly spread and visited in order to keep moves short, two
    tilts each (half and full *max_tilt_deg*). *phase* is the sweep number:
    it turns the azimuths by the golden angle and steps the roll and distance
    cycles, so a top-up sweep adds poses rather than repeating them.
    """
    sight = np.asarray(line_of_sight, dtype=np.float64)
    sight = sight / np.linalg.norm(sight)
    # Any direction off the sight line seeds the perpendicular pair.
    seed = np.array([0.0, 0.0, 1.0]) if abs(sight[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(sight, seed)
    u /= np.linalg.norm(u)
    v = np.cross(sight, u)
    center = np.asarray(center, dtype=np.float64)
    anchor = _pose_to_mat(anchor_pose)

    n_az = int(np.ceil(n / 2))
    targets = []
    for j in range(n_az):
        azimuth = phase * _GOLDEN_ANGLE + 2.0 * np.pi * j / n_az
        axis = np.cos(azimuth) * u + np.sin(azimuth) * v
        for k, tilt in enumerate((0.5 * max_tilt_deg, max_tilt_deg)):
            if len(targets) >= n:
                break
            roll = _SWEEP_ROLLS_DEG[(2 * j + k + phase) % len(_SWEEP_ROLLS_DEG)]
            radial = _SWEEP_RADIAL_M[(j + k + phase) % len(_SWEEP_RADIAL_M)]
            rot = st.Rotation.from_rotvec(np.radians(tilt) * axis) * st.Rotation.from_rotvec(
                np.radians(roll) * sight
            )
            orbit = np.eye(4)
            orbit[:3, :3] = rot.as_matrix()
            orbit[:3, 3] = center - rot.apply(center) - radial * rot.apply(sight)
            targets.append(
                {
                    "pose": _mat_to_pose(orbit @ anchor),
                    "orbit": orbit,
                    "tilt_deg": float(tilt),
                    "roll_deg": float(roll),
                    "radial_m": float(radial),
                    "azimuth_deg": float(np.degrees(azimuth) % 360.0),
                }
            )
    return targets


def _plan_sweep(arm, anchor, prior, n, max_tilt_deg, phase, guard=None):
    """``(targets, skipped)``: up to *n* orbit poses worth visiting, and why
    the rest were dropped before the arm moved.

    _CANDIDATES_PER_TARGET times as many candidates are generated as wanted,
    and the survivors are thinned evenly, so filtering costs count rather than
    coverage. With a :class:`~solarian.sim.collision.CollisionGuard`, a
    target is also dropped
    when the gripper would come too close to the arm or the table there, or on
    the way to it from the target before.
    """
    cam_in_base = np.linalg.inv(anchor["tx_base2tcp"]) @ np.linalg.inv(prior)
    tag_in_base = cam_in_base @ anchor["tx_world2cam"]
    center, normal = tag_in_base[:3, 3], tag_in_base[:3, 2]
    camera = cam_in_base[:3, 3]

    skipped: Counter = Counter()
    keep = []
    for target in _orbit_targets(
        anchor["pose"], center, center - camera, _CANDIDATES_PER_TARGET * n,
        max_tilt_deg, phase,
    ):
        seen_from = target["orbit"][:3, :3] @ (camera - center)
        view_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(abs(normal @ seen_from) / np.linalg.norm(seen_from), 0.0, 1.0)
                )
            )
        )
        if view_deg < _MIN_TILT_DEG:
            skipped["too flat-on to the tag"] += 1
            continue
        if view_deg > _MAX_VIEW_DEG:
            skipped["too oblique to the tag"] += 1
            continue
        pose = target["pose"]
        if pose[2] < center[2] + _TCP_CLEARANCE_M:
            skipped["fingertips too close to the table"] += 1
            continue
        achieved, blocked, q = arm.reachable_pose(pose)
        if (
            blocked
            or np.linalg.norm(achieved[:3] - pose[:3]) > _REACH_POS_M
            or _rot_between_deg(achieved, pose) > _REACH_ROT_DEG
        ):
            skipped["out of the arm's reach"] += 1
            continue
        if guard is not None and guard.clearance(q)[0] < _COLLISION_MARGIN_M:
            skipped["the gripper would hit the arm or the table"] += 1
            continue
        keep.append({**target, "q": q})
    if len(keep) > n:
        keep = [keep[i] for i in np.unique(np.linspace(0, len(keep) - 1, n).round().astype(int))]
    if guard is not None:
        keep = _drop_colliding_moves(guard, anchor["q"], keep, skipped)
    return keep, skipped


def _drop_colliding_moves(guard, anchor_q, targets, skipped) -> list[dict]:
    """*targets* minus those whose move in, or the final move home, collides.

    The arm moves in straight Cartesian lines, which differential IK follows
    with joints that pass close to the straight joint-space line between the
    two ends -- so that line is what is checked, densely enough that no part
    can slip through the margin between two steps.

    The anchor was set by hand and may already sit inside the margin, by the
    hulls if not in fact; a move may then come as close as its ends do, but
    no closer, or nothing could ever leave the anchor.
    """

    def clear(q_from, q_to) -> bool:
        limit = min(
            _COLLISION_MARGIN_M, guard.clearance(q_from)[0], guard.clearance(q_to)[0]
        )
        steps = int(np.ceil(np.degrees(np.max(np.abs(q_to - q_from))) / _PATH_STEP_DEG))
        return all(
            guard.clearance(q_from + t * (q_to - q_from))[0] >= limit
            for t in np.linspace(0.0, 1.0, steps + 1)[1:-1]
        )

    kept, here = [], anchor_q
    for target in targets:
        if clear(here, target["q"]):
            kept.append(target)
            here = target["q"]
        else:
            skipped["the gripper would hit the arm on the way there"] += 1
    while kept and not clear(kept[-1]["q"], anchor_q):
        kept.pop()
        skipped["the gripper would hit the arm on the way there"] += 1
    return kept


def _teach_anchor(arm, camera, rig, max_tilt_deg):
    """A pose set by hand, measured, or None when the user gives up.

    The camera runs live throughout, with the tag's distance, view angle and
    ambiguity drawn on it, so the user aims by what the measurement will see.
    """
    print(
        "\nAnchor pose. Walk the arm until the fingers point at the tag from "
        f"about\n0.3 m, tilted at least {_MIN_TILT_DEG:g} deg off flat-on (the "
        "tag line turns green), with\nroom to swing "
        f"{max_tilt_deg:g} deg every way from there. The arm then orbits the "
        "tag by itself.\nENTER confirms and q stops, from either the window or "
        "this shell."
    )
    while True:
        if not _confirm_live(
            camera, rig, "Take hold of the arm, then confirm to release torque.",
            "take hold of the arm, ENTER releases torque",
        ):
            return None
        arm.start_calibration_mode()
        aimed = _confirm_live(
            camera, rig, "Torque is OFF. Aim at the tag, then confirm while still holding it.",
            "torque OFF - aim at the tag, ENTER holds here",
        )
        # Goal_Position survives the torque-off, so the hold is latched to
        # where the arm now is, or it snaps back to the pre-free-drive goal.
        arm.end_calibration_mode()
        arm.enable_torque_holding()
        if not aimed:
            return None
        time.sleep(_SETTLE_S)
        if not _confirm_live(
            camera, rig, "Torque is ON. Let go of the arm, then confirm.",
            "holding - let go, ENTER measures",
        ):
            return None
        _wait_until_still(arm)
        sample, reason = _measure(arm, camera, *rig)
        if sample is not None:
            print(
                f"  anchor: tag at {sample['tag_m']:.3f} m, reprojection "
                f"{sample['rms_px']:.2f} px, ambiguity {sample['ratio']:.0f}"
            )
            return sample
        print(f"  nothing recorded: {reason}. Re-seat the anchor.")


def _open_preview(intr) -> bool:
    """Open the camera window at the frame's aspect; False when there is no display."""
    import cv2

    frame_w, frame_h = int(intr["DIM"][0]), int(intr["DIM"][1])
    scale = min(1.0, _WINDOW_W / frame_w)
    try:
        cv2.namedWindow(_CAM_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_CAM_WINDOW, round(frame_w * scale), round(frame_h * scale))
    except cv2.error:
        return False
    return True


def _confirm_live(camera, rig, prompt: str, overlay: str) -> bool:
    """Stream the camera until Enter, from the window or the shell. False on q.

    Without a display it is a plain prompt.
    """
    import cv2

    print(f"  {prompt}")
    if _open_preview(rig[4]):
        try:
            while True:
                key = _show_frame(
                    camera, rig, [(overlay, (0, 255, 255)), ("q stops", (200, 200, 200))]
                )
                line = _shell_line()
                if key == ord("q") or (line is not None and line.lower() == "q"):
                    return False
                if key in _CONFIRM_KEYS or line is not None:
                    return True
        except cv2.error:
            pass
    return input("  Enter, or q to stop: ").strip().lower() != "q"


def _show_frame(camera, rig, lines) -> int:
    """Draw the tag, how it reads, and *lines* on a fresh frame; the key pressed."""
    import cv2

    sensor_name, detector, tag_id, obj_pts, intr, color_mode, gate_wh, _tx, warp = rig
    full = _opencv_bgr(_read_frame(camera, sensor_name, warp), color_mode)
    corners, _seen = _detect(
        cv2.cvtColor(full, cv2.COLOR_BGR2GRAY), detector, tag_id, gate_wh
    )
    solved = _tag_in_camera(corners, obj_pts, intr) if corners is not None else None
    if solved is None:
        reading = (f"tag {tag_id} not found", (0, 0, 255))
    else:
        rvec, tvec, _rms, ratio = solved
        tvec = np.ravel(tvec)
        normal = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0][:, 2]
        view_deg = float(
            np.degrees(np.arccos(np.clip(abs(normal @ tvec) / np.linalg.norm(tvec), 0.0, 1.0)))
        )
        usable = (
            _MIN_TILT_DEG <= view_deg <= _MAX_VIEW_DEG and ratio >= _MIN_AMBIGUITY_RATIO
        )
        reading = (
            f"tag {np.linalg.norm(tvec):.2f} m, {view_deg:.0f} deg off flat-on, "
            f"ambiguity {ratio:.0f}",
            (0, 255, 0) if usable else (0, 165, 255),
        )
    view = full.copy()
    k = int(intr["DIM"][0]) / _OVERLAY_REF_W
    _draw_tag(view, corners, solved, obj_pts, intr, k)
    _draw_status(view, [reading, *lines], k)
    cv2.imshow(_CAM_WINDOW, view)
    return cv2.waitKey(1) & 0xFF


def _wait_until_still(arm) -> tuple[float, bool]:
    """``(seconds waited, settled)``: until no joint creeps past _STILL_DEG.

    Joint encoders, not the camera, because they see the integral term pulling
    a loaded joint onto its goal long after the visible ringing has stopped.
    """
    start = time.monotonic()
    history: list[tuple[float, np.ndarray]] = []
    while True:
        now = time.monotonic()
        history = [(t, q) for t, q in history if now - t <= _STILL_WINDOW_S]
        history.append((now, arm._read_joints_rad()))
        if now - start >= _STILL_WINDOW_S:
            qs = np.stack([q for _t, q in history])
            # Wrapped against the first read, so a wrist sitting on +-pi
            # does not look like a full turn of motion.
            moved = np.angle(np.exp(1j * (qs - qs[0])))
            if np.degrees(np.max(moved.max(axis=0) - moved.min(axis=0))) <= _STILL_DEG:
                return now - start, True
        if now - start >= _STILL_TIMEOUT_S:
            return now - start, False
        time.sleep(_STILL_POLL_S)


def _auto_sweep(
    arm, camera, rig, anchor, prior, n, max_tilt_deg, phase, first_pose, guard=None
):
    """Drive the arm through an orbit about the tag, one sample per pose.

    The arm is energised throughout and comes back to the anchor however
    this exits, so the next sweep plans from the same place and the user
    is never handed an arm parked somewhere they did not choose.
    """
    import cv2

    targets, skipped = _plan_sweep(arm, anchor, prior, n, max_tilt_deg, phase, guard)
    dropped = ", ".join(f"{count} {why}" for why, count in skipped.most_common())
    print(
        f"\nSweep of {len(targets)} poses about the tag, up to {max_tilt_deg:g} "
        f"deg of tilt, numbered from {first_pose}."
        + (f" Dropped before moving: {dropped}." if dropped else "")
        + " q in the window or this shell, or Ctrl-C, stops it."
    )
    if not targets:
        _para(
            "No orbit pose survived planning. Re-seat the anchor nearer the "
            "middle of the arm's reach, or pass a smaller tilt=<deg>."
        )
        return []

    preview = _open_preview(rig[4])
    samples: list[dict] = []
    try:
        with returns_to_start(arm, _SWEEP_SPEED_M_S):
            for i, target in enumerate(targets, 1):
                arm.moveL(
                    target["pose"], speed=_SWEEP_SPEED_M_S, rot_speed=_SWEEP_ROT_SPEED
                )
                run_control_loop(
                    arm, arm.last_waypoint_time - time.monotonic() + _SETTLE_S
                )
                waited_s, still = _wait_until_still(arm)
                sample, reason = _measure(arm, camera, *rig)
                label = (
                    f"{i}/{len(targets)}: tilt {target['tilt_deg']:.0f} deg at "
                    f"{target['azimuth_deg']:.0f}, roll {target['roll_deg']:+.0f}, "
                    f"{target['radial_m'] * 100:+.0f} cm, "
                    + (f"still after {waited_s:.1f} s" if still else f"still creeping at {waited_s:.0f} s")
                )
                if sample is None:
                    print(f"  {label} -- skipped: {reason}.")
                else:
                    samples.append(sample)
                    print(
                        f"  pose {first_pose + len(samples) - 1} ({label}): tag at "
                        f"{sample['tag_m']:.3f} m, reprojection "
                        f"{sample['rms_px']:.2f} px, ambiguity {sample['ratio']:.0f}"
                    )
                stop = False
                if preview:
                    try:
                        stop = _show_frame(
                            camera,
                            rig,
                            [
                                (f"{label}  ({len(samples)} measured)", (0, 255, 255)),
                                ("q stops the sweep", (200, 200, 200)),
                            ],
                        ) == ord("q")
                    except cv2.error:
                        preview = False
                line = _shell_line()
                if stop or (line is not None and line.lower() == "q"):
                    print("  Stopped.")
                    break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if preview:
            try:
                cv2.destroyWindow(_CAM_WINDOW)
            except cv2.error:
                pass
    print(f"  {len(samples)} of {len(targets)} sweep poses measured.")
    return samples


def _collision_guard(arm_name, joint_names):
    """The robot scene's collision check for *arm_name*, or None, said aloud.

    The arm's own IK model has no gripper in it, so this is the only thing
    standing between an orbit target and the camera driven into the forearm.
    """
    from solarian.conf import settings
    from solarian.sim.collision import CollisionGuard
    from solarian.utils import get_class

    scene = get_class(settings.ROBOT_CLASS).scene
    if scene is None or not any(a.module == arm_name for a in scene.attachments):
        print(
            f"\n  WARNING: the robot declares no scene with {arm_name} in it, so the "
            "sweep cannot check for the gripper hitting the arm. Watch it, and "
            "stop it with q if anything gets close."
        )
        return None
    try:
        return CollisionGuard(scene, arm_name, joint_names)
    except Exception as exc:  # noqa: BLE001 - a broken scene must not block calibrating
        print(
            f"\n  WARNING: the robot scene would not load for collision checks "
            f"({exc}). Watch the sweep, and stop it with q if anything gets close."
        )
        return None


@contextmanager
def _taken_before_release():
    """Hold the exit until a hand is on the arm.

    The automatic sweep ends energised, and leaving ``module_running`` cuts
    torque, which drops a servo arm where it stands.
    """
    try:
        yield
    finally:
        try:
            input("\nTake hold of the arm, then press Enter to release torque...")
        except (EOFError, KeyboardInterrupt):
            print()


def _pose_spread_deg(samples) -> np.ndarray:
    """Standard deviation of the TCP orientations about their mean, per tool axis."""
    rots = st.Rotation.from_rotvec(np.stack([s["pose"][3:6] for s in samples]))
    return np.degrees((rots.mean().inv() * rots).as_rotvec().std(axis=0))


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

    The loss is soft-L1 at ``_ROBUST_SCALE_M``: least squares for the
    millimetres a healthy pose is out by, linear for a misdetection, so one
    bad pose cannot drag every jackknife solve that contains it.
    """
    from scipy.optimize import least_squares

    # Radians and metres only become comparable through a lever arm, and the
    # tag's own distance is the honest one: it is how far an orientation error
    # actually displaces the thing the camera is looking at.
    lever = float(np.mean([s["tag_m"] for s in samples]))
    cam2world = np.linalg.inv(np.stack([s["tx_world2cam"] for s in samples]))
    base2tcp = np.stack([s["tx_base2tcp"] for s in samples])

    def residual(x):
        world2base = np.linalg.inv(_pose_to_mat(x[:6]))
        err = cam2world @ (_pose_to_mat(x[6:]) @ base2tcp @ world2base)
        rotvec = st.Rotation.from_matrix(err[:, :3, :3]).as_rotvec()
        return np.concatenate([err[:, :3, 3], rotvec * lever], axis=1).ravel()

    fit = least_squares(
        residual,
        np.concatenate(
            [_mat_to_pose(tx_base2world), _mat_to_pose(tx_tcp2camera)]
        ),
        method="trf",
        loss="soft_l1",
        f_scale=_ROBUST_SCALE_M,
        xtol=1e-12,
        ftol=1e-12,
    )
    return _pose_to_mat(fit.x[:6]), _pose_to_mat(fit.x[6:])


def _distance_scale(samples) -> dict:
    """The factor the measured tag distances fit best when it is left free.

    Everything metric on the camera side rests on two inputs -- the printed
    marker size and the lens focal length -- and both scale every tag
    distance by the same factor. The arm's distances are independent of
    both, so poses at more than one distance pin the factor down: the
    solve runs again with it free, and a scale off 1 says one of those two
    inputs is wrong. It is reported rather than applied, because the cure
    is to fix the input, and ``marker=`` is where any scale is corrected.

    ``scale`` > 1 means the true distances are longer than measured. The
    standard error comes from the Jacobian at the solution.
    """
    from scipy.optimize import least_squares

    tx_base2world, tx_tcp2camera = _solve(samples)
    lever = float(np.mean([s["tag_m"] for s in samples]))
    rot_w2c = np.stack([s["tx_world2cam"][:3, :3] for s in samples])
    t_w2c = np.stack([s["tx_world2cam"][:3, 3] for s in samples])
    base2tcp = np.stack([s["tx_base2tcp"] for s in samples])

    def residual(x):
        cam2world = np.tile(np.eye(4), (len(samples), 1, 1))
        cam2world[:, :3, :3] = np.transpose(rot_w2c, (0, 2, 1))
        cam2world[:, :3, 3] = -np.einsum("nji,nj->ni", rot_w2c, x[12] * t_w2c)
        world2base = np.linalg.inv(_pose_to_mat(x[:6]))
        err = cam2world @ (_pose_to_mat(x[6:12]) @ base2tcp @ world2base)
        rotvec = st.Rotation.from_matrix(err[:, :3, :3]).as_rotvec()
        return np.concatenate([err[:, :3, 3], rotvec * lever], axis=1).ravel()

    fit = least_squares(
        residual,
        np.concatenate(
            [_mat_to_pose(tx_base2world), _mat_to_pose(tx_tcp2camera), [1.0]]
        ),
        method="trf",
        loss="soft_l1",
        f_scale=_ROBUST_SCALE_M,
        xtol=1e-12,
        ftol=1e-12,
    )
    dof = max(fit.fun.size - fit.x.size, 1)
    sigma2 = float(np.sum(fit.fun**2) / dof)
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * sigma2
        stderr = float(np.sqrt(max(cov[12, 12], 0.0)))
    except np.linalg.LinAlgError:
        stderr = float("inf")
    scale = float(fit.x[12])
    return {
        "scale": scale,
        "stderr": stderr,
        "resolved": abs(scale - 1.0) > max(_SCALE_SIGMAS * stderr, _SCALE_FLOOR),
        "tip_in_camera": _pose_to_mat(fit.x[6:12])[:3, 3],
    }


def _report_scale(scale, marker_mm: float) -> None:
    """The fitted distance scale, and what it means when it is not 1."""
    print(
        f"\n  distance scale        {scale['scale']:.3f} +/- {scale['stderr']:.3f}"
        "   (1 = marker size and lens agree with the arm)"
    )
    if not scale["resolved"]:
        return
    tip = scale["tip_in_camera"]
    print()
    _para(
        f"The tag distances fit best scaled by {scale['scale']:.3f} (+/- "
        f"{scale['stderr']:.3f}), which is not 1. Either the marker is not "
        f"{marker_mm:.0f} mm or the lens intrinsics do not match the lens "
        "fitted -- both scale every distance alike, and the fingertip depth "
        "comes out wrong by about the same factor. With the scale left free "
        f"the fingertips sit at [{tip[0]:+.4f}, {tip[1]:+.4f}, {tip[2]:+.4f}] m "
        "in camera axes."
    )
    _para(
        "Measure the black square edge to edge, and check that the mounted "
        "lens is the one calibrate-umi-camera fitted (refocusing counts). If "
        f"both check out, re-run with marker={marker_mm * scale['scale']:.1f}, "
        "which corrects the distances whichever input caused it."
    )


def _with_joint_offsets(samples, backend, delta) -> list[dict]:
    """The samples re-posed by forward kinematics at their joint angles + *delta*."""
    from solarian.core.ik.diagnostics import fk

    out = []
    for s in samples:
        pose = fk(backend, s["q"] + delta)
        out.append({**s, "pose": pose, "tx_base2tcp": np.linalg.inv(_pose_to_mat(pose))})
    return out


def _fit_joint_offsets(samples, backend, fitted) -> dict:
    """The joint zeros the camera measures, solved with both hand-eye transforms.

    The tag is fixed, so every pose has to agree on one tag-in-base. A joint
    whose zero is off by a degree bends forward kinematics differently at every
    configuration, which no rigid camera mount can explain -- the plain solve
    pays for it by moving the camera along its axis and leaving a rotation
    residual. Freeing one offset per joint in *fitted* (indices into ``q``)
    removes that. The first joint and the last cannot be among them: a base
    yaw zero is a turn of tag-in-base, and a flange roll zero a turn of the
    camera mount, so neither changes any residual.

    ``delta`` > 0 means the joint sits further along than it reads. Standard
    errors come from the Jacobian at the solution, as in :func:`_distance_scale`.
    """
    from scipy.optimize import least_squares

    from solarian.core.ik.diagnostics import fk

    fitted = list(fitted)
    q = np.stack([s["q"] for s in samples])
    tx_base2world, tx_tcp2camera = _solve(samples)
    lever = float(np.mean([s["tag_m"] for s in samples]))
    cam2world = np.linalg.inv(np.stack([s["tx_world2cam"] for s in samples]))

    def delta_of(x):
        delta = np.zeros(q.shape[1])
        delta[fitted] = x[12:]
        return delta

    # Finite differences step the 12 transform parameters with the offsets
    # held, so most residual calls repeat the forward kinematics of the last.
    cache: dict[bytes, np.ndarray] = {}

    def base2tcp_at(delta):
        key = delta.tobytes()
        if key not in cache:
            if len(cache) > 32:
                cache.clear()
            cache[key] = np.linalg.inv(
                np.stack([_pose_to_mat(fk(backend, qi + delta)) for qi in q])
            )
        return cache[key]

    def residual(x):
        world2base = np.linalg.inv(_pose_to_mat(x[:6]))
        err = cam2world @ (_pose_to_mat(x[6:12]) @ base2tcp_at(delta_of(x)) @ world2base)
        rotvec = st.Rotation.from_matrix(err[:, :3, :3]).as_rotvec()
        return np.concatenate([err[:, :3, 3], rotvec * lever], axis=1).ravel()

    fit = least_squares(
        residual,
        np.concatenate(
            [_mat_to_pose(tx_base2world), _mat_to_pose(tx_tcp2camera), np.zeros(len(fitted))]
        ),
        method="trf",
        loss="soft_l1",
        f_scale=_ROBUST_SCALE_M,
        x_scale="jac",
        xtol=1e-12,
        ftol=1e-12,
    )
    dof = max(fit.fun.size - fit.x.size, 1)
    sigma2 = float(np.sum(fit.fun**2) / dof)
    stderr = np.full(q.shape[1], np.nan)
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * sigma2
        stderr[fitted] = np.sqrt(np.clip(np.diag(cov)[12:], 0.0, None))
    except np.linalg.LinAlgError:
        stderr[fitted] = np.inf
    delta = delta_of(fit.x)
    before = _fit_error(samples, tx_base2world, tx_tcp2camera)
    after = _fit_error(
        _with_joint_offsets(samples, backend, delta),
        _pose_to_mat(fit.x[:6]),
        _pose_to_mat(fit.x[6:12]),
    )
    return {
        "delta": delta,
        "stderr": stderr,
        "fitted": fitted,
        "resolved": bool(np.all(np.degrees(stderr[fitted]) <= _OFFSET_STDERR_MAX_DEG)),
        "residual_mm": (_rms(before[0]), _rms(after[0])),
        "residual_deg": (_rms(before[1]), _rms(after[1])),
    }


def _rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _offer_joint_zeros(arm, zeros) -> np.ndarray | None:
    """Print the fitted joint zeros, and write them to the arm's yaml if they
    are resolved and the user agrees.

    Returns the offsets as written -- whole ticks, so slightly off what was
    fitted -- or None when nothing was written. The running arm keeps the
    calibration it connected with, so its reads stay in the old zeros for the
    rest of this run; the caller corrects them with what this returns.
    """
    from dataclasses import replace

    from solarian.cli.commands._ik_common import counts_per_rev_resolver
    from solarian.core.ik.alignment import offset_ticks

    names = arm._ik_joint_names()
    limits = arm._joint_limits()
    # Best effort: without a resolution the encoder bounds go unchecked.
    cprs = counts_per_rev_resolver(arm) or {}
    calibration = dict(arm.calibration)
    applied = np.zeros(len(names))
    off_encoder = []
    print("\n  joint zeros, measured by the camera (+ = the joint sits further along than it reads)")
    for i in zeros["fitted"]:
        name = names[i]
        cal = calibration[name]
        lo, hi = limits[name]
        ticks, applied[i] = offset_ticks(cal, lo, hi, zeros["delta"][i])
        if arm.actuators[name].is_continuous:
            calibration[name] = replace(cal, homing_offset=cal.homing_offset + ticks)
        else:
            calibration[name] = replace(
                cal, range_min=cal.range_min + ticks, range_max=cal.range_max + ticks
            )
            cpr = cprs.get(name)
            if calibration[name].range_min < 0 or (
                cpr and calibration[name].range_max > cpr - 1
            ):
                off_encoder.append(name)
        print(
            f"    {name:<16}{np.degrees(zeros['delta'][i]):+6.2f} deg +/- "
            f"{np.degrees(zeros['stderr'][i]):.2f}   ({ticks:+d} ticks)"
        )
    fixed = [name for i, name in enumerate(names) if i not in zeros["fitted"]]
    print(f"    {', '.join(fixed)}: not observable from one tag, left as they are")
    (mm_before, mm_after), (deg_before, deg_after) = zeros["residual_mm"], zeros["residual_deg"]
    print(
        f"    per-pose residual {mm_before:.1f} mm / {deg_before:.2f} deg rms with the "
        f"current zeros, {mm_after:.1f} mm / {deg_after:.2f} deg with these"
    )
    if not zeros["resolved"]:
        print()
        _para(
            f"Not written: every zero has to be known to {_OFFSET_STDERR_MAX_DEG:g} "
            "deg or better. They are resolved by how differently the joints are "
            "bent between poses, so re-run with the anchor nearer the middle of "
            "the arm's reach, where more of the orbit survives planning, or with "
            "a larger sweep=<poses>."
        )
        return None
    if off_encoder:
        print()
        _para(
            f"Not written: shifting {', '.join(off_encoder)} would push its range "
            "past the end of the encoder. Re-run `solarian calibrate` for that "
            "joint first."
        )
        return None
    if input("\n  Write the corrected zeros to the arm's yaml? [y/N] ").strip().lower() != "y":
        return None
    # The same physical home pose, re-read in the new zeros.
    home = {
        name: arm._rad_to_norm(name, arm._norm_to_rad(name, value) + applied[names.index(name)])
        if name in names
        else value
        for name, value in arm.cfg.home.items()
    }
    arm._save_calibration(calibration)
    if home:
        arm._save_home(home)
    print("  Written. They take effect the next time the arm connects.")
    return applied


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
    trans_rms = _rms(trans_mm)
    rot_rms = _rms(rot_deg)
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


def _print_checks(samples, grade) -> None:
    for label, value, limit, ok in grade["checks"]:
        print(f"  {label:<22}{value:<16}{limit:<12}{'ok' if ok else 'FAIL'}")
    # Not a check: the lever arm is resolved by rotation between poses, so a
    # narrow spread is the usual reason a wide error bar stays wide.
    spread = _pose_spread_deg(samples)
    print(
        f"  {'pose spread':<22}[{spread[0]:.0f}, {spread[1]:.0f}, {spread[2]:.0f}] "
        "deg std about the tool axes"
    )


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


def _hint_joint_zeros(grade) -> None:
    """Name the joint zeros when the residual is too big for the tag alone."""
    if grade["trans_rms"] <= _KINEMATIC_RESIDUAL_MM:
        return
    print()
    _para(
        f"{grade['trans_rms']:.1f} mm rms per pose is more than tag noise "
        f"explains (a calibrated arm fits to about {_KINEMATIC_RESIDUAL_MM / 2:g} "
        "mm). The usual cause is the arm's joint zeros, which bias every pose "
        "and that more poses only average down slowly. Re-run with joints=fit "
        "to measure and write them."
    )


def _report_fail(samples, trans_mm, grade) -> None:
    _banner("CALIBRATION FAILED -- DO NOT USE THESE NUMBERS.")
    _print_checks(samples, grade)
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


def _report_pass(
    samples,
    derived,
    grade,
    arm_name,
    ik_tcp,
    frame_wh,
    marker_mm=_MARKER_MM,
    rerun="solarian calibrate-hand-eye",
) -> None:
    res = f"{frame_wh[0]}x{frame_wh[1]}"
    _banner("CALIBRATION PASSED")
    _print_checks(samples, grade)
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
        f"{marker_mm:g} mm edge to edge, the offsets below are wrong by "
        "that same percentage and this error bar stays just as tight. The "
        "same goes for the arm's joint zeros, which joints=fit measures."
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
    move = -delta
    # Policy axes are OpenCV's on the remapped image: x right, y down -- the
    # side the fingers are on -- and z out of the lens. The raw preview may be
    # rotated against that image, so the directions are named by the gripper.
    words = (
        ("right", "left", "as you look out through the lens, fingers below"),
        ("down, toward the fingers", "up, away from the fingers", ""),
        ("forward, toward the fingertips", "back, away from the fingertips", ""),
    )
    print(
        f"  {step}. Move the camera "
        f"[{move[0]:+.1f}, {move[1]:+.1f}, {move[2]:+.1f}] mm along its own axes:"
    )
    for axis, value, (plus, minus, note) in zip("XYZ", move, words):
        print(
            f"       {axis} {value:+5.1f} mm  {plus if value >= 0 else minus}"
            + (f"  ({note})" if note else "")
        )
    print(
        "     Moving the camera beats moving the\n"
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
        f"  {step}. Re-run to check:\n\n"
        f"       {rerun}\n\n"
        "     joints=fit is not needed again: the zeros it writes stay in the\n"
        "     arm's yaml and every run uses them, until `solarian calibrate`\n"
        "     redoes the arm. The edit landed if the difference above\n"
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
                        "q": [float(v) for v in s["q"]],
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
    category = "Robot"
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

    By default you set one anchor pose by hand, looking at the tag, and the arm
    then orbits the tag by itself through a spread of tilts and rolls, a few
    seconds per pose. mode=manual instead has you pose the arm by hand for every
    sample. Needs an arm with an ik_tcp and a camera whose yaml points at a
    GoPro optical remap (`solarian calibrate-umi-camera`).

    joints=fit also measures the arm's joint zeros, which `solarian calibrate`
    can only set from a home pose matched by eye. The sweep is longer and
    tilts further, the camera sees how the arm's model bends away from the
    truth across it, and the corrected zeros are offered for the arm's yaml
    before the hand-eye answer is solved against them.
    """
    params = {
        "arm": "Name of the IK arm module to measure; prompts when omitted.",
        "tag": "ArUco id of the table marker, DICT_4X4_50 (default: 13).",
        "marker": "Measured side of the marker's black square in mm (default: 160).",
        "mode": "auto: the arm orbits the tag from one taught pose; manual: pose every sample by hand (default: auto).",
        "sweep": "Poses per automatic sweep, anchor included (default: 36, 48 with joints=fit).",
        "tilt": "Largest orbit tilt in degrees for the automatic sweep (default: 30, 40 with joints=fit).",
        "joints": "fit: also measure the arm's joint zeros from the sweep and offer to write them; off (default: off).",
        "samples": "Poses to collect in manual mode; 8 minimum (default: 12).",
    }
    examples = (
        "solarian calibrate-hand-eye",
        "solarian calibrate-hand-eye marker=158.5 tilt=25",
        "solarian calibrate-hand-eye joints=fit marker=159",
        "solarian calibrate-hand-eye mode=manual arm=right_arm samples=16",
    )

    def handle(self, cfg) -> None:
        tag_id = int(OmegaConf.select(cfg, "tag") or _TAG_ID)
        marker_mm = float(OmegaConf.select(cfg, "marker") or _MARKER_MM)
        n_samples = max(int(OmegaConf.select(cfg, "samples") or _SAMPLES), _MIN_SAMPLES)
        mode = str(OmegaConf.select(cfg, "mode") or "auto").lower()
        if mode not in ("auto", "manual"):
            print(f"mode={mode} is not a mode: auto or manual.")
            return
        auto = mode == "auto"
        joints = str(OmegaConf.select(cfg, "joints") or "off").lower()
        if joints not in ("off", "fit"):
            print(f"joints={joints} is not an option: off or fit.")
            return
        fit_joints = joints == "fit"
        if fit_joints and not auto:
            print("joints=fit needs mode=auto: the joint zeros come out of the orbit sweep.")
            return
        n_sweep = max(
            int(
                OmegaConf.select(cfg, "sweep")
                or (_JOINT_SWEEP_POSES if fit_joints else _SWEEP_POSES)
            ),
            _MIN_SAMPLES,
        )
        max_tilt = float(
            OmegaConf.select(cfg, "tilt")
            or (_JOINT_SWEEP_TILT_DEG if fit_joints else _SWEEP_TILT_DEG)
        )

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
        from solarian.core.conf_paths import conf_dir as project_conf_dir

        conf_dir = project_conf_dir()
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
            "hand on\nthe power. "
            + (
                "After you set one anchor pose by hand it moves by itself,\n"
                "orbiting the tag, and it stays energised until you take it back."
                if auto
                else "It goes limp as soon as the run starts."
            ),
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
            rig = (
                sensor_name,
                detector,
                tag_id,
                obj_pts,
                intr,
                color_mode,
                _gate_wh(intr),
                tx_cam2policy,
                warp,
            )
            with module_running(arm, "Connecting the arm...", confirm=False), (
                _taken_before_release() if auto else nullcontext()
            ):
                if auto:
                    prior = _prior_tcp2camera(
                        conf_dir / "calibrations" / f"{camera.name}_hand_eye.json",
                        template.ik_tcp,
                    )
                    guard = _collision_guard(arm_name, arm._ik_joint_names())
                    anchor = _teach_anchor(arm, camera, rig, max_tilt)
                    if anchor is None:
                        return
                    sweeps = 0

                    def gather(count, first_pose):
                        nonlocal sweeps
                        taken = _auto_sweep(
                            arm, camera, rig, anchor, prior, count, max_tilt,
                            sweeps, first_pose, guard,
                        )
                        sweeps += 1
                        return taken

                    samples = [anchor] + gather(n_sweep - 1, 2)
                else:

                    def gather(count, first_pose):
                        return _collect(
                            arm,
                            camera,
                            sensor_name,
                            detector,
                            tag_id,
                            obj_pts,
                            intr,
                            color_mode,
                            tx_cam2policy,
                            count,
                            warp,
                            first_pose=first_pose,
                        )

                    samples = gather(n_samples, 1)

                zeros = joint_delta = written = None
                if fit_joints and len(samples) >= _MIN_SAMPLES:
                    fk_backend = arm.ik_backend.build()
                    zeros = _fit_joint_offsets(
                        samples, fk_backend, range(1, len(samples[0]["q"]) - 1)
                    )
                    written = _offer_joint_zeros(arm, zeros)
                    # Solved against the fitted zeros even when they were not
                    # written: the current ones are what the fit just showed
                    # to be off, and they bias the camera by millimetres.
                    joint_delta = zeros["delta"] if written is None else written
                    samples = _with_joint_offsets(samples, fk_backend, joint_delta)
                    if written is None:
                        print(
                            "  The solve below still uses the fitted zeros; only "
                            "the arm's yaml keeps the old ones."
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
                    _report_scale(_distance_scale(samples), marker_mm)
                    if not fit_joints:
                        _hint_joint_zeros(grade)
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
                        extra = gather(add, len(samples) + 1)
                    if extra and joint_delta is not None:
                        extra = _with_joint_offsets(extra, fk_backend, joint_delta)
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
                rerun = " ".join(
                    ["solarian"] + [a for a in sys.argv[1:] if not a.startswith("joints=")]
                )
                _report_pass(
                    samples,
                    derived,
                    grade,
                    arm_name,
                    template.ik_tcp,
                    frame_wh,
                    marker_mm,
                    rerun,
                )
                scale = _distance_scale(samples)
                _report_scale(scale, marker_mm)
                if not fit_joints:
                    _hint_joint_zeros(grade)

        names = arm._ik_joint_names()
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
                        # Best-fit factor on the measured tag distances; 1
                        # means the marker size and intrinsics agree with
                        # the arm. Reported, not applied.
                        "distance_scale": scale["scale"],
                        "distance_scale_stderr": scale["stderr"],
                        # Joint zeros fitted by joints=fit, which every
                        # transform here is solved against; null without it.
                        "joint_offsets_written": written is not None,
                        "joint_offsets_deg": None
                        if joint_delta is None
                        else {
                            names[i]: float(np.degrees(joint_delta[i]))
                            for i in zeros["fitted"]
                        },
                        "joint_offsets_stderr_deg": None
                        if joint_delta is None
                        else {
                            names[i]: float(np.degrees(zeros["stderr"][i]))
                            for i in zeros["fitted"]
                        },
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
