"""Robot-world hand-eye: frame conventions, PnP path, verdict and capture flow.

A hand-eye solve has no self-evident answer to eyeball, and every quantity it
produces is one inversion away from a number that looks just as plausible and
is wrong. So these build a rig whose transforms are known, push it through the
same code the command runs on hardware, and require the ground truth back.

The verdict tests matter for the same reason from the other side: the command
is trusted to refuse, so "prints nothing on a bad fit" has to be enforced
rather than hoped for.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.spatial.transform as st

from solarian.cli.commands.calibrate_hand_eye import (
    UMI_TIP_IN_CAMERA,
    _MARKER_MM,
    _MAX_RESIDUAL_MM,
    _MAX_ROT_RMS_DEG,
    _MIN_AMBIGUITY_RATIO,
    _MIN_TILT_DEG,
    _TAG_ID,
    _ask_replacements,
    _default_replacements,
    _derive,
    _diagnose,
    _fit_error,
    _grade,
    _parse_replacements,
    _marker_points,
    _offset_to_mat,
    _pose_to_mat,
    _report_fail,
    _report_pass,
    _solve,
    _tag_in_camera,
)
from solarian.core.ik.spec import FrameOffset
from solarian.core.optical_remap import load_shipped_gopro_intrinsics

# The resolution the command measures at: it drops the bus resize for the run
# precisely so the tag is read at the remap's native size, and corner
# precision scales with focal length in pixels. Testing at the 640x360 the
# policy consumes would set a noise floor three times too pessimistic.
FRAME_WH = (1920, 1080)
MARKER_M = _MARKER_MM / 1000.0

# The camera sits back and to the side of the flange, yawed 3 deg off it: the
# umi-slim CAD mount plus exactly the kind of build error this has to recover.
CAMERA_IN_FLANGE = np.eye(4)
CAMERA_IN_FLANGE[:3, :3] = st.Rotation.from_rotvec(
    np.deg2rad([1.0, 3.0, -0.5])
).as_matrix()
CAMERA_IN_FLANGE[:3, 3] = [-0.0002, -0.0847, 0.0217]

# Declared TCP: the umi-slim CAD tip with identity rotation, as so106 wires it.
IK_TCP = FrameOffset(pos=(0.0, -0.0157, 0.2057), quat=(1.0, 0.0, 0.0, 0.0))

TAG_IN_BASE = np.eye(4)
TAG_IN_BASE[:3, :3] = st.Rotation.from_rotvec([0.0, 0.0, 0.3]).as_matrix()
TAG_IN_BASE[:3, 3] = [0.28, -0.05, 0.002]


def _mat_to_pose(mat: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [mat[:3, 3], st.Rotation.from_matrix(mat[:3, :3]).as_rotvec()]
    )


def _project(tx_world2cam, obj_pts, intr, rng, pixel_noise):
    import cv2

    px, _ = cv2.fisheye.projectPoints(
        obj_pts.reshape(-1, 1, 3),
        np.ascontiguousarray(cv2.Rodrigues(tx_world2cam[:3, :3])[0]),
        np.ascontiguousarray(tx_world2cam[:3, 3].reshape(3, 1)),
        intr["K"],
        intr["D"],
    )
    return px.reshape(-1, 2) + rng.normal(0.0, pixel_noise, (len(obj_pts), 2))


def _simulate(ik_tcp, n=14, pixel_noise=0.0, seed=0):
    """The samples ``_collect`` would produce on a known rig.

    Corners go out through the real fisheye model and come back through
    ``_tag_in_camera``, so the undistort plus planar-PnP path is exercised
    rather than assumed, and poses are tilted off the tag normal by at least
    what the command asks the user for -- a fronto-parallel view would leave
    the four-corner pose ambiguous, which is a property of the target and so
    belongs in the simulation too.
    """
    import cv2

    rng = np.random.default_rng(seed)
    obj_pts = _marker_points(MARKER_M)
    intr = load_shipped_gopro_intrinsics(FRAME_WH)
    tcp_in_flange = _offset_to_mat(ik_tcp)

    samples = []
    for _ in range(4000):
        if len(samples) >= n:
            break
        tilt = np.radians(rng.uniform(_MIN_TILT_DEG + 5.0, 45.0))
        about = rng.uniform(-np.pi, np.pi)
        flange_in_base = np.eye(4)
        flange_in_base[:3, :3] = (
            st.Rotation.from_rotvec([np.pi, 0.0, 0.0])
            # Roll about the lens axis conditions the solve; tilt off the tag
            # normal conditions each pose. Both are needed and they are not
            # the same motion.
            * st.Rotation.from_rotvec([0.0, 0.0, rng.uniform(-np.pi, np.pi)])
            * st.Rotation.from_rotvec(
                tilt * np.array([np.cos(about), np.sin(about), 0.0])
            )
        ).as_matrix()
        flange_in_base[:3, 3] = [0.25, -0.1, 0.32] + rng.uniform(-0.05, 0.05, 3)

        camera_in_base = flange_in_base @ CAMERA_IN_FLANGE
        tx_world2cam = np.linalg.inv(camera_in_base) @ TAG_IN_BASE
        if tx_world2cam[2, 3] < 0.12:
            continue

        px = _project(tx_world2cam, obj_pts, intr, rng, pixel_noise)
        if px.min() < 0 or px[:, 0].max() > FRAME_WH[0] or px[:, 1].max() > FRAME_WH[1]:
            continue

        solved = _tag_in_camera(px.astype(np.float32), obj_pts, intr)
        if solved is None:
            continue
        rvec, tvec, rms_px, ratio = solved
        if ratio < _MIN_AMBIGUITY_RATIO:
            continue
        recovered = np.eye(4)
        recovered[:3, :3] = cv2.Rodrigues(rvec)[0]
        recovered[:3, 3] = np.asarray(tvec).squeeze()

        pose = _mat_to_pose(flange_in_base @ tcp_in_flange)
        samples.append(
            {
                "tx_base2tcp": np.linalg.inv(_pose_to_mat(pose)),
                "tx_world2cam": recovered,
                "pose": pose,
                "rms_px": rms_px,
                "ratio": ratio,
                "tag_m": float(np.linalg.norm(tvec)),
            }
        )
    assert len(samples) == n, f"only staged {len(samples)} of {n} poses"
    return samples


def test_tag_in_camera_recovers_a_known_tag_pose():
    """Fisheye pixels -> tag pose, the input every other number rests on."""
    import cv2

    obj_pts = _marker_points(MARKER_M)
    intr = load_shipped_gopro_intrinsics(FRAME_WH)
    truth = np.eye(4)
    truth[:3, :3] = st.Rotation.from_rotvec([0.45, -0.3, 0.05]).as_matrix()
    truth[:3, 3] = [-0.02, -0.01, 0.34]

    px = _project(truth, obj_pts, intr, np.random.default_rng(0), 0.0)
    rvec, tvec, rms_px, ratio = _tag_in_camera(
        px.astype(np.float32), obj_pts, intr
    )

    assert rms_px < 0.05
    assert np.allclose(np.asarray(tvec).squeeze(), truth[:3, 3], atol=1e-4)
    assert np.allclose(cv2.Rodrigues(rvec)[0], truth[:3, :3], atol=1e-3)
    # Noise-free, so the mirrored solution is nowhere near as good.
    assert ratio > _MIN_AMBIGUITY_RATIO


def test_marker_points_are_centred_and_scale_with_the_printed_size():
    pts = _marker_points(0.16)

    assert pts.shape == (4, 3)
    assert np.allclose(pts[:, 2], 0.0)
    # Centred, which is both cv2.aruco's convention and what IPPE_SQUARE needs.
    assert np.allclose(pts.mean(axis=0), 0.0)
    assert pts[:, 0].max() == pytest.approx(0.08)
    assert pts[:, 1].max() == pytest.approx(0.08)
    # Corner order: top-left, top-right, bottom-right, bottom-left.
    assert pts[0].tolist() == [-0.08, 0.08, 0.0]
    assert pts[2].tolist() == [0.08, -0.08, 0.0]


def test_ambiguity_ratio_separates_a_tilted_tag_from_a_flat_on_one():
    """The gate has to earn its threshold, in both directions.

    Four coplanar corners fit two poses mirrored through the tag plane. Near
    fronto-parallel both reproject so well that noise decides, and the loser is
    wrong by tens of degrees. This is the check that the ratio actually tracks
    that, so the threshold is a measurement rather than a guess.
    """
    obj_pts = _marker_points(MARKER_M)
    intr = load_shipped_gopro_intrinsics(FRAME_WH)
    rng = np.random.default_rng(11)

    def trial(tilt_deg):
        out = []
        for _ in range(80):
            truth = np.eye(4)
            rotvec = np.array([np.radians(tilt_deg), 0.04, 0.02])
            truth[:3, :3] = st.Rotation.from_rotvec(rotvec).as_matrix()
            truth[:3, 3] = [0.0, 0.0, 0.30]
            px = _project(truth, obj_pts, intr, rng, 0.3)
            rvec, _tvec, _rms, ratio = _tag_in_camera(
                px.astype(np.float32), obj_pts, intr
            )
            off = np.degrees(np.linalg.norm(np.asarray(rvec).ravel() - rotvec))
            out.append((ratio, off > 2.0))
        return out

    flat = trial(5.0)
    tilted = trial(_MIN_TILT_DEG)

    # The gate's promise: nothing it accepts is the mirrored solution. This is
    # the assertion the threshold exists to satisfy, and it holds across both
    # tilts rather than depending on the user following the guidance.
    accepted = [(r, w) for r, w in flat + tilted if r >= _MIN_AMBIGUITY_RATIO]
    assert accepted, "the gate rejected everything, so it measures nothing"
    assert not any(wrong for _r, wrong in accepted)

    # And it is not rejecting at random: the mirrored solutions all live in
    # what it threw away, which is only true if the ratio tracks ambiguity.
    rejected = [(r, w) for r, w in flat + tilted if r < _MIN_AMBIGUITY_RATIO]
    assert sum(wrong for _r, wrong in rejected) > 0

    # Tilting past the guidance is what keeps good poses out of the reject
    # pile, so the advice and the gate agree.
    assert np.median([r for r, _w in tilted]) > _MIN_AMBIGUITY_RATIO
    assert np.median([r for r, _w in flat]) < np.median([r for r, _w in tilted])


def test_solve_recovers_the_rig_exactly():
    """Noise-free: any inverted or transposed frame shows up here immediately."""
    samples = _simulate(IK_TCP)
    tx_base2world, tx_tcp2camera = _solve(samples)

    assert np.allclose(tx_base2world, np.linalg.inv(TAG_IN_BASE), atol=1e-6)
    # tcp -> camera is constant across poses and equals the two fixed mounts.
    expected = np.linalg.inv(CAMERA_IN_FLANGE) @ _offset_to_mat(IK_TCP)
    assert np.allclose(tx_tcp2camera, expected, atol=1e-6)

    # Bounds are round-trip float noise from the iterative fisheye undistort,
    # not a tolerance on anything physical: an inverted or transposed frame
    # lands degrees and centimetres out, not microradians.
    trans_mm, rot_deg = _fit_error(samples, tx_base2world, tx_tcp2camera)
    assert trans_mm.max() < 1e-2
    assert rot_deg.max() < 1e-3


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
    """Four corners are enough: the arm, not the tag, is the error budget."""
    samples = _simulate(IK_TCP, pixel_noise=0.3, seed=3)
    _, tx_tcp2camera = _solve(samples)
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


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def test_grade_passes_a_clean_fit_and_fails_a_loose_one():
    samples = _simulate(IK_TCP, pixel_noise=0.3, seed=5)
    good = _grade(*_fit_error(samples, *_solve(samples)))

    assert good["passed"]
    assert good["trans_rms"] < _MAX_RESIDUAL_MM
    assert good["rot_rms"] < _MAX_ROT_RMS_DEG
    # The error bar that decides whether a correction is worth acting on.
    assert good["stderr_mm"] == pytest.approx(
        good["trans_rms"] / np.sqrt(good["n"]), rel=1e-9
    )

    loose = _grade(
        np.full(12, 28.5), np.full(12, 3.97)
    )
    assert not loose["passed"]
    assert [ok for *_, ok in loose["checks"]] == [False, False, True]

    # Too few poses fails on its own, however tight the residual.
    assert not _grade(np.full(4, 0.1), np.full(4, 0.01))["passed"]


def test_grade_flags_an_outlier_pose_but_not_uniform_error():
    spread = _grade(np.full(12, 28.5), np.full(12, 3.97))
    assert spread["outliers"] == []

    with_outlier = np.full(12, 1.0)
    with_outlier[7] = 40.0
    assert _grade(with_outlier, np.full(12, 0.2))["outliers"] == [7]

    # Three times the median but still inside the pass mark is not an outlier:
    # scaling noise is not the same as a wrong detection.
    small = np.full(12, 1.0)
    small[3] = 4.0
    assert _grade(small, np.full(12, 0.2))["outliers"] == []


def test_failed_report_prints_the_verdict_and_no_offsets(capsys):
    """The whole point of the verdict: a bad fit must hand over no numbers.

    A residual this loose cannot resolve the offset being measured, so printing
    it anyway -- which is what the old report did under a warning -- invites
    acting on noise. These assertions are what stop that coming back.
    """
    samples = _simulate(IK_TCP, pixel_noise=0.3, seed=7)
    trans_mm = np.full(len(samples), 28.5)
    grade = _grade(trans_mm, np.full(len(samples), 3.97))
    _report_fail(samples, trans_mm, grade)

    out = capsys.readouterr().out
    assert "CALIBRATION FAILED" in out
    assert "DO NOT USE THESE NUMBERS" in out
    assert "RUN IT AGAIN" not in out
    assert "28.5 mm rms" in out and "FAIL" in out
    # Nothing that could be pasted or measured against.
    for leak in ("quat=", "pos=", "fingertips in camera", "what to do with this"):
        assert leak not in out


def test_failed_report_names_a_bad_detection_and_clears_the_rig(capsys):
    """One wrecked pose is a re-run, not a rebuild, and it must say which."""
    samples = _simulate(IK_TCP, seed=9)
    # Wrench one tag pose off its true orientation: a misdetection, not noise.
    samples[4] = dict(samples[4])
    samples[4]["tx_world2cam"] = samples[4]["tx_world2cam"] @ _pose_to_mat(
        np.concatenate([[0.03, 0.0, 0.0], np.deg2rad([0.0, 12.0, 0.0])])
    )
    samples[4]["rms_px"] = 2.4

    trans_mm, rot_deg = _fit_error(samples, *_solve(samples))
    grade = _grade(trans_mm, rot_deg)
    assert not grade["passed"]
    _report_fail(samples, trans_mm, grade)

    out = capsys.readouterr().out
    assert "pose 5:" in out
    assert "2.40 px" in out
    assert "bad detections" in out
    assert "recapture" in out


def test_failed_report_blames_the_arm_when_the_error_is_spread(capsys):
    """Uniform error with a big rotation residual can only be kinematics.

    The distinction is the useful half of the diagnosis: re-running fixes a
    misdetection and reproduces an arm problem exactly, so saying which one
    this is decides what the user does next.
    """
    samples = _simulate(IK_TCP, seed=13)
    rng = np.random.default_rng(2)
    for s in samples:
        # Perturb the reported TCP orientation, which is what a hand-held arm
        # does: the vision is untouched and only the pose side is wrong.
        wobble = _pose_to_mat(
            np.concatenate([np.zeros(3), np.deg2rad(rng.normal(0.0, 4.0, 3))])
        )
        s["tx_base2tcp"] = wobble @ s["tx_base2tcp"]

    trans_mm, rot_deg = _fit_error(samples, *_solve(samples))
    grade = _grade(trans_mm, rot_deg)
    assert not grade["passed"]
    _diagnose(samples, trans_mm, grade)

    out = capsys.readouterr().out
    assert "spread across the poses" in out
    assert "forward kinematics" in out
    assert "one cause, whole budget" in out


def test_default_replacements_are_the_outliers_or_the_worst_pose():
    """A failed fit has to name what to recapture, not dump the whole run."""
    with_outlier = np.full(12, 1.0)
    with_outlier[7] = 40.0
    with_outlier[2] = 38.0
    assert _default_replacements(
        with_outlier, _grade(with_outlier, np.full(12, 0.2))
    ) == [2, 7]

    spread = np.linspace(10.0, 21.0, 12)
    grade = _grade(spread, np.full(12, 2.0))
    assert grade["outliers"] == []
    assert _default_replacements(spread, grade) == [int(np.argmax(spread))]


def test_parse_replacements_accepts_enter_numbers_and_quit():
    default = [0]
    assert _parse_replacements("", 12, default) == [0]
    assert _parse_replacements("1 9 12", 12, default) == [0, 8, 11]
    assert _parse_replacements("1, 9", 12, default) == [0, 8]
    assert _parse_replacements("q", 12, default) is None
    with pytest.raises(ValueError, match="not in 1..12"):
        _parse_replacements("0", 12, default)
    with pytest.raises(ValueError, match="not a pose number"):
        _parse_replacements("all", 12, default)


def test_ask_replacements_defaults_to_the_named_outliers(monkeypatch, capsys):
    samples = _simulate(IK_TCP, n=12, seed=9)
    trans_mm = np.full(12, 2.0)
    trans_mm[0] = 42.6
    grade = _grade(trans_mm, np.full(12, 0.2))
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")

    assert _ask_replacements(samples, trans_mm, grade) == [0]
    out = capsys.readouterr().out
    assert "Pose 1 (42.6 mm)" in out
    assert "poisoning the fit" in out


def test_substituting_an_outlier_recovers_a_clean_fit():
    """The recapture path has to actually clear a wrecked pose, not just ask."""
    clean = _simulate(IK_TCP, seed=9)
    wrecked = [dict(s) for s in clean]
    wrecked[0] = dict(wrecked[0])
    wrecked[0]["tx_world2cam"] = wrecked[0]["tx_world2cam"] @ _pose_to_mat(
        np.concatenate([[0.03, 0.0, 0.0], np.deg2rad([0.0, 12.0, 0.0])])
    )

    assert not _grade(*_fit_error(wrecked, *_solve(wrecked)))["passed"]
    wrecked[0] = clean[0]
    assert _grade(*_fit_error(wrecked, *_solve(wrecked)))["passed"]


def test_passing_report_prints_both_deliverables_and_the_verify_loop(capsys):
    samples = _simulate(IK_TCP, pixel_noise=0.3, seed=17)
    trans_mm, rot_deg = _fit_error(samples, *_solve(samples))
    grade = _grade(trans_mm, rot_deg)
    _, tx_tcp2camera = _solve(samples)
    derived = _derive(tx_tcp2camera, IK_TCP)
    _report_pass(derived, grade, "right_arm", IK_TCP, FRAME_WH)

    out = capsys.readouterr().out
    assert "CALIBRATION PASSED" in out
    assert "fingertips in camera axes" in out
    assert "what to do with this" in out
    # The camera shim, the ik_tcp quaternion (3 deg of yaw is well over the
    # floor for a clean fit), and the instruction to re-run and confirm.
    assert "Move the camera" in out
    assert "quat=(" in out
    assert "Re-run this command" in out
    assert "Cross-check" in out


def test_passing_report_withholds_a_rotation_it_cannot_resolve(capsys):
    """An angle inside the fit's own floor is not a measurement of anything."""
    samples = _simulate(IK_TCP, pixel_noise=0.3, seed=19)
    grade = _grade(*_fit_error(samples, *_solve(samples)))
    derived = dict(_derive(_solve(samples)[1], IK_TCP))
    derived["misalign_deg"] = grade["rot_floor_deg"] * 0.5

    _report_pass(derived, grade, "right_arm", IK_TCP, FRAME_WH)
    out = capsys.readouterr().out

    assert "Leave ik_tcp alone" in out
    assert "quat=(" not in out
    # The hardware half is unaffected by the rotation being unresolvable.
    assert "Move the camera" in out


def test_umi_training_tip_matches_the_dataset_plan_constants():
    """The constant the whole command compares against, from UMI's own terms."""
    cam_to_center_height = 0.086
    cam_to_mount_offset = 0.01465
    tcp_offset = 0.205
    assert np.allclose(
        UMI_TIP_IN_CAMERA,
        [0.0, cam_to_center_height, tcp_offset - cam_to_mount_offset],
    )


# --------------------------------------------------------------------------
# Detection gate and capture cycle
# --------------------------------------------------------------------------


class _FakeDetector:
    """Records the resolution of every image it is asked about."""

    def __init__(self, respond):
        self.respond = respond
        self.seen: list[tuple[int, int]] = []

    def detectMarkers(self, gray):
        self.seen.append(gray.shape[:2])
        return self.respond(gray)


def _quad(gray):
    h, w = gray.shape[:2]
    # A quad whose coordinates name the resolution it was found at.
    corners = np.array(
        [[[0.0, 0.0], [w / 2.0, 0.0], [w / 2.0, h / 2.0], [0.0, h / 2.0]]],
        dtype=np.float32,
    )
    return [corners], np.array([[_TAG_ID]]), []


def test_detect_measures_on_the_full_frame_not_the_gate():
    """The cheap gate runs on a small copy; the corners come from the full one.

    Returning the gate's corners would throw away the resolution the frame was
    read at, which is the whole reason the bus resize is dropped for the run.
    """
    import solarian.cli.commands.calibrate_hand_eye as mod

    det = _FakeDetector(_quad)
    corners, ids = mod._detect(
        np.zeros((1080, 1920), np.uint8), det, _TAG_ID, (640, 360)
    )

    assert det.seen == [(360, 640), (1080, 1920)]
    assert corners[1].tolist() == [960.0, 0.0]
    assert ids == [_TAG_ID]


def test_detect_skips_the_full_frame_when_the_gate_sees_nothing():
    """Misses are the common case while aiming, so they stop at the gate."""
    import solarian.cli.commands.calibrate_hand_eye as mod

    det = _FakeDetector(lambda _g: ([], None, []))
    gray = np.zeros((1080, 1920), np.uint8)

    assert mod._detect(gray, det, _TAG_ID, (640, 360)) == (None, [])
    assert det.seen == [(360, 640)]


def test_detect_ignores_other_tags_but_reports_seeing_them():
    """Ids 12-19 are all 160 mm and all on the table; only one is the world.

    Only one page of the sheet gets printed, so aiming at the wrong tag is the
    likely first-run mistake -- and to the detector it looks identical to an
    empty table. Handing the ids back is what lets the command tell them apart.
    """
    import solarian.cli.commands.calibrate_hand_eye as mod

    def others(gray):
        corners, _ids, rejected = _quad(gray)
        return corners, np.array([[_TAG_ID + 1]]), rejected

    det = _FakeDetector(others)
    gray = np.zeros((720, 1280), np.uint8)

    assert mod._detect(gray, det, _TAG_ID, (640, 360)) == (None, [_TAG_ID + 1])


def test_a_wrong_sheet_page_is_named_with_the_flag_that_fixes_it():
    """"Not found" and "that is tag 13" are different problems to the user."""
    import solarian.cli.commands.calibrate_hand_eye as mod

    empty = mod._missing_tag_reason(_TAG_ID, [])
    assert empty == f"tag {_TAG_ID} is not in this frame"
    assert "tag=" not in empty

    wrong = mod._missing_tag_reason(_TAG_ID, [_TAG_ID + 1])
    assert f"tag {_TAG_ID + 1} is" in wrong
    assert f"tag={_TAG_ID + 1}" in wrong


class _FakeArm:
    """Records torque calls, and slides its pose a little on every read."""

    def __init__(self, poses, drift_m=0.0, drift_deg=0.0):
        self.poses = poses
        self.index = 0
        self.reads = 0
        self.torque: list[str] = []
        self.drift_m = drift_m
        self.drift_deg = drift_deg

    def read_pose(self):
        self.reads += 1
        pose = np.array(
            self.poses[min(self.index, len(self.poses) - 1)], dtype=float
        )
        pose[0] += self.drift_m * self.reads
        pose[5] += np.radians(self.drift_deg) * self.reads
        return pose

    def start_calibration_mode(self):
        self.torque.append("free")

    def hold_measured_pose(self):
        self.torque.append("latch")
        return np.zeros(6)

    def enable_torque(self):
        self.torque.append("on")

    def enable_torque_holding(self):
        self.hold_measured_pose()
        self.enable_torque()
        self.torque.append("rewrite")
        return np.zeros(6)

    def joint_command_delta(self, _reference):
        return 0.0

    def disable_torque(self):
        self.torque.append("off")
        # The user walks the arm somewhere else for the next pose.
        self.index += 1

    def end_calibration_mode(self):
        self.torque.append("end")


class _FakeCam:
    """One read per measurement, then stops the loop from outside."""

    frequency = 30.0

    def __init__(self, stop_after):
        self.stop_after = stop_after
        self.reads = 0

    def read_sensors(self, _cls, names):
        self.reads += 1
        if self.reads > self.stop_after:
            raise KeyboardInterrupt
        return {names[0]: {"value": np.zeros((360, 640, 3), dtype=np.uint8)}}


def _pose(x, yaw_deg):
    return np.concatenate([[x, 0.0, 0.2], np.deg2rad([0.0, 0.0, yaw_deg])])


def _run_collect(
    monkeypatch,
    poses,
    stop_after,
    n_samples=99,
    corners=True,
    ratio=50.0,
    drift_m=0.0,
    drift_deg=0.0,
    arm_cls=_FakeArm,
    recapture=None,
):
    """Drive _collect headless, counting the confirmations it asks for.

    With no display _collect takes the blocking-prompt path, and ``input`` is
    patched to answer instantly, so one loop iteration is one confirmation and
    a pose costs three: hold, measure, release.
    """
    import cv2

    import solarian.cli.commands.calibrate_hand_eye as mod

    monkeypatch.setattr(
        mod,
        "_detect",
        lambda *_a: ((np.zeros((4, 2), np.float32), [_TAG_ID]) if corners else (None, [])),
    )
    monkeypatch.setattr(
        mod,
        "_tag_in_camera",
        lambda *_a: (np.zeros((3, 1)), np.array([0.0, 0.0, 0.3]), 0.2, ratio),
    )
    # No preview window in a test run; this is also the headless code path.
    monkeypatch.setattr(
        cv2,
        "namedWindow",
        lambda *_a, **_k: (_ for _ in ()).throw(cv2.error("no display")),
    )
    prompts: list[str] = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None, raising=False)

    arm = arm_cls(poses, drift_m=drift_m, drift_deg=drift_deg)
    cam = _FakeCam(stop_after)
    samples = mod._collect(
        arm,
        cam,
        "camera",
        _FakeDetector(_quad),
        _TAG_ID,
        _marker_points(MARKER_M),
        load_shipped_gopro_intrinsics(FRAME_WH),
        "bgr",
        n_samples,
        recapture,
    )
    return arm, cam, samples, prompts


def test_collect_holds_the_arm_for_every_sample_and_leaves_it_limp(monkeypatch):
    """The torque cycle is the whole point of the three-step capture.

    Latch before energising or the arm snaps to a stale goal; energise before
    the sample or the pose belongs to the user's grip instead of the servos;
    and come back off at the end, because end_calibration_mode does not touch
    torque and the user is holding the arm.
    """
    poses = [_pose(0.0, 30.0 * i) for i in range(6)]
    arm, _cam, samples, _prompts = _run_collect(
        monkeypatch, poses, stop_after=99, n_samples=2
    )

    assert len(samples) == 2
    assert arm.torque == [
        "free",
        "latch", "on", "rewrite", "off",
        "latch", "on", "rewrite", "off",
        # The finally, which is what a mid-hold interrupt relies on.
        "off", "end",
    ]
    # Torque stays on across the sample: off comes after the release confirm,
    # not after the measurement.


def test_collect_leaves_torque_off_when_interrupted_mid_hold(monkeypatch):
    """Interrupted while energised, the arm still has to come back limp."""
    poses = [_pose(0.0, 30.0 * i) for i in range(6)]
    # One camera read allowed, so the interrupt lands inside the first measure,
    # after torque went on.
    arm, _cam, samples, _prompts = _run_collect(monkeypatch, poses, stop_after=0)

    assert samples == []
    assert "on" in arm.torque
    assert arm.torque[-2:] == ["off", "end"]


def test_collect_does_not_drop_the_arm_until_the_release_confirm(monkeypatch):
    """After the sample the servos still hold; limp comes on the next Enter."""
    poses = [_pose(0.0, 40.0 * i) for i in range(8)]
    arm, _cam, samples, prompts = _run_collect(
        monkeypatch, poses, stop_after=99, n_samples=1
    )
    assert len(samples) == 1
    assert "hold the arm" in prompts[2]
    # Release confirm then finally. Measurement itself does not limp the arm.
    assert arm.torque == ["free", "latch", "on", "rewrite", "off", "off", "end"]


def test_collect_takes_one_sample_per_three_confirmations(monkeypatch):
    """Hold, measure, then take the arm before torque comes off."""
    poses = [_pose(0.0, 40.0 * i) for i in range(20)]
    _arm, cam, samples, prompts = _run_collect(
        monkeypatch, poses, stop_after=99, n_samples=3
    )

    assert len(samples) == 3
    assert len(prompts) == 9
    assert "position the arm" in prompts[0]
    assert "let go" in prompts[1]
    assert "hold the arm" in prompts[2]
    # One frame per measurement, and none grabbed while merely aiming.
    assert cam.reads == 3


def test_collect_names_the_pose_being_recaptured(monkeypatch):
    """A retry has to say which poison pose is being replaced."""
    poses = [_pose(0.0, 40.0 * i) for i in range(8)]
    _arm, _cam, samples, prompts = _run_collect(
        monkeypatch, poses, stop_after=99, n_samples=1, recapture=[1]
    )

    assert len(samples) == 1
    assert "Recapture pose 1" in prompts[0]
    assert "position the arm" in prompts[0]


def test_collect_drops_torque_when_a_joint_jumps_on_energising(monkeypatch):
    """A wrist_roll spin is invisible in TCP xyz; the joint check has to catch it."""

    class _JumpArm(_FakeArm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._jumps = 0

        def joint_command_delta(self, _reference):
            self._jumps += 1
            if self._jumps > 1:
                raise KeyboardInterrupt
            return 2000.0

    poses = [_pose(0.0, 30.0 * i) for i in range(6)]
    arm, _cam, samples, _prompts = _run_collect(
        monkeypatch, poses, stop_after=4, n_samples=2, arm_cls=_JumpArm
    )

    assert samples == []
    assert "on" in arm.torque
    assert arm.torque[-2:] == ["off", "end"]


def test_collect_rejects_a_sample_when_the_arm_moves_while_measuring(monkeypatch):
    """A frame paired with a pose the arm has already left is a silent lie.

    It is indistinguishable in the fit from a genuinely misaligned rig, which
    is exactly why it has to be caught at the source rather than diagnosed
    later.
    """
    poses = [_pose(0.0, 30.0 * i) for i in range(20)]
    _arm, _cam, samples, _prompts = _run_collect(
        monkeypatch, poses, stop_after=6, drift_deg=4.0
    )

    assert samples == []


def test_collect_records_nothing_without_the_tag(monkeypatch):
    """A confirmation with no tag in view is refused, not recorded blank."""
    poses = [_pose(0.0, 30.0 * i) for i in range(20)]
    _arm, _cam, samples, _prompts = _run_collect(
        monkeypatch, poses, stop_after=6, corners=False
    )

    assert samples == []


def test_collect_refuses_a_tag_too_flat_on_to_disambiguate(monkeypatch):
    """The ambiguity gate has to reject at capture time, not average it in."""
    poses = [_pose(0.0, 30.0 * i) for i in range(20)]
    _arm, _cam, samples, _prompts = _run_collect(
        monkeypatch, poses, stop_after=6, ratio=_MIN_AMBIGUITY_RATIO * 0.5
    )

    assert samples == []


def test_shell_line_reports_nothing_waiting_on_an_unselectable_stdin(monkeypatch):
    """No stdin to poll must read as "no keypress", not as a confirmation.

    Under pytest, systemd, or a pipe, sys.stdin may have no usable fileno at
    all. Treating that as a waiting line would fire a sample every frame.
    """
    import io as _io

    import solarian.cli.commands.calibrate_hand_eye as mod

    monkeypatch.setattr("sys.stdin", _io.StringIO("ignored"))
    assert mod._shell_line() is None
