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
    _FRAMES_PER_SAMPLE,
    _MARKER_MM,
    _MAX_TOPUP_BATCH,
    _MAX_RESIDUAL_MM,
    _MAX_ROT_RMS_DEG,
    _MAX_TIP_STDERR_MM,
    _MIN_AMBIGUITY_RATIO,
    _MIN_TILT_DEG,
    _SCALE_SUSPECT_FACTOR,
    _TAG_ID,
    _TOPUP_SAMPLES,
    _ask_topup,
    _bar_is_responding,
    _derive,
    _diagnose,
    _fit_error,
    _grade,
    _marker_points,
    _more_poses_would_help,
    _offset_to_mat,
    _parse_topup,
    _pose_to_mat,
    _projected_poses,
    _refine,
    _report_fail,
    _report_pass,
    _scale_excess,
    _solve,
    _tag_in_camera,
    _uncertainty,
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


def _verdict(samples):
    """The grade on a pose set, reached the way the command reaches it."""
    return _grade(*_fit_error(samples, *_solve(samples)), _uncertainty(samples))


def _bar(mm, rot_floor_deg=0.1):
    """A stand-in error bar, for grading residuals that have no poses behind
    them. The checks table is a pure function of its inputs, and pairing a
    synthetic residual with a synthetic bar is what lets each be varied
    without the other moving."""
    return {
        "tip_stderr_mm": mm,
        "tip_stderr_axes_mm": np.full(3, mm),
        "rot_floor_deg": rot_floor_deg,
    }


def _tip_error_mm(tx_tcp2camera, ik_tcp=None):
    """How far a solved transform puts the fingertips from where they are."""
    ik_tcp = IK_TCP if ik_tcp is None else ik_tcp
    truth = (np.linalg.inv(CAMERA_IN_FLANGE) @ _offset_to_mat(ik_tcp))[:3, 3]
    return float(np.linalg.norm(tx_tcp2camera[:3, 3] - truth) * 1000.0)


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


def test_refinement_beats_the_closed_form_it_starts_from():
    """SHAH is a starting point, not the answer.

    It recovers the two rotations from the rotation parts alone and then lets
    the translations follow, so an error in the rotations is never paid for
    by the translations and the pair it returns sits near the minimum of the
    residual rather than at it. On a rig whose arm misreports its own tool
    orientation -- which is every real one -- closing that gap is most of the
    accuracy available without collecting anything more.
    """
    import cv2

    closed_form, refined = [], []
    for seed in range(8):
        samples = _simulate(IK_TCP, n=14, pixel_noise=0.3, seed=900 + seed)
        rng = np.random.default_rng(seed)
        for s in samples:
            wobble = _pose_to_mat(
                np.concatenate([np.zeros(3), np.deg2rad(rng.normal(0.0, 1.0, 3))])
            )
            s["tx_base2tcp"] = wobble @ s["tx_base2tcp"]

        r_b2w, t_b2w, r_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
            R_world2cam=[s["tx_world2cam"][:3, :3] for s in samples],
            t_world2cam=[s["tx_world2cam"][:3, 3] for s in samples],
            R_base2gripper=[s["tx_base2tcp"][:3, :3] for s in samples],
            t_base2gripper=[s["tx_base2tcp"][:3, 3] for s in samples],
            method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
        )
        start_b2w, start_g2c = np.eye(4), np.eye(4)
        start_b2w[:3, :3], start_b2w[:3, 3] = r_b2w, np.asarray(t_b2w).squeeze()
        start_g2c[:3, :3], start_g2c[:3, 3] = r_g2c, np.asarray(t_g2c).squeeze()

        end_b2w, end_g2c = _refine(samples, start_b2w, start_g2c)
        closed_form.append(_tip_error_mm(start_g2c))
        refined.append(_tip_error_mm(end_g2c))

        # The residual is what it descends, and Levenberg-Marquardt cannot
        # come back up it. Distance to the truth is a different quantity and
        # is allowed to wander on any single run -- only its average is the
        # claim here -- so the monotonicity belongs on the residual.
        before = _fit_error(samples, start_b2w, start_g2c)[0]
        after = _fit_error(samples, end_b2w, end_g2c)[0]
        assert np.sqrt(np.mean(after**2)) <= np.sqrt(np.mean(before**2)) + 1e-9

    # Not a marginal improvement: it roughly halves the distance to the truth.
    assert np.mean(refined) < 0.7 * np.mean(closed_form)


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
    good = _verdict(samples)

    assert good["passed"]
    assert good["stderr_mm"] <= _MAX_TIP_STDERR_MM
    # The error bar is the jackknife, not a function of the residual.
    assert good["stderr_mm"] == pytest.approx(
        _uncertainty(samples)["tip_stderr_mm"], rel=1e-9
    )

    loose = _grade(np.full(12, 28.5), np.full(12, 3.97), _bar(9.0))
    assert not loose["passed"]

    # Too few poses fails on its own, however tight everything else is.
    assert not _grade(np.full(4, 0.1), np.full(4, 0.01), _bar(0.2))["passed"]


def test_the_error_bar_decides_the_verdict_not_the_residual():
    """The whole point of the rewrite, asserted from both sides.

    A residual that sits at the rig's own noise floor has to be able to pass,
    because no amount of care brings it down and grading on it is what made a
    failed run ask for replacement poses forever. A wide error bar has to
    fail even when every pose agrees with every other, because that is the
    number the printed offsets are acted on with.
    """
    # Scatter well past the old 8 mm / 1 deg pass marks, answer nailed down.
    noisy_rig = _grade(np.full(14, 6.9), np.full(14, 0.99), _bar(1.5))
    assert noisy_rig["passed"]

    # Tight residual, but the answer moves when any one pose is dropped.
    unresolved = _grade(np.full(14, 0.4), np.full(14, 0.05), _bar(7.0))
    assert not unresolved["passed"]

    # The ceilings are still ceilings: they catch a gross problem.
    assert not _grade(np.full(14, 40.0), np.full(14, 0.6), _bar(1.0))["passed"]
    assert not _grade(np.full(14, 6.0), np.full(14, 5.0), _bar(1.0))["passed"]


def test_grade_flags_an_outlier_pose_but_not_uniform_error():
    spread = _grade(np.full(12, 28.5), np.full(12, 3.97), _bar(9.0))
    assert spread["outliers"] == []

    with_outlier = np.full(12, 1.0)
    with_outlier[7] = 40.0
    assert _grade(with_outlier, np.full(12, 0.2), _bar(1.0))["outliers"] == [7]

    # Three times the median but still inside the floor is not an outlier:
    # scaling noise is not the same as a wrong detection.
    small = np.full(12, 1.0)
    small[3] = 4.0
    assert _grade(small, np.full(12, 0.2), _bar(1.0))["outliers"] == []


def test_the_error_bar_tracks_the_true_error_and_shrinks_with_poses():
    """The jackknife has to earn being the verdict.

    It is only worth gating on if it says something true about how far the
    answer is from the rig, and if taking more poses actually moves it --
    that second half is what the per-pose residual could not do, and so what
    left the old loop with nothing to converge towards.
    """
    bars, errors = {}, {}
    for n in (10, 20):
        bar, err = [], []
        for seed in range(6):
            samples = _simulate(IK_TCP, n=n, pixel_noise=0.3, seed=950 + seed)
            rng = np.random.default_rng(seed)
            for s in samples:
                wobble = _pose_to_mat(
                    np.concatenate(
                        [np.zeros(3), np.deg2rad(rng.normal(0.0, 1.0, 3))]
                    )
                )
                s["tx_base2tcp"] = wobble @ s["tx_base2tcp"]
            bar.append(_uncertainty(samples)["tip_stderr_mm"])
            err.append(_tip_error_mm(_solve(samples)[1]))
        bars[n], errors[n] = np.mean(bar), np.mean(err)

    # Honest: the same order as the error it is standing in for, and not
    # under it, since a bar that understates is worse than no bar.
    for n in (10, 20):
        assert 0.8 * errors[n] < bars[n] < 3.0 * errors[n]

    # And it responds to the one action the user can take. Doubling the poses
    # should buy about root two; assert clearly less than the whole of it so
    # the test is not measuring noise.
    assert bars[20] < 0.85 * bars[10]


def test_projected_poses_says_how_many_it_would_take():
    """The offer to add poses has to be quantified or it is just a retry.

    One over the root of the count, inverted: four times the poses for half
    the error bar. A user looking at "about forty" knows to go and fix the
    rig instead.
    """
    assert _projected_poses({"n": 12, "stderr_mm": _MAX_TIP_STDERR_MM}) == 12
    # Twice the limit means four times the poses.
    assert _projected_poses({"n": 12, "stderr_mm": 2 * _MAX_TIP_STDERR_MM}) == 48
    # Already inside it: nothing more needed.
    assert _projected_poses({"n": 12, "stderr_mm": 0.5 * _MAX_TIP_STDERR_MM}) == 3


def test_failed_report_prints_the_verdict_and_no_offsets(capsys):
    """The whole point of the verdict: a bad fit must hand over no numbers.

    A residual this loose cannot resolve the offset being measured, so printing
    it anyway -- which is what the old report did under a warning -- invites
    acting on noise. These assertions are what stop that coming back.
    """
    samples = _simulate(IK_TCP, pixel_noise=0.3, seed=7)
    trans_mm = np.full(len(samples), 28.5)
    grade = _grade(trans_mm, np.full(len(samples), 3.97), _bar(9.0))
    _report_fail(samples, trans_mm, grade)

    out = capsys.readouterr().out
    assert "CALIBRATION FAILED" in out
    assert "DO NOT USE THESE NUMBERS" in out
    assert "RUN IT AGAIN" not in out
    assert "28.5 mm rms" in out and "FAIL" in out
    # Nothing that could be pasted or measured against.
    for leak in ("quat=", "pos=", "fingertips in camera", "what to do with this"):
        assert leak not in out


def _wreck(samples, index, rms_px=2.4):
    """Wrench one tag pose off its truth: a misdetection, not noise."""
    samples[index] = dict(samples[index])
    samples[index]["tx_world2cam"] = samples[index]["tx_world2cam"] @ _pose_to_mat(
        np.concatenate([[0.03, 0.0, 0.0], np.deg2rad([0.0, 12.0, 0.0])])
    )
    samples[index]["rms_px"] = rms_px
    return samples


def test_failed_report_names_a_bad_detection_and_clears_the_rig(capsys):
    """One wrecked pose must be named, and the rig cleared of blame."""
    samples = _wreck(_simulate(IK_TCP, seed=9), 4)

    trans_mm, rot_deg = _fit_error(samples, *_solve(samples))
    grade = _grade(trans_mm, rot_deg, _uncertainty(samples))
    _report_fail(samples, trans_mm, grade)

    out = capsys.readouterr().out
    assert "pose 5:" in out
    assert "2.40 px" in out
    assert "bad detections" in out
    # Nothing is asked of the user for a pose the command can simply drop.
    assert "recapture" not in out.lower()


def _wobble_arm(samples, deg, seed=2):
    """Perturb the reported TCP orientation, which is what a loose arm does:
    the vision is untouched and only the pose side is wrong."""
    rng = np.random.default_rng(seed)
    for s in samples:
        s["tx_base2tcp"] = (
            _pose_to_mat(
                np.concatenate([np.zeros(3), np.deg2rad(rng.normal(0.0, deg, 3))])
            )
            @ s["tx_base2tcp"]
        )
    return samples


def test_failed_report_blames_the_arm_when_the_error_is_spread(capsys):
    """Uniform error with a big rotation residual can only be kinematics.

    The distinction is the useful half of the diagnosis: more poses narrow an
    error bar but cannot move a residual, so saying which of the two is wrong
    decides whether the user keeps going or goes and fixes the arm.
    """
    samples = _wobble_arm(_simulate(IK_TCP, seed=13), 4.0)

    trans_mm, rot_deg = _fit_error(samples, *_solve(samples))
    grade = _grade(trans_mm, rot_deg, _uncertainty(samples))
    assert not grade["passed"]
    assert grade["rot_rms"] > _MAX_ROT_RMS_DEG
    _diagnose(samples, trans_mm, grade)

    out = capsys.readouterr().out
    assert "spread across the poses" in out
    assert "forward kinematics" in out
    # And it must not send the user back to do the same thing again.
    assert "reproduces it exactly" in out


def test_diagnosis_names_the_marker_when_the_rotation_cannot_explain_it(capsys):
    """A translation residual with no rotation behind it is a scale error.

    It is the one failure the error bar cannot see -- every pose shares it,
    so every leave-one-out solve moves with it and the jackknife stays
    tight. Distances in this problem come from the tag and nowhere else, so
    a residual the orientation cannot account for points at the one number
    the user typed in.
    """
    samples = _simulate(IK_TCP, n=14, pixel_noise=0.05, seed=23)
    # The marker really is 4 percent larger than the command was told, so
    # every measured tag distance comes back short by the same factor.
    for s in samples:
        s["tx_world2cam"] = s["tx_world2cam"].copy()
        s["tx_world2cam"][:3, 3] *= 1.04

    trans_mm, rot_deg = _fit_error(samples, *_solve(samples))
    grade = _grade(trans_mm, rot_deg, _bar(0.5))
    _diagnose(samples, trans_mm, grade)

    out = capsys.readouterr().out
    assert "marker=<mm>" in out
    assert "edge to edge" in out
    # Not blamed on the arm, which is the wrong thing to go and check.
    assert "forward kinematics" not in out


def test_parse_topup_accepts_enter_a_count_and_quit():
    assert _parse_topup("", 4) == 4
    assert _parse_topup("6", 4) == 6
    assert _parse_topup("q", 4) is None
    assert _parse_topup("no", 4) is None
    with pytest.raises(ValueError, match="at least one pose"):
        _parse_topup("0", 4)
    with pytest.raises(ValueError, match="not a number of poses"):
        _parse_topup("pose 3", 4)


def test_ask_topup_offers_more_poses_and_never_a_replacement(monkeypatch, capsys):
    """What the user is asked for has to be the thing that helps.

    Swapping a pose for another draws a fresh residual from the same
    distribution, which on a rig sitting at its own noise floor is why the
    old prompt could repeat forever. Adding poses narrows the error bar being
    graded, so the prompt has to ask for that and say how many.
    """
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")

    # The default has to be the number just quoted, not a token handful.
    # Offering four against a stated shortfall of 36 is the command
    # disagreeing with itself, and it is what capped a real run at twelve
    # extra poses when it had asked for thirty-seven.
    assert _ask_topup({"n": 12, "stderr_mm": 6.0}) == _MAX_TOPUP_BATCH
    # _para wraps to the terminal, so the prose is matched unwrapped.
    out = " ".join(capsys.readouterr().out.split())
    assert "6.0 mm of error bar" in out
    # 12 poses at twice the limit needs four times as many.
    assert "48 poses in total" in out
    assert "about 36 more" in out
    assert "recapture" not in out.lower()
    assert "replace" not in out.lower()

    # A small shortfall asks for a small round, with a floor so a round is
    # never a single pose.
    assert _ask_topup({"n": 20, "stderr_mm": 3.2}) == _TOPUP_SAMPLES
    capsys.readouterr()


def test_dropping_an_outlier_recovers_a_clean_fit():
    """The auto-drop has to actually clear a wrecked pose.

    This is the whole justification for dropping rather than asking: if the
    remaining poses pass on their own, a recapture round buys the user
    nothing but time. The loop drops before it checks the verdict for the
    same reason -- a misdetection left in a fit that scrapes a pass still
    drags the answer, and the poses to answer with are already in hand.
    """
    clean = _simulate(IK_TCP, n=14, seed=9)
    wrecked = _wreck([dict(s) for s in clean], 0)

    grade = _verdict(wrecked)
    assert grade["outliers"] == [0]

    kept = [s for i, s in enumerate(wrecked) if i not in grade["outliers"]]
    assert _verdict(kept)["passed"]

    # The drop is worth making: it is millimetres on the answer, not a
    # tidier-looking residual.
    assert _tip_error_mm(_solve(wrecked)[1]) > 2.0
    assert _tip_error_mm(_solve(kept)[1]) < 0.5


def test_the_error_bar_sees_the_damage_a_bad_pose_does():
    """A misdetection has to show up in the number that gates the run.

    Dropping it is an optimisation; the guarantee is that a fit dragged off
    by a bad pose cannot quietly report itself as precise. Leaving that pose
    out moves the answer, and moving the answer is exactly what the
    jackknife measures.
    """
    wrecked = _wreck([dict(s) for s in _simulate(IK_TCP, n=14, seed=9)], 0)
    bar = _uncertainty(wrecked)["tip_stderr_mm"]

    # It should land near the bias the bad pose actually causes, which is
    # what makes it a usable warning rather than a coincidence.
    assert bar == pytest.approx(_tip_error_mm(_solve(wrecked)[1]), rel=0.5)


def test_only_a_wide_error_bar_is_offered_more_poses():
    """Failures route to the action that can actually fix them.

    Offering poses for a failure poses cannot fix is precisely the loop that
    does not terminate, so which failures get the offer is the load-bearing
    decision in the whole rewrite.
    """
    # Poses disagreeing at random about an answer they nearly share: the
    # average of more of them is better, so ask for more.
    assert _more_poses_would_help(
        _grade(np.full(14, 6.0), np.full(14, 0.9), _bar(5.0))
    )

    # The rig itself, past a ceiling. Every further pose reproduces it.
    assert not _more_poses_would_help(
        _grade(np.full(14, 40.0), np.full(14, 0.9), _bar(5.0))
    )
    assert not _more_poses_would_help(
        _grade(np.full(14, 6.0), np.full(14, 5.0), _bar(5.0))
    )


def test_a_suspected_scale_error_does_not_block_the_top_up():
    """The two problems are independent and must be treated as such.

    A mismeasured marker biases every pose identically, so it adds no
    scatter and is not why the error bar is wide. Refusing to continue on
    the suspicion would stall a run that poses would have finished -- the
    old non-termination wearing a new hat -- so it is reported and the
    top-up goes ahead.
    """
    samples = _simulate(IK_TCP, n=14, pixel_noise=0.05, seed=23)
    for s in samples:
        s["tx_world2cam"] = s["tx_world2cam"].copy()
        s["tx_world2cam"][:3, 3] *= 1.04

    trans_mm, rot_deg = _fit_error(samples, *_solve(samples))
    grade = _grade(trans_mm, rot_deg, _bar(5.0))

    assert _scale_excess(samples, grade) > _SCALE_SUSPECT_FACTOR
    assert _more_poses_would_help(grade)


def test_the_loop_stops_on_a_stalled_bar_not_on_a_round_count():
    """What ends the loop has to be evidence, not a counter.

    A round count cannot tell a rig that needs forty poses and is walking
    towards them from one with a systematic floor. Cutting off the first is
    how a real run got told it needed thirty-seven more poses and then
    allowed twelve.
    """
    first = {"n": 12, "stderr_mm": 6.0}

    # Tracking 1/sqrt(n) from where it started: 24 poses should be near
    # 4.2 mm, and it is. Still failing, still worth continuing.
    assert _bar_is_responding(first, {"n": 24, "stderr_mm": 3.9})
    # Even running somewhat behind the curve, within the slack.
    assert _bar_is_responding(first, {"n": 24, "stderr_mm": 5.5})
    # Flat despite doubling the poses: something else is holding it up.
    assert not _bar_is_responding(first, {"n": 24, "stderr_mm": 6.4})
    assert not _bar_is_responding(first, {"n": 48, "stderr_mm": 5.0})


def test_measure_waits_out_an_arm_that_has_not_settled(monkeypatch):
    """A wrist still ringing is a moment, not a verdict on the pose.

    Handing it back costs the user the whole three-confirmation cycle and a
    walk back to the same place, for something that a pause fixes. A real
    run lost fourteen poses this way.
    """
    import solarian.cli.commands.calibrate_hand_eye as mod

    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky(*_a, **_k):
        calls["n"] += 1
        if calls["n"] < 3:
            return None, "the corners moved 4.2 px between frames", True
        return {"ok": True}, None, False

    monkeypatch.setattr(mod, "_measure_once", flaky)
    sample, reason = mod._measure(*[None] * 10)
    assert sample == {"ok": True} and reason is None
    assert calls["n"] == 3, "it should have waited, not given up"

    # A refusal about where the arm is pointed comes straight back: waiting
    # cannot turn a tag that is out of frame into one that is in it.
    calls["n"] = 0
    monkeypatch.setattr(
        mod, "_measure_once", lambda *_a, **_k: (None, "no tag in view", False)
    )
    sample, reason = mod._measure(*[None] * 10)
    assert sample is None and reason == "no tag in view"


def test_topping_up_converges_where_recapturing_could_not():
    """The loop has to reach a verdict, which is the entire point.

    Replacing the worst pose drew a fresh residual from the same
    distribution and promoted a different pose to worst, forever. Adding
    poses moves the graded number every time, so the same rig that used to
    spin now finishes -- in a bounded number of rounds, from a start that
    genuinely fails.
    """
    from solarian.cli.commands.calibrate_hand_eye import (
        _MAX_TOPUP_ROUNDS,
        _TOPUP_SAMPLES,
    )

    pool = _wobble_arm(_simulate(IK_TCP, n=32, pixel_noise=0.3, seed=47), 1.6)
    samples = pool[:8]

    grade = _verdict(samples)
    assert not grade["passed"], "start from a fit that actually fails"
    started_at = grade["trans_rms"]

    rounds = 0
    while not grade["passed"]:
        assert _more_poses_would_help(grade)
        rounds += 1
        assert rounds <= _MAX_TOPUP_ROUNDS, "did not converge inside the cap"
        samples = pool[: len(samples) + _TOPUP_SAMPLES]
        grade = _verdict(samples)

    assert grade["stderr_mm"] <= _MAX_TIP_STDERR_MM
    # It converged by narrowing the bar, not because the rig got better: the
    # residual is no smaller than it started. That is the distinction the old
    # gate could not make, and the reason it had nothing to converge towards.
    assert grade["trans_rms"] >= started_at


def test_a_noise_floor_rig_is_not_asked_to_recapture_anything(capsys):
    """The bug this rewrite exists to kill, asserted end to end.

    An arm that misreports its tool orientation by half a degree leaves a
    residual no capture technique brings down. The old command graded that
    residual, so every round replaced the worst pose, a different pose became
    the worst, and it never terminated. Here it simply passes.
    """
    samples = _wobble_arm(_simulate(IK_TCP, n=14, pixel_noise=0.3, seed=31), 0.5)
    grade = _verdict(samples)

    # Exactly the shape of the run that used to loop: scatter past the old
    # 8 mm / 1 deg pass marks, and no single pose to blame for it.
    assert grade["outliers"] == []
    assert grade["trans_rms"] > 1.5
    assert grade["passed"]
    assert grade["stderr_mm"] <= _MAX_TIP_STDERR_MM


def test_passing_report_prints_both_deliverables_and_the_verify_loop(capsys):
    samples = _simulate(IK_TCP, pixel_noise=0.3, seed=17)
    grade = _verdict(samples)
    _, tx_tcp2camera = _solve(samples)
    derived = _derive(tx_tcp2camera, IK_TCP)
    _report_pass(samples, derived, grade, "right_arm", IK_TCP, FRAME_WH)

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


def test_passing_report_says_the_error_bar_is_precision_not_truth(capsys):
    """A tight bar around a biased answer must not read as a guarantee.

    The jackknife cannot see anything every pose shares, and the marker's
    printed size is exactly that: get it wrong and every offset is wrong by
    the same percentage while the bar stays just as tight. A passing report
    that did not say so would be inviting the user to trust the wrong thing.
    """
    samples = _simulate(IK_TCP, pixel_noise=0.3, seed=29)
    grade = _verdict(samples)
    _report_pass(
        samples, _derive(_solve(samples)[1], IK_TCP), grade, "right_arm", IK_TCP, FRAME_WH
    )

    out = capsys.readouterr().out
    assert "precision, not on truth" in out
    assert f"{_MARKER_MM:.0f} mm edge to edge" in out


def test_passing_report_warns_when_the_residual_smells_of_scale(capsys):
    """The silent failure has to be loud somewhere, and a pass is where.

    A mismeasured marker is common-mode: it biases every pose alike, so the
    poses agree, the error bar is tight and the run passes with a
    confidently wrong answer. The residual is the only witness -- more
    translation than the orientation error can account for at that distance
    -- so a passing run says so rather than leaving it to be discovered by
    a robot missing its grasp.
    """
    clean = _simulate(IK_TCP, n=14, pixel_noise=0.3, seed=29)
    grade = _verdict(clean)
    assert grade["passed"]
    _report_pass(clean, _derive(_solve(clean)[1], IK_TCP), grade, "a", IK_TCP, FRAME_WH)
    assert "WARNING" not in capsys.readouterr().out, "healthy rig must stay quiet"

    off = _simulate(IK_TCP, n=14, pixel_noise=0.05, seed=23)
    for s in off:
        s["tx_world2cam"] = s["tx_world2cam"].copy()
        s["tx_world2cam"][:3, 3] *= 1.04
    grade = _verdict(off)
    assert grade["passed"], "the error bar cannot see this, which is the point"
    _report_pass(off, _derive(_solve(off)[1], IK_TCP), grade, "a", IK_TCP, FRAME_WH)

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "marker=<mm>" in out


def test_passing_report_withholds_a_rotation_it_cannot_resolve(capsys):
    """An angle inside the fit's own floor is not a measurement of anything."""
    samples = _simulate(IK_TCP, pixel_noise=0.3, seed=19)
    grade = _verdict(samples)
    derived = dict(_derive(_solve(samples)[1], IK_TCP))
    derived["misalign_deg"] = grade["rot_floor_deg"] * 0.5

    _report_pass(samples, derived, grade, "right_arm", IK_TCP, FRAME_WH)
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
    """Counts frame grabs, then stops the loop from outside."""

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
    first_pose=1,
    jitter_px=0.0,
):
    """Drive _collect headless, counting the confirmations it asks for.

    With no display _collect takes the blocking-prompt path, and ``input`` is
    patched to answer instantly, so one loop iteration is one confirmation and
    a pose costs three: hold, measure, release.
    """
    import cv2

    import solarian.cli.commands.calibrate_hand_eye as mod

    # Detection is patched at the frame level, so a measurement still makes
    # _FRAMES_PER_SAMPLE of them and the averaging path is the one exercised.
    # jitter_px walks the corners between frames, which is what the sample
    # has to notice and refuse.
    frames = {"n": 0}

    def fake_detect(*_a):
        if not corners:
            return None, []
        frames["n"] += 1
        return (
            np.full((4, 2), jitter_px * (frames["n"] % 2), np.float32),
            [_TAG_ID],
        )

    monkeypatch.setattr(mod, "_detect", fake_detect)
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
        np.eye(4),
        n_samples,
        first_pose,
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
    # A measurement averages a burst of frames, and none are grabbed while
    # merely aiming.
    assert cam.reads == 3 * _FRAMES_PER_SAMPLE


def test_collect_numbers_a_top_up_round_on_from_the_poses_already_taken(
    monkeypatch,
):
    """Added poses continue the count; nothing is being replaced.

    The number in the prompt is the number the failure report just used, so
    the two have to agree or the user cannot tell which pose is which.
    """
    poses = [_pose(0.0, 40.0 * i) for i in range(8)]
    _arm, _cam, samples, prompts = _run_collect(
        monkeypatch, poses, stop_after=99, n_samples=2, first_pose=13
    )

    assert len(samples) == 2
    assert "Pose 13 (1/2 this round)" in prompts[0]
    assert "Pose 14 (2/2 this round)" in prompts[3]
    assert "recapture" not in " ".join(prompts).lower()


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


def test_collect_averages_a_burst_of_frames_into_one_sample(monkeypatch):
    """Corner jitter is independent between frames; the corner is not.

    Averaging is the cheapest precision in the command, but only if the
    frames really are taken and really are averaged, so both are asserted:
    the burst is grabbed, and the corners that come out sit at the mean of
    what the frames said rather than at whichever one arrived last.
    """
    poses = [_pose(0.0, 40.0 * i) for i in range(8)]
    _arm, cam, samples, _prompts = _run_collect(
        monkeypatch, poses, stop_after=99, n_samples=1, jitter_px=1.0
    )

    assert len(samples) == 1
    assert cam.reads == _FRAMES_PER_SAMPLE
    # Corners alternate between 0 and 1 px across the burst, so the spread
    # about the mean is reported and is under half the swing.
    assert 0.0 < samples[0]["jitter_px"] <= 1.0


def test_collect_refuses_a_sample_whose_corners_will_not_hold_still(monkeypatch):
    """Averaging assumes the corner underneath the jitter is fixed.

    Whole pixels of wander mean it is not -- something is being knocked, or
    the exposure is too long to freeze it -- and averaging that produces a
    confident number for a pose that never existed.
    """
    poses = [_pose(0.0, 30.0 * i) for i in range(20)]
    _arm, _cam, samples, _prompts = _run_collect(
        monkeypatch, poses, stop_after=99, n_samples=1, jitter_px=40.0
    )

    assert samples == []


def test_collect_refuses_a_tag_that_only_shows_up_in_some_frames(monkeypatch):
    """A tag on the edge of detectable is not a pose worth averaging.

    The frames that did land are the ones flattered by noise, so the average
    of them is biased rather than merely noisy.
    """
    import solarian.cli.commands.calibrate_hand_eye as mod

    seen = {"n": 0}

    def flaky(*_a):
        seen["n"] += 1
        # Found in every other frame: enough to look like a detection, not
        # enough to be one.
        if seen["n"] % 2:
            return None, []
        return np.zeros((4, 2), np.float32), [_TAG_ID]

    monkeypatch.setattr(mod, "_detect", flaky)
    sample, reason = mod._measure(
        _FakeArm([_pose(0.0, 0.0)]),
        _FakeCam(99),
        "camera",
        _FakeDetector(_quad),
        _TAG_ID,
        _marker_points(MARKER_M),
        load_shipped_gopro_intrinsics(FRAME_WH),
        "bgr",
        (640, 360),
        np.eye(4),
    )

    assert sample is None
    assert "only found in" in reason


def test_measure_reports_the_pose_in_policy_axes(monkeypatch):
    """Measured on the sensor's frame, reported in the one the policy uses.

    The remap rotation is the only thing separating the two, so applying it
    is what makes a raw-frame measurement mean the same as a warped-frame
    one. Getting it backwards would be silent: the numbers stay plausible.
    """
    import solarian.cli.commands.calibrate_hand_eye as mod

    monkeypatch.setattr(
        mod, "_detect", lambda *_a: (np.zeros((4, 2), np.float32), [_TAG_ID])
    )
    monkeypatch.setattr(
        mod,
        "_tag_in_camera",
        lambda *_a: (np.zeros((3, 1)), np.array([0.1, 0.2, 0.3]), 0.2, 50.0),
    )

    turn = np.eye(4)
    turn[:3, :3] = st.Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_matrix()
    args = (
        "camera",
        _FakeDetector(_quad),
        _TAG_ID,
        _marker_points(MARKER_M),
        load_shipped_gopro_intrinsics(FRAME_WH),
        "bgr",
        (640, 360),
    )
    poses = [_pose(0.0, 0.0)]
    plain, _ = mod._measure(_FakeArm(poses), _FakeCam(99), *args, np.eye(4))
    turned, _ = mod._measure(_FakeArm(poses), _FakeCam(99), *args, turn)

    assert np.allclose(plain["tx_world2cam"][:3, 3], [0.1, 0.2, 0.3])
    # A quarter turn about z takes (x, y) to (-y, x).
    assert np.allclose(turned["tx_world2cam"][:3, 3], [-0.2, 0.1, 0.3])
    # The distance to the tag is a property of the rig, not of the axes.
    assert turned["tag_m"] == pytest.approx(plain["tag_m"])


def test_shell_line_reports_nothing_waiting_on_an_unselectable_stdin(monkeypatch):
    """No stdin to poll must read as "no keypress", not as a confirmation.

    Under pytest, systemd, or a pipe, sys.stdin may have no usable fileno at
    all. Treating that as a waiting line would fire a sample every frame.
    """
    import io as _io

    import solarian.cli.commands.calibrate_hand_eye as mod

    monkeypatch.setattr("sys.stdin", _io.StringIO("ignored"))
    assert mod._shell_line() is None
