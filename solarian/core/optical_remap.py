"""Warp a third-party fisheye camera into GoPro Hero 10 + Max Lens Mod space.

Production entry point: :func:`load_aligner` + :func:`apply_optical_remap`.
The saved JSON embeds both cameras' UMI-schema intrinsics so it is portable.
"""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np

from solarian.core.fisheye import load_intrinsics

GOPRO_TARGET = "gopro_hero10_max_lens_mod_1080p"
GOPRO_OUT_SIZE = (1920, 1080)
DATA_PACKAGE = "solarian.data.umi_gopro"
GOPRO_INTRINSICS_NAME = "gopro_hero10_max_lens_mod_1080p.json"
COVERAGE_MASK_NAME = "gopro_coverage_mask.png"
CHECKERBOARD_NAME = "checkerboard_9x6.png"
GOLD_IMAGE_NAME = "umi_gold_camera0_rgb_t0.png"
CHECKERBOARD_URL = "https://github.com/opencv/opencv/blob/master/doc/pattern.png"
REPROJ_PASS_PX = 2.0
REPROJ_GREAT_PX = 1.0
# Homography translation of the optical axis beyond this is almost always
# "two cameras were not on the same mount", not a gripper alignment tweak.
HOMOGRAPHY_SHIFT_WARN_DEG = 5.0


def package_data_path(name: str) -> Path:
    """Filesystem path to a shipped ``solarian.data.umi_gopro`` file."""
    return Path(str(resources.files(DATA_PACKAGE).joinpath(name)))


def shipped_gopro_intrinsics_path() -> Path:
    return package_data_path(GOPRO_INTRINSICS_NAME)


def shipped_coverage_mask_path() -> Path:
    return package_data_path(COVERAGE_MASK_NAME)


def shipped_checkerboard_path() -> Path:
    return package_data_path(CHECKERBOARD_NAME)


def shipped_gold_image_path() -> Path:
    return package_data_path(GOLD_IMAGE_NAME)


def load_shipped_gopro_intrinsics(
    actual_wh: Optional[tuple[int, int]] = None,
) -> dict:
    return load_intrinsics(str(shipped_gopro_intrinsics_path()), actual_wh)


def load_shipped_gopro_schema() -> dict:
    with open(shipped_gopro_intrinsics_path(), "r") as fp:
        return json.load(fp)


def _undistort_to_norm(pts_xy: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    import cv2

    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.fisheye.undistortPoints(pts, K, D).reshape(-1, 2)


def _undistort_to_rays(pts_xy: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    norm = _undistort_to_norm(pts_xy, K, D)
    rays = np.concatenate([norm, np.ones((norm.shape[0], 1))], axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays


def estimate_homography(
    gopro_pts: np.ndarray,
    rpi_pts: np.ndarray,
    K_g,
    D_g,
    K_r,
    D_r,
) -> tuple[np.ndarray, float]:
    """Planar homography H mapping user -> GoPro normalized coords."""
    import cv2

    ng = _undistort_to_norm(gopro_pts, K_g, D_g)
    nr = _undistort_to_norm(rpi_pts, K_r, D_r)
    H, _mask = cv2.findHomography(nr, ng, method=0)
    if H is None:
        raise RuntimeError("findHomography failed (degenerate correspondences)")
    nr_h = np.concatenate([nr, np.ones((len(nr), 1))], axis=1)
    pg = (H @ nr_h.T).T
    pg = pg[:, :2] / pg[:, 2:3]
    rms = float(np.sqrt(np.mean(np.sum((pg - ng) ** 2, axis=1))))
    return H, rms


def homography_optical_axis_shift_deg(H: np.ndarray) -> float:
    """Degrees the homography yaws/pitches the user optical axis in GoPro space."""
    H = np.asarray(H, dtype=np.float64)
    x = float(H[0, 2] / H[2, 2])
    y = float(H[1, 2] / H[2, 2])
    return float(np.degrees(np.arctan(np.hypot(x, y))))


def estimate_relative_rotation(
    gopro_pts: np.ndarray,
    rpi_pts: np.ndarray,
    K_g,
    D_g,
    K_r,
    D_r,
) -> tuple[np.ndarray, float]:
    """Rotation R such that ray_gopro ~= R @ ray_user (Wahba / Kabsch)."""
    rg = _undistort_to_rays(gopro_pts, K_g, D_g)
    rr = _undistort_to_rays(rpi_pts, K_r, D_r)
    H = rr.T @ rg
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    pred = (R @ rr.T).T
    cos = np.clip(np.sum(pred * rg, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    rms = float(np.sqrt(np.mean(ang**2)))
    return R, rms


class Aligner:
    """Warps a user fisheye image into the GoPro image plane via cv2.remap."""

    def __init__(
        self,
        gopro_intr,
        camera_intr,
        transform: np.ndarray,
        out_size: Optional[tuple[int, int]] = None,
        model: str = "rotation",
    ):
        self.Kg = gopro_intr["K"]
        self.Dg = gopro_intr["D"]
        self.Kr = camera_intr["K"]
        self.Dr = camera_intr["D"]
        self.H = np.asarray(transform, dtype=np.float64)
        self.Hinv = np.linalg.inv(self.H)
        self.model = model
        if out_size is None:
            out_size = (int(gopro_intr["DIM"][0]), int(gopro_intr["DIM"][1]))
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self.map1, self.map2, self.valid = self._build_maps()

    def _build_maps(self):
        import cv2

        w, h = self.out_size
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
        ng = _undistort_to_norm(grid, self.Kg, self.Dg)
        ng_h = np.concatenate([ng, np.ones((len(ng), 1))], axis=1)
        nr_h = (self.Hinv @ ng_h.T).T
        valid = nr_h[:, 2] > 1e-9
        nr = np.full((len(nr_h), 2), -1.0, dtype=np.float64)
        nr[valid] = nr_h[valid, :2] / nr_h[valid, 2:3]
        dist = cv2.fisheye.distortPoints(
            nr.reshape(-1, 1, 2), self.Kr, self.Dr
        ).reshape(-1, 2)
        dist[~valid] = -1.0
        map_x = dist[:, 0].reshape(h, w).astype(np.float32)
        map_y = dist[:, 1].reshape(h, w).astype(np.float32)
        valid_mask = valid.reshape(h, w)
        return map_x, map_y, valid_mask

    def warp(self, img: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.remap(
            img,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def forward_points(self, camera_pts: np.ndarray) -> np.ndarray:
        import cv2

        nr = _undistort_to_norm(camera_pts, self.Kr, self.Dr)
        nr_h = np.concatenate([nr, np.ones((len(nr), 1))], axis=1)
        ng_h = (self.H @ nr_h.T).T
        ng = ng_h[:, :2] / ng_h[:, 2:3]
        return cv2.fisheye.distortPoints(
            ng.reshape(-1, 1, 2), self.Kg, self.Dg
        ).reshape(-1, 2)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "transform": self.H.tolist(),
            "out_size": list(self.out_size),
        }


def identity_aligner(gopro_intr, camera_intr, out_size=None) -> Aligner:
    return Aligner(
        gopro_intr,
        camera_intr,
        np.eye(3, dtype=np.float64),
        out_size=out_size or GOPRO_OUT_SIZE,
        model="rotation",
    )


def compute_coverage_mask(frames: list[np.ndarray], thresh: int = 10) -> np.ndarray:
    import cv2
    import scipy.ndimage as ndi

    acc = None
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        acc = g if acc is None else np.maximum(acc, g)
    mask = acc > thresh
    mask = ndi.binary_opening(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = lbl == biggest
    return mask


def load_coverage_mask(
    path: Optional[str] = None,
    out_size: tuple[int, int] = GOPRO_OUT_SIZE,
) -> np.ndarray:
    import cv2

    w, h = out_size
    src = Path(path) if path else shipped_coverage_mask_path()
    if src.is_file():
        cov = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if cov is None:
            raise FileNotFoundError(f"could not read coverage mask: {src}")
        if cov.shape[:2] != (h, w):
            cov = cv2.resize(cov, (w, h), interpolation=cv2.INTER_NEAREST)
        return cov > 127
    return np.ones((h, w), dtype=bool)


def apply_optical_remap(
    frame: np.ndarray,
    aligner: Aligner,
    coverage: Optional[np.ndarray] = None,
) -> np.ndarray:
    """User native frame -> GoPro-resolution BGR/RGB with coverage blanked."""
    warped = aligner.warp(frame)
    keep = aligner.valid
    if coverage is not None:
        cov = coverage
        if cov.shape[:2] != keep.shape:
            import cv2

            cov = cv2.resize(
                cov.astype(np.uint8),
                (keep.shape[1], keep.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        keep = keep & cov
    out = warped.copy()
    out[~keep] = 0
    return out


def approximate_fov_deg(intr: dict) -> tuple[float, float]:
    """Horizontal / vertical FOV (deg) from edge pixels via fisheye undistort."""
    K, D = intr["K"], intr["D"]
    w, h = int(intr["DIM"][0]), int(intr["DIM"][1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    left = _undistort_to_rays(np.array([[0.0, cy]]), K, D)[0]
    right = _undistort_to_rays(np.array([[w - 1.0, cy]]), K, D)[0]
    top = _undistort_to_rays(np.array([[cx, 0.0]]), K, D)[0]
    bot = _undistort_to_rays(np.array([[cx, h - 1.0]]), K, D)[0]
    optical = np.array([0.0, 0.0, 1.0])

    def ang(a, b):
        c = float(np.clip(np.dot(a, b), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    fov_h = ang(left, optical) + ang(right, optical)
    fov_v = ang(top, optical) + ang(bot, optical)
    return fov_h, fov_v


def coverage_fraction(aligner: Aligner, coverage: Optional[np.ndarray] = None) -> float:
    gopro = coverage if coverage is not None else np.ones(aligner.valid.shape, dtype=bool)
    denom = max(int(gopro.sum()), 1)
    return float((aligner.valid & gopro).sum()) / float(denom)


def save_remap(
    path: str,
    aligner: Aligner,
    gopro_schema: dict,
    camera_schema: dict,
    extra: Optional[dict] = None,
) -> dict:
    data = {
        "target": GOPRO_TARGET,
        "gopro_intrinsics": gopro_schema,
        "camera_intrinsics": camera_schema,
        **aligner.to_dict(),
    }
    if extra:
        data["metrics"] = extra
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fp:
        json.dump(data, fp, indent=2)
    return data


def _opencv_from_payload(payload, actual_wh=None) -> dict:
    if isinstance(payload, str):
        return load_intrinsics(payload, actual_wh)
    if isinstance(payload, dict) and payload.get("intrinsic_type") == "FISHEYE":
        return load_intrinsics(payload, actual_wh)
    # Already OpenCV-style
    return payload


def load_aligner(path: str) -> Aligner:
    with open(path, "r") as fp:
        data = json.load(fp)
    gopro_src = data.get("gopro_intrinsics")
    cam_src = data.get("camera_intrinsics") or data.get("rpi_intrinsics")
    if gopro_src is None or cam_src is None:
        raise ValueError(f"remap JSON missing intrinsics: {path}")
    gopro_intr = _opencv_from_payload(gopro_src)
    camera_intr = _opencv_from_payload(cam_src)
    out_size = tuple(data["out_size"]) if data.get("out_size") else GOPRO_OUT_SIZE
    if "transform" in data:
        transform = np.array(data["transform"], dtype=np.float64)
        model = data.get("model", "rotation")
    else:
        transform = np.array(data["R"], dtype=np.float64)
        model = "rotation"
    return Aligner(
        gopro_intr, camera_intr, transform, out_size=out_size, model=model
    )


def build_remap_from_images(
    user_dir: str,
    gopro_dir: Optional[str] = None,
    board: tuple[int, int] = (9, 6),
    out_path: Optional[str] = None,
    geometry_dir: Optional[str] = None,
    init_json_path: Optional[str] = None,
    min_samples: int = 8,
) -> tuple[Aligner, dict]:
    """Calibrate a user camera folder and remap it onto shipped GoPro optics.

    If ``gopro_dir`` is set, solves a homography from paired checkerboards
    (sorted filename order). Otherwise uses identity rotation.
    Returns ``(aligner, remap_json_dict)``.
    """
    import cv2

    from solarian.core.fisheye import (
        MIN_SAMPLES,
        calibrate_fisheye,
        detect_boards_in_folder,
        find_chessboard,
        imread_color,
        list_images,
        umi_schema_dict,
    )

    min_n = max(min_samples, MIN_SAMPLES)
    print(f"Detecting {board[0]}x{board[1]} chessboard in {user_dir}...")
    samples, image_size, kept_paths = detect_boards_in_folder(user_dir, board)
    if len(samples) < min_n:
        raise RuntimeError(
            f"only {len(samples)} detections in {user_dir}; need >= {min_n}"
        )
    print(f"Calibrating on {len(samples)} detections...")
    if init_json_path is None:
        init_json_path = str(shipped_gopro_intrinsics_path())
    rms, K, D, kept_idxs = calibrate_fisheye(
        samples, image_size, board, init_json_path=init_json_path
    )
    print(
        f"  RMS reprojection error: {rms:.4f} px on "
        f"{len(kept_idxs)}/{len(samples)} samples"
    )
    camera_schema = umi_schema_dict(K, D, image_size, len(kept_idxs), rms)
    gopro_schema = load_shipped_gopro_schema()
    camera_intr = load_intrinsics(camera_schema)
    gopro_intr = load_intrinsics(gopro_schema, GOPRO_OUT_SIZE)

    metrics: dict = {
        "camera_calib_rms_px": float(rms),
        "n_calib_images": int(len(kept_idxs)),
    }
    cam_fov = approximate_fov_deg(camera_intr)
    go_fov = approximate_fov_deg(gopro_intr)
    metrics["camera_fov_h_deg"] = cam_fov[0]
    metrics["camera_fov_v_deg"] = cam_fov[1]
    metrics["gopro_fov_h_deg"] = go_fov[0]
    metrics["gopro_fov_v_deg"] = go_fov[1]
    print(
        f"  FOV user ~ {cam_fov[0]:.1f}x{cam_fov[1]:.1f} deg  "
        f"GoPro ~ {go_fov[0]:.1f}x{go_fov[1]:.1f} deg"
    )

    reproj_rms = None
    if gopro_dir:
        g_paths = list_images(gopro_dir)
        u_paths = list_images(user_dir)
        n = min(len(g_paths), len(u_paths))
        if n == 0:
            raise RuntimeError("no paired GoPro/user images")
        # Prefer the first pair where BOTH boards detect at native size.
        g_pts = r_pts = None
        pair_g = pair_u = None
        for gp, up in zip(g_paths[:n], u_paths[:n]):
            gbgr = imread_color(gp)
            ubgr = imread_color(up)
            gc = find_chessboard(cv2.cvtColor(gbgr, cv2.COLOR_BGR2GRAY), board)
            uc = find_chessboard(cv2.cvtColor(ubgr, cv2.COLOR_BGR2GRAY), board)
            if gc is None or uc is None:
                continue
            g_pts = gc.reshape(-1, 2)
            r_pts = uc.reshape(-1, 2)
            pair_g, pair_u = gbgr, ubgr
            print(f"  homography pair: {os.path.basename(gp)} + {os.path.basename(up)}")
            break
        if g_pts is None:
            raise RuntimeError(
                "checkerboard not found in any paired GoPro+user image"
            )
        gopro_pair_intr = load_intrinsics(gopro_schema, (pair_g.shape[1], pair_g.shape[0]))
        user_pair_intr = load_intrinsics(camera_schema, (pair_u.shape[1], pair_u.shape[0]))
        transform, norm_rms = estimate_homography(
            g_pts, r_pts,
            gopro_pair_intr["K"], gopro_pair_intr["D"],
            user_pair_intr["K"], user_pair_intr["D"],
        )
        print(f"  planar homography; normalized residual RMS = {norm_rms:.6f}")
        shift_deg = homography_optical_axis_shift_deg(transform)
        metrics["homography_axis_shift_deg"] = shift_deg
        if shift_deg > HOMOGRAPHY_SHIFT_WARN_DEG:
            print(
                f"  WARNING: homography yaws the optical axis by ~{shift_deg:.1f} deg.\n"
                "  That happens when the GoPro and user camera were not on the same\n"
                "  mount; the gripper overlay will look shifted/warped. For a wrist\n"
                "  camera, skip the GoPro (optics-only identity remap)."
            )
        aligner = Aligner(
            gopro_pair_intr, user_pair_intr, transform,
            out_size=GOPRO_OUT_SIZE, model="homography",
        )
        pred = aligner.forward_points(r_pts)
        err = np.linalg.norm(pred - g_pts, axis=1)
        reproj_rms = float(np.sqrt(np.mean(err ** 2)))
        reproj_mean = float(np.mean(err))
        reproj_max = float(np.max(err))
        verdict = (
            "EXCELLENT (<1px)" if reproj_rms < REPROJ_GREAT_PX
            else "PASS (<2px)" if reproj_rms < REPROJ_PASS_PX
            else "FAIL (>=2px)"
        )
        print(
            f"  checkerboard reproj: mean={reproj_mean:.3f}px  "
            f"rms={reproj_rms:.3f}px  max={reproj_max:.3f}px  -> {verdict}"
        )
        metrics.update({
            "reproj_rms_px": reproj_rms,
            "reproj_mean_px": reproj_mean,
            "reproj_max_px": reproj_max,
            "homography_norm_rms": float(norm_rms),
        })
        if geometry_dir:
            os.makedirs(geometry_dir, exist_ok=True)
            warped = apply_optical_remap(
                pair_u, aligner, load_coverage_mask(out_size=aligner.out_size)
            )
            g_resized = pair_g
            if (pair_g.shape[1], pair_g.shape[0]) != GOPRO_OUT_SIZE:
                g_resized = cv2.resize(pair_g, GOPRO_OUT_SIZE)
            cv2.imwrite(os.path.join(geometry_dir, "gopro.png"), g_resized)
            cv2.imwrite(os.path.join(geometry_dir, "user_warped.png"), warped)
            diff = cv2.absdiff(g_resized, warped)
            diff[~aligner.valid] = 0
            sbs = np.concatenate([g_resized, warped, diff], axis=1)
            cv2.imwrite(os.path.join(geometry_dir, "side_by_side.png"), sbs)
            print(f"  wrote geometry checks -> {geometry_dir}")
    else:
        aligner = identity_aligner(gopro_intr, camera_intr, out_size=GOPRO_OUT_SIZE)
        print("  identity rotation (no paired GoPro); optics-only remap")

    cov = load_coverage_mask(out_size=aligner.out_size)
    frac = coverage_fraction(aligner, cov)
    metrics["gopro_coverage_frac"] = frac
    print(f"  GoPro pixels filled by user camera: {frac*100:.1f}%")

    data = {
        "target": GOPRO_TARGET,
        "gopro_intrinsics": gopro_schema,
        "camera_intrinsics": camera_schema,
        **aligner.to_dict(),
        "metrics": metrics,
    }
    if out_path:
        save_remap(out_path, aligner, gopro_schema, camera_schema, extra=metrics)
        print(f"  wrote remap -> {out_path}")
    return aligner, data


def peek_remap_out_size(path: str, conf_dir: Optional[Path] = None) -> tuple[int, int]:
    """``(width, height)`` of remap output without building cv2 maps."""
    resolved = resolve_remap_path(path, conf_dir)
    with open(resolved, "r") as fp:
        data = json.load(fp)
    if data.get("out_size"):
        w, h = data["out_size"]
        return int(w), int(h)
    return GOPRO_OUT_SIZE


def camera_output_hw(cfg, conf_dir: Optional[Path] = None) -> tuple[int, int]:
    """``(height, width)`` of frames emitted by a camera bus after remap/resize."""
    if cfg.resize_width is not None and cfg.resize_height is not None:
        return int(cfg.resize_height), int(cfg.resize_width)
    if getattr(cfg, "optical_remap", None):
        try:
            w, h = peek_remap_out_size(str(cfg.optical_remap), conf_dir)
            return h, w
        except FileNotFoundError:
            return GOPRO_OUT_SIZE[1], GOPRO_OUT_SIZE[0]
    if cfg.rotation in (90, -90):
        return int(cfg.width), int(cfg.height)
    return int(cfg.height), int(cfg.width)


def resolve_remap_path(raw: str, conf_dir: Optional[Path] = None) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    if conf_dir is not None:
        cand = Path(conf_dir) / raw
        if cand.is_file():
            return cand
        cand = Path(conf_dir).parent / raw
        if cand.is_file():
            return cand
    project = Path(os.environ.get("SOLARIAN_PROJECT_PATH", os.getcwd()))
    cand = project / raw
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"optical_remap not found: {raw}")


class OpticalRemap:
    """Cached aligner + coverage for per-frame warp on a camera bus."""

    def __init__(self, remap_path: str, conf_dir: Optional[Path] = None):
        self.path = resolve_remap_path(remap_path, conf_dir)
        self.aligner = load_aligner(str(self.path))
        self.coverage = load_coverage_mask(out_size=self.aligner.out_size)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return apply_optical_remap(frame, self.aligner, self.coverage)


def load_bus_remap(cfg) -> Optional[OpticalRemap]:
    path = getattr(cfg, "optical_remap", None)
    if not path:
        return None
    conf_dir = Path(os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf")))
    return OpticalRemap(str(path), conf_dir)


def finish_camera_frame(processed: np.ndarray, cfg, remap: Optional[OpticalRemap]):
    """Optical remap (native res) then optional resize. Input/output match bus color."""
    import cv2

    if remap is not None:
        processed = remap.apply(processed)
    if cfg.resize_width is not None and cfg.resize_height is not None:
        if processed.shape[0] != cfg.resize_height or processed.shape[1] != cfg.resize_width:
            processed = cv2.resize(
                processed,
                (int(cfg.resize_width), int(cfg.resize_height)),
                interpolation=cv2.INTER_AREA,
            )
    return processed
