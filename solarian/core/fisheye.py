"""Fisheye intrinsics IO and OpenCV calibration (UMI JSON schema).

Ported from ``umi.common.cv_util`` and ``calibrate_hdmi_intrinsics.py`` so
Solarian does not import the UMI repo at runtime.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

MIN_SAMPLES = 8
DEFAULT_BOARD = (9, 6)


def parse_fisheye_intrinsics(json_data: dict) -> dict:
    """UMI OpenCameraImuCalibration JSON -> OpenCV ``{DIM, K, D}``."""
    if json_data.get("intrinsic_type") != "FISHEYE":
        raise ValueError(
            f"expected intrinsic_type FISHEYE, got {json_data.get('intrinsic_type')!r}"
        )
    intr_data = json_data["intrinsics"]
    h = int(json_data["image_height"])
    w = int(json_data["image_width"])
    f = float(intr_data["focal_length"])
    px = float(intr_data["principal_pt_x"])
    py = float(intr_data["principal_pt_y"])
    kb8 = [
        float(intr_data["radial_distortion_1"]),
        float(intr_data["radial_distortion_2"]),
        float(intr_data["radial_distortion_3"]),
        float(intr_data["radial_distortion_4"]),
    ]
    return {
        "DIM": np.array([w, h], dtype=np.int64),
        "K": np.array(
            [
                [f, 0, px],
                [0, f, py],
                [0, 0, 1],
            ],
            dtype=np.float64,
        ),
        "D": np.array([kb8], dtype=np.float64).T,
    }


def convert_fisheye_intrinsics_resolution(
    opencv_intr_dict: dict,
    target_resolution: tuple[int, int],
) -> dict:
    """Rescale fisheye K to another resolution (UMI: height-normalized, symmetric H crop)."""
    import copy

    iw, ih = opencv_intr_dict["DIM"]
    iK = opencv_intr_dict["K"]
    ifx = iK[0, 0]
    ify = iK[1, 1]
    ipx = iK[0, 2]
    ipy = iK[1, 2]

    ow, oh = int(target_resolution[0]), int(target_resolution[1])
    ofx = ifx / ih * oh
    ofy = ify / ih * oh
    opx = (ipx - (iw / 2)) / ih * oh + (ow / 2)
    opy = ipy / ih * oh
    oK = np.array(
        [
            [ofx, 0, opx],
            [0, ofy, opy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    out = copy.deepcopy(opencv_intr_dict)
    out["DIM"] = np.array([ow, oh], dtype=np.int64)
    out["K"] = oK
    return out


def load_intrinsics(
    source: str | dict,
    actual_wh: Optional[tuple[int, int]] = None,
) -> dict:
    """Load UMI-schema JSON (path or dict) -> ``{K, D, DIM}``, optionally rescaled."""
    if isinstance(source, str):
        with open(source, "r") as fp:
            data = json.load(fp)
    else:
        data = source
    intr = parse_fisheye_intrinsics(data)
    if actual_wh is not None:
        cw, ch = int(intr["DIM"][0]), int(intr["DIM"][1])
        aw, ah = int(actual_wh[0]), int(actual_wh[1])
        if (cw, ch) != (aw, ah):
            intr = convert_fisheye_intrinsics_resolution(intr, (aw, ah))
    return intr


def umi_schema_dict(
    K: np.ndarray,
    D: np.ndarray,
    image_size: tuple[int, int],
    n_samples: int,
    rms: float,
    fps: float = 60.0,
) -> dict:
    """Build the UMI fisheye JSON object (does not write a file)."""
    w, h = image_size
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    f = 0.5 * (fx + fy)
    d = np.asarray(D, dtype=np.float64).reshape(-1)
    return {
        "final_reproj_error": float(rms),
        "fps": float(fps),
        "image_height": int(h),
        "image_width": int(w),
        "intrinsic_type": "FISHEYE",
        "intrinsics": {
            "aspect_ratio": (fx / fy) if fy != 0 else 1.0,
            "focal_length": f,
            "principal_pt_x": cx,
            "principal_pt_y": cy,
            "radial_distortion_1": float(d[0]),
            "radial_distortion_2": float(d[1]),
            "radial_distortion_3": float(d[2]),
            "radial_distortion_4": float(d[3]),
            "skew": 0.0,
        },
        "nr_calib_images": int(n_samples),
        "stabelized": False,
    }


def write_json_in_umi_schema(
    out_path: str,
    K: np.ndarray,
    D: np.ndarray,
    image_size: tuple[int, int],
    n_samples: int,
    rms: float,
    fps: float = 60.0,
) -> dict:
    data = umi_schema_dict(K, D, image_size, n_samples, rms, fps=fps)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(data, fp, indent=2)
    return data


def find_chessboard(gray: np.ndarray, board: tuple[int, int]):
    import cv2

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, board, flags)
    if not found:
        return None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)


def parse_board(s: str) -> tuple[int, int]:
    cols, rows = (int(x) for x in s.lower().split("x"))
    return cols, rows


def build_init_K(image_size, init_json_path: Optional[str] = None) -> np.ndarray:
    w, h = image_size
    if init_json_path and os.path.exists(init_json_path):
        with open(init_json_path, "r") as fp:
            data = json.load(fp)
        intr = data["intrinsics"]
        src_w = int(data["image_width"])
        src_h = int(data["image_height"])
        s_uniform = h / src_h
        sx = w / src_w
        sy = h / src_h
        f = float(intr["focal_length"])
        cx = float(intr["principal_pt_x"])
        cy = float(intr["principal_pt_y"])
        return np.array(
            [
                [f * s_uniform, 0, cx * sx],
                [0, f * s_uniform, cy * sy],
                [0, 0, 1.0],
            ],
            dtype=np.float64,
        )
    return np.array(
        [
            [420.0, 0, w / 2.0],
            [0, 420.0, h / 2.0],
            [0, 0, 1.0],
        ],
        dtype=np.float64,
    )


def _make_object_points(samples, board):
    cols, rows = board
    objp = np.zeros((1, cols * rows, 3), dtype=np.float64)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_points = [objp.copy() for _ in samples]
    img_points = [c.reshape(1, -1, 2).astype(np.float64) for c in samples]
    return obj_points, img_points


def _calibrate_once(samples, image_size, board, K_init, flags=None):
    import cv2

    obj_points, img_points = _make_object_points(samples, board)
    K = K_init.copy()
    D = np.zeros((4, 1), dtype=np.float64)
    n = len(samples)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n)]
    if flags is None:
        flags = (
            cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
            | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_FIX_SKEW
        )
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_points, img_points, image_size, K, D, rvecs, tvecs, flags, term
    )
    return rms, K, D, rvecs, tvecs


def _per_sample_rms(samples, board, K, D, rvecs, tvecs):
    import cv2

    obj_points, img_points = _make_object_points(samples, board)
    per_rms = np.zeros(len(samples), dtype=np.float64)
    for i, (op, ip) in enumerate(zip(obj_points, img_points)):
        proj, _ = cv2.fisheye.projectPoints(op, rvecs[i], tvecs[i], K, D)
        diff = proj.reshape(-1, 2) - ip.reshape(-1, 2)
        per_rms[i] = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return per_rms


def _auto_prune(
    samples,
    board,
    K,
    D,
    rvecs,
    tvecs,
    kept_idxs,
    image_size,
    K_init,
    target_rms=1.0,
    max_drop_frac=0.4,
):
    import cv2

    cur_samples = list(samples)
    cur_kept = list(kept_idxs)
    cur_K, cur_D = K, D
    cur_rvecs, cur_tvecs = rvecs, tvecs
    orig_n = len(samples)
    max_drop = int(round(orig_n * max_drop_frac))

    while True:
        per_rms = _per_sample_rms(
            cur_samples, board, cur_K, cur_D, cur_rvecs, cur_tvecs
        )
        med = float(np.median(per_rms))
        worst = float(per_rms.max())
        global_rms = float(np.sqrt(np.mean(per_rms**2)))
        print(
            f"  per-sample RMS: median={med:.3f} worst={worst:.3f} "
            f"global={global_rms:.3f} on {len(cur_samples)} samples"
        )
        if global_rms <= target_rms:
            return global_rms, cur_K, cur_D, cur_kept
        if (orig_n - len(cur_samples)) >= max_drop:
            print(
                f"  reached max drop fraction; stopping prune at "
                f"global RMS={global_rms:.3f}"
            )
            return global_rms, cur_K, cur_D, cur_kept

        threshold = max(target_rms, 2.0 * med)
        worst_idx = int(np.argmax(per_rms))
        if per_rms[worst_idx] <= threshold:
            print(
                f"  worst sample is within {threshold:.3f} px threshold; "
                f"stopping prune at global RMS={global_rms:.3f}"
            )
            return global_rms, cur_K, cur_D, cur_kept

        original_global_idx = cur_kept[worst_idx]
        print(
            f"  dropping sample (original idx {original_global_idx}) "
            f"with RMS={per_rms[worst_idx]:.3f} px"
        )
        cur_samples = [s for k, s in enumerate(cur_samples) if k != worst_idx]
        cur_kept = [g for k, g in enumerate(cur_kept) if k != worst_idx]
        if len(cur_samples) < MIN_SAMPLES:
            print(
                f"  cannot drop further: would go below MIN_SAMPLES={MIN_SAMPLES}"
            )
            return global_rms, cur_K, cur_D, cur_kept
        try:
            _new_rms, cur_K, cur_D, cur_rvecs, cur_tvecs = _calibrate_once(
                cur_samples, image_size, board, K_init
            )
        except cv2.error as e:
            print(f"  refit after drop failed: {e}; stopping prune")
            return global_rms, cur_K, cur_D, cur_kept


def calibrate_fisheye(
    samples,
    image_size,
    board,
    init_json_path: Optional[str] = None,
    target_rms: float = 1.0,
):
    """Robust fisheye calibration with leave-one-out + residual pruning."""
    import cv2

    K_init = build_init_K(image_size, init_json_path)
    try:
        rms, K, D, rvecs, tvecs = _calibrate_once(
            samples, image_size, board, K_init
        )
        print(f"  initial fit on all {len(samples)} samples: RMS={rms:.3f} px")
        return _auto_prune(
            samples,
            board,
            K,
            D,
            rvecs,
            tvecs,
            list(range(len(samples))),
            image_size,
            K_init,
            target_rms=target_rms,
        )
    except cv2.error as e:
        print(
            f"  full-set calibration failed: "
            f"{e.err if hasattr(e, 'err') else e}"
        )

    print(
        f"  running leave-one-out to find bad sample(s) "
        f"(this is O(N), N={len(samples)})..."
    )
    bad_idxs = []
    for i in range(len(samples)):
        trial = [s for j, s in enumerate(samples) if j != i]
        try:
            _calibrate_once(trial, image_size, board, K_init)
            bad_idxs.append(i)
        except cv2.error:
            pass

    if not bad_idxs:
        raise RuntimeError(
            "Calibration failed and leave-one-out found no single bad sample. "
            "Recapture with more tilted views and full-frame coverage."
        )

    surviving = [s for i, s in enumerate(samples) if i not in bad_idxs]
    print(
        f"  identified {len(bad_idxs)} bad sample(s) at indices "
        f"{bad_idxs}; recalibrating on {len(surviving)} survivors"
    )
    if len(surviving) < MIN_SAMPLES:
        raise RuntimeError(
            f"Only {len(surviving)} surviving samples after filtering; "
            f"need at least {MIN_SAMPLES}."
        )
    rms, K, D, rvecs, tvecs = _calibrate_once(
        surviving, image_size, board, K_init
    )
    kept = [i for i in range(len(samples)) if i not in bad_idxs]
    return _auto_prune(
        surviving,
        board,
        K,
        D,
        rvecs,
        tvecs,
        kept,
        image_size,
        K_init,
        target_rms=target_rms,
    )


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder: str) -> list[str]:
    """Sorted list of image paths in a folder (non-recursive)."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)
    names = [n for n in os.listdir(folder) if n.lower().endswith(IMG_EXTS)]
    names.sort()
    return [os.path.join(folder, n) for n in names]


def imread_color(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def detect_boards_in_folder(
    folder: str,
    board: tuple[int, int] = DEFAULT_BOARD,
) -> tuple[list[np.ndarray], tuple[int, int], list[str]]:
    """Detect chessboards in a folder of images. Returns (corners, image_size, kept_paths)."""
    import cv2

    paths = list_images(folder)
    samples: list[np.ndarray] = []
    kept: list[str] = []
    image_size = None
    for p in paths:
        img = imread_color(p)
        h, w = img.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            print(f"  skipping {os.path.basename(p)}: size {(w, h)} != {image_size}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = find_chessboard(gray, board)
        if corners is None:
            print(f"  {os.path.basename(p)}: no board")
            continue
        samples.append(corners.astype(np.float64))
        kept.append(p)
        print(f"  {os.path.basename(p)}: detected")
    if image_size is None:
        raise FileNotFoundError(f"no images in {folder}")
    return samples, image_size, kept
