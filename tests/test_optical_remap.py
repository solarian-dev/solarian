"""Optical remap onto shipped GoPro Hero 10 + Max Lens Mod HDMI 1080p optics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from solarian.core.configs import SensorConfig
from solarian.core.optical_remap import (
    GOPRO_OUT_SIZE,
    REPROJ_PASS_PX,
    Aligner,
    apply_optical_remap,
    approximate_fov_deg,
    build_remap_from_images,
    camera_output_hw,
    finish_camera_frame,
    identity_aligner,
    load_aligner,
    load_coverage_mask,
    load_shipped_gopro_intrinsics,
    save_remap,
    shipped_checkerboard_path,
    shipped_coverage_mask_path,
    shipped_gold_image_path,
    shipped_gopro_intrinsics_path,
)

# tests/ -> solarian repo; sibling UMI checkout lives next to it.
CAM_COMPARE = (
    Path(__file__).resolve().parents[1].parent
    / "universal_manipulation_interface"
    / "data"
    / "cam_compare"
)


def test_shipped_gopro_assets_exist():
    assert shipped_gopro_intrinsics_path().is_file()
    assert shipped_coverage_mask_path().is_file()
    assert shipped_checkerboard_path().is_file()
    assert shipped_gold_image_path().is_file()
    data = json.loads(shipped_gopro_intrinsics_path().read_text())
    assert data["image_width"] == 1920
    assert data["image_height"] == 1080
    assert data["intrinsic_type"] == "FISHEYE"
    import cv2

    gold = cv2.imread(str(shipped_gold_image_path()), cv2.IMREAD_COLOR)
    assert gold is not None
    assert gold.shape[:2] == (224, 224)


def test_resolve_gold_defaults_to_shipped_and_honors_override(tmp_path):
    import cv2
    from omegaconf import OmegaConf

    from solarian.cli.commands.calibrate_umi_camera import _resolve_gold

    default = _resolve_gold(OmegaConf.create({}), OmegaConf.create({}))
    assert default == shipped_gold_image_path()

    custom = tmp_path / "custom_gold.png"
    cv2.imwrite(str(custom), np.zeros((224, 224, 3), dtype=np.uint8))
    override = _resolve_gold(OmegaConf.create({"gold": str(custom)}), OmegaConf.create({}))
    assert override == custom


def test_identity_aligner_roundtrip_same_intrinsics():
    intr = load_shipped_gopro_intrinsics()
    aligner = identity_aligner(intr, intr)
    cx, cy = float(intr["K"][0, 2]), float(intr["K"][1, 2])
    pts = np.array(
        [
            [cx, cy],
            [cx, cy - 80],
            [cx + 60, cy],
        ],
        dtype=np.float64,
    )
    pred = aligner.forward_points(pts)
    assert pred.shape == pts.shape
    assert np.allclose(pred, pts, atol=2.0)

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[int(cy) - 20 : int(cy) + 20, int(cx) - 20 : int(cx) + 20] = 255
    warped = apply_optical_remap(frame, aligner, coverage=np.ones((1080, 1920), dtype=bool))
    assert warped.shape == frame.shape
    assert warped[int(cy), int(cx)].min() > 200


def test_save_load_embeds_intrinsics(tmp_path):
    gopro = load_shipped_gopro_intrinsics()
    aligner = identity_aligner(gopro, gopro)
    gopro_schema = json.loads(shipped_gopro_intrinsics_path().read_text())
    out = tmp_path / "remap.json"
    save_remap(
        str(out),
        aligner,
        gopro_schema,
        gopro_schema,
        extra={"camera_calib_rms_px": 0.5},
    )
    data = json.loads(out.read_text())
    assert data["target"] == "gopro_hero10_max_lens_mod_1080p"
    assert data["gopro_intrinsics"]["intrinsic_type"] == "FISHEYE"
    assert data["camera_intrinsics"]["intrinsic_type"] == "FISHEYE"
    loaded = load_aligner(str(out))
    assert loaded.out_size == GOPRO_OUT_SIZE
    assert loaded.model == "rotation"
    assert np.allclose(loaded.H, np.eye(3))


def test_camera_config_optical_remap_optional():
    cam = SensorConfig(
        port="0",
        fps=30,
        width=2028,
        height=1520,
        rotation=0,
        channels=3,
        color_mode="rgb",
        video_format="MJPG",
        resize_width=640,
        resize_height=360,
        optical_remap="conf/calibrations/right_wrist_camera_gopro_remap.json",
    )
    assert cam.optical_remap.endswith("gopro_remap.json")
    h, w = camera_output_hw(cam)
    assert (h, w) == (360, 640)


def test_finish_camera_frame_identity_then_16_9_resize():
    gopro = load_shipped_gopro_intrinsics()
    aligner = identity_aligner(gopro, gopro)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[500:580, 900:1020] = (0, 0, 255)
    warped = apply_optical_remap(frame, aligner, load_coverage_mask())
    assert warped.shape[:2] == (1080, 1920)

    class _Cfg:
        resize_width = 640
        resize_height = 360
        optical_remap = None

    class _Remap:
        def apply(self, img):
            return warped

    out = finish_camera_frame(frame, _Cfg(), _Remap())
    assert out.shape == (360, 640, 3)


def test_approximate_fov_positive():
    intr = load_shipped_gopro_intrinsics()
    fov_h, fov_v = approximate_fov_deg(intr)
    assert fov_h > fov_v > 90.0


def _cam_compare_ok() -> bool:
    return (CAM_COMPARE / "checker" / "rpi").is_dir() and (
        CAM_COMPARE / "checker" / "gopro"
    ).is_dir()


@pytest.mark.skipif(not _cam_compare_ok(), reason="cam_compare checker data not present")
def test_homography_from_existing_intrinsics():
    import cv2

    from solarian.core.fisheye import find_chessboard, imread_color, list_images, load_intrinsics
    from solarian.core.optical_remap import estimate_homography

    gopro_dir = CAM_COMPARE / "checker" / "gopro"
    rpi_dir = CAM_COMPARE / "checker" / "rpi"
    g_paths = list_images(str(gopro_dir))
    r_paths = list_images(str(rpi_dir))
    g_pts = r_pts = None
    gbgr = rbgr = None
    for gp, rp in zip(g_paths, r_paths):
        gbgr = imread_color(gp)
        rbgr = imread_color(rp)
        gc = find_chessboard(cv2.cvtColor(gbgr, cv2.COLOR_BGR2GRAY), (9, 6))
        rc = find_chessboard(cv2.cvtColor(rbgr, cv2.COLOR_BGR2GRAY), (9, 6))
        if gc is not None and rc is not None:
            g_pts = gc.reshape(-1, 2)
            r_pts = rc.reshape(-1, 2)
            break
    assert g_pts is not None, "no paired board detections"
    gopro_intr = load_intrinsics(
        str(CAM_COMPARE / "gopro_intrinsics.json"), (gbgr.shape[1], gbgr.shape[0])
    )
    rpi_intr = load_intrinsics(
        str(CAM_COMPARE / "rpi_intrinsics.json"), (rbgr.shape[1], rbgr.shape[0])
    )
    H, _ = estimate_homography(
        g_pts, r_pts, gopro_intr["K"], gopro_intr["D"], rpi_intr["K"], rpi_intr["D"]
    )
    aligner = Aligner(
        gopro_intr, rpi_intr, H, out_size=(gbgr.shape[1], gbgr.shape[0]), model="homography"
    )
    pred = aligner.forward_points(r_pts)
    err = np.linalg.norm(pred - g_pts, axis=1)
    rms = float(np.sqrt(np.mean(err**2)))
    assert rms < REPROJ_PASS_PX, f"reproj rms {rms:.3f} px"


@pytest.mark.skipif(not _cam_compare_ok(), reason="cam_compare checker data not present")
def test_build_remap_from_cam_compare_images(tmp_path):
    out = tmp_path / "remap.json"
    geom = tmp_path / "geometry"
    aligner, data = build_remap_from_images(
        str(CAM_COMPARE / "checker" / "rpi"),
        gopro_dir=str(CAM_COMPARE / "checker" / "gopro"),
        board=(9, 6),
        out_path=str(out),
        geometry_dir=str(geom),
    )
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded["camera_intrinsics"]["intrinsic_type"] == "FISHEYE"
    assert loaded["gopro_intrinsics"]["image_width"] == 1920
    rms = data["metrics"]["reproj_rms_px"]
    assert rms < REPROJ_PASS_PX, f"reproj rms {rms:.3f} px"
    assert (geom / "side_by_side.png").is_file()
    from solarian.core.fisheye import imread_color, list_images

    rpi = imread_color(list_images(str(CAM_COMPARE / "checker" / "rpi"))[0])
    warped = apply_optical_remap(rpi, aligner, load_coverage_mask())
    assert warped.shape[:2] == (1080, 1920)


def test_remapped_gopro_frame_umi_crop_is_224():
    import cv2

    gopro = load_shipped_gopro_intrinsics()
    aligner = identity_aligner(gopro, gopro)
    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    frame[400:700, 700:1200] = (200, 180, 160)
    warped = apply_optical_remap(frame, aligner, load_coverage_mask())
    assert warped.shape[:2] == (1080, 1920)
    # UMI eval: height-normalized resize then center-crop to 224 (same as PolicyImagePipeline).
    oh = ow = 224
    ih, iw = warped.shape[:2]
    rh = oh
    rw = int(np.ceil(rh / ih * iw))
    resized = cv2.resize(warped, (rw, rh), interpolation=cv2.INTER_AREA)
    x0 = (rw - ow) // 2
    crop = resized[:, x0 : x0 + ow]
    assert crop.shape == (224, 224, 3)


def test_yaml_patch_sets_optical_remap_and_16_9_resize():
    from solarian.cli.commands.calibrate_umi_camera import _upsert_yaml_key

    text = (
        "sensors:\n"
        "  camera:\n"
        "    width: 2028\n"
        "    height: 1520\n"
        "    resize_width: 480\n"
        "    resize_height: 360\n"
        "    type: picamera\n"
    )
    text = _upsert_yaml_key(text, "optical_remap", "conf/calibrations/cam_gopro_remap.json")
    text = _upsert_yaml_key(text, "resize_width", "640")
    text = _upsert_yaml_key(text, "resize_height", "360")
    assert "optical_remap: conf/calibrations/cam_gopro_remap.json" in text
    assert "resize_width: 640" in text
    assert "resize_height: 360" in text
    assert "resize_width: 480" not in text
