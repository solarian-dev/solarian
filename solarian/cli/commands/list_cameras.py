import sys
import cv2
from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry


BACKENDS = {
    "linux": cv2.CAP_V4L2,
    "darwin": cv2.CAP_AVFOUNDATION,
    "win32": cv2.CAP_MSMF,
}


def _find_cameras_linux(max_index: int = 16) -> list[dict]:
    """Enumerate cameras on Linux via /dev/video* and V4L2."""
    from pathlib import Path

    cameras = []
    seen_indices = set()

    for dev in sorted(Path("/dev").glob("video*")):
        idx = int(dev.name.removeprefix("video"))
        if idx in seen_indices:
            continue
        seen_indices.add(idx)

        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend_name = cap.getBackendName()
            cap.release()
            cameras.append({
                "index": idx,
                "port": str(dev),
                "width": w,
                "height": h,
                "fps": fps,
                "backend": backend_name,
            })
        else:
            cap.release()

    return cameras


def _find_cameras_probe(backend, max_index: int = 16) -> list[dict]:
    """Enumerate cameras by probing indices 0..max_index-1."""
    cameras = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend_name = cap.getBackendName()
            cap.release()
            cameras.append({
                "index": idx,
                "port": str(idx),
                "width": w,
                "height": h,
                "fps": fps,
                "backend": backend_name,
            })
        else:
            cap.release()
    return cameras


def _find_cameras_picamera2() -> list[dict]:
    """Enumerate CSI / libcamera cameras via picamera2 when available.

    ``port`` is the index passed to ``Picamera2(camera_num=...)`` / PiCameraBus
    (position in ``global_camera_info()``, typically 0 = cam0 / CSI 0).
    """
    try:
        from picamera2 import Picamera2
    except ImportError:
        return []

    cameras = []
    try:
        infos = Picamera2.global_camera_info()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: picamera2 available but global_camera_info() failed: {e}")
        return []

    for i, info in enumerate(infos):
        model = info.get("Model", "unknown")
        cam_id = info.get("Id", "")
        cameras.append({
            "index": i,
            "port": str(i),
            "width": None,
            "height": None,
            "fps": None,
            "backend": "picamera2",
            "model": model,
            "id": cam_id,
        })
    return cameras


def find_cameras(max_index: int = 16) -> list[dict]:
    platform = sys.platform
    if platform == "linux":
        cameras = _find_cameras_linux(max_index)
    else:
        backend = BACKENDS.get(platform)
        if backend is None:
            print(f"Warning: unsupported platform '{platform}', using default OpenCV backend")
            backend = cv2.CAP_ANY
        cameras = _find_cameras_probe(backend, max_index)

    cameras.extend(_find_cameras_picamera2())
    return cameras


@CommandRegistry.register('list-cameras')
class ListCamerasCommand(BaseCommand):
    help = 'Detects and lists all cameras available on the system.'

    def handle(self, cfg: DictConfig) -> None:
        print("Scanning for cameras...\n")
        cameras = find_cameras()

        if not cameras:
            print("No cameras detected.")
            return

        print(f"Found {len(cameras)} camera(s):\n")
        for cam in cameras:
            lines = [
                f"  [Camera {cam['index']}]",
                f"  port    : {cam['port']}",
                f"  backend : {cam['backend']}",
            ]
            if cam.get("model"):
                lines.append(f"  model   : {cam['model']}")
            if cam.get("id"):
                lines.append(f"  id      : {cam['id']}")
            if cam.get("width") is not None and cam.get("height") is not None:
                lines.append(f"  size    : {cam['width']}x{cam['height']}")
            if cam.get("fps") is not None:
                lines.append(f"  fps     : {cam['fps']:.1f}")
            if cam["backend"] == "picamera2":
                lines.append("  note    : use camera_port with this index for PiCameraModule")
            print("\n".join(lines) + "\n")
