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


def find_cameras(max_index: int = 16) -> list[dict]:
    platform = sys.platform
    if platform == "linux":
        return _find_cameras_linux(max_index)

    backend = BACKENDS.get(platform)
    if backend is None:
        print(f"Warning: unsupported platform '{platform}', using default OpenCV backend")
        backend = cv2.CAP_ANY

    return _find_cameras_probe(backend, max_index)


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
            print(
                f"  [Camera {cam['index']}]\n"
                f"  port    : {cam['port']}\n"
                f"  backend : {cam['backend']}\n"
                f"  size    : {cam['width']}x{cam['height']}\n"
                f"  fps     : {cam['fps']:.1f}\n"
            )
