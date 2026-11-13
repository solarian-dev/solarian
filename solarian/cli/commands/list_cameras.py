import ctypes
import fcntl
import os
import sys
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.camera_backend_registry import discover_camera_backends


BACKENDS = {
    "linux": "CAP_V4L2",
    "darwin": "CAP_AVFOUNDATION",
    "win32": "CAP_MSMF",
}

# Raspberry Pi pipeline / codec nodes that OpenCV can open but are not cameras.
_INTERNAL_V4L2_DRIVERS = {
    "bcm2835-codec",
    "bcm2835-isp",
    "bcm2835-unicam",
    "pispbe",
    "rp1-cfe",
    "rpi-hevc-dec",
    "rpivid",
    "unicam",
}

V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
V4L2_CAP_META_CAPTURE = 0x00800000
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE = 9
V4L2_FRMSIZE_TYPE_DISCRETE = 1
V4L2_FRMSIZE_TYPE_CONTINUOUS = 2
V4L2_FRMSIZE_TYPE_STEPWISE = 3


def _IOC(dir_, type_, nr, size):
    return (dir_ << 30) | (size << 16) | (ord(type_) << 8) | nr


def _IOR(type_, nr, size):
    return _IOC(2, type_, nr, size)


def _IOWR(type_, nr, size):
    return _IOC(3, type_, nr, size)


class _v4l2_capability(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_char * 16),
        ("card", ctypes.c_char * 32),
        ("bus_info", ctypes.c_char * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class _v4l2_fmtdesc(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("description", ctypes.c_char * 32),
        ("pixelformat", ctypes.c_uint32),
        ("mbus_code", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class _v4l2_frmsize_discrete(ctypes.Structure):
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32)]


class _v4l2_frmsize_stepwise(ctypes.Structure):
    _fields_ = [
        ("min_width", ctypes.c_uint32),
        ("max_width", ctypes.c_uint32),
        ("step_width", ctypes.c_uint32),
        ("min_height", ctypes.c_uint32),
        ("max_height", ctypes.c_uint32),
        ("step_height", ctypes.c_uint32),
    ]


class _v4l2_frmsizeenum(ctypes.Structure):
    class _u(ctypes.Union):
        _fields_ = [
            ("discrete", _v4l2_frmsize_discrete),
            ("stepwise", _v4l2_frmsize_stepwise),
        ]

    _anonymous_ = ("_u",)
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("_u", _u),
        ("reserved", ctypes.c_uint32 * 2),
    ]


VIDIOC_QUERYCAP = _IOR("V", 0, ctypes.sizeof(_v4l2_capability))
VIDIOC_ENUM_FMT = _IOWR("V", 2, ctypes.sizeof(_v4l2_fmtdesc))
VIDIOC_ENUM_FRAMESIZES = _IOWR("V", 74, ctypes.sizeof(_v4l2_frmsizeenum))


def _clean_model(name: str) -> str:
    """Collapse UVC names like 'Elgato HD60 X: Elgato HD60 X'."""
    name = name.strip()
    if ": " in name:
        left, right = name.split(": ", 1)
        if left.strip() == right.strip():
            return left.strip()
    return name


def _fourcc_to_str(pixelformat: int) -> str:
    return "".join(chr((pixelformat >> (8 * i)) & 0xFF) for i in range(4))


def _v4l2_query(path: str) -> dict | None:
    cap = _v4l2_capability()
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap)
    except OSError:
        os.close(fd)
        return None
    os.close(fd)

    caps = cap.device_caps or cap.capabilities
    driver = cap.driver.decode("utf-8", errors="replace").rstrip("\x00")
    card = cap.card.decode("utf-8", errors="replace").rstrip("\x00")
    bus_info = cap.bus_info.decode("utf-8", errors="replace").rstrip("\x00")
    return {
        "path": path,
        "driver": driver,
        "card": card,
        "bus_info": bus_info,
        "caps": caps,
    }


def _fourcc_priority(pixelformat: int) -> int:
    name = _fourcc_to_str(pixelformat)
    return {"YUYV": 0, "YUY2": 0, "MJPG": 1, "RGB3": 2, "BGR3": 2, "NV12": 3}.get(name, 9)


def _v4l2_enumerate_modes(path: str) -> list[tuple[int, int, int]]:
    """Return (width, height, pixelformat) for every advertised capture mode."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return []

    modes: list[tuple[int, int, int]] = []
    try:
        for buf_type in (V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE):
            fmt_index = 0
            while fmt_index < 64:
                fmt = _v4l2_fmtdesc()
                fmt.index = fmt_index
                fmt.type = buf_type
                try:
                    fcntl.ioctl(fd, VIDIOC_ENUM_FMT, fmt)
                except OSError:
                    break
                size_index = 0
                while size_index < 128:
                    fs = _v4l2_frmsizeenum()
                    fs.index = size_index
                    fs.pixel_format = fmt.pixelformat
                    try:
                        fcntl.ioctl(fd, VIDIOC_ENUM_FRAMESIZES, fs)
                    except OSError:
                        break
                    sizes: list[tuple[int, int]] = []
                    if fs.type == V4L2_FRMSIZE_TYPE_DISCRETE:
                        sizes.append((fs.discrete.width, fs.discrete.height))
                    elif fs.type in (V4L2_FRMSIZE_TYPE_STEPWISE, V4L2_FRMSIZE_TYPE_CONTINUOUS):
                        sizes.append((fs.stepwise.max_width, fs.stepwise.max_height))
                    for width, height in sizes:
                        modes.append((int(width), int(height), int(fmt.pixelformat)))
                    if fs.type != V4L2_FRMSIZE_TYPE_DISCRETE:
                        break
                    size_index += 1
                fmt_index += 1
    finally:
        os.close(fd)
    return modes


def _v4l2_candidate_modes(path: str) -> list[tuple[int, int, int]]:
    """Largest unique sizes first; prefer packed formats OpenCV handles well."""
    by_size: dict[tuple[int, int], int] = {}
    for width, height, pixelformat in _v4l2_enumerate_modes(path):
        key = (width, height)
        current = by_size.get(key)
        if current is None or _fourcc_priority(pixelformat) < _fourcc_priority(current):
            by_size[key] = pixelformat
    return sorted(
        [(w, h, pf) for (w, h), pf in by_size.items()],
        key=lambda mode: (mode[0] * mode[1], -_fourcc_priority(mode[2])),
        reverse=True,
    )


def _frame_looks_valid(frame) -> bool:
    """Reject no-signal / wrong-resolution placeholders from HDMI capture cards."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    if int(frame.max()) == 0 or float(frame.std()) < 0.5:
        return False
    small = frame[::8, ::8]
    if small.ndim == 3:
        packed = (
            small[..., 0].astype("uint32") << 16
            | small[..., 1].astype("uint32") << 8
            | small[..., 2].astype("uint32")
        )
        unique = int(len(set(packed.ravel().tolist())))
    else:
        unique = int(len(set(small.ravel().tolist())))
    # Status screens are a few colors on a solid background; live video is not.
    return unique >= 32


def _find_cameras_linux() -> list[dict]:
    """One V4L2 entry per physical capture device (USB card, webcam, ...)."""
    grouped: dict[str, list[dict]] = {}
    for dev in sorted(Path("/dev").glob("video*"), key=lambda p: int(p.name.removeprefix("video"))):
        info = _v4l2_query(str(dev))
        if info is None:
            continue
        if info["driver"] in _INTERNAL_V4L2_DRIVERS:
            continue
        caps = info["caps"]
        is_capture = bool(caps & V4L2_CAP_VIDEO_CAPTURE) or bool(caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE)
        is_meta_only = bool(caps & V4L2_CAP_META_CAPTURE) and not (caps & V4L2_CAP_VIDEO_CAPTURE)
        if not is_capture or is_meta_only:
            continue
        key = info["bus_info"] or info["path"]
        grouped.setdefault(key, []).append(info)

    cameras = []
    for group in grouped.values():
        group.sort(
            key=lambda n: (
                0 if n["caps"] & V4L2_CAP_VIDEO_CAPTURE else 1,
                int(Path(n["path"]).name.removeprefix("video")),
            )
        )
        chosen = group[0]
        cameras.append({
            "port": chosen["path"],
            "backend": "V4L2",
            "model": _clean_model(chosen["card"]) or "unknown",
        })
    return cameras


def _find_cameras_probe(backend, max_index: int = 16) -> list[dict]:
    """Enumerate cameras by probing indices 0..max_index-1 (macOS / Windows)."""
    import cv2

    cameras = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        opened = cap.isOpened()
        backend_name = cap.getBackendName() if opened else None
        cap.release()
        if opened:
            cameras.append({
                "port": str(idx),
                "backend": backend_name or "OpenCV",
                "model": "unknown",
            })
    return cameras


def find_cameras(max_index: int = 16) -> list[dict]:
    cameras = []
    for name, backend in discover_camera_backends().items():
        try:
            cameras.extend(backend.find())
        except Exception as e:  # noqa: BLE001 - one broken backend must not hide the rest
            print(f"Warning: camera backend {name!r} failed to list cameras: {e}")
    platform = sys.platform
    if platform == "linux":
        cameras.extend(_find_cameras_linux())
        return cameras

    import cv2

    backend_name = BACKENDS.get(platform)
    if backend_name is None:
        print(f"Warning: unsupported platform '{platform}', using default OpenCV backend")
        backend = cv2.CAP_ANY
    else:
        backend = getattr(cv2, backend_name)
    cameras.extend(_find_cameras_probe(backend, max_index))
    return cameras


def _print_cameras(cameras: list[dict]) -> None:
    print(f"Found {len(cameras)} camera(s):\n")
    for i, cam in enumerate(cameras, start=1):
        print(f"  [{i}]")
        print(f"  port    : {cam['port']}")
        print(f"  backend : {cam['backend']}")
        print(f"  model   : {cam['model']}")
        print()


def _prompt_camera_choice(n: int) -> int | None:
    while True:
        try:
            raw = input(f"Which camera to open? (1-{n}, or q to quit): ").strip().lower()
        except EOFError:
            print()
            return None
        if raw in {"", "q", "quit"}:
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= n:
                return idx
        print(f"Please enter a number between 1 and {n}, or q to quit.")


def _fit_window(width: int, height: int, max_w: int = 1280, max_h: int = 720) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return max_w, max_h
    scale = min(max_w / width, max_h / height, 1.0)
    return max(1, int(width * scale)), max(1, int(height * scale))


def _preview_loop(title: str, frames):
    import cv2

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    window_sized = False
    try:
        for frame in frames:
            if frame is None:
                print("Warning: dropped empty frame")
                continue
            if not window_sized:
                h, w = frame.shape[:2]
                win_w, win_h = _fit_window(w, h)
                cv2.resizeWindow(title, win_w, win_h)
                window_sized = True
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            try:
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        print()
    finally:
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass


def _apply_v4l2_mode(cap, width: int, height: int, pixelformat: int) -> None:
    import cv2

    cap.set(cv2.CAP_PROP_FOURCC, pixelformat)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))


def _open_v4l2_capture(port: str, width: int | None = None, height: int | None = None, pixelformat: int | None = None):
    import cv2

    cap = cv2.VideoCapture(port, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    if width is not None and height is not None:
        if pixelformat is not None:
            _apply_v4l2_mode(cap, width, height, pixelformat)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap


def _grab_valid_v4l2_frame(cap, attempts: int = 6, skip: int = 2):
    last = None
    for i in range(skip + attempts):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if i < skip:
            continue
        last = frame
        if _frame_looks_valid(frame):
            return frame
    return last if last is not None and _frame_looks_valid(last) else None


def _iter_v4l2_frames(port: str, width: int | None = None, height: int | None = None):
    import cv2

    if width and height:
        candidates = _v4l2_candidate_modes(port)
        matching = [m for m in candidates if m[0] == int(width) and m[1] == int(height)]
        ordered = matching + [m for m in candidates if m not in matching]
        if not ordered:
            ordered = [(int(width), int(height), None)]
    else:
        ordered = _v4l2_candidate_modes(port)

    cap = None
    first_frame = None
    try:
        for w, h, pixelformat in ordered:
            if cap is not None:
                cap.release()
            if pixelformat is not None:
                cap = _open_v4l2_capture(port, w, h, pixelformat)
            else:
                cap = _open_v4l2_capture(port, w, h)
            if cap is None:
                continue
            first_frame = _grab_valid_v4l2_frame(cap)
            if first_frame is not None:
                break
        if first_frame is None:
            if cap is not None:
                cap.release()
            cap = _open_v4l2_capture(port)
            if cap is None:
                raise RuntimeError(f"Could not open camera {port}")
            ok, first_frame = cap.read()
            if not ok:
                first_frame = None
        if first_frame is None or cap is None:
            raise RuntimeError(f"Could not read a frame from {port}")

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Opening {port} at {actual_w}x{actual_h}.")
        yield first_frame
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read a frame from {port}")
            yield frame
    finally:
        if cap is not None:
            cap.release()


def _iter_opencv_frames(port: str, width: int | None = None, height: int | None = None):
    import cv2

    try:
        index = int(port)
    except ValueError:
        index = port

    backend_name = BACKENDS.get(sys.platform)
    backend = getattr(cv2, backend_name) if backend_name else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera {port}")

    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Opening camera {port} at {actual_w}x{actual_h}.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read a frame from camera {port}")
            yield frame
    finally:
        cap.release()


def iter_camera_frames(
    cam: dict,
    width: int | None = None,
    height: int | None = None,
):
    """Yield BGR frames from a ``find_cameras()`` entry."""
    backend = cam["backend"]
    plugin = discover_camera_backends().get(backend)
    if plugin is not None:
        yield from plugin.iter_frames(cam["port"], width, height)
    elif backend == "V4L2":
        yield from _iter_v4l2_frames(cam["port"], width, height)
    else:
        yield from _iter_opencv_frames(cam["port"], width, height)


def open_camera_preview(cam: dict) -> None:
    title = f"{cam['model']} ({cam['backend']})"
    frames = iter_camera_frames(cam)
    try:
        _preview_loop(title, frames)
    finally:
        frames.close()


@CommandRegistry.register("list-cameras")
class ListCamerasCommand(BaseCommand):
    category = "Utilities"
    uses_config = False
    help = "Detect connected cameras and optionally open a live preview."
    description = """
    Enumerates every camera backend available on this machine (USB/UVC, network
    streams, and plugin backends such as the Pi's CSI cameras from
    solarian-picamera), prints model, backend and
    port for each, then opens a live preview of the one you pick so you can tell
    which physical camera it is before naming its port in a camera module yaml.

    Run it with the robot stopped: run-robot and run-server hold the camera open,
    and a CSI camera admits one reader at a time.
    """
    usage = ""
    examples = ("solarian list-cameras",)

    def handle(self, cfg: DictConfig) -> None:
        try:
            import cv2

            cv2.setLogLevel(getattr(cv2, "LOG_LEVEL_ERROR", 2))
        except Exception:  # noqa: BLE001
            pass

        print("Scanning for cameras...\n")
        cameras = find_cameras()

        if not cameras:
            print("No cameras detected.")
            return

        _print_cameras(cameras)
        choice = _prompt_camera_choice(len(cameras))
        if choice is None:
            return

        cam = cameras[choice - 1]
        try:
            open_camera_preview(cam)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to open camera: {e}")
