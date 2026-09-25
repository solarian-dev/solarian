"""Measure camera observation latency with UMI's QR-timestamp method.

The UMI paper's formula, ported from ``scripts/calibrate_uvc_camera_latency.py``:

    l_camera = t_recv - t_display - l_display,   l_display ~= 1.5 / f_monitor

A screen shows a QR code encoding the instant it was drawn; the camera films
that screen; the gap between the decoded instant and the instant our pipeline
stamped the frame is the camera's observation latency.

Solarian splits this into two roles because the camera is usually on the robot
and the screen on your laptop. Both sides read :func:`synced_time`, so the
subtraction is valid across machines as long as both have synced to the relay.

    # on the machine with a monitor
    solarian calibrate-camera-latency role=display

    # on the robot, pointing the camera at that monitor
    solarian calibrate-camera-latency role=capture monitor_hz=60

The capture window shows the remapped observation a skill would see, not the
native rpicam-hello preview. Aim using that window.

Both roles may share one machine when the robot has a screen — nothing here
opens a bus, and one OS clock makes the subtraction exact — but a Pi's display
pipeline and the CPU the two processes then share both inflate the result.
"""

from __future__ import annotations

import time

import numpy as np
from omegaconf import OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.cli.commands._calibration import (
    _opencv_bgr,
    module_running,
    pick_module,
    save_module_field,
)
from solarian.core.camera import Camera
from solarian.core.clock_sync import sync_with_server, synced_time
from solarian.core.data import ImageData
from solarian.utils.precise_sleep import precise_wait

_QR_WINDOW = "Timestamp QRCode"
_CAM_WINDOW = "Camera"
_QR_SIZE = 720
_QR_QUIET_ZONE = 4
_DUMP_FRAME = "camera_latency_last_frame.png"
# Panel refresh is l_display on the capture side. A new QR every vsync at 120 Hz
# blends several payloads into one exposure: finder patterns still match, decode
# returns empty.
_QR_UPDATE_HZ = 15.0
# UMI's conservative estimate: a VSynced frame is on the glass ~1.5 refreshes
# after the draw call returns.
_DISPLAY_FRAMES = 1.5
_N_FRAMES = 120
_MONITOR_HZ_DEFAULT = 60.0


def _cfg_get(cfg, key: str, default=None):
    val = OmegaConf.select(cfg, key)
    return default if val is None else val


def _sync_clock(cfg) -> bool:
    """Put this process on the relay's timebase; both roles must agree."""
    conn = cfg.connection
    try:
        result = sync_with_server(conn.ip, int(conn.clock_sync_port), samples=12)
    except Exception as exc:  # noqa: BLE001
        print(f"Clock sync to {conn.ip}:{conn.clock_sync_port} failed: {exc}")
        return False
    print(
        f"Clock synced: offset {result.offset_s * 1000.0:+.2f} ms, "
        f"rtt {result.rtt_s * 1000.0:.2f} ms"
    )
    return True


def _render_qr(text: str, size: int) -> np.ndarray:
    import cv2

    params = cv2.QRCodeEncoder.Params()
    params.correction_level = cv2.QRCodeEncoder_CORRECT_LEVEL_H
    img = cv2.QRCodeEncoder.create(params).encode(text)
    img = cv2.copyMakeBorder(
        img,
        _QR_QUIET_ZONE,
        _QR_QUIET_ZONE,
        _QR_QUIET_ZONE,
        _QR_QUIET_ZONE,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def _qr_gray(frame) -> np.ndarray:
    """Channel-mean gray. QR is black/white, so RGB vs BGR must not matter."""
    if frame.ndim == 2:
        return frame
    return frame.mean(axis=2).astype(np.uint8)


def _qr_detectors():
    """Aruco first: it reads smaller codes than QRCodeDetector after 640x360."""
    import cv2

    detectors = []
    if hasattr(cv2, "QRCodeDetectorAruco"):
        detectors.append(cv2.QRCodeDetectorAruco())
    detectors.append(cv2.QRCodeDetector())
    return detectors


def _stamp_from_code(code) -> float | None:
    if not code:
        return None
    try:
        return float(code)
    except ValueError:
        return None


def _unwarp_qr(gray, corners, size: int = 400):
    import cv2

    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        return None
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    return cv2.warpPerspective(
        gray, cv2.getPerspectiveTransform(pts, dst), (size, size)
    )


def _decode_qr(detectors, frame) -> float | None:
    """Decode a timestamp QR. ``None`` if absent."""
    gray = _qr_gray(frame)
    for detector in detectors:
        for name in ("detectAndDecode", "detectAndDecodeCurved"):
            decode = getattr(detector, name, None)
            if decode is None:
                continue
            code, corners = decode(gray)[:2]
            got = _stamp_from_code(code)
            if got is not None:
                return got
            if corners is None:
                continue
            warped = _unwarp_qr(gray, corners)
            if warped is None:
                continue
            got = _stamp_from_code(detector.detectAndDecode(warped)[0])
            if got is not None:
                return got
    return None


def _run_display(qr_size: int, monitor_hz: float) -> None:
    import cv2

    cv2.namedWindow(_QR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        _QR_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )
    print(
        f"\nShowing a full-screen timestamp QR code. Panel {monitor_hz:g} Hz, "
        f"QR updates at {min(monitor_hz, _QR_UPDATE_HZ):g} Hz so a 30 fps "
        "exposure is not a blend of several codes.\n"
        "Point the robot's camera at this screen, fill as much of the frame as\n"
        "you can, and check focus. Press q here when the capture side is done."
    )

    # Redrawing faster than the panel presents buys nothing and costs a core,
    # which matters when the capture side shares this machine's CPU.
    dt = 1.0 / monitor_hz
    qr_dt = 1.0 / min(monitor_hz, _QR_UPDATE_HZ)
    t_start = time.monotonic()
    frames = 0
    n_qr = 0
    render_s = 0.0
    qr_img = None
    t_qr = -1e9
    try:
        while True:
            now = time.monotonic()
            if qr_img is None or now - t_qr >= qr_dt:
                # Stamp as late as possible: the delay we want to exclude is
                # everything after this line, not the QR encode before it.
                t_stamp = time.monotonic()
                qr_img = _render_qr(f"{synced_time():.6f}", qr_size)
                render_s += time.monotonic() - t_stamp
                t_qr = now
                n_qr += 1
            cv2.imshow(_QR_WINDOW, qr_img)
            frames += 1
            if cv2.pollKey() == ord("q"):
                break
            precise_wait(t_end=t_start + dt * frames, time_func=time.monotonic)
    finally:
        cv2.destroyWindow(_QR_WINDOW)

    # Encoding happens after the stamp, so it lands inside whatever the capture
    # side measures. On a fast machine it is noise; on a slow one it is not.
    print(
        f"\nDrew {frames} frames, {n_qr} QR updates, "
        f"{render_s / max(n_qr, 1) * 1000.0:.1f} ms mean encode+draw. That "
        "much of the capture side's reading is this machine's QR generation, "
        "not camera latency — move the display to a faster machine if it is "
        "a large share of the result."
    )


def _run_capture(
    module,
    sensor_name: str,
    monitor_hz: float,
    n_frames: int,
    plot: bool,
    color_mode: str,
) -> None:
    import cv2

    detectors = _qr_detectors()
    l_display = _DISPLAY_FRAMES / monitor_hz

    # Keyed by display timestamp: a QR seen in several frames only measures the
    # latency on the first, afterwards it is the monitor that has not refreshed.
    first_recv: dict[float, float] = {}
    n_read = 0
    n_decoded = 0
    t_report = time.monotonic()
    dumped = False
    last_frame = None
    preview = True
    try:
        cv2.namedWindow(_CAM_WINDOW, cv2.WINDOW_NORMAL)
    except cv2.error:
        preview = False
    print(
        f"\nCollecting {n_frames} distinct QR timestamps "
        "(a held code only counts once). Ctrl+C prints the mean from what we have."
    )
    if preview:
        print(
            "Camera window is the observation the decoder sees, not "
            "rpicam-hello. Fill it with the QR. Press q to stop."
        )

    try:
        while len(first_recv) < n_frames:
            frame = module.read_sensors(ImageData, [sensor_name])[sensor_name]["value"]
            # Same stamp Module.update_observations writes, and it is already
            # backdated by whatever the bus knew -- so what this measures is the
            # error left in it, and the result corrects the current
            # receive_latency rather than replacing it.
            t_recv = module.module_bus.capture_timestamps.get(
                sensor_name, synced_time() - module.cfg.receive_latency
            )
            n_read += 1
            last_frame = frame
            t_display = _decode_qr(detectors, frame)
            if t_display is not None:
                n_decoded += 1
                first_recv.setdefault(t_display, t_recv)

            if preview:
                view = _opencv_bgr(frame, color_mode)
                label = "decoded" if t_display is not None else "no QR"
                cv2.putText(
                    view,
                    label,
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0) if t_display is not None else (0, 0, 255),
                    2,
                )
                try:
                    cv2.imshow(_CAM_WINDOW, view)
                    if cv2.pollKey() == ord("q"):
                        break
                except cv2.error:
                    preview = False

            if time.monotonic() - t_report > 1.0:
                t_report = time.monotonic()
                print(
                    f"  {len(first_recv)}/{n_frames} unique QR, "
                    f"{n_decoded}/{n_read} frames decoded"
                )

            if n_read > 20 and not first_recv and not dumped:
                dumped = True
                cv2.imwrite(_DUMP_FRAME, _opencv_bgr(frame, color_mode))
                print(
                    f"\nNo QR in the first 20 frames. Saved detector input to "
                    f"{_DUMP_FRAME} (this is the remapped observation, not the "
                    "native rpicam preview). Keep the display running, fill the "
                    "Camera window with the code, check focus. Press q to stop.\n"
                )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if preview:
            try:
                cv2.destroyWindow(_CAM_WINDOW)
            except cv2.error:
                pass

    if not first_recv:
        if last_frame is not None:
            cv2.imwrite(_DUMP_FRAME, _opencv_bgr(last_frame, color_mode))
        print(
            "\nNo QR code decoded. Check that the display side is running, that "
            "the Camera window shows the screen, and that the code is in "
            f"focus. Last detector input: {_DUMP_FRAME}"
        )
        return
    if len(first_recv) < n_frames:
        print(f"\nStopped after {len(first_recv)}/{n_frames} unique QR timestamps.")

    # A held QR is drawn once and filmed later, so the code in an exposure is
    # already part-way through its hold window. Keeping the first of k sightings
    # lands ~hold/(k+1) in; at k=1 the dedup above never fired and the full
    # half-window is still in the number.
    hold = 1.0 / min(monitor_hz, _QR_UPDATE_HZ)
    sightings = n_decoded / len(first_recv)
    l_hold = hold / (sightings + 1.0)

    latencies = (
        np.array([r - d for d, r in first_recv.items()]) - l_display - l_hold
    )
    print(
        "\n=== camera receive_latency residual "
        "(l_residual = t_stamp - t_display - l_display - l_hold) ==="
    )
    print(f"  frames read:      {n_read}")
    print(f"  frames decoded:   {n_decoded} ({n_decoded / max(n_read, 1):.0%})")
    print(f"  unique QR:        {len(first_recv)}")
    print(f"  l_display:        {l_display * 1000.0:.1f} ms ({monitor_hz:g} Hz monitor)")
    print(
        f"  l_hold:           {l_hold * 1000.0:.1f} ms "
        f"({hold * 1000.0:.0f} ms hold, {sightings:.1f} sightings per QR)"
    )
    print(
        f"  measured:         mean {latencies.mean() * 1000.0:.1f} ms, "
        f"std {latencies.std() * 1000.0:.1f} ms, "
        f"min {latencies.min() * 1000.0:.1f} ms, "
        f"max {latencies.max() * 1000.0:.1f} ms"
    )
    if sightings < 1.5:
        print(
            f"\nEach QR was decoded about once ({n_read / max(n_decoded, 1):.0f} "
            "reads per decode), so l_hold is an assumed half-window rather than "
            "something the sightings measured, and it is the largest term in the "
            "spread above. Decoding faster than the display updates would shrink "
            "the correction and tighten the result."
        )
    corrected = module.cfg.receive_latency + float(latencies.mean())
    print(f"  receive_latency:  {module.cfg.receive_latency:.4f} -> {corrected:.4f} s")
    print(
        "\nThis anchors every action timestamp in a UMI chunk, so an over-estimate "
        "backdates the whole chunk and the first waypoints land in the past.\n"
        "A residual near zero means the stamp already names the exposure and the "
        "constant is doing its job; re-run after any change to the capture path."
    )
    if corrected < 0.0:
        # The residual may be either sign -- it corrects an over-estimate as
        # readily as an under-estimate -- but the total cannot be: a frame
        # cannot be stamped before the screen drew what it shows. So the two
        # machines disagree about the time, and writing it would backdate every
        # chunk by a clock error.
        print(
            "\nNegative total, which is not physical, so nothing was written. "
            "Check that both roles are synced to the same relay clock."
        )
    else:
        save_module_field(module, "receive_latency", round(corrected, 4))

    if plot:
        from matplotlib import pyplot as plt

        recv = np.array(list(first_recv.values()))
        plt.scatter(recv - recv[0], latencies * 1000.0, alpha=0.6)
        plt.axhline(latencies.mean() * 1000.0, color="r", label="mean")
        plt.xlabel("receive time (s)")
        plt.ylabel("camera latency (ms)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


@CommandRegistry.register("calibrate-camera-latency", "robot")
class CalibrateCameraLatencyCommand(BaseCommand):
    help = "Measure the camera's receive_latency by filming a timestamp QR code (UMI method)."
    description = """
    UMI's method: one machine displays a QR code encoding the instant it was drawn,
    the other films that screen, and the gap between the decoded instant and the
    instant the pipeline stamped the frame is the camera's observation latency.
    Added to the camera module's yaml receive_latency, which the module then
    subtracts as it stamps every frame.

    It corrects rather than replaces, because the stamp it reads is the one the
    module publishes and that is already backdated. Where the hardware dates its
    own frames the constant only has to cover the stages after the sensor, so
    the honest number is whatever error is left over -- and it converges, since
    a correctly set constant measures zero.

    Two invocations, one per role, because the camera is usually on the robot and
    the screen on a laptop:

      role=display   on the machine with the monitor
      role=capture   on the machine the camera is wired to

    Both sides subtract timestamps from the relay's timebase, so the relay has to
    be running (`solarian run-server`) unless both roles are on this same machine,
    which is what allow_unsynced=true declares. Aim the camera with the Camera
    window the capture role opens, not with an external preview.
    """
    params = {
        "role": "display on the machine with the monitor, capture on the machine "
                "with the camera. Required.",
        "monitor_hz": "Refresh rate of the screen being filmed. Required for "
                      "role=capture, since l_display = 1.5 / f_monitor is part of "
                      "the arithmetic; paces the redraw for role=display "
                      "(default: 60).",
        "qr_size": "Side of the displayed QR code in pixels (default: 720).",
        "n_frames": "How many decoded frames role=capture averages (default: 120).",
        "plot": "true to show the per-frame latency plot when capture finishes "
                "(default: false).",
        "allow_unsynced": "true to measure without the relay, valid only when the "
                          "display and the camera are on this same machine "
                          "(default: false).",
    }
    examples = (
        "solarian calibrate-camera-latency role=display monitor_hz=60",
        "solarian calibrate-camera-latency role=capture monitor_hz=60",
        "solarian calibrate-camera-latency role=capture monitor_hz=120 n_frames=240 plot=true",
    )

    def handle(self, cfg) -> None:
        role = str(_cfg_get(cfg, "role", "")).lower()
        if role not in {"display", "capture"}:
            print(
                "Pass role=display (on the machine with a monitor) or "
                "role=capture (on the robot)."
            )
            return

        allow_unsynced = bool(_cfg_get(cfg, "allow_unsynced", False))
        if not _sync_clock(cfg) and not allow_unsynced:
            print(
                "\nBoth roles must share the relay's timebase or the subtraction "
                "is meaningless. Start the relay (solarian run-server) and retry, "
                "or pass allow_unsynced=true if the display and the camera are on "
                "this same machine."
            )
            return

        if role == "display":
            # Only paces the redraw here, so a default is harmless; on the
            # capture side the same number enters the arithmetic and is required.
            _run_display(
                int(_cfg_get(cfg, "qr_size", _QR_SIZE)),
                float(_cfg_get(cfg, "monitor_hz", _MONITOR_HZ_DEFAULT)),
            )
            return

        monitor_hz = _cfg_get(cfg, "monitor_hz")
        if monitor_hz is None:
            print(
                "monitor_hz is required: l_display = 1.5 / f_monitor is a real "
                "part of the measurement. Pass the refresh rate of the screen "
                "the camera is filming, e.g. monitor_hz=60."
            )
            return

        module = pick_module("camera", sensor_cls=Camera)
        if module is None:
            return
        sensor_name = next(
            name for name, s in module.sensors.items() if isinstance(s, Camera)
        )

        with module_running(
            module,
            "The camera will be opened. Point it at the QR display. Aim with "
            "the Camera window (remapped observation), not rpicam-hello.",
        ):
            _run_capture(
                module,
                sensor_name,
                float(monitor_hz),
                int(_cfg_get(cfg, "n_frames", _N_FRAMES)),
                bool(_cfg_get(cfg, "plot", False)),
                str(module.sensors[sensor_name].cfg.color_mode),
            )
