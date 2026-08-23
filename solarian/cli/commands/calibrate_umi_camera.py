"""Live gold vs wrist-camera overlay for UMI policy-image alignment.

Mac-side: subscribe to the running robot's observation stream (same relay
as ``run-controller``), crop/mask to 224×224 like cup-arrange, and show
gold | live | absdiff plus orange / white-shell / screw superposition.

Requires ``solarian run-server`` and ``solarian run-robot`` already up.
Does not open the CSI camera (``run-robot`` keeps that lock).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry

_WINDOW = "calibrate-umi-camera"
_SCALE = 3
_FIRST_WAIT_MS = 10_000
_POLL_MS = 200
_DUMP_DEFAULT = ".cache/umi_image_compare"


def _conf_dir() -> Path:
    return Path(os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf")))


def _load_skill_cfg() -> DictConfig:
    path = _conf_dir() / "cup_arrange.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run this from the so106 project with "
            f"--conf conf or --conf conf_dev."
        )
    return OmegaConf.load(str(path))


def _resolve_gold(cfg, skill_cfg) -> Path:
    raw = None
    if cfg is not None and OmegaConf.select(cfg, "gold") is not None:
        raw = str(cfg.gold)
    elif skill_cfg.get("umi_gold_image"):
        raw = str(skill_cfg.umi_gold_image)
    if not raw:
        raise FileNotFoundError(
            "No gold image. Set umi_gold_image in cup_arrange.yaml or pass gold=<path>."
        )
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"Gold image not found: {path}")
    return path


def _wrist_rgb(observation, cam_module: str):
    cam = observation[cam_module]
    if cam is None:
        return None
    try:
        img = cam["image"]
    except KeyError:
        return None
    value = getattr(img, "value", None)
    if value is None:
        return None
    import numpy as np

    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


@CommandRegistry.register("calibrate-umi-camera", "controller")
class CalibrateUmiCameraCommand(BaseCommand):
    help = (
        "Live gold vs wrist-camera overlay (orange / white / screw superposition). "
        "Run on the laptop; keep run-server and run-robot up."
    )

    def handle(self, cfg: DictConfig) -> None:
        try:
            import so106.observation  # noqa: F401 — register So106Observation
        except ImportError as exc:
            raise RuntimeError(
                "calibrate-umi-camera is for the so106 project. "
                "Run it from projects/so106."
            ) from exc

        import cv2
        import numpy as np
        import zmq

        from solarian.core.message import Message
        from solarian.core.observation import Observation
        from so106.skills.umi.compare_policy_image import (
            _imread_rgb,
            _imwrite_rgb,
            render_contact_sheet,
        )
        from so106.skills.umi.gripper_align import compute_align_metrics
        from so106.skills.umi.policy_image import PolicyImagePipeline

        skill_cfg = _load_skill_cfg()
        gold_path = _resolve_gold(cfg, skill_cfg)
        gold = _imread_rgb(gold_path)
        cam_module = str(skill_cfg.get("modules", {}).get("camera", "right_wrist_camera"))
        dump_dir = Path(str(skill_cfg.get("debug_image_path", _DUMP_DEFAULT)))

        pipeline = PolicyImagePipeline(
            output_res=(gold.shape[1], gold.shape[0]) if gold.shape[0] == gold.shape[1] else (224, 224),
            letterbox_to_square=bool(skill_cfg.get("letterbox_to_square", False)),
            no_mirror=bool(skill_cfg.get("no_mirror", True)),
            mirror_swap=bool(skill_cfg.get("mirror_swap", False)),
        )

        ip = str(cfg.connection.ip)
        port = int(cfg.connection.state_sender_port)
        endpoint = f"tcp://{ip}:{port}"
        print(
            f"Gold: {gold_path}\n"
            f"Subscribing to {endpoint}  camera={cam_module}\n"
            f"Need run-server + run-robot. Press q to quit, s to save {dump_dir / 'compare.png'}."
        )

        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(endpoint)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        got_any = False
        live_policy = None
        last_lines: list[str] = []
        deadline = time.monotonic() + _FIRST_WAIT_MS / 1000.0

        try:
            cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
            while True:
                events = dict(poller.poll(_POLL_MS))
                if sock in events:
                    raw = sock.recv()
                    msg = Message.deserialize(raw)
                    obs_dict = (msg.data or {}).get("observation")
                    if obs_dict:
                        observation = Observation(obs_dict)
                        rgb = _wrist_rgb(observation, cam_module)
                        if rgb is not None:
                            live_policy = pipeline.process_frame(rgb)
                            metrics = compute_align_metrics(gold, live_policy)
                            last_lines = metrics.lines()
                            got_any = True

                if not got_any:
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            f"No observations from {endpoint} after {_FIRST_WAIT_MS // 1000}s. "
                            "Start `solarian run-server` and `solarian run-robot` "
                            "with the same --conf."
                        )
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        return
                    continue

                sheet = render_contact_sheet(
                    gold, live_policy, extra_lines=last_lines, scale=_SCALE
                )
                cv2.imshow(_WINDOW, np.ascontiguousarray(sheet[:, :, ::-1]))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return
                if key == ord("s") and live_policy is not None:
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    out = dump_dir / "compare.png"
                    _imwrite_rgb(
                        out,
                        render_contact_sheet(gold, live_policy, extra_lines=last_lines),
                    )
                    print(f"Wrote {out}")
                    print("\n".join(last_lines))
        finally:
            try:
                cv2.destroyWindow(_WINDOW)
            except Exception:
                pass
            sock.close(0)
            ctx.term()
