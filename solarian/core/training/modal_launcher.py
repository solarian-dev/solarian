"""
Generic Modal launcher for policy training.

``ModalLauncher`` turns a trainer's
:class:`~solarian.core.ai.system1.policy_trainer.ModalSpec` into a
``modal.Image`` and runs the trainer on cloud GPUs.  The framework never
hardcodes any library; the plugin declares its image needs via
:meth:`~solarian.core.ai.system1.policy_trainer.PolicyTrainer.modal_spec`,
and the orchestration knobs (GPU type/count, volume, secrets, timeout) come
from :class:`~solarian.core.configs.ModalConfig`.

Flow (mirrors UMI's ``train_modal.py`` but generalized):

1. Build an image from the spec (python version, apt, pip + ``accelerate``,
   any local python sources).
2. Mount a ``modal.Volume`` at ``/data`` for datasets and outputs and attach
   the configured secrets.
3. In the container, write the resolved config to the volume and run
   ``accelerate launch -m solarian.core.training.modal_entry`` with
   ``--num_processes = gpus`` for multi-GPU.
4. Commit the volume so checkpoints/logs persist.

Datasets must already live on the volume; reference them in the config with
``/data/...`` paths (dataset placement is plugin/trainer-specific, so the
launcher does not rewrite dataset paths — only ``output_dir``).
"""

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

#: Mount point of the Modal volume inside the container.
DATA_DIR = "/data"


class ModalLauncher:
    """Runs a :class:`PolicyTrainer` on Modal using its :class:`ModalSpec`."""

    def __init__(self, trainer_cls: type, cfg: Any):
        self.trainer_cls = trainer_cls
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Image construction
    # ------------------------------------------------------------------

    def _build_image(self, modal, spec, gpus: int):
        image = modal.Image.debian_slim(python_version=spec.python_version)
        if spec.apt_packages:
            image = image.apt_install(*spec.apt_packages)

        pip_packages = list(spec.pip_packages)
        # Multi-GPU launches go through `accelerate launch`; make sure it's
        # available even if the plugin forgot to declare it.
        if gpus >= 1 and not any(
            p == "accelerate" or p.startswith("accelerate") for p in pip_packages
        ):
            pip_packages.append("accelerate")
        if pip_packages:
            image = image.pip_install(*pip_packages)

        if spec.env:
            image = image.env(spec.env)

        # Ship unpublished/editable packages (e.g. a local UMI fork, or the
        # plugin itself during development) into the image.
        for module_name in spec.local_python_packages:
            try:
                image = image.add_local_python_source(module_name)
            except Exception:  # noqa: BLE001 - older modal versions / missing module
                logger.exception(
                    "Could not add local python source %r to the Modal image; "
                    "ensure it is installed/importable or listed in pip_packages.",
                    module_name,
                )
        return image

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        import modal

        modal_cfg = self.cfg.modal
        spec = self.trainer_cls.modal_spec()
        gpus = int(modal_cfg.gpus)
        gpu_spec = f"{modal_cfg.gpu_type}:{gpus}" if gpus > 1 else modal_cfg.gpu_type

        image = self._build_image(modal, spec, gpus)
        app = modal.App(modal_cfg.app_name, image=image)
        volume = modal.Volume.from_name(modal_cfg.volume, create_if_missing=True)
        secrets = [modal.Secret.from_name(name) for name in modal_cfg.secrets]

        # Point outputs at the volume so checkpoints/logs survive the container.
        run_id = time.strftime("%Y.%m.%d_%H.%M.%S")
        job_name = getattr(self.cfg, "job_name", "train")
        remote_output_dir = f"{DATA_DIR}/outputs/{job_name}/{run_id}"

        cfg_dict = self.cfg.model_dump()
        cfg_dict["output_dir"] = remote_output_dir
        # Avoid an infinite hop: the in-container run must not re-trigger Modal.
        if isinstance(cfg_dict.get("modal"), dict):
            cfg_dict["modal"]["enabled"] = False
        cfg_json = json.dumps(cfg_dict)

        mixed_precision = modal_cfg.mixed_precision

        @app.function(
            gpu=gpu_spec,
            timeout=int(modal_cfg.timeout),
            volumes={DATA_DIR: volume},
            secrets=secrets,
        )
        def _train_remote(cfg_json: str, output_dir: str, num_processes: int):
            import os
            import subprocess
            import sys

            os.makedirs(output_dir, exist_ok=True)
            cfg_path = os.path.join(output_dir, "train_config.json")
            with open(cfg_path, "w") as f:
                f.write(cfg_json)
            volume.commit()

            cmd = [
                "accelerate", "launch",
                "--num_processes", str(num_processes),
                "--mixed_precision", mixed_precision,
                "-m", "solarian.core.training.modal_entry",
                "--config", cfg_path,
            ]
            print(">>>", " ".join(cmd), flush=True)
            try:
                subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
            finally:
                # Persist whatever made it to disk, even on partial failures.
                volume.commit()
            print(f"Training complete. Outputs on volume {modal_cfg.volume}:{output_dir}")

        logger.info(
            "Launching Modal training: app=%s gpu=%s volume=%s output=%s",
            modal_cfg.app_name, gpu_spec, modal_cfg.volume, remote_output_dir,
        )
        with app.run():
            _train_remote.remote(cfg_json, remote_output_dir, gpus)
