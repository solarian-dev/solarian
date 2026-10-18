"""
PolicyTrainer — base class for training a policy inside a plugin.

A *policy plugin* (e.g. ``solarian-policy-diffusion`` or
``solarian-policy-lerobot``) ships two things keyed to a model family /
library: a :class:`~solarian.core.ai.system1.policy_server.PolicyServer`
(inference) and a :class:`PolicyTrainer` (training).  The framework's
``solarian train`` command is deliberately library-agnostic: it never
imports a concrete trainer.  Instead it reads the ``trainer`` key from the
config, resolves the matching :class:`PolicyTrainer` subclass through the
``solarian.trainers`` entry-point group (see
:mod:`solarian.core.ai.system1.trainer_registry`), and runs it.

Subclasses implement a single required method, :meth:`train`, and may
override:

* :meth:`config_class`  — the pydantic schema used to validate the config
  (defaults to the thin :class:`~solarian.core.configs.TrainConfig`).
* :meth:`modal_spec`    — the container image spec for cloud training.
* :meth:`resolvers`     — OmegaConf resolvers the config interpolates with.
* :meth:`add_arguments` / :meth:`from_args` — standalone CLI launching (so a
  trainer can also be driven directly under ``accelerate launch``).

Typical plugin usage::

    class MyTrainer(PolicyTrainer):
        @classmethod
        def config_class(cls):
            return MyTrainConfig

        def train(self):
            ...  # the real training loop

    def main():
        MyTrainer.main()
"""

import abc
import argparse
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModalSpec:
    """Describes the container image a trainer needs to run on Modal.

    The ``solarian-modal`` plugin (``solarian modal-train``) turns this into a
    ``modal.Image``.  A plugin only declares *what* it needs (apt + pip
    packages); it never imports ``modal`` itself.
    """

    def __init__(
        self,
        python_version: str = "3.10",
        apt_packages: Optional[List[str]] = None,
        pip_packages: Optional[List[str]] = None,
        local_python_packages: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        #: Python version for the micromamba/base image.
        self.python_version = python_version
        #: System packages installed with apt (e.g. ``ffmpeg``, ``libgl1``).
        self.apt_packages = list(apt_packages or [])
        #: Pip requirements (e.g. ``"torch==2.8.0"``). They take precedence
        #: over the runtime requirements of ``local_python_packages``, which
        #: the launcher adds, as it does ``accelerate``.
        self.pip_packages = list(pip_packages or [])
        #: Importable local packages shipped into the image as source, with
        #: their package data (e.g. ``"solarian"`` and the plugin itself).
        self.local_python_packages = list(local_python_packages or [])
        #: Extra environment variables baked into the image.
        self.env = dict(env or {})


class PolicyTrainer(abc.ABC):
    """Abstract base class for policy trainers.

    Subclasses implement :meth:`train`.  The framework constructs an
    instance with the validated config and an output directory, then calls
    :meth:`train` (locally or, for Modal runs, inside a GPU container).
    """

    #: Human-readable title (override in subclasses if desired).
    title: str = "Solarian Policy Trainer"

    def __init__(self, cfg: Any, output_dir: Optional[str] = None):
        """
        Args:
            cfg: The validated config (an instance of :meth:`config_class`).
                Because the base :class:`~solarian.core.configs.TrainConfig`
                allows extra keys, plugin-specific fields are preserved and
                accessible as attributes / via ``cfg.model_dump()``.
            output_dir: Resolved directory for checkpoints and logs.
        """
        self.cfg = cfg
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # Methods to implement
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def train(self) -> None:
        """Run training to completion (blocking)."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    @classmethod
    def config_class(cls) -> type:
        """Return the pydantic config class used to validate the config.

        Defaults to the thin framework :class:`~solarian.core.configs.TrainConfig`.
        Plugins override this with a subclass that adds their own fields.
        """
        from solarian.core.configs import TrainConfig

        return TrainConfig

    @classmethod
    def modal_spec(cls) -> ModalSpec:
        """Return the :class:`ModalSpec` describing the cloud image.

        Default is an empty spec; trainers that support Modal override it.
        """
        return ModalSpec()

    @classmethod
    def resolvers(cls) -> Dict[str, Callable[..., Any]]:
        """OmegaConf resolvers the training config uses, ``{name: function}``.

        Registered before the config is resolved, so a config written for the
        library's own tooling (``${eval:...}``) loads unchanged.
        """
        return {}

    # ------------------------------------------------------------------
    # Standalone CLI helpers
    # ------------------------------------------------------------------

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add subclass-specific CLI arguments. Override as needed."""

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PolicyTrainer":
        """Build a trainer from parsed CLI args.

        The default loads a config file (JSON or YAML) given by ``--config``,
        validates it with :meth:`config_class`, and uses ``--output-dir`` (or
        the config's ``output_dir``) as the output directory.
        """
        from omegaconf import OmegaConf

        from solarian.core.ai.system1.trainer_registry import register_resolvers

        register_resolvers(cls)
        cfg_dict = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        cfg = cls.config_class().model_validate(cfg_dict)
        output_dir = getattr(args, "output_dir", None) or getattr(cfg, "output_dir", None)
        return cls(cfg, output_dir=output_dir)

    @classmethod
    def main(cls, argv: Optional[list] = None) -> None:
        """Console entry point: parse args, build the trainer and :meth:`train`."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        parser = argparse.ArgumentParser(description=f"{cls.__name__} policy trainer")
        parser.add_argument(
            "--config", "-c", required=True,
            help="Path to a resolved config file (JSON or YAML).",
        )
        parser.add_argument(
            "--output-dir", "-o", default=None,
            help="Output directory for checkpoints/logs (overrides config).",
        )
        cls.add_arguments(parser)
        args = parser.parse_args(argv)
        trainer = cls.from_args(args)
        trainer.train()
