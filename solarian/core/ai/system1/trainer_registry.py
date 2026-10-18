"""
Trainer registry — discovery of installed policy trainers via entry points.

Policy plugins advertise their :class:`~solarian.core.ai.system1.policy_trainer.PolicyTrainer`
subclasses under the ``solarian.trainers`` entry-point group, e.g. in a
plugin's ``pyproject.toml``::

    [project.entry-points."solarian.trainers"]
    diffusion = "solarian_policy_diffusion.trainer:DiffusionTrainer"

The framework's ``solarian train`` command reads the ``trainer`` key from the
config (``trainer: diffusion``) and calls :func:`get_trainer_class` to resolve
it to the trainer class — without importing the plugin directly.  This keeps
the framework library-agnostic and mirrors how skills are discovered (see
:mod:`solarian.core.ai.system1.registry`).
"""

import importlib.metadata
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

#: Entry-point group policy trainers register under.
TRAINER_ENTRY_POINT_GROUP = "solarian.trainers"


def _iter_entry_points(group: str):
    """Yield entry points for *group* across importlib.metadata API versions."""
    try:
        eps = importlib.metadata.entry_points()
    except Exception:  # noqa: BLE001 - metadata access can fail in odd envs
        logger.exception("Failed to read entry points")
        return []
    # Python 3.10+: EntryPoints.select; older: dict-like .get
    if hasattr(eps, "select"):
        return list(eps.select(group=group))
    return list(eps.get(group, []))  # type: ignore[attr-defined]


def discover_trainer_classes() -> Dict[str, type]:
    """Return ``{entry_point_name: PolicyTrainer subclass}`` for installed plugins."""
    out: Dict[str, type] = {}
    for ep in _iter_entry_points(TRAINER_ENTRY_POINT_GROUP):
        try:
            out[ep.name] = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not kill discovery
            logger.exception("Failed to load trainer entry point %r", ep.name)
    return out


def available_trainer_names() -> List[str]:
    """Return the names of all installed trainers (without importing them)."""
    return sorted(ep.name for ep in _iter_entry_points(TRAINER_ENTRY_POINT_GROUP))


def get_trainer_class(name: str) -> type:
    """Resolve a trainer *name* to its :class:`PolicyTrainer` subclass.

    Raises:
        KeyError: if no installed plugin registers a trainer under *name*.
            The error lists the trainers that *are* available to make typos
            and missing installs obvious.
    """
    for ep in _iter_entry_points(TRAINER_ENTRY_POINT_GROUP):
        if ep.name == name:
            return ep.load()
    available = available_trainer_names()
    hint = ", ".join(available) if available else "(none installed)"
    raise KeyError(
        f"No policy trainer registered as {name!r} under the "
        f"{TRAINER_ENTRY_POINT_GROUP!r} entry-point group. "
        f"Available trainers: {hint}. "
        f"Install the policy plugin that provides {name!r} "
        f"(e.g. `pip install solarian-policy-{name}`)."
    )


def register_resolvers(trainer_cls: type) -> None:
    """Register the OmegaConf resolvers *trainer_cls* declares."""
    from omegaconf import OmegaConf

    for name, function in trainer_cls.resolvers().items():
        OmegaConf.register_new_resolver(name, function, replace=True)


def load_train_config(cfg: Any) -> Tuple[type, dict, Any]:
    """The trainer a composed training config names, the config resolved, and validated.

    *cfg* is unresolved: it may interpolate with resolvers only its trainer
    registers (``${eval:...}``), so the ``trainer`` key is read first, that
    trainer's resolvers registered, and only then is the whole config
    resolved, ``${deployment.*}`` included.

    Returns ``(trainer_cls, cfg_dict, train_cfg)``: the resolved plain dict
    (what is shipped to a remote run) and the trainer's :meth:`config_class`
    instance built from it.
    """
    from omegaconf import OmegaConf

    from solarian.utils.configs import resolve_with_deployment

    name = OmegaConf.select(cfg, "trainer")
    if not name:
        available = ", ".join(available_trainer_names()) or "(none installed)"
        raise ValueError(
            "A training config must set 'trainer' to select a trainer; pass one with "
            f"--conf conf/training/<name>. Available trainers: {available}."
        )
    trainer_cls = get_trainer_class(str(name))
    register_resolvers(trainer_cls)
    cfg_dict = resolve_with_deployment(cfg)
    return trainer_cls, cfg_dict, trainer_cls.config_class().model_validate(cfg_dict)
