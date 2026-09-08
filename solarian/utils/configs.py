from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from typing import TYPE_CHECKING, Any
from pydantic import ValidationError
import os

if TYPE_CHECKING:
    from solarian.core.configs import ModuleConfig


def parse_with_friendly_errors(model_class, data):
    """Parse config data with user-friendly error messages"""
    try:
        return model_class(**data)
    except ValidationError as e:
        # Get the config filename if possible
        config_path = os.environ.get("SOLARIAN_CURRENT_CONFIG", "")
        
        # Format a user-friendly error message
        error_msg = f"Configuration Error in {model_class.__name__}:\n"
        error_msg += f"  {str(e)}\n"
        
        if config_path:
            error_msg += f"  Please check your configuration file: {config_path}\n"
            
        # Re-raise with the friendly message
        raise ValueError(error_msg) from e


DEFAULT_DEPLOYMENT = "localhost"

# Cached because every module and skill config load resolves against it.
_deployment_cache: dict[tuple[str, str], DictConfig] = {}


#: Names another deployment to inherit from, so a profile that changes one thing
#: does not have to restate the device paths it shares.
EXTENDS_KEY = "extends"


def load_deployment_config() -> DictConfig:
    """The active deployment: relay address and device paths for this host layout.

    Read straight from ``{SOLARIAN_CONF_PATH}/deployment/{SOLARIAN_DEPLOYMENT}.yaml``
    rather than from the Hydra tree, so command configs, module configs, skill
    configs and child processes (``solarian run-policy``) all see the same values.
    Empty for projects with no ``deployment/`` directory.
    """
    conf_path = os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))
    name = os.environ.get("SOLARIAN_DEPLOYMENT") or DEFAULT_DEPLOYMENT

    key = (conf_path, name)
    if key in _deployment_cache:
        return _deployment_cache[key]

    deployment_dir = os.path.join(conf_path, "deployment")
    if not os.path.isdir(deployment_dir):
        conf = OmegaConf.create({})
    else:
        conf = _load_deployment_file(deployment_dir, name)

    _deployment_cache[key] = conf
    return conf


def _load_deployment_file(
    deployment_dir: str, name: str, extended_by: tuple[str, ...] = ()
) -> DictConfig:
    """One deployment file, merged over whatever it ``extends``.

    Deployments differ from each other in one or two values far more often than
    they differ throughout — moving the relay to another host changes an address,
    not which port the arm is on — and a copy of the whole profile per variation
    is a set of device paths that drift apart silently.
    """
    if name in extended_by:
        chain = " -> ".join([*extended_by, name])
        raise ValueError(f"Deployment '{name}' extends itself: {chain}")

    yaml_path = os.path.join(deployment_dir, f"{name}.yaml")
    if not os.path.exists(yaml_path):
        choices = sorted(
            f[: -len(".yaml")] for f in os.listdir(deployment_dir) if f.endswith(".yaml")
        )
        if extended_by:
            raise ValueError(
                f"Deployment '{extended_by[-1]}' extends '{name}', which does not "
                f"exist: {yaml_path}\n"
                f"  Available deployments: {', '.join(choices) or '(none)'}"
            )
        raise ValueError(
            f"Unknown deployment '{name}': {yaml_path} does not exist.\n"
            f"  Available deployments: {', '.join(choices) or '(none)'}\n"
            f"  Select one with `--deployment <name>` or SOLARIAN_DEPLOYMENT=<name>."
        )

    conf = OmegaConf.load(yaml_path)
    base_name = conf.pop(EXTENDS_KEY, None)
    if base_name is None:
        return conf
    if not isinstance(base_name, str):
        raise ValueError(
            f"Deployment '{name}' has {EXTENDS_KEY}: {base_name!r}; expected the "
            f"name of one other deployment."
        )
    base = _load_deployment_file(deployment_dir, base_name, (*extended_by, name))
    return OmegaConf.merge(base, conf)


def resolve_with_deployment(cfg: DictConfig | dict) -> dict:
    """Resolve ``${deployment.*}`` against the deployment selected for this run.

    The injected node is dropped from the result so it never reaches a Pydantic
    model.
    """
    merged = OmegaConf.merge({"deployment": load_deployment_config()}, cfg)
    data = OmegaConf.to_container(merged, resolve=True)
    del data["deployment"]
    return data


def load_config_yaml(yaml_path: str, config_class: type) -> Any:
    data = resolve_with_deployment(OmegaConf.load(yaml_path))
    return parse_with_friendly_errors(config_class, data)


def load_module_config_yaml(yaml_path: str) -> ModuleConfig:
    # Imported here because solarian.core's package init reaches back into this
    # module, so a top-level import makes it depend on which side was imported
    # first.
    from solarian.core.configs import ModuleConfig

    return load_config_yaml(yaml_path, ModuleConfig)


def load_module_configs_for_robot(robot_class) -> dict[str, tuple[type, ModuleConfig]]:
    """
    Load module configs from separate YAML files based on a Robot class's annotations.
    
    For each annotation on the robot class that is a Module subclass, looks for
    a corresponding {name}.yaml file in the conf directory and loads it using
    the module's own `config_class` (which must extend ModuleConfig).
    
    Args:
        robot_class: The Robot subclass (e.g. MySolarianRobot) with module annotations.
        
    Returns:
        Dict mapping module name to (module_type, config_instance) tuples.
    """
    from solarian.core.module import Module
    
    conf_path = os.environ.get("SOLARIAN_CONF_PATH", os.path.join(os.getcwd(), "conf"))
    module_configs = {}
    
    # Walk the MRO to find all Module annotations (not just the leaf class)
    seen = set()
    for cls in robot_class.__mro__:
        for name, anno in getattr(cls, '__annotations__', {}).items():
            if name in seen:
                continue
            seen.add(name)
            if isinstance(anno, type) and issubclass(anno, Module):
                yaml_path = os.path.join(conf_path, f"{name}.yaml")
                if not os.path.exists(yaml_path):
                    raise FileNotFoundError(
                        f"Module config file not found: {yaml_path}\n"
                        f"  Robot class '{robot_class.__name__}' declares module '{name}: {anno.__name__}'\n"
                        f"  Expected a config file at: {yaml_path}"
                    )
                # Use the module's own config_class (defaults to ModuleConfig)
                cfg_class = anno.config_class
                module_cfg = load_config_yaml(yaml_path, cfg_class)
                module_configs[name] = (anno, module_cfg)
    
    return module_configs

