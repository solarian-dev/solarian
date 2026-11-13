from __future__ import annotations

import difflib
import inspect
import os
import re
import textwrap
import types
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

if TYPE_CHECKING:
    from solarian.core.configs import ModuleConfig


#: Where a config class without its own ``Docs:`` line sends the reader.
CONFIG_DOCS_URL = "https://solarian.dev/docs/project-configs"

_DOCS_LINE = re.compile(r"^\s*Docs:\s*(\S+)", re.MULTILINE)
_FIELD_LINE = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*:(?!=)")
_RST_ROLE = re.compile(r":\w+:`~?(?:[\w.]+\.)?([^`]+)`")

_WIDTH = 88
_MAX_INPUT_REPR = 60


class ConfigError(ValueError):
    """A config file that cannot be loaded, already formatted for a person.

    Either it does not validate, or it names a ``${deployment.*}`` value the
    deployment file does not set. The message is the whole report, so it reads
    the same wherever it surfaces: caught by the CLI, or printed as the last
    line of a child process's traceback.
    """

    #: A runner logs this as an outcome, not as a crash with a traceback.
    deliberate = True

    def __init__(
        self,
        model_class: type[BaseModel] | None = None,
        errors: list[dict[str, Any]] | None = None,
        config_path: str | None = None,
        cli_overridable: bool = False,
        *,
        message: str | None = None,
    ):
        self.model_class = model_class
        self.errors = errors or []
        self.config_path = config_path
        if message is not None:
            super().__init__(message)
            return
        assert model_class is not None and errors is not None
        super().__init__(
            format_config_errors(model_class, errors, config_path, cli_overridable)
        )


def parse_with_friendly_errors(model_class, data, config_path: str | None = None):
    """``model_class(**data)``, raising :class:`ConfigError` if it does not validate.

    ``config_path`` names the file ``data`` came from. Unset, it is the command's
    own config, which is also the one command-line overrides apply to.
    """
    try:
        return model_class(**data)
    except ValidationError as e:
        cli_overridable = config_path is None
        if config_path is None:
            config_path = os.environ.get("SOLARIAN_CURRENT_CONFIG") or None
        raise ConfigError(
            model_class, e.errors(include_url=False), config_path, cli_overridable
        ) from None


def docs_url(model_class: type) -> str | None:
    """The ``Docs:`` link of the nearest class in ``model_class``'s MRO that has one."""
    for klass in model_class.__mro__:
        if klass is BaseModel:
            break
        doc = klass.__dict__.get("__doc__")
        if doc:
            match = _DOCS_LINE.search(doc)
            if match:
                return match.group(1)
    return None


def field_help(model_class: type[BaseModel]) -> dict[str, str]:
    """Each field's help text: ``Field(description=...)``, else its ``#:`` comment."""
    comments: dict[str, str] = {}
    for klass in reversed(model_class.__mro__):
        if isinstance(klass, type) and issubclass(klass, BaseModel) and klass is not BaseModel:
            comments.update(_field_comments(klass))
    return {
        name: field.description or comments.get(name, "")
        for name, field in model_class.model_fields.items()
    }


@lru_cache(maxsize=None)
def _field_comments(klass: type) -> dict[str, str]:
    """The ``#:`` comment block directly above each field declared in ``klass``'s body."""
    try:
        lines, _ = inspect.getsourcelines(klass)
    except (OSError, TypeError):
        return {}

    body_indent = None
    pending: list[str] = []
    comments: dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            pending = []
            continue
        indent = len(line) - len(line.lstrip())
        if body_indent is None:
            body_indent = indent
        if stripped.startswith("#:"):
            pending.append(stripped[2:].strip())
            continue
        match = _FIELD_LINE.match(line)
        if match and pending and indent == body_indent:
            comments[match.group(2)] = " ".join(pending)
        pending = []
    return comments


def format_config_errors(
    model_class: type[BaseModel],
    errors: list[dict[str, Any]],
    config_path: str | None = None,
    cli_overridable: bool = False,
) -> str:
    """A validation failure as a person fixing the yaml wants to read it."""
    missing: dict[str, list[str]] = {}
    invalid: dict[str, list[dict[str, Any]]] = {}
    unknown: dict[str, list[str]] = {}
    top_docs = docs_url(model_class) or CONFIG_DOCS_URL

    for error in errors:
        path, owner, field = _resolve(model_class, error["loc"])
        if error["type"] == "missing":
            if path not in missing:
                missing[path] = _describe(path, field, owner, top_docs, missing=True)
        elif error["type"] == "extra_forbidden":
            unknown.setdefault(path, _describe_unknown(path, owner))
        else:
            invalid.setdefault(path, []).append(error)

    where = _display_path(config_path) if config_path else None
    title = f"Configuration error in {where} ({model_class.__name__})" if where else (
        f"Configuration error in {model_class.__name__}"
    )
    out = [title, ""]

    if missing:
        out.append("  Missing required fields:")
        for block in missing.values():
            out.extend(block)
        out.append("")

    if invalid:
        out.append("  Invalid values:")
        for path, errs in invalid.items():
            _, owner, field = _resolve(model_class, errs[0]["loc"])
            out.extend(_describe_invalid(path, errs, field, owner, top_docs))
        out.append("")

    if unknown:
        out.append("  Unknown fields:")
        for block in unknown.values():
            out.extend(block)
        out.append("")

    if missing:
        first = next(iter(missing))
        hint = f"  Add the missing fields to {where}" if where else "  Add the missing fields"
        if cli_overridable:
            hint += f", or set them on the command line, e.g. {first}=<value>"
        out.append(hint + ".")
    elif where:
        out.append(f"  Fix the values above in {where}.")
    out.append(f"  Docs: {top_docs}")
    return "\n".join(out)


def _describe(
    path: str, field: FieldInfo | None, owner: type | None, top_docs: str, missing: bool
) -> list[str]:
    head = f"    {path}"
    if field is not None:
        head += f": {_type_name(field.annotation)}"
    lines = [head]
    if owner is not None and field is not None:
        name = path.rsplit(".", 1)[-1]
        lines.extend(_wrap(field_help(owner).get(name, "")))
        choices = _choices(field.annotation)
        if missing and choices:
            lines.extend(_wrap("One of: " + ", ".join(repr(c) for c in choices)))
    lines.extend(_nested_docs(owner, top_docs))
    return lines


def _describe_invalid(
    path: str,
    errors: list[dict[str, Any]],
    field: FieldInfo | None,
    owner: type | None,
    top_docs: str,
) -> list[str]:
    label = path or "(whole section)"
    head = f"    {label}"
    if "input" in errors[0] and path:
        head += f": got {_short_repr(errors[0]['input'])}"
    lines = [head]
    # A union reports one error per member it tried; the declared type says the
    # same thing once.
    if len(errors) > 1 and field is not None:
        lines.extend(_wrap(f"Expected {_type_name(field.annotation)}"))
    else:
        lines.extend(_wrap(_clean_msg(errors[0]["msg"])))
        if field is not None:
            lines.extend(_wrap(f"Type: {_type_name(field.annotation)}"))
    if field is not None and owner is not None:
        name = path.rsplit(".", 1)[-1]
        lines.extend(_wrap(field_help(owner).get(name, "")))
        default = _default_repr(field)
        if default is not None:
            lines.extend(_wrap(f"Default: {default}"))
    lines.extend(_nested_docs(owner, top_docs))
    return lines


def _describe_unknown(path: str, owner: type | None) -> list[str]:
    name = path.rsplit(".", 1)[-1]
    line = f"    {path}"
    if owner is not None and issubclass(owner, BaseModel):
        close = difflib.get_close_matches(name, owner.model_fields, n=1)
        line += f": not a field of {owner.__name__}"
        if close:
            line += f" (did you mean '{close[0]}'?)"
    return [line]


def _nested_docs(owner: type | None, top_docs: str) -> list[str]:
    if owner is None:
        return []
    url = docs_url(owner)
    return _wrap(f"Docs: {url}") if url and url != top_docs else []


def _resolve(
    model_class: type[BaseModel], loc: tuple[Any, ...]
) -> tuple[str, type | None, FieldInfo | None]:
    """``loc`` as a dotted path, plus the model declaring its last field and that field.

    Pydantic puts the name of the union member it tried into ``loc``; those
    parts name no field, so they are dropped from the path.
    """
    current: Any = model_class
    owner: type | None = None
    field: FieldInfo | None = None
    path = ""
    for part in loc:
        if isinstance(part, int):
            current = _element_type(current, part)
            path += f"[{part}]"
            field = None
            continue
        model = _model_with_field(current, part)
        if model is not None:
            owner, field = model, model.model_fields[part]
            current = field.annotation
            path = f"{path}.{part}" if path else part
            continue
        value_type = _dict_value_type(current)
        if value_type is not None:
            current = value_type
            path = f"{path}.{part}" if path else part
            field = None
            continue
        member = _union_member(current, part)
        if member is not None:
            current = member
            continue
        # Unknown key: an extra field, or a model the walk could not follow.
        owner = _first_model(current) or owner
        path = f"{path}.{part}" if path else part
        field = None
    return path, owner, field


def _members(tp: Any) -> list[Any]:
    if get_origin(tp) in (Union, types.UnionType):
        members = []
        for arg in get_args(tp):
            members.extend(_members(arg))
        return members
    return [] if tp is type(None) else [tp]


def _model_with_field(tp: Any, name: str) -> type[BaseModel] | None:
    for member in _members(tp):
        if isinstance(member, type) and issubclass(member, BaseModel) and name in member.model_fields:
            return member
    return None


def _first_model(tp: Any) -> type[BaseModel] | None:
    for member in _members(tp):
        if isinstance(member, type) and issubclass(member, BaseModel):
            return member
    return None


def _dict_value_type(tp: Any) -> Any:
    for member in _members(tp):
        if get_origin(member) is dict:
            args = get_args(member)
            return args[1] if len(args) == 2 else Any
    return None


def _element_type(tp: Any, index: int) -> Any:
    for member in _members(tp):
        origin = get_origin(member)
        args = get_args(member)
        if origin in (list, set, frozenset) and args:
            return args[0]
        if origin is tuple and args:
            if len(args) == 2 and args[1] is Ellipsis:
                return args[0]
            return args[index] if index < len(args) else Any
    return Any


def _union_member(tp: Any, tag: str) -> Any:
    for member in _members(tp):
        if _type_name(member) == tag or getattr(member, "__name__", None) == tag:
            return member
    return None


def _type_name(tp: Any) -> str:
    if tp is type(None):
        return "None"
    if tp is Any:
        return "any"
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        return " | ".join(_type_name(arg) for arg in get_args(tp))
    if origin is Literal:
        return " | ".join(repr(arg) for arg in get_args(tp))
    if origin is not None:
        args = ", ".join("..." if a is Ellipsis else _type_name(a) for a in get_args(tp))
        return f"{getattr(origin, '__name__', str(origin))}[{args}]"
    if isinstance(tp, type):
        return tp.__name__
    return str(tp)


def _choices(tp: Any) -> list[Any]:
    choices: list[Any] = []
    for member in _members(tp):
        if get_origin(member) is Literal:
            choices.extend(get_args(member))
        elif isinstance(member, type) and issubclass(member, Enum):
            choices.extend(m.value for m in member)
    return choices


def _default_repr(field: FieldInfo) -> str | None:
    if field.default is PydanticUndefined or field.default_factory is not None:
        return None
    value = field.default
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, BaseModel):
        return None
    return _short_repr(value)


def _short_repr(value: Any) -> str:
    text = repr(value)
    if len(text) > _MAX_INPUT_REPR:
        text = text[: _MAX_INPUT_REPR - 3] + "..."
    return text


def _clean_msg(msg: str) -> str:
    return msg.removeprefix("Value error, ")


def _wrap(text: str) -> list[str]:
    if not text:
        return []
    text = _RST_ROLE.sub(r"`\1`", text).replace("``", "`")
    return textwrap.wrap(
        text, width=_WIDTH, initial_indent="      ", subsequent_indent="      "
    )


def _display_path(path: str) -> str:
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return path
    return path if rel.startswith("..") else rel


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
    configs and child processes (``solarian run-service``) all see the same values.
    Empty for projects with no ``deployment/`` directory.
    """
    from solarian.core.conf_paths import DEPLOYMENT, conf_dir

    conf_path = str(conf_dir())
    name = os.environ.get("SOLARIAN_DEPLOYMENT") or DEFAULT_DEPLOYMENT

    key = (conf_path, name)
    if key in _deployment_cache:
        return _deployment_cache[key]

    deployment_dir = os.path.join(conf_path, DEPLOYMENT)
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

    conf = load_yaml(yaml_path)
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


_MISSING_KEY = re.compile(r"Interpolation key '([^']+)' not found")


def _deployment_file() -> str:
    from solarian.core.conf_paths import DEPLOYMENT, conf_dir

    name = os.environ.get("SOLARIAN_DEPLOYMENT") or DEFAULT_DEPLOYMENT
    return _display_path(os.path.join(str(conf_dir()), DEPLOYMENT, f"{name}.yaml"))


def _source_label(source: str | os.PathLike | None) -> str:
    """Which file a config came from, for an error message.

    Unset, it is the command's own config, the one Hydra composed.
    """
    source = source or os.environ.get("SOLARIAN_CURRENT_CONFIG")
    return _display_path(str(source)) if source else "the configuration"


def load_yaml(path: str | os.PathLike) -> DictConfig:
    """``OmegaConf.load``, with a syntax error reported as a :class:`ConfigError`."""
    import yaml

    try:
        return OmegaConf.load(path)
    except yaml.YAMLError as exc:
        raise invalid_yaml(path, exc) from None


def invalid_yaml(path: str | os.PathLike, exc: Exception) -> "ConfigError":
    """A YAML syntax error in ``path`` as a :class:`ConfigError` that names the line."""
    mark = getattr(exc, "problem_mark", None)
    where = f", line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
    problem = getattr(exc, "problem", None) or str(exc)
    return ConfigError(
        message=f"{_display_path(str(path))} is not valid YAML{where}: {problem}."
    )


def resolve_with_deployment(
    cfg: DictConfig | dict, source: str | os.PathLike | None = None
) -> dict:
    """Resolve ``${deployment.*}`` against the deployment selected for this run.

    The injected node is dropped from the result so it never reaches a Pydantic
    model. ``source`` names the file ``cfg`` came from, for the error raised
    when an interpolation does not resolve.
    """
    from omegaconf.errors import InterpolationKeyError, InterpolationResolutionError

    merged = OmegaConf.merge({"deployment": load_deployment_config()}, cfg)
    try:
        data = OmegaConf.to_container(merged, resolve=True)
    except InterpolationKeyError as exc:
        match = _MISSING_KEY.search(exc.msg)
        missing = match.group(1) if match else str(exc.child_node)
        where = f"{exc.full_key} requires " if exc.full_key else ""
        defined_in = (
            _deployment_file() if missing.startswith("deployment.") else _source_label(source)
        )
        raise ConfigError(
            message=(
                f"Could not load {_source_label(source)}: {where}${{{missing}}}, "
                f"which is not set in {defined_in}."
            )
        ) from None
    except InterpolationResolutionError as exc:
        key = f"{exc.full_key}: " if exc.full_key else ""
        reason = exc.msg.splitlines()[0]
        raise ConfigError(
            message=f"Could not load {_source_label(source)}: {key}{reason}"
        ) from None
    del data["deployment"]
    return data


def config_error_in(traceback_str: str) -> "ConfigError | None":
    """The :class:`ConfigError` a child process's traceback ends in, if it does.

    A worker reports a crash as its formatted traceback. A config error is the
    reader's to fix rather than a crash, so the host re-raises it as one.
    """
    marker = f"{ConfigError.__module__}.{ConfigError.__qualname__}: "
    _, found, message = traceback_str.rpartition(marker)
    return ConfigError(message=message.strip()) if found else None


def load_config_yaml(yaml_path: str, config_class: type) -> Any:
    data = resolve_with_deployment(load_yaml(yaml_path), source=yaml_path)
    return parse_with_friendly_errors(config_class, data, config_path=yaml_path)


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
    a corresponding conf/modules/{name}.yaml and loads it using the module's
    own `config_class` (which must extend ModuleConfig).
    
    Args:
        robot_class: The Robot subclass (e.g. MySolarianRobot) with module annotations.
        
    Returns:
        Dict mapping module name to (module_type, config_instance) tuples.
    """
    from solarian.core.conf_paths import module_config_path
    from solarian.core.module import Module

    module_configs = {}
    
    # Walk the MRO to find all Module annotations (not just the leaf class)
    seen = set()
    for cls in robot_class.__mro__:
        for name, anno in getattr(cls, '__annotations__', {}).items():
            if name in seen:
                continue
            seen.add(name)
            if isinstance(anno, type) and issubclass(anno, Module):
                yaml_path = str(module_config_path(name))
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

