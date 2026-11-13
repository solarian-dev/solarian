"""What to run for ``solarian run-service <name>``: a project conf or a bare service.

A project conf that needs a server says which, in one block of the same shape
whether it is a processor's conf or a skill's::

    # conf/services/bbox_tracker.yaml        # conf/skills/umi_target.yaml
    server:                                  server:
      service: sam2-tracker                    service: diffusion-policy
      port: 8765                               checkpoint: ${deployment.umi_target_policy_checkpoint}
      args: [--model, small]                   host: 0.0.0.0
                                               port: 8003

``service`` names a ``solarian.services`` entry point; ``checkpoint`` goes to
the server's ``checkpoint_arg``. ``run-service``, a skill's managed server,
``solarian modal-serve`` and Solarian Base all resolve this one block, so
they cannot disagree on what a conf serves.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The key of the server block in a conf.
SERVER_KEY = "server"
#: Bind address when neither the conf nor the caller names one: the robot or
#: relay reaches a service over the network, never via loopback.
DEFAULT_HOST = "0.0.0.0"


@dataclass
class ServerSpec:
    #: ``solarian.services`` entry-point name.
    service: str
    #: Module run with ``python -m``.
    module: str
    host: str
    port: int
    checkpoint: str | None = None
    #: The server's flag taking ``checkpoint``.
    checkpoint_arg: str | None = None
    #: Passed to the server unchanged.
    args: list[str] = field(default_factory=list)
    #: Whether a skill launches this server itself (``server.managed``).
    managed: bool = False

    def command(self) -> list[str]:
        cmd = [sys.executable, "-m", self.module, "--host", self.host, "--port", str(self.port)]
        if self.checkpoint is not None:
            cmd += [str(self.checkpoint_arg), self.checkpoint]
        return cmd + self.args


def server_spec(
    block: dict[str, Any],
    *,
    default_service: str | None = None,
    checkpoint: str | None = None,
    host: str | None = None,
    port: int | None = None,
    source: str = "the conf",
) -> ServerSpec:
    """The server a resolved ``server:`` block describes; arguments override it."""
    from solarian.services.service import discover_services

    name = block.get("service") or default_service
    if not name:
        raise ValueError(f"{source} has no server.service naming the service to run.")
    services = discover_services()
    service_cls = services.get(name)
    if service_cls is None:
        available = ", ".join(sorted(services)) or "(none installed)"
        raise ValueError(f"{source} names service {name!r}, which is not installed. Installed: {available}.")
    checkpoint = checkpoint or block.get("checkpoint")
    if service_cls.checkpoint_arg is None and checkpoint:
        raise ValueError(f"Service {name!r} takes no checkpoint, but {source} gives one.")
    if service_cls.checkpoint_arg is not None and not checkpoint:
        raise ValueError(
            f"Service {name!r} needs a checkpoint: set server.checkpoint in {source} "
            "or pass --checkpoint."
        )
    return ServerSpec(
        service=name,
        module=service_cls.server_module,
        host=str(host or block.get("host") or DEFAULT_HOST),
        port=int(port if port is not None else block.get("port") or service_cls.default_port),
        checkpoint=str(checkpoint) if checkpoint else None,
        checkpoint_arg=service_cls.checkpoint_arg,
        args=[str(a) for a in block.get("args") or []],
        managed=bool(block.get("managed", False)),
    )


def server_confs(name: str) -> list[Path]:
    """Project confs called ``name`` that carry a server block.

    ``name`` is a conf name (``bbox_tracker``), a filename or a path; a name is
    looked up in both ``conf/services/`` and ``conf/skills/``.
    """
    from solarian.core.conf_paths import service_config_path, skill_config_path

    path = Path(name)
    if path.suffix == ".yaml" and (path.is_absolute() or len(path.parts) > 1):
        candidates = [path]
    else:
        stem = path.stem if path.suffix == ".yaml" else name
        candidates = [service_config_path(stem), skill_config_path(stem)]
    return [p for p in candidates if p.is_file() and SERVER_KEY in read_conf(p)]


def read_conf(path: Path) -> dict[str, Any]:
    """A conf as written, ``${...}`` left unresolved."""
    import yaml

    from solarian.utils.configs import invalid_yaml

    with open(path) as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise invalid_yaml(path, exc) from None


def resolve_server_block(
    raw: dict[str, Any], checkpoint: str | None = None, source: Path | None = None
) -> dict[str, Any]:
    """``raw``'s server block with ``${deployment.*}`` resolved for this host.

    Only the block is resolved, and ``checkpoint`` replaces the conf's before
    resolution, so a host whose deployment lacks the conf's other keys (a
    Solarian Base checkout has no ``conf/deployment``) can still serve it.
    ``source`` is the conf ``raw`` was read from, named if a value is missing.
    """
    from solarian.utils.configs import resolve_with_deployment

    block = dict(raw.get(SERVER_KEY) or {})
    if checkpoint:
        block["checkpoint"] = checkpoint
    return resolve_with_deployment({SERVER_KEY: block}, source=source)[SERVER_KEY]


def resolve_server(
    name: str,
    checkpoint: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> ServerSpec:
    """The server ``name`` means: a project conf with a server block, or an installed service."""
    from solarian.services.service import discover_services

    confs = server_confs(name)
    is_service = name in discover_services()
    if len(confs) + is_service > 1:
        matches = [str(p) for p in confs] + ([f"the installed service {name!r}"] if is_service else [])
        raise ValueError(f"{name!r} is ambiguous: {', '.join(matches)}. Pass a conf's path instead.")
    if confs:
        block = resolve_server_block(read_conf(confs[0]), checkpoint, source=confs[0])
        return server_spec(block, host=host, port=port, source=str(confs[0]))
    if is_service:
        return server_spec({"service": name}, checkpoint=checkpoint, host=host, port=port, source=f"service {name!r}")
    available = ", ".join(sorted(discover_services())) or "(none installed)"
    raise ValueError(
        f"{name!r} is neither a conf with a server block in conf/services/ or conf/skills/ "
        f"nor an installed service. Installed services: {available}."
    )
