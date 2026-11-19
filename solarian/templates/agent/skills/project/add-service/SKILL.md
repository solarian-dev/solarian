---
name: add-service
description: Add a Solarian service - a model or other heavy computation served over HTTP (a tracker, a depth model, a policy server) - with its conf/services yaml, run it with `solarian run-service`, and call it from a skill or bind it to an observation field. Use when work is too heavy or too GPU-bound for a skill's own loop.
---

# Add a service

Docs: `solarian docs show services`, `policies`, `project-configs`. Check for a published one first (`solarian docs show plugins`); `solarian install <plugin>` then seeds its `conf/services/<name>.yaml`.

## 1. The server

```python
from solarian.services import ServiceServer


class DepthServer(ServiceServer):
    def load(self):                       # load the model once
        ...

    @property
    def info(self) -> dict:               # served at /info
        return {"model": "..."}

    def add_routes(self, app):            # a FastAPI app
        @app.post("/depth")
        def depth(payload: dict) -> dict:
            ...


if __name__ == "__main__":
    DepthServer.main()
```

`/`, `/health` and `/info` come built in. When `SOLARIAN_ENDPOINT_TOKEN` is set, a bearer token is required.

A **policy** server subclasses `PolicyServer` instead (default port 8001) and implements `load`, `predict(obs_dict, **kw) -> {"action": ...}`, `model_info` and `shape_meta`; the client is `PolicyClient`.

## 2. The service description

```python
from solarian.services import Service


class DepthService(Service):
    server_module = "{{ package }}.depth_server"   # run as python -m
    default_port = 8765
    checkpoint_arg = None                          # e.g. "--input" when a checkpoint is required
```

Installed services are found through the `solarian.services` entry point; for a plugin, `solarian create-plugin policy <name>` scaffolds server, service and skill together.

## 3. Its config

```yaml
# conf/services/depth.yaml
execution_place: ${deployment.depth_place}
server_url: ${deployment.depth_url}
token: ${oc.select:deployment.solarian_base_endpoint_token,null}
server: {service: depth, port: 8765, host: 0.0.0.0, checkpoint: null, args: [], managed: false}
```

URLs, checkpoints and tokens belong in `conf/deployment/<name>.yaml`, never literally here. A skill that serves a policy has the same `server:` block in its `conf/skills/<skill>.yaml`.

## 4. Run and call it

```bash
solarian run-service depth                    # or: solarian run-service <skill> -c <checkpoint>
```

It looks the name up in `conf/services/`, `conf/skills/`, then the installed services. Call it with `ServiceClient(server_url, timeout=5.0, token=...)`, or bind it to an observation with `CalculatedField(inputs, processor=<ServiceProcessor subclass>, output, conf=...)` and start it from the controller with `start_service(module, field, **params)`.

## 5. Check it

`curl <server_url>/health` while it runs, then exercise the calling skill in simulation (`solarian run-robot --sim`).
