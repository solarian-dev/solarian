"""One server/client framework: auth, timing, the policy API, and what run-service runs."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from solarian.services import PolicyClient, PolicyServer, Service, ServiceClient, ServiceServer
from solarian.services import launch, service as service_mod
from solarian.services.server import SERVER_MS_HEADER
from solarian.utils import configs


class _Echo(ServiceServer):
    def load(self):
        pass

    @property
    def info(self):
        return {"kind": "echo"}

    def add_routes(self, app):
        @app.post("/echo")
        def echo(body: dict):
            return body


class _Doubler(PolicyServer):
    def load(self):
        self.resets = 0

    def predict(self, obs_dict, **kwargs):
        return {"action": obs_dict["x"] * 2, "info": {"scale": kwargs.get("scale")}}

    def reset(self):
        self.resets += 1

    @property
    def model_info(self):
        return {"model_name": "doubler"}

    @property
    def shape_meta(self):
        return {"action": {"shape": [3]}}


def _loaded(server):
    server.load()
    server.loaded = True
    return server


@pytest.fixture
def serve():
    """Serve an app on a free local port for the test; yields ``serve(app) -> url``."""
    import socket
    import threading
    import time

    import uvicorn

    servers = []

    def start(app) -> str:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        deadline = time.monotonic() + 5.0
        while not server.started:
            assert time.monotonic() < deadline, "server never started"
            time.sleep(0.01)
        servers.append(server)
        return f"http://127.0.0.1:{port}"

    yield start
    for server in servers:
        server.should_exit = True


def test_a_token_is_required_on_every_route_once_set():
    app = _loaded(_Echo()).build_app(token="s3cret")
    http = TestClient(app)

    assert http.get("/health").status_code == 401
    assert http.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert http.get("/health", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_the_token_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("SOLARIAN_ENDPOINT_TOKEN", "from-env")
    http = TestClient(_loaded(_Echo()).build_app())
    assert http.get("/info").status_code == 401

    monkeypatch.delenv("SOLARIAN_ENDPOINT_TOKEN")
    assert TestClient(_loaded(_Echo()).build_app()).get("/info").json() == {"kind": "echo"}


def test_a_client_with_the_token_gets_through_and_times_the_call(serve):
    app = _loaded(_Echo()).build_app(token="s3cret")
    client = ServiceClient(serve(app), token="s3cret")

    assert client.post("/echo", {"a": 1}) == {"a": 1}
    assert client.last_timing["server_ms"] > 0.0
    assert client.last_timing["payload_mb"] > 0.0
    assert SERVER_MS_HEADER in TestClient(app).get("/", headers={"Authorization": "Bearer s3cret"}).headers


def test_a_policy_client_drives_any_policy_server(serve):
    server = _loaded(_Doubler())
    client = PolicyClient(serve(server.build_app(token="")))

    assert client.get_model_info() == {"model_name": "doubler"}
    assert client.get_shape_meta() == {"action": {"shape": [3]}}
    action, raw = client.predict_with_raw({"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
    assert action.tolist() == [2.0, 4.0, 6.0]
    assert raw.tolist() == action.tolist(), "raw_action defaults to action"
    full = client.predict_full({"x": np.ones(3)}, scale=0.5)
    assert full["info"] == {"scale": 0.5}
    client.reset()
    assert server.resets == 1
    assert client.info() == {"model_name": "doubler"}, "a policy's /info is its model_info"


def test_a_policy_that_returns_no_action_is_a_500():
    class _Broken(_Doubler):
        def predict(self, obs_dict, **kwargs):
            return {"nope": 1}

    http = TestClient(_loaded(_Broken()).build_app(token=""))
    reply = http.post("/predict", json={"obs_dict": {"x": [1.0]}})
    assert reply.status_code == 500
    assert "action" in reply.json()["detail"]


# -- what run-service runs ---------------------------------------------------


class _Tracker(Service):
    server_module = "tracker.server"
    default_port = 8765


class _Policy(Service):
    server_module = "policy.server"
    default_port = 8001
    checkpoint_arg = "--input"


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A conf/ tree, two installed services, and an empty deployment cache."""
    for folder in ("services", "skills"):
        (tmp_path / folder).mkdir()
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(tmp_path))
    monkeypatch.delenv("SOLARIAN_DEPLOYMENT", raising=False)
    monkeypatch.setattr(configs, "_deployment_cache", {})
    monkeypatch.setattr(
        service_mod, "discover_services", lambda: {"tracker": _Tracker, "policy": _Policy}
    )
    return tmp_path


def test_a_service_conf_runs_its_server_block(project):
    (project / "services" / "bbox.yaml").write_text(
        "server_url: x\nserver:\n  service: tracker\n  args: [--model, small]\n"
    )

    spec = launch.resolve_server("bbox")

    assert spec.command() == [
        sys.executable, "-m", "tracker.server", "--host", "0.0.0.0", "--port", "8765",
        "--model", "small",
    ]


def test_a_skill_conf_serves_its_checkpoint_from_the_deployment(project):
    (project / "deployment").mkdir()
    (project / "deployment" / "localhost.yaml").write_text("pick_ckpt: /models/pick.ckpt\n")
    (project / "skills" / "pick.yaml").write_text(
        "server:\n  service: policy\n  checkpoint: ${deployment.pick_ckpt}\n  port: 8003\n"
        "frequency: ${deployment.not_on_this_host}\n"
    )

    spec = launch.resolve_server("pick")

    assert spec.checkpoint == "/models/pick.ckpt"
    assert spec.command()[-4:] == ["--port", "8003", "--input", "/models/pick.ckpt"]


def test_a_checkpoint_argument_replaces_a_key_this_host_lacks(project):
    (project / "skills" / "pick.yaml").write_text(
        "server:\n  service: policy\n  checkpoint: ${deployment.pick_ckpt}\n"
    )

    assert launch.resolve_server("pick", checkpoint="/data/run/last.ckpt").checkpoint == "/data/run/last.ckpt"


def test_a_policy_without_a_checkpoint_is_refused(project):
    (project / "skills" / "pick.yaml").write_text(
        "server:\n  service: policy\n  checkpoint: ${oc.select:deployment.pick_ckpt,null}\n"
    )
    with pytest.raises(ValueError, match="needs a checkpoint"):
        launch.resolve_server("pick")


def test_a_bare_service_runs_with_its_defaults(project):
    spec = launch.resolve_server("policy", checkpoint="a.ckpt", port=9000)
    assert (spec.module, spec.port, spec.checkpoint) == ("policy.server", 9000, "a.ckpt")


def test_a_name_both_a_service_and_a_skill_conf_is_ambiguous(project):
    for folder in ("services", "skills"):
        (project / folder / "both.yaml").write_text("server:\n  service: tracker\n")
    with pytest.raises(ValueError, match="ambiguous"):
        launch.resolve_server("both")


def test_a_conf_without_a_server_block_is_not_servable(project):
    (project / "services" / "plain.yaml").write_text("server_url: x\n")
    with pytest.raises(ValueError, match="neither a conf"):
        launch.resolve_server("plain")


def test_a_skill_supplies_the_service_its_conf_leaves_out(project):
    spec = launch.server_spec({"checkpoint": "c.ckpt", "managed": True}, default_service="policy")
    assert (spec.service, spec.managed) == ("policy", True)
