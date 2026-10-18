"""solarian.tracking: backend selection, fan-out, and the trainer adapters."""

import sys
import types

import pytest

from solarian import tracking
from solarian.tracking import registry
from solarian.tracking.tracker import Tracker


class Recorder(Tracker):
    name = "recorder"
    instances: list["Recorder"] = []

    def __init__(self, spec):
        super().__init__(spec)
        self.rows, self.summary, self.exit_code = [], {}, None
        Recorder.instances.append(self)

    def log(self, values, step):
        self.rows.append((step, values))

    def update_summary(self, values):
        self.summary.update(values)

    def finish(self, exit_code):
        self.exit_code = exit_code


class Broken(Tracker):
    name = "broken"

    def log(self, values, step):
        raise ConnectionError("down")


@pytest.fixture(autouse=True)
def fake_backends(monkeypatch):
    from solarian.tracking.backends.none import NoneTracker

    Recorder.instances = []
    monkeypatch.delenv("SOLARIAN_TRACKING", raising=False)
    monkeypatch.setattr(registry, "discover_trackers", lambda: {
        "recorder": Recorder, "broken": Broken, "none": NoneTracker, "wandb": Recorder,
        "solarian_base": Recorder,
    })


def test_clean_tags_drops_empty_truncates_and_dedupes():
    long = "x" * 80
    assert tracking.clean_tags(["a", "", None, "None", "a", long, " b "]) == ["a", "x" * 64, "b"]


def test_resolve_backends():
    assert registry.resolve_backends(None) == ["solarian_base"]
    assert registry.resolve_backends("none") == []
    assert registry.resolve_backends(["recorder", "wandb", "recorder"]) == ["recorder", "wandb"]
    assert registry.resolve_backends("recorder,broken") == ["recorder", "broken"]


def test_the_default_backend_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr(registry, "discover_trackers", lambda: {"wandb": Recorder})
    with pytest.raises(ValueError, match="pip install solarian-base"):
        registry.resolve_backends(None)


def test_env_overrides_the_config(monkeypatch):
    monkeypatch.setenv("SOLARIAN_TRACKING", "none")
    assert registry.resolve_backends("recorder") == []
    monkeypatch.setenv("SOLARIAN_TRACKING", "recorder")
    assert registry.resolve_backends("wandb") == ["recorder"]


def test_run_fans_out_and_survives_a_broken_backend(capsys):
    run = tracking.init(project="umi", name="r1", tags=["a", ""], config={"lr": 1e-4},
                        backend=["recorder", "broken"])
    run.log({"loss": 0.5}, step=1)
    run.log({"loss": 0.4}, step=2)
    run.summary["best"] = 0.4
    run.finish()
    (rec,) = Recorder.instances
    assert rec.spec.tags == ["a"]
    assert rec.spec.config == {"lr": 1e-4}
    assert rec.spec.system["hostname"]
    assert rec.rows == [(1, {"loss": 0.5}), (2, {"loss": 0.4})]
    assert rec.summary == {"best": 0.4}
    assert rec.exit_code == 0
    assert capsys.readouterr().err.count("broken.log failed") == 1


def test_module_level_log_uses_the_current_run():
    tracking.init(name="r2", backend="recorder")
    tracking.log({"x": 1}, step=0)
    tracking.finish(3)
    assert Recorder.instances[0].rows == [(0, {"x": 1})]
    assert Recorder.instances[0].exit_code == 3
    assert tracking.current() is None


def test_train_logger_goes_through_tracking():
    from solarian.analytics import LoggerFactory
    from solarian.analytics.train_logger import TrainDataLog
    from solarian.core.configs import TrainConfig, WandBLoggerConfig

    cfg = TrainConfig(trainer="act", train_dataset="d", job_name="job", output_dir="/tmp/x")
    logger = LoggerFactory.create_logger(WandBLoggerConfig(type="recorder", project="p"), train_cfg=cfg)
    data = TrainDataLog()
    data.update_metrics({"loss": 2.0, "step": 7})
    logger.log(data)
    (rec,) = Recorder.instances
    assert rec.spec.name == "job" and rec.spec.project == "p"
    assert "trainer:act" in rec.spec.tags
    assert rec.rows == [(7, {"loss": 2.0})]


def test_accelerate_tracker_starts_with_the_config(monkeypatch):
    try:
        import accelerate.tracking  # noqa: F401
    except ImportError:
        class GeneralTracker:
            def start(self):
                pass

        monkeypatch.setitem(sys.modules, "accelerate", types.ModuleType("accelerate"))
        monkeypatch.setitem(sys.modules, "accelerate.tracking",
                            types.SimpleNamespace(GeneralTracker=GeneralTracker))
        monkeypatch.delitem(sys.modules, "solarian.tracking.accelerate", raising=False)
    from solarian.tracking.accelerate import SolarianTracker, accelerate_trackers

    assert accelerate_trackers({"project": "umi", "backend": "none"}) == ([], {})
    (tracker,), kwargs = accelerate_trackers({
        "project": "umi", "name": "n", "tags": ["t", ""], "mode": "online", "backend": "recorder",
        "entity": "team",
    })
    assert isinstance(tracker, SolarianTracker) and kwargs == {}
    tracker.start()
    tracker.store_init_configuration({"batch_size": 128})
    tracker.log({"val_loss": 0.1}, step=5)
    tracker.finish()
    (rec,) = Recorder.instances
    assert rec.spec.project == "umi" and rec.spec.tags == ["t"] and rec.spec.mode == "online"
    assert rec.spec.config == {"batch_size": 128}
    assert rec.spec.extra == {"entity": "team"}
    assert rec.rows == [(5, {"val_loss": 0.1})] and rec.exit_code == 0
