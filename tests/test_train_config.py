"""A training file names its trainer, whose resolvers it may use."""

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

from solarian.core.ai.system1 import trainer_registry
from solarian.core.ai.system1.trainer_registry import load_train_config


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trainer: str
    job_name: str
    steps: int


class _Trainer:
    @classmethod
    def resolvers(cls):
        return {"times_ten": lambda x: int(x) * 10}

    @classmethod
    def config_class(cls):
        return _Config


@pytest.fixture
def trainer(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(tmp_path))
    monkeypatch.setattr(trainer_registry, "get_trainer_class",
                        lambda name: _Trainer if name == "probe" else pytest.fail(name))


def test_the_trainers_resolvers_resolve_its_file(trainer):
    cfg = OmegaConf.create({"trainer": "probe", "job_name": "run", "steps": "${times_ten:5}"})

    trainer_cls, cfg_dict, train_cfg = load_train_config(cfg)

    assert trainer_cls is _Trainer
    assert cfg_dict["steps"] == 50
    assert train_cfg == _Config(trainer="probe", job_name="run", steps=50)


def test_a_file_without_a_trainer_says_where_training_files_live(trainer):
    with pytest.raises(ValueError, match="conf/training/<name>"):
        load_train_config(OmegaConf.create({"job_name": "run"}))
