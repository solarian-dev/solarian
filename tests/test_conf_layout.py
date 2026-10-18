"""conf/ is organized by consumer, and --conf names a config in any folder of it."""

import sys

import pytest
from omegaconf import OmegaConf

from solarian.cli import manage
from solarian.cli.base import BaseCommand, CommandRegistry
from solarian.core.conf_paths import (
    cameras_dir,
    conf_dir,
    module_config_path,
    service_config_path,
    skill_config_path,
)


@pytest.fixture
def conf(tmp_path, monkeypatch):
    conf = tmp_path / "conf"
    conf.mkdir()
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(conf))
    return conf


def test_conf_dir_falls_back_to_the_project(tmp_path, monkeypatch):
    monkeypatch.delenv("SOLARIAN_CONF_PATH", raising=False)
    monkeypatch.setenv("SOLARIAN_PROJECT_PATH", str(tmp_path))

    assert conf_dir() == tmp_path / "conf"


def test_each_kind_of_config_has_its_folder(conf):
    assert module_config_path("right_arm") == conf / "modules" / "right_arm.yaml"
    assert service_config_path("bbox_tracker") == conf / "services" / "bbox_tracker.yaml"
    assert cameras_dir() == conf / "calibrations" / "cameras"


@pytest.mark.parametrize("given", ["umi", "umi.yaml"])
def test_a_skill_name_or_filename_is_in_the_skills_folder(conf, given):
    assert skill_config_path(given) == conf / "skills" / "umi.yaml"


def test_a_skill_path_with_a_directory_is_the_callers(conf, tmp_path):
    assert skill_config_path("runs/umi.yaml").as_posix() == "runs/umi.yaml"
    assert skill_config_path(str(tmp_path / "umi.yaml")) == tmp_path / "umi.yaml"


@pytest.mark.parametrize("lookup, flat, folder", [
    (lambda: module_config_path("right_arm"), "right_arm.yaml", "modules"),
    (lambda: skill_config_path("umi"), "umi.yaml", "skills"),
    (lambda: service_config_path("bbox_tracker"), "bbox_tracker.yaml", "services"),
])
def test_a_config_left_at_the_root_says_where_to_move_it(conf, lookup, flat, folder):
    (conf / flat).write_text("a: 1\n")

    with pytest.raises(FileNotFoundError, match=f"Move {conf / flat} to {conf / folder / flat}"):
        lookup()

    (conf / folder).mkdir()
    (conf / folder / flat).write_text("a: 1\n")
    assert lookup() == conf / folder / flat


# ---- --conf ------------------------------------------------------------------

class _ConfProbe(BaseCommand):
    help = "Record the config solarian composed."
    description = "Test command."
    examples = ("solarian conf-probe",)
    setup_modules = ()
    resolve_config = False
    composed = None

    def get_config_class(self):
        return None

    def handle(self, cfg) -> None:
        _ConfProbe.composed = OmegaConf.to_container(cfg)


@pytest.fixture
def project(tmp_path, monkeypatch):
    CommandRegistry.register("conf-probe")(_ConfProbe)
    _ConfProbe.composed = None
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "__init__.py").write_text("")
    (tmp_path / "proj" / "settings.py").write_text("")
    conf = tmp_path / "conf"
    (conf / "commands").mkdir(parents=True)
    (conf / "training").mkdir()
    (conf / "controller.yaml").write_text("fps: 30\nui: web\n")
    (conf / "commands" / "record.yaml").write_text(
        "defaults:\n  - /controller@_global_\n  - _self_\nnum_episodes: 3\n")
    (conf / "training" / "pick.yaml").write_text("trainer: umi\njob_name: pick\ntask:\n  name: pick\n")
    monkeypatch.chdir(tmp_path)
    for name in ("SOLARIAN_CONF_PATH", "SOLARIAN_PROJECT_PATH", "SOLARIAN_SETTINGS",
                 "SOLARIAN_CURRENT_CONFIG", "SOLARIAN_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield conf
    CommandRegistry._commands.pop("conf-probe")
    CommandRegistry._config_names.pop("conf-probe")


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["solarian", "conf-probe", *argv])
    manage.main()
    return _ConfProbe.composed


def test_a_config_in_a_folder_keeps_its_keys_at_the_top_level(project, monkeypatch):
    cfg = _run(monkeypatch, "--conf", "conf/training/pick", "job_name=pick_v2")

    assert cfg == {"trainer": "umi", "job_name": "pick_v2", "task": {"name": "pick"}}


def test_a_command_config_merges_the_controller_from_the_conf_root(project, monkeypatch):
    cfg = _run(monkeypatch, "--conf", "conf/commands/record.yaml", "num_episodes=5")

    assert cfg == {"fps": 30, "ui": "web", "num_episodes": 5}
