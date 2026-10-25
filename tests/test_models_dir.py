from pathlib import Path

from solarian.utils.models import models_dir


def test_the_env_root_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLARIAN_MODELS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(tmp_path / "conf"))
    (tmp_path / "conf").mkdir()
    assert models_dir("sam2") == tmp_path / "store" / "sam2"


def test_a_project_keeps_its_models_next_to_its_conf(tmp_path, monkeypatch):
    monkeypatch.delenv("SOLARIAN_MODELS_DIR", raising=False)
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(tmp_path / "conf"))
    (tmp_path / "conf").mkdir()
    assert models_dir("depth_anything") == tmp_path / "models" / "depth_anything"


def test_outside_a_project_it_is_the_user_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("SOLARIAN_MODELS_DIR", raising=False)
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(tmp_path / "no_conf"))
    assert models_dir() == Path("~/.cache/solarian/models").expanduser()
