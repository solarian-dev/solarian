"""solarian install: where a plugin comes from, and how its config is seeded."""

from pathlib import Path

import pytest

from solarian.cli.commands.install import (
    InstallCommand,
    InstallError,
    add_dependency,
    config_sources,
    resolve_source,
    seed_configs,
)


def _package(root: Path, name: str = "solarian-widget") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    return root


def test_an_explicit_path_does_not_ask_pypi(tmp_path):
    plugin = _package(tmp_path / "plugins" / "solarian-move-skill")

    def boom(_name):
        raise AssertionError("PyPI should not be consulted for a path")

    package, local = resolve_source("plugins/solarian-move-skill", tmp_path, boom)

    assert package == "solarian-move-skill"
    assert local == plugin.resolve()


def test_a_missing_path_fails(tmp_path):
    with pytest.raises(InstallError, match="does not exist"):
        resolve_source("plugins/missing", tmp_path, lambda _name: True)


def test_a_path_that_is_not_a_package_fails(tmp_path):
    (tmp_path / "bare").mkdir()
    with pytest.raises(InstallError, match="not a plugin directory"):
        resolve_source("./bare", tmp_path, lambda _name: True)


def test_pypi_wins_over_a_local_directory(tmp_path):
    _package(tmp_path / "solarian-ur5")

    package, local = resolve_source("ur5", tmp_path, lambda name: name == "solarian-ur5")

    assert package == "solarian-ur5"
    assert local is None


def test_a_name_missing_from_pypi_uses_the_current_folder_then_plugins(tmp_path):
    _package(tmp_path / "plugins" / "solarian-move-skill")
    package, local = resolve_source(
        "solarian-move-skill", tmp_path, lambda _name: False
    )
    assert local == (tmp_path / "plugins" / "solarian-move-skill").resolve()
    assert package == "solarian-move-skill"

    nearer = _package(tmp_path / "solarian-move-skill")
    package, local = resolve_source("move-skill", tmp_path, lambda _name: False)
    assert package == "solarian-move-skill"
    assert local == nearer.resolve()


def test_a_name_found_nowhere_fails(tmp_path):
    with pytest.raises(InstallError, match="not on PyPI"):
        resolve_source("solarian-missing", tmp_path, lambda _name: False)


def test_an_unreachable_pypi_still_uses_a_local_package(tmp_path):
    plugin = _package(tmp_path / "plugins" / "solarian-move-skill")
    package, local = resolve_source(
        "solarian-move-skill", tmp_path, lambda _name: None
    )
    assert package == "solarian-move-skill"
    assert local == plugin.resolve()


def test_an_unreachable_pypi_with_no_local_package_fails(tmp_path):
    with pytest.raises(InstallError, match="Could not reach PyPI"):
        resolve_source("solarian-missing", tmp_path, lambda _name: None)


def test_templates_are_copied_once(tmp_path):
    plugin = tmp_path / "solarian-move-skill"
    templates = plugin / "solarian_move_skill" / "templates"
    templates.mkdir(parents=True)
    (templates / "move.yaml").write_text("poses: {home: {}}\n", encoding="utf-8")
    (plugin / "solarian_move_skill" / "conf.yaml").write_text("ignored: true\n", encoding="utf-8")
    _package(plugin)

    sources = config_sources("solarian-move-skill", plugin)
    assert [name for _src, name in sources] == ["move.yaml"]

    conf = tmp_path / "conf"
    written = seed_configs(sources, conf)
    assert written == [conf / "move.yaml"]
    (conf / "move.yaml").write_text("poses: {edited: {}}\n", encoding="utf-8")
    assert seed_configs(sources, conf) == []
    assert "edited" in (conf / "move.yaml").read_text(encoding="utf-8")


def test_a_legacy_conf_yaml_is_named_after_the_plugin(tmp_path):
    plugin = tmp_path / "solarian-ur5"
    pkg = plugin / "solarian_ur5"
    pkg.mkdir(parents=True)
    (pkg / "conf.yaml").write_text("port: /dev/ttyUSB0\n", encoding="utf-8")
    _package(plugin)

    sources = config_sources("solarian-ur5", plugin)
    assert sources == [(pkg / "conf.yaml", "ur5.yaml")]


def test_an_installed_package_is_found_by_import(tmp_path, monkeypatch):
    pkg = tmp_path / "solarian_widget"
    (pkg / "templates").mkdir(parents=True)
    (pkg / "templates" / "widget.yaml").write_text("a: 1\n", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib
    importlib.invalidate_caches()

    sources = config_sources("solarian-widget", None)
    assert sources == [(pkg / "templates" / "widget.yaml", "widget.yaml")]


def test_dependency_is_inserted_once(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\nname = 'demo'\ndependencies = [\n    \"solarian>=0.1.0\",\n]\n",
        encoding="utf-8",
    )
    assert add_dependency(pyproject, "solarian-move-skill") == "added"
    assert add_dependency(pyproject, "solarian-move-skill") == "present"
    text = pyproject.read_text(encoding="utf-8")
    assert text.count("solarian-move-skill") == 1
    assert '"solarian-move-skill>=0.1.0"' in text


def test_install_from_a_local_path_seeds_config_and_the_dependency(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "solarian-move-skill"
    templates = plugin / "solarian_move_skill" / "templates"
    templates.mkdir(parents=True)
    (templates / "move.yaml").write_text("frequency: 10\n", encoding="utf-8")
    _package(plugin)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\ndependencies = [\n]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOLARIAN_INSTALL_TARGET", "plugins/solarian-move-skill")
    monkeypatch.setenv("SOLARIAN_PROJECT_PATH", str(tmp_path))

    def reject_pypi(_name):
        raise AssertionError("PyPI should not be consulted for a path")

    monkeypatch.setattr(
        "solarian.cli.commands.install.pypi_has_project", reject_pypi
    )
    installed = []
    monkeypatch.setattr(
        "solarian.cli.commands.install.pip_install",
        lambda package, local_path: installed.append((package, local_path)),
    )

    InstallCommand().handle(None)

    assert installed == [("solarian-move-skill", plugin.resolve())]
    assert (tmp_path / "conf" / "move.yaml").read_text(encoding="utf-8") == "frequency: 10\n"
    assert "solarian-move-skill>=0.1.0" in (tmp_path / "pyproject.toml").read_text(
        encoding="utf-8"
    )
