"""``solarian doctor``: the checks it runs on a project or plugin."""

from __future__ import annotations

import pytest

from solarian.cli import agent_files
from solarian.cli.doctor import ERROR, OK, WARN, _dependencies, run_checks
from solarian.cli.plugin_scaffolder import scaffold_plugin
from solarian.cli.scaffolder import scaffold_project


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_files, "installed_plugins", lambda: [])
    return scaffold_project("newbot", target_dir=tmp_path)


def statuses(root, status):
    return [check.message for check in run_checks(root)[1] if check.status == status]


def test_a_new_project_has_no_errors(project):
    kind, checks = run_checks(project)
    assert kind == "project"
    assert not [c for c in checks if c.status == ERROR]
    assert any("settings.py names" in c.message for c in checks if c.status == OK)
    assert any("AGENTS.md and the skills match" in c.message for c in checks if c.status == OK)


def test_a_module_without_its_yaml_is_an_error(project):
    (project / "conf/modules/arm.yaml").unlink()
    [message] = statuses(project, ERROR)
    assert "`arm` module has no conf/modules/arm.yaml" in message


def test_settings_naming_a_missing_class_is_an_error(project):
    settings = project / "newbot/settings.py"
    settings.write_text(settings.read_text().replace("NewbotServer", "GoneServer"))
    assert any("defines no class GoneServer" in m for m in statuses(project, ERROR))


def test_a_deployment_profile_without_a_key_conf_reads_is_a_warning(project):
    robot = project / "conf/robot.yaml"
    robot.write_text(robot.read_text() + "\nlidar_port: ${deployment.lidar_port}\n")
    assert any("Deployment `localhost` lacks lidar_port (read by conf/robot.yaml)" in m
               for m in statuses(project, WARN))


def test_an_old_or_edited_agents_md_is_a_warning(project):
    agents_md = project / "AGENTS.md"
    text = agents_md.read_text()
    agents_md.write_text(text.replace(f"solarian {agent_files.solarian_version()};", "solarian 0.0.1;", 1))
    warnings = statuses(project, WARN)
    assert any("written by Solarian 0.0.1" in m for m in warnings)
    assert any("were edited" in m and "AGENTS.md" in m for m in warnings)


def test_a_plugin_is_checked_for_its_agent_files(tmp_path):
    root, _, _ = scaffold_plugin("module", "servo", target_dir=tmp_path)
    kind, checks = run_checks(root)
    assert kind == "plugin"
    assert any("not installed" in c.message for c in checks if c.status == WARN)
    assert not [c for c in checks if c.status == ERROR]


def test_outside_a_project_there_is_one_error(tmp_path):
    kind, [check] = run_checks(tmp_path)
    assert kind is None and check.status == ERROR


def test_dependencies_are_read_from_the_project_table():
    pyproject = '[project]\ndependencies = [\n  "solarian[sim]>=0.5",  # core\n  "solarian-feetech>=0.1",\n]\n' \
                '[project.optional-dependencies]\nhw = ["solarian-robstride"]\n'
    assert _dependencies(pyproject) == ["solarian", "solarian-feetech"]
    assert _dependencies('dependencies = ["a", "b[x]"]\nother = ["c"]\n') == ["a", "b"]
