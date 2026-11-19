"""The agent files ``create-project``, ``create-plugin`` and ``solarian agent`` write."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from solarian.cli import agent_files
from solarian.cli.agent_files import (
    MANIFEST,
    NOTES,
    SKILL_DIRS,
    TEMPLATES,
    detect_kind,
    frontmatter,
    install_bootstrap_skill,
    installed_plugins,
    stamp_version,
    sync_agent_files,
)
from solarian.cli.plugin_scaffolder import scaffold_plugin
from solarian.cli.scaffolder import scaffold_project

PROJECT_SKILLS = sorted(p.name for p in (TEMPLATES / "skills" / "project").iterdir() if p.is_dir())


@pytest.fixture
def no_plugins(monkeypatch):
    monkeypatch.setattr(agent_files, "installed_plugins", lambda: [])


@pytest.fixture
def project(tmp_path, no_plugins):
    return scaffold_project("newbot", target_dir=tmp_path)


def test_a_new_project_gets_agents_md_claude_md_skills_and_notes(project):
    agents_md = (project / "AGENTS.md").read_text()
    assert stamp_version(agents_md) == agent_files.solarian_version()
    assert "# newbot: guide for coding agents" in agents_md
    assert "newbot/robot.py" in agents_md
    assert (project / "CLAUDE.md").read_text().endswith("\n@AGENTS.md\n")
    for skill_dir in SKILL_DIRS:
        assert sorted(p.name for p in (project / skill_dir).iterdir()) == PROJECT_SKILLS
    assert (project / NOTES).is_file()
    recorded = json.loads((project / MANIFEST).read_text())["files"]
    assert "AGENTS.md" in recorded and NOTES not in recorded


def test_every_skill_is_named_after_its_folder_and_rendered(project):
    for skill_dir in SKILL_DIRS:
        for skill in (project / skill_dir).iterdir():
            text = (skill / "SKILL.md").read_text()
            fields = frontmatter(text)
            assert fields["name"] == skill.name
            assert len(fields["description"]) > 40
            assert "{{" not in text and "{%" not in text
    assert "newbot/modules/" in (project / ".agents/skills/add-module/SKILL.md").read_text()


def test_a_second_sync_changes_nothing(project):
    report = sync_agent_files(project, "project")
    assert not (report.written or report.removed or report.conflicts or report.notes_created)
    assert report.unchanged > 0


def test_an_edited_file_is_left_alone_until_forced(project):
    agents_md = project / "AGENTS.md"
    agents_md.write_text(agents_md.read_text() + "my rule\n")

    for _ in range(2):
        report = sync_agent_files(project, "project")
        assert report.conflicts == ["AGENTS.md"]
        assert agents_md.read_text().endswith("my rule\n")

    report = sync_agent_files(project, "project", force=True)
    assert report.written == ["AGENTS.md"]
    assert not agents_md.read_text().endswith("my rule\n")


def test_an_agents_md_solarian_did_not_write_is_kept(tmp_path, no_plugins):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "handmade"\n')
    (tmp_path / "AGENTS.md").write_text("# mine\n")
    report = sync_agent_files(tmp_path, "project")
    assert "AGENTS.md" in report.conflicts
    assert (tmp_path / "AGENTS.md").read_text() == "# mine\n"


def test_notes_are_never_rewritten(project):
    notes = project / NOTES
    notes.write_text("Use the left arm only.\n")
    sync_agent_files(project, "project", force=True)
    assert notes.read_text() == "Use the left arm only.\n"


def test_a_skill_no_longer_shipped_is_removed_unless_edited(project):
    manifest_path = project / MANIFEST
    manifest = json.loads(manifest_path.read_text())
    for name in ("gone", "kept"):
        path = project / ".agents/skills" / name / "SKILL.md"
        path.parent.mkdir()
        path.write_text(f"---\nname: {name}\n---\n")
        manifest["files"][path.relative_to(project).as_posix()] = agent_files._sha(path.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    (project / ".agents/skills/kept/SKILL.md").write_text("edited\n")

    report = sync_agent_files(project, "project")

    assert report.removed == [".agents/skills/gone/SKILL.md"]
    assert not (project / ".agents/skills/gone").exists()
    assert report.conflicts == [".agents/skills/kept/SKILL.md"]
    assert (project / ".agents/skills/kept/SKILL.md").read_text() == "edited\n"


def test_plugins_add_their_section_and_namespaced_skills(project, monkeypatch, tmp_path):
    skill = tmp_path / "pkg" / "agent" / "skills" / "calibrate"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: calibrate\ndescription: Calibrate a servo.\n---\n\nSteps.\n")
    plugin = agent_files.PluginAgent(
        dist="solarian-servo", short="servo", summary="Servo arm", groups=("modules",),
        section="### solarian-servo\n\n- Provides: `ServoArm`.", skills=(skill,),
    )
    monkeypatch.setattr(agent_files, "installed_plugins", lambda: [plugin])

    sync_agent_files(project, "project")

    agents_md = (project / "AGENTS.md").read_text()
    assert "- `solarian-servo` (modules): Servo arm" in agents_md
    assert "- Provides: `ServoArm`." in agents_md
    assert "- `servo-calibrate`: Calibrate a servo." in agents_md
    for skill_dir in SKILL_DIRS:
        text = (project / skill_dir / "servo-calibrate" / "SKILL.md").read_text()
        assert frontmatter(text)["name"] == "servo-calibrate"


def test_an_installed_plugin_is_read_from_its_agent_folder(tmp_path, monkeypatch):
    site = tmp_path / "site"
    package = site / "fakeplug"
    (package / "agent" / "skills" / "use-it").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "agent" / "AGENTS.md").write_text("### solarian-fakeplug\n<!-- a hint -->\n- Provides: `Fake`.\n")
    (package / "agent" / "skills" / "use-it" / "SKILL.md").write_text("---\nname: use-it\ndescription: d\n---\n")
    info = site / "solarian_fakeplug-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: solarian-fakeplug\nVersion: 0.1\nSummary: A fake\n")
    (info / "entry_points.txt").write_text("[solarian.modules]\nfake = fakeplug:Fake\n")
    monkeypatch.syspath_prepend(str(site))

    plugin = next(p for p in installed_plugins() if p.dist == "solarian-fakeplug")

    assert (plugin.short, plugin.summary, plugin.groups) == ("fakeplug", "A fake", ("modules",))
    assert plugin.section == "### solarian-fakeplug\n- Provides: `Fake`."
    assert [p.name for p in plugin.skills] == ["use-it"]


def test_a_new_plugin_gets_its_guide_and_a_contribution_stub(tmp_path):
    root, _, written = scaffold_plugin("module", "servo", target_dir=tmp_path)

    assert detect_kind(root) == "plugin"
    agents_md = (root / "AGENTS.md").read_text()
    assert "`solarian.modules`" in agents_md
    assert "solarian create-plugin <type> servo" in agents_md
    assert (root / ".agents/skills/plugin-dev/SKILL.md").is_file()
    assert (root / "solarian_servo/agent/AGENTS.md").read_text().startswith("### solarian-servo")
    assert '"agent/*.md"' in (root / "pyproject.toml").read_text()
    assert {"AGENTS.md", "CLAUDE.md", "skill plugin-dev"} <= set(written)


def test_adding_a_component_refreshes_the_plugin_guide(tmp_path):
    root, _, _ = scaffold_plugin("actuator", "servo", target_dir=tmp_path)
    assert "`solarian.modules`" not in (root / "AGENTS.md").read_text()
    scaffold_plugin("module", "servo", target_dir=tmp_path)
    assert "`solarian.modules`" in (root / "AGENTS.md").read_text()


def test_detect_kind(tmp_path, project):
    assert detect_kind(tmp_path / "nothing") is None
    assert detect_kind(project) == "project"


@pytest.fixture
def user_dirs(tmp_path, monkeypatch):
    dirs = {name: tmp_path / name / "skills" for name in ("agents", "claude", "cursor")}
    monkeypatch.setattr(agent_files, "USER_SKILL_DIRS", dirs)
    return dirs


def test_the_bootstrap_skill_installs_once_and_reports_unchanged(user_dirs):
    [(dest, outcome)] = install_bootstrap_skill(["agents"])
    assert (dest, outcome) == (user_dirs["agents"] / "solarian", "installed")
    text = (dest / "SKILL.md").read_text()
    assert frontmatter(text)["name"] == "solarian"
    assert stamp_version(text) == agent_files.solarian_version()
    assert install_bootstrap_skill(["agents"]) == [(dest, "unchanged")]


def test_the_marketplace_plugin_matches_the_templates():
    sync = Path(__file__).resolve().parents[1] / "integrations" / "sync_agent_plugin.py"
    if not sync.is_file():
        pytest.skip("not a source checkout")
    spec = importlib.util.spec_from_file_location("agent_plugin_sync", sync)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.stale() == [], "run python integrations/sync_agent_plugin.py"


def test_the_marketplace_plugin_holds_no_scripts():
    # Anthropic's directory blocks scripts in a plugin from an unverified organization.
    plugin = Path(__file__).resolve().parents[1] / "integrations" / "agent-plugin"
    if not plugin.is_dir():
        pytest.skip("not a source checkout")
    files = [p for p in plugin.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert [p for p in files if p.suffix not in {".md", ".json", ".png"}] == []


def test_the_bootstrap_skill_replaces_only_its_own_copy(user_dirs):
    dest = user_dirs["cursor"] / "solarian"
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text("---\nname: solarian\n---\nmine\n")
    assert install_bootstrap_skill(["cursor"]) == [(dest, "skipped")]
    assert install_bootstrap_skill(["cursor"], force=True) == [(dest, "updated")]

    (dest / "SKILL.md").write_text("<!-- installed by solarian 0.0.1; old -->\n")
    assert install_bootstrap_skill(["cursor"]) == [(dest, "updated")]
