"""Ops a project defines in <package>/dataset_ops/, and the scaffolding that writes them."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from test_dataset_ops import make_session

from solarian.cli.commands.add_op import AddOpCommand
from solarian.cli.plugin_scaffolder import scaffold_plugin
from solarian.cli.scaffolder import scaffold_project
from solarian.core.recording.archive import Session
from solarian.datasets import run_op
from solarian.datasets.op import _project_commit, discover, find_project

OP = '''
from pydantic import BaseModel

from solarian.datasets import DatasetOp, OpKind
from {package}.helpers import LIMIT


class Params(BaseModel):
    pass


class {cls}(DatasetOp):
    {name}kind = OpKind.FILTER
    Params = Params

    def keep(self, episode, params):
        return episode.index < LIMIT
'''


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A project ``proj_<test>`` whose package holds a few op files, good and bad."""
    package = f"proj_{tmp_path.name.replace('-', '_')}"
    root = tmp_path / "project"
    (root / package / "dataset_ops").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "{package}"\n')
    (root / package / "__init__.py").write_text("")
    (root / package / "settings.py").write_text("")
    (root / package / "helpers.py").write_text("LIMIT = 2\n")
    ops = root / package / "dataset_ops"
    (ops / "first_two.py").write_text(OP.format(package=package, cls="FirstTwo", name=""))
    (ops / "mirror.py").write_text(OP.format(package=package, cls="NotAMirror", name=""))
    (ops / "broken.py").write_text("import not_a_module\n")
    (ops / "two.py").write_text(OP.format(package=package, cls="A", name="")
                                + "\n\nclass B(A):\n    name = 'b'\n")
    (ops / "_shared.py").write_text("raise RuntimeError('a helper module is not an op')\n")
    yield root
    for name in [m for m in sys.modules if m == package or m.startswith(f"{package}.")]:
        del sys.modules[name]
    monkeypatch.delenv("SOLARIAN_PROJECT_DIR", raising=False)


def _package(project: Path) -> str:
    return next(p.name for p in project.iterdir() if (p / "__init__.py").is_file())


def test_project_ops_load_by_file_name(project):
    catalog = discover(project)
    package = _package(project)
    op = catalog.ops["first-two"]
    assert op.source == f"project:{package}/dataset_ops/first_two.py"
    assert op.title == "First two"
    assert op.package == package
    assert op.commit is None
    assert catalog.ops["mirror"].overrides == "builtin"
    assert catalog.ops["mirror"].describe()["source"].endswith("mirror.py")
    errors = dict(catalog.errors)
    assert "ModuleNotFoundError" in errors[f"{package}/dataset_ops/broken.py"]
    assert "defines 2" in errors[f"{package}/dataset_ops/two.py"]
    assert not any("_shared" in where for where in errors)
    assert "delete-episodes" in catalog.ops


def test_a_project_op_runs_where_the_project_is(project, tmp_path, monkeypatch):
    session = make_session(tmp_path / "session", episodes=4)
    monkeypatch.chdir(project / _package(project))
    assert find_project() == project
    [summary] = run_op("first-two", [session], {"out": tmp_path / "kept"})
    assert summary.episodes == 2
    assert Session(tmp_path / "kept").lineage()[0]["op"]["source"].startswith("project:")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOLARIAN_PROJECT_DIR", str(project))
    assert find_project() == project


def test_the_commit_is_recorded_and_marked_dirty(project):
    def git(*args):
        subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)

    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    commit = discover(project).ops["first-two"].commit
    assert len(commit) == 40
    package = _package(project)
    (project / package / "dataset_ops" / "first_two.py").write_text(
        (project / package / "dataset_ops" / "first_two.py").read_text() + "\n# edited\n")
    _project_commit.cache_clear()
    assert discover(project).ops["first-two"].commit == f"{commit}-dirty"


def test_outside_a_project_there_are_no_project_ops(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOLARIAN_PROJECT_DIR", raising=False)
    catalog = discover()
    assert catalog.project is None
    assert all(not op.source.startswith("project:") for op in catalog.ops.values())


def test_add_op_writes_a_working_op(project, tmp_path, monkeypatch):
    monkeypatch.chdir(project)
    command = AddOpCommand()
    command.parse_args(["drop_short"])
    command.handle(None)
    package = _package(project)
    assert (project / package / "dataset_ops" / "drop_short.py").is_file()
    with pytest.raises(SystemExit, match="exists"):
        command.handle(None)

    session = make_session(tmp_path / "session", episodes=2)
    [summary] = run_op("drop-short", [session], {"out": tmp_path / "out"}, {"min_frames": 1000})
    assert summary.episodes == 0


def test_the_project_template_carries_an_example_op(tmp_path):
    root = scaffold_project("newbot", target_dir=tmp_path)
    example = root / "newbot" / "dataset_ops" / "drop_short_episodes.py"
    text = example.read_text()
    assert "class DropShortEpisodesOp(DatasetOp)" in text and "${" not in text
    compile(text, str(example), "exec")


def test_the_op_plugin_type_writes_the_stub_and_entry_point(tmp_path):
    root, created, written = scaffold_plugin("op", "blur-faces", target_dir=tmp_path)
    assert created and "solarian_blur_faces/op.py" in written
    pyproject = (root / "pyproject.toml").read_text()
    assert '[project.entry-points."solarian.dataset_ops"]' in pyproject
    assert 'blur-faces = "solarian_blur_faces.op:BlurFacesOp"' in pyproject
    text = (root / "solarian_blur_faces" / "op.py").read_text()
    assert "${" not in text
    compile(text, "op.py", "exec")
    assert textwrap.dedent(text).count("class BlurFacesOp(DatasetOp)") == 1
