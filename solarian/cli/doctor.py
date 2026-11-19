"""The checks behind ``solarian doctor``: what is wrong with a project or plugin before it runs.

Nothing here imports the project, so a broken project still gets a report.
"""

from __future__ import annotations

import ast
import importlib.metadata
import re
from dataclasses import dataclass
from pathlib import Path

from solarian.cli import agent_files
from solarian.cli.agent_files import NOTES, USER_SKILL_DIRS, solarian_version, stamp_version

OK, WARN, ERROR = "ok", "warn", "error"

SETTINGS_KEYS = ("ROBOT_CLASS", "CONTROLLER_CLASS", "SERVER_CLASS")
DEPLOYMENT_REF = re.compile(r"\$\{deployment\.([A-Za-z0-9_.]+)\}")


@dataclass(frozen=True)
class Check:
    status: str
    message: str
    fix: str | None = None


def run_checks(root: Path) -> tuple[str | None, list[Check]]:
    """``(kind, checks)`` for the project or plugin at *root*; kind is None without a pyproject."""
    kind = agent_files.detect_kind(root)
    if kind is None:
        return None, [Check(ERROR, f"No pyproject.toml in {root}.",
                            "Run doctor in a Solarian project or plugin, or create one: solarian create-project <name>")]
    checks = _installed(root, kind)
    if kind == "project":
        checks += _settings_and_modules(root)
        checks += _deployments(root)
    checks += _agent_files(root, kind)
    checks += _bootstrap_skill()
    return kind, checks


# ── Installed ─────────────────────────────────────────────────────────


def _dependencies(pyproject: str) -> list[str]:
    """Distribution names in ``[project] dependencies``."""
    lines = pyproject.split("\n")
    start = next((i for i, line in enumerate(lines) if re.match(r"dependencies\s*=\s*\[", line)), None)
    if start is None:
        return []
    names = []
    for line in lines[start:]:
        code = line.split("#")[0]
        names += [re.match(r"[A-Za-z0-9_.-]+", spec)[0] for spec in re.findall(r'"([^"]+)"', code)]
        if code.rstrip().endswith("]"):
            break
    return names


def _declared_entry_points(pyproject: str) -> set[tuple[str, str, str]]:
    """``(group, name, value)`` for every ``[project.entry-points."solarian.*"]`` line."""
    declared = set()
    group = None
    for line in pyproject.split("\n"):
        table = re.match(r'^\[project\.entry-points\."(solarian\.[a-z_]+)"\]', line)
        if table:
            group = table[1]
        elif line.startswith("["):
            group = None
        elif group:
            entry = re.match(r'^\s*"?([\w.-]+)"?\s*=\s*"([^"]+)"', line)
            if entry:
                declared.add((group, entry[1], entry[2]))
    return declared


def _installed(root: Path, kind: str) -> list[Check]:
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    name = agent_files.pyproject_name(root)
    reinstall = 'pip install -e ".[all]"' if kind == "project" else "pip install -e ."
    checks = []
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        checks.append(Check(WARN, f"{name} is not installed in this Python environment.", reinstall))
    else:
        if kind == "plugin":
            installed = {(ep.group, ep.name, ep.value) for ep in dist.entry_points if ep.group.startswith("solarian.")}
            if installed != _declared_entry_points(pyproject):
                checks.append(Check(WARN, "The installed entry points differ from pyproject.toml, so Solarian "
                                          "still sees the old ones.", reinstall))
            else:
                checks.append(Check(OK, f"{name} is installed with the entry points pyproject.toml declares."))
        else:
            checks.append(Check(OK, f"{name} is installed."))

    missing = []
    for dep in _dependencies(pyproject):
        canonical = re.sub(r"[-_.]+", "-", dep).lower()
        if canonical == "solarian" or not canonical.startswith("solarian-"):
            continue
        try:
            importlib.metadata.distribution(canonical)
        except importlib.metadata.PackageNotFoundError:
            missing.append(canonical)
    if missing:
        checks.append(Check(ERROR, f"pyproject.toml depends on {', '.join(missing)}, not installed here.", reinstall))
    return checks


# ── Project layout ────────────────────────────────────────────────────


def _robot_modules(robot_file: Path, class_name: str) -> list[str]:
    """Attributes of *class_name* assigned a ``...Module(...)``: the robot's modules, by name."""
    tree = ast.parse(robot_file.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                target, value = stmt.targets[0].id, stmt.value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                target, value = stmt.target.id, stmt.value
            else:
                continue
            if isinstance(value, ast.Call):
                func = value.func
                called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if called.endswith("Module"):
                    names.append(target)
    return names


def _settings_and_modules(root: Path) -> list[Check]:
    package = agent_files.pyproject_name(root).replace("-", "_")
    settings = root / package / "settings.py"
    if not settings.is_file():
        return [Check(ERROR, f"{package}/settings.py is missing.",
                      "It names ROBOT_CLASS, CONTROLLER_CLASS and SERVER_CLASS: solarian docs show settings")]
    source = settings.read_text(encoding="utf-8")
    checks, classes = [], {}
    for key in SETTINGS_KEYS:
        match = re.search(rf"^{key}\s*=\s*['\"]([\w.]+)['\"]", source, re.M)
        if not match:
            checks.append(Check(ERROR, f"{package}/settings.py does not set {key}.",
                                f"{key} = '<package>.<module>.<Class>'"))
            continue
        module, _, class_name = match[1].rpartition(".")
        path = root / Path(*module.split(".")).with_suffix(".py")
        defined = path.is_file() and any(
            isinstance(node, ast.ClassDef) and node.name == class_name
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
        if not defined:
            checks.append(Check(ERROR, f"{key} names {match[1]}, but {path.relative_to(root)} defines no "
                                       f"class {class_name}.", f"Fix {key} in {package}/settings.py."))
            continue
        classes[key] = (path, class_name)
    if len(classes) == len(SETTINGS_KEYS):
        checks.append(Check(OK, "settings.py names a robot, controller and server class that exist."))

    if "ROBOT_CLASS" in classes:
        modules = _robot_modules(*classes["ROBOT_CLASS"])
        conf = root / "conf"
        absent = [name for name in modules if not (conf / "modules" / f"{name}.yaml").is_file()]
        for name in absent:
            flat = conf / f"{name}.yaml"
            fix = (f"Move conf/{name}.yaml to conf/modules/{name}.yaml." if flat.is_file() else
                   f"solarian install <plugin> seeds it, or copy the plugin's default yaml to conf/modules/{name}.yaml.")
            checks.append(Check(ERROR, f"The robot's `{name}` module has no conf/modules/{name}.yaml, "
                                       "so the robot fails to start.", fix))
        if modules and not absent:
            checks.append(Check(OK, f"conf/modules/ has a yaml for each of the robot's {len(modules)} modules."))
    return checks


def _deployments(root: Path) -> list[Check]:
    from omegaconf import OmegaConf

    from solarian.utils.configs import DEFAULT_DEPLOYMENT, load_deployment_file

    conf = root / "conf"
    deployment_dir = conf / "deployment"
    refs: dict[str, str] = {}
    for path in sorted(conf.rglob("*.yaml")) if conf.is_dir() else []:
        if deployment_dir in path.parents:
            continue
        for key in DEPLOYMENT_REF.findall(path.read_text(encoding="utf-8")):
            refs.setdefault(key, path.relative_to(root).as_posix())

    profiles = sorted(p.stem for p in deployment_dir.glob("*.yaml")) if deployment_dir.is_dir() else []
    if not profiles:
        if refs:
            return [Check(ERROR, f"conf/ reads {len(refs)} ${{deployment.*}} keys but there is no conf/deployment/.",
                          "Create conf/deployment/localhost.yaml: solarian docs show project-configs")]
        return []

    checks = []
    if DEFAULT_DEPLOYMENT not in profiles:
        checks.append(Check(WARN, f"No conf/deployment/{DEFAULT_DEPLOYMENT}.yaml, the profile used without -d.",
                            "Create it, or pass -d <profile> to every command."))
    complete = 0
    for name in profiles:
        try:
            profile = load_deployment_file(str(deployment_dir), name)
        except Exception as exc:
            checks.append(Check(ERROR, f"conf/deployment/{name}.yaml does not load: {exc}"))
            continue
        absent = [key for key in refs if OmegaConf.select(profile, key, throw_on_missing=False) is None]
        if absent:
            used_in = sorted({refs[key] for key in absent})
            checks.append(Check(WARN, f"Deployment `{name}` lacks {', '.join(absent)} (read by {', '.join(used_in)}).",
                                f"Add them to conf/deployment/{name}.yaml if `-d {name}` runs those parts."))
        else:
            complete += 1
    if complete == len(profiles):
        checks.append(Check(OK, f"Every deployment profile ({len(profiles)}) sets the {len(refs)} keys conf/ reads."))
    return checks


# ── Agent files ───────────────────────────────────────────────────────


def _agent_files(root: Path, kind: str) -> list[Check]:
    installed = solarian_version()
    agents_md = root / "AGENTS.md"
    report = agent_files.sync_agent_files(root, kind, dry_run=True)
    conflicts = list(report.conflicts)
    checks = []
    if not agents_md.is_file():
        checks.append(Check(WARN, f"No AGENTS.md, so coding agents get no guide to this {kind}.",
                            "solarian agent sync"))
    else:
        written_by = stamp_version(agents_md.read_text(encoding="utf-8"))
        if written_by is None:
            conflicts.remove("AGENTS.md")
            checks.append(Check(WARN, "AGENTS.md was not written by Solarian, so agents miss the generated guide.",
                                f"Move its content into {NOTES}, then run solarian agent sync --force"))
        elif written_by != installed:
            checks.append(Check(WARN, f"AGENTS.md was written by Solarian {written_by}; {installed} is installed.",
                                "solarian agent sync"))
        elif report.written or report.removed:
            changes = ", ".join(agent_files.summarize(report.written + report.removed))
            checks.append(Check(WARN, f"The agent files are behind the installed plugins ({changes}).",
                                "solarian agent sync"))
        else:
            checks.append(Check(OK, f"AGENTS.md and the skills match Solarian {installed} and the installed plugins."))
    if conflicts:
        checks.append(Check(WARN, f"Generated files were edited, so sync leaves them alone: {', '.join(conflicts)}.",
                            f"Move the changes into {NOTES}, then run solarian agent sync --force"))
    return checks


def _bootstrap_skill() -> list[Check]:
    installed = solarian_version()
    checks = []
    for folder in USER_SKILL_DIRS.values():
        skill = folder / agent_files.BOOTSTRAP_SKILL / "SKILL.md"
        if not skill.is_file():
            continue
        written_by = stamp_version(skill.read_text(encoding="utf-8"))
        if written_by is not None and written_by != installed:
            checks.append(Check(WARN, f"The solarian skill in {folder} is from Solarian {written_by}.",
                                "solarian agent install"))
    return checks
