"""Apple Silicon install fixes that pip and uv cannot express as version specifiers.

The ``atomics`` wheels for macOS load on Apple Silicon and then fail at
``dlopen``. A source build is the one that works. ``xformers`` publishes no
macOS wheel, and its source build passes ``-fopenmp``, which Apple clang
rejects. ``pip`` and ``uv`` call the build backend, which builds both; a wheel
built elsewhere still fixes itself on first import.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ATOMICS_VERSION = "1.0.3"
# The xformers release built for torch==2.8.0. Newer ones require a newer torch.
XFORMERS_VERSION = "0.0.32.post1"
TORCH_VERSION = "2.8.0"
_ready = False
_xformers_ready = False
_xformers_attempted = False


def apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _cache_dir() -> Path:
    # The dylib has to match this interpreter. A build env and the venv it
    # installs into share a cache when they are the same Python.
    impl = f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}"
    return Path.home() / ".cache" / "solarian" / f"atomics-{ATOMICS_VERSION}" / f"{impl}-arm64"


def _prefer_source_atomics() -> None:
    root = _cache_dir()
    if not (root / "atomics").is_dir():
        return
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _drop_cached_atomics() -> None:
    for name in list(sys.modules):
        if name == "atomics" or name.startswith("atomics."):
            del sys.modules[name]


def _atomics_works() -> bool:
    _prefer_source_atomics()
    try:
        from atomics import UINT, atomicview

        buf = ctypes.create_string_buffer(8)
        with atomicview(buffer=buf, atype=UINT):
            pass
        return True
    except Exception:
        _drop_cached_atomics()
        return False


def _install_command(target: Path) -> list[str]:
    """pip when this interpreter has it, otherwise uv. A uv venv may omit pip."""
    spec = importlib.util.find_spec("pip")
    if spec is not None:
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-binary=atomics",
            "--target",
            str(target),
            f"atomics=={ATOMICS_VERSION}",
        ]
    uv = os.environ.get("UV") or shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Cannot build atomics from source: this interpreter has no pip and uv is not on PATH."
        )
    return [
        uv,
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-binary",
        "atomics",
        "--target",
        str(target),
        f"atomics=={ATOMICS_VERSION}",
    ]


def _install_atomics_from_source() -> None:
    target = _cache_dir()
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    env = os.environ.copy()
    # A nested pip during a PEP 517 build deadlocks on the build tracker.
    env.pop("PIP_BUILD_TRACKER", None)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    subprocess.run(_install_command(target), check=True, env=env)


def ensure_atomics() -> None:
    """Build ``atomics`` from source on Apple Silicon when the wheel cannot load."""
    global _ready
    if _ready or not apple_silicon():
        return
    if _atomics_works():
        _ready = True
        return
    try:
        _install_atomics_from_source()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Apple Silicon needs atomics built from source; the published wheel "
            "fails at dlopen. The build needs git and a C compiler "
            "(Xcode Command Line Tools: xcode-select --install)."
        ) from exc
    if not _atomics_works():
        raise RuntimeError(
            "Built atomics from source on Apple Silicon, but importing it still failed."
        )
    _ready = True


def _in_uv_build_env() -> bool:
    # uv compiles the project in a throwaway env. Installing into it is discarded.
    return "uv/builds-" in sys.prefix


def _ancestors() -> list[tuple[str, str]]:
    """(command, environment) of each parent, stop at launchd.

    The environment is separate so a shell script that mentions ``[ai]``
    cannot mark a ``pip install .`` as an AI install.
    """
    rows = []
    pid = os.getpid()
    for _ in range(12):
        parent = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        ppid = parent.stdout.strip()
        if not ppid.isdigit() or int(ppid) <= 1 or int(ppid) == pid:
            break
        pid = int(ppid)
        command = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # `eww` appends the environment. uv's own process still has the venv
        # the user is installing into; the build subprocess has replaced its own.
        environ = subprocess.run(
            ["ps", "eww", "-o", "command=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
        if command:
            rows.append((command, environ))
    return rows


def _installer(ancestors: list[tuple[str, str]]) -> tuple[str, str]:
    """The pip or uv command, not a parent shell whose script mentions both."""
    for command, environ in ancestors:
        # The executable is often a path (`.../bin/pip install`), so pip is
        # preceded by a slash rather than whitespace.
        if re.search(r"(?:^|\s|/)(?:uv\s+pip|pip3?)\s+install\b", command):
            return command, environ
        if re.search(r"(?:^|\s|/)uv\s+build\b", command):
            return command, environ
    return "", ""


def _ai_extra_requested(ancestors: list[tuple[str, str]]) -> bool:
    command, _environ = _installer(ancestors)
    for group in re.findall(r"\[([^\]]+)\]", command):
        extras = {part.strip() for part in group.split(",")}
        if "ai" in extras or "all" in extras:
            return True
    return False


def _no_deps_requested(ancestors: list[tuple[str, str]]) -> bool:
    command, _environ = _installer(ancestors)
    return "--no-deps" in command


def _uv_target_python(ancestors: list[tuple[str, str]]) -> str | None:
    command, environ = _installer(ancestors)
    flagged = re.search(r"--python(?:=|\s+)(\S+)", command)
    if flagged and Path(flagged.group(1)).is_file():
        return flagged.group(1)
    virtual = re.search(r"(?:^|\s)VIRTUAL_ENV=(\S+)", environ)
    if virtual and "uv/builds-" not in virtual.group(1):
        candidate = Path(virtual.group(1)) / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return None


def _child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    # The build hook inherits PYTHONPATH pointed at pip's isolated env, so a
    # nested `python -m pip` is not the venv's pip. Pass a normal environment.
    keep = (
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "UV_CACHE_DIR",
    )
    env = {key: os.environ[key] for key in keep if key in os.environ}
    if extra:
        env.update(extra)
    return env


def _same_interpreter(python: str) -> bool:
    return Path(python).resolve() == Path(sys.executable).resolve()


def _torch_ready(python: str, *, external: bool = False) -> bool:
    # The build hook's sys.path is pip's isolated env, so an in-process import
    # can miss a torch that is installed in the target.
    if not external and _same_interpreter(python):
        importlib.invalidate_caches()
        try:
            import torch
        except Exception:
            return False
        return torch.__version__.split("+")[0] == TORCH_VERSION
    statement = (
        "import torch,sys;"
        f"sys.exit(0 if torch.__version__.split('+')[0]=='{TORCH_VERSION}' else 1)"
    )
    return (
        subprocess.run(
            [python, "-c", statement],
            env=_child_env(),
            check=False,
        ).returncode
        == 0
    )


def _xformers_works(python: str, *, external: bool = False) -> bool:
    if not external and _same_interpreter(python):
        importlib.invalidate_caches()
        try:
            import xformers.ops  # noqa: F401
        except Exception:
            return False
        return True
    return (
        subprocess.run(
            [python, "-c", "import xformers.ops"],
            env=_child_env(),
            check=False,
        ).returncode
        == 0
    )


def _clang_without_openmp() -> Path:
    # Apple clang has no -fopenmp. The extension still builds and imports
    # without the flag; unknown OpenMP pragmas are ignored.
    path = Path.home() / ".cache" / "solarian" / "clang-no-openmp.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "args = [arg for arg in sys.argv[1:] if arg != '-fopenmp']\n"
        "os.execv('/usr/bin/clang++', ['/usr/bin/clang++', *args])\n"
    )
    path.chmod(0o755)
    return path


def _run_install(python: str, args: list[str], env: dict[str, str]) -> None:
    # Don't resolve() the interpreter: a venv python is a symlink into the
    # framework, and pip lives beside the symlink. A nested `uv pip` from
    # inside `uv`'s build deadlocks on uv's lock.
    pip = Path(python).absolute().parent / "pip"
    if pip.is_file():
        cmd = [str(pip), "install", "--disable-pip-version-check", *args]
    else:
        uv = os.environ.get("UV") or shutil.which("uv")
        if uv is None:
            raise RuntimeError(
                "Cannot build xformers on Apple Silicon: this environment has no pip and uv is not on PATH."
            )
        cmd = [uv, "pip", "install", "--python", python, *args]
    subprocess.run(cmd, check=True, env=env)


def _install_xformers(python: str, *, allow_torch: bool, external: bool) -> None:
    if not _torch_ready(python, external=external):
        if not allow_torch:
            return
        print(
            f"Installing torch {TORCH_VERSION} so xformers can build on Apple Silicon.",
            file=sys.stderr,
        )
        _run_install(python, [f"torch=={TORCH_VERSION}"], _child_env())
    if not _torch_ready(python, external=external) or _xformers_works(python, external=external):
        return
    wrapper = str(_clang_without_openmp())
    print(
        "Building xformers from source for Apple Silicon "
        "(the published build passes -fopenmp, which Apple clang rejects).",
        file=sys.stderr,
    )
    _run_install(
        python,
        [
            "--no-build-isolation",
            "--no-deps",
            f"xformers=={XFORMERS_VERSION}",
        ],
        _child_env({"CC": wrapper, "CXX": wrapper}),
    )


def ensure_xformers(*, installing: bool = False) -> None:
    """Build ``xformers`` on Apple Silicon when the AI stack is being installed.

    A declared extra dependency fails the whole install: Apple clang rejects
    ``-fopenmp``. ``installing`` is set from the build backend, which can see
    the ``ai`` extra on the pip/uv command line. Import time covers a prebuilt
    wheel and a uv build env, once torch is already in the interpreter.
    """
    global _xformers_ready, _xformers_attempted
    if _xformers_ready or _xformers_attempted or not apple_silicon():
        return
    if installing:
        ancestors = _ancestors()
        if not _ai_extra_requested(ancestors):
            return
        python = _uv_target_python(ancestors) if _in_uv_build_env() else sys.executable
        if python is None:
            return
        allow_torch = not _no_deps_requested(ancestors)
    else:
        python = sys.executable
        allow_torch = False
        if not _torch_ready(python):
            return
    _xformers_attempted = True
    if _xformers_works(python, external=installing):
        if _same_interpreter(python):
            _xformers_ready = True
        return
    try:
        _install_xformers(python, allow_torch=allow_torch, external=installing)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(
            "Solarian could not build xformers on Apple Silicon. "
            "The build needs git and a C compiler "
            "(Xcode Command Line Tools: xcode-select --install). "
            f"({exc})",
            file=sys.stderr,
        )
        return
    if _same_interpreter(python) and _xformers_works(python, external=installing):
        _xformers_ready = True
