"""Unit tests for the ``solarian --version`` flag."""

from __future__ import annotations

import importlib.metadata

import pytest

from solarian.cli.manage import BANNER, _get_version, main


def test_get_version_matches_installed_metadata():
    assert _get_version() == importlib.metadata.version("solarian")


def test_get_version_falls_back_when_not_installed(monkeypatch):
    def raise_not_found(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", raise_not_found)
    assert _get_version() == "unknown"


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_and_exits(flag, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["solarian", flag])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"solarian {_get_version()}\n"


def test_banner_only_on_a_terminal(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["solarian", "--version"])
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    with pytest.raises(SystemExit):
        main()

    assert capsys.readouterr().out.startswith(f"\n{BANNER}")
