"""Every command has to be able to describe itself.

``solarian <command> --help`` is rendered from metadata on the command class, so
these checks are what keeps a new command from shipping without documentation.
"""

import pytest

import solarian.cli.commands  # noqa: F401 - registers the built-in commands
from solarian.cli.base import CommandRegistry
from solarian.cli.help import WIDTH, format_command_help
from solarian.cli.manage import NO_HYDRA_COMMANDS

COMMANDS = sorted(CommandRegistry.get_all_command_names())


def _render(name: str) -> str:
    return format_command_help(
        name,
        CommandRegistry.get_command(name),
        config_name=CommandRegistry.get_config_name(name),
        uses_config=name not in NO_HYDRA_COMMANDS,
    )


@pytest.mark.parametrize("name", COMMANDS)
def test_command_documents_itself(name):
    cmd = CommandRegistry.get_command(name)
    assert cmd.help, f"{name} has no one-line help"
    assert cmd.description.strip(), f"{name} has no description"
    assert cmd.examples, f"{name} has no examples"


@pytest.mark.parametrize("name", COMMANDS)
def test_examples_invoke_the_command(name):
    for example in CommandRegistry.get_command(name).examples:
        assert example.startswith(f"solarian {name}"), (
            f"{name} example does not invoke it: {example!r}"
        )


@pytest.mark.parametrize("name", COMMANDS)
def test_arguments_are_described(name):
    cmd = CommandRegistry.get_command(name)
    for term, text in {**cmd.params, **cmd.options}.items():
        assert text.strip(), f"{name} documents {term} with an empty string"


@pytest.mark.parametrize("name", COMMANDS)
def test_usage_lists_the_arguments_the_command_parses(name):
    """A command that reads positionals or flags has to show them in its usage."""
    cmd = CommandRegistry.get_command(name)
    for term in cmd.options:
        assert term.split()[0] in cmd.usage, f"{name} usage omits {term!r}"


@pytest.mark.parametrize("name", COMMANDS)
def test_listed_help_matches_the_command_page(name):
    """The one-liner in the command list is the one the command's help prints."""
    assert CommandRegistry.get_command_help(name) == CommandRegistry.get_command(name).help


@pytest.mark.parametrize("name", COMMANDS)
def test_help_renders_within_the_terminal_width(name):
    text = _render(name)
    assert text.startswith(f"usage: solarian {name}")
    assert "examples:" in text
    too_wide = [line for line in text.splitlines() if len(line) > WIDTH]
    assert not too_wide, f"{name} help is not wrapped: {too_wide}"


def test_help_names_the_config_it_composes():
    text = _render("run-robot")
    assert "conf/robot.yaml" in text
    assert "--deployment" in text


def test_help_of_a_config_free_command_hides_the_conf_flags():
    text = _render("create-project")
    assert "--conf" not in text
    assert "<project_name>" in text


def test_overrides_are_listed_with_their_help():
    text = _render("calibrate-umi-camera")
    assert "config overrides (key=value)" in text
    assert "overlay_only" in text
    assert "solarian calibrate-umi-camera overlay_only=true" in text
