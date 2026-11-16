"""Every command has to be able to describe itself.

``solarian <command> --help`` is rendered from metadata on the command class, so
these checks are what keeps a new command from shipping without documentation.
"""

import pytest

import solarian.cli.commands  # noqa: F401 - registers the built-in commands
from solarian.cli.base import CommandRegistry
from solarian.cli.help import WIDTH, format_command_help

COMMANDS = sorted(CommandRegistry.get_all_command_names())


def _render(name: str) -> str:
    cmd = CommandRegistry.get_command(name)
    return format_command_help(
        name,
        cmd,
        config_name=CommandRegistry.get_config_name(name),
        uses_config=CommandRegistry.uses_config(name),
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


@pytest.mark.parametrize("name", COMMANDS)
def test_category_matches_the_lazy_record(name):
    """Help groups an unloaded command from register_lazy, which must match the class."""
    cmd = CommandRegistry.get_command(name)
    lazy = CommandRegistry._lazy_commands[name]
    assert cmd.category
    assert lazy.category == cmd.category
    assert CommandRegistry.get_category(name) == cmd.category


def test_plugin_command_declares_its_own_section_and_config():
    """A plugin command is grouped and excused from Hydra by its class, not manage.py."""
    from solarian.cli.base import BaseCommand
    from solarian.cli.manage import _build_command_list

    @CommandRegistry.register("widget-ping")
    class WidgetPing(BaseCommand):
        help = "Ping a widget."
        category = "Widgets"
        uses_config = False

        def handle(self, cfg):
            return None

    try:
        assert CommandRegistry.get_category("widget-ping") == "Widgets"
        assert CommandRegistry.uses_config("widget-ping") is False
        listing = _build_command_list()
        assert "Widgets:" in listing
        assert "widget-ping" in listing
        assert "Ping a widget." in listing
    finally:
        CommandRegistry._commands.pop("widget-ping", None)
        CommandRegistry._config_names.pop("widget-ping", None)


def test_commands_with_their_own_argv_store_it_on_the_command():
    cases = {
        "install": (["solarian-move"], {"target": "solarian-move"}),
        "create-project": (
            ["my_robot", "--force"],
            {"project_name": "my_robot", "force": True},
        ),
        "create-plugin": (
            ["actuator", "feetech"],
            {"plugin_type": "actuator", "plugin_name": "feetech", "force": False},
        ),
        "cad-import": (
            ["assets/arm", "--clean-only"],
            {"assembly_dir": "assets/arm", "clean_only": True, "urdf": None},
        ),
        "export-dataset": (
            [
                "datasets/session", "out/ds", "--fps", "30",
                "--episodes", "0,1", "--no-remap", "--overwrite",
            ],
            {
                "session_dir": "datasets/session",
                "output_dir": "out/ds",
                "fps": "30",
                "episodes": "0,1",
                "no_remap": True,
                "overwrite": True,
            },
        ),
        "run-service": (
            [
                "umi_pnp", "--checkpoint", "cup.ckpt",
                "--host", "0.0.0.0", "--port", "8001", "--model", "small",
            ],
            {
                "name": "umi_pnp",
                "checkpoint": "cup.ckpt",
                "host": "0.0.0.0",
                "port": 8001,
                "server_args": ["--model", "small"],
            },
        ),
    }
    for name, (argv, expected) in cases.items():
        cmd = CommandRegistry.get_command(name)
        assert cmd.parse_args(argv) == []
        for attr, value in expected.items():
            assert getattr(cmd, attr) == value, f"{name}.{attr}"


def test_a_hydra_command_forwards_its_overrides():
    cmd = CommandRegistry.get_command("run-robot")
    assert cmd.parse_args(["fps=60", "connection.ip=10.0.0.2"]) == [
        "fps=60",
        "connection.ip=10.0.0.2",
    ]


def test_install_requires_a_target():
    cmd = CommandRegistry.get_command("install")
    with pytest.raises(SystemExit):
        cmd.parse_args([])


def test_overrides_are_listed_with_their_help():
    text = _render("derive-model-limits")
    assert "config overrides (key=value)" in text
    assert "sweep" in text
    assert "solarian derive-model-limits module=right_arm sweep=true" in text
