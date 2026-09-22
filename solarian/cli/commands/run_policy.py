"""
``solarian run-policy <skill>`` — run the policy server associated with a skill.

A helper that launches the :class:`~solarian.core.ai.system1.PolicyServer`
that a given skill drives.  The skill declares its server via the
:attr:`~solarian.core.ai.system1.skill.Skill.policy_server_module` class
attribute and the checkpoint location in its bundled config
(``policy_server.checkpoint``); this command resolves both and runs the server
(``python -m <module> --input <checkpoint> --host <host> --port <port>``).

Usage::

    solarian run-policy <skill_name> [--checkpoint PATH] [--host HOST] [--port PORT]

Examples::

    solarian run-policy cup_arrange
    solarian run-policy cup_arrange --checkpoint data/models/cup.ckpt --port 8001
"""

import os
import subprocess
import sys

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register('run-policy')
class RunPolicyCommand(BaseCommand):
    help = 'Run the policy server associated with a skill'
    description = """
    Looks up the named skill, reads the policy server module it declares and the
    checkpoint from its bundled config, and runs that server as a subprocess. The
    controller then reaches it over HTTP at policy_server_url.

    Run it on the machine that holds the checkpoint and the GPU, which is often not
    the robot. The flags override what the skill's config says.
    """
    usage = "<skill_name> [--checkpoint PATH] [--host HOST] [--port PORT]"
    options = {
        "<skill_name>": "Skill whose policy server to run; unknown names print the "
                        "installed skills.",
        "--checkpoint PATH": "Checkpoint to serve, overriding the skill's configured "
                             "one.",
        "--host HOST": "Address to bind (default: the skill's config).",
        "--port PORT": "Port to bind (default: the skill's config).",
    }
    examples = (
        "solarian run-policy cup_arrange",
        "solarian run-policy cup_arrange --checkpoint data/models/cup.ckpt --port 8001",
    )

    def handle(self, cfg: DictConfig) -> None:
        skill_name = os.environ.get("SOLARIAN_RUN_POLICY_SKILL")
        if not skill_name:
            print("Error: a skill name is required")
            print(
                "Usage: solarian run-policy <skill_name> "
                "[--checkpoint PATH] [--host HOST] [--port PORT]"
            )
            sys.exit(1)

        from solarian.core.ai.system1.registry import discover_skill_classes

        skills = discover_skill_classes()
        skill_cls = skills.get(skill_name)
        if skill_cls is None:
            available = ", ".join(sorted(skills)) or "(none installed)"
            print(f"Error: unknown skill {skill_name!r}. Available skills: {available}.")
            sys.exit(1)

        # Resolve the project-level config override (conf/<skill>.yaml) by the
        # name the skill was looked up under, if the skill didn't set one.
        if not getattr(skill_cls, "config_name", None):
            skill_cls.config_name = skill_name

        checkpoint = os.environ.get("SOLARIAN_RUN_POLICY_CHECKPOINT")
        host = os.environ.get("SOLARIAN_RUN_POLICY_HOST")
        port = os.environ.get("SOLARIAN_RUN_POLICY_PORT")

        try:
            spec = skill_cls.get_policy_server_spec(
                checkpoint=checkpoint, host=host, port=port
            )
        except Exception as e:  # noqa: BLE001 - surface config errors cleanly
            print(f"Error: {e}")
            sys.exit(1)

        if spec is None:
            print(
                f"Error: skill {skill_name!r} does not define a policy server "
                f"(its 'policy_server_module' is not set), so there is nothing to run."
            )
            sys.exit(1)

        cmd = [
            sys.executable,
            "-m",
            spec["module"],
            "--input",
            spec["checkpoint"],
            "--host",
            spec["host"],
            "--port",
            str(spec["port"]),
        ]
        print(
            f"Running policy server for skill {skill_name!r} "
            f"(module={spec['module']}, checkpoint={spec['checkpoint']}, "
            f"host={spec['host']}, port={spec['port']}):"
        )
        print("  " + " ".join(cmd))
        try:
            completed = subprocess.run(cmd)
        except KeyboardInterrupt:
            return
        if completed.returncode != 0:
            sys.exit(completed.returncode)
