"""Onshape → MJCF export command.

Turns an Onshape assembly into an attachable MuJoCo (MJCF) model that a
[Module] can contribute to a simulated [Robot]'s scene. See the reusable engine
in :mod:`solarian.sim.onshape` and the docs page "Onshape → Sim Models".

Usage:
    solarian onshape-export <assembly_dir> [--clean-only] [--keys-file PATH]

``<assembly_dir>`` is a directory containing a ``config.json`` with the
onshape-to-robot fields (``document_id``/``workspace_id``/``element_id`` or
``url``, ``output_format: mujoco``, ``output_filename``, ``assets_directory``,
``robot_name``, ``joint_properties``) plus a ``solarian`` block describing how to
turn the raw export into an attachable Solarian model
(:class:`solarian.sim.onshape.SolarianExportSpec`).

Credentials are read from the environment by onshape-to-robot
(``ONSHAPE_API``, ``ONSHAPE_ACCESS_KEY``, ``ONSHAPE_SECRET_KEY``). If the keys
are not already in the environment, this command loads them from a keys file
(``--keys-file``, default ``onshape_keys.txt`` in the current directory) that
contains ``KEY=VALUE`` lines.

Examples:
    solarian onshape-export assets/so106_description
    solarian onshape-export assets/umi_slim_gripper_description --clean-only
"""

import os
import sys
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry

#: Public Onshape stack; overridden if ``ONSHAPE_API`` is already set.
DEFAULT_ONSHAPE_API = "https://cad.onshape.com"

#: Default file searched for credentials when they are not in the environment.
DEFAULT_KEYS_FILE = "onshape_keys.txt"


def _load_keys_file(path: Path) -> None:
    """Load ``KEY=VALUE`` lines from *path* into the environment (never
    overwriting values already set)."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _ensure_credentials(keys_file: str | None) -> None:
    """Make sure the Onshape API env vars onshape-to-robot expects are set."""
    os.environ.setdefault("ONSHAPE_API", DEFAULT_ONSHAPE_API)

    have_keys = os.environ.get("ONSHAPE_ACCESS_KEY") and os.environ.get("ONSHAPE_SECRET_KEY")
    if not have_keys:
        candidate = Path(keys_file) if keys_file else Path(DEFAULT_KEYS_FILE)
        if candidate.exists():
            _load_keys_file(candidate)
            print(f"Loaded Onshape credentials from {candidate}")

    if not (os.environ.get("ONSHAPE_ACCESS_KEY") and os.environ.get("ONSHAPE_SECRET_KEY")):
        raise RuntimeError(
            "Onshape API credentials not found. Set ONSHAPE_ACCESS_KEY and "
            "ONSHAPE_SECRET_KEY in the environment, or put them in a keys file "
            f"(default '{DEFAULT_KEYS_FILE}', or pass --keys-file). Create keys at "
            "https://dev-portal.onshape.com/keys."
        )


@CommandRegistry.register("onshape-export")
class OnshapeExportCommand(BaseCommand):
    help = "Export an Onshape assembly into an attachable MuJoCo sim model"

    def handle(self, cfg: DictConfig) -> None:
        assembly_dir = os.environ.get("SOLARIAN_ONSHAPE_EXPORT_DIR")
        if not assembly_dir:
            print("Error: onshape-export requires an assembly directory argument")
            print("Usage: solarian onshape-export <assembly_dir> [--clean-only] [--keys-file PATH]")
            sys.exit(1)

        clean_only = os.environ.get("SOLARIAN_ONSHAPE_EXPORT_CLEAN_ONLY") == "true"
        keys_file = os.environ.get("SOLARIAN_ONSHAPE_EXPORT_KEYS_FILE") or None

        assembly_path = Path(assembly_dir)
        if not assembly_path.is_dir():
            raise FileNotFoundError(f"Assembly directory not found: {assembly_path}")
        if not (assembly_path / "config.json").exists():
            raise FileNotFoundError(f"No config.json in {assembly_path}")

        if not clean_only:
            _ensure_credentials(keys_file)

        # Imported here so `import solarian` never requires the sim/onshape extras.
        from solarian.sim.onshape import export

        print(f"\n{'='*60}")
        print(f"Exporting Onshape assembly: {assembly_path}")
        print(f"Mode: {'clean-only (no network)' if clean_only else 'full export'}")
        print(f"{'='*60}\n")

        out_xml = export(str(assembly_path), clean_only=clean_only)

        print(f"\n{'='*60}")
        print(f"Done. Attachable MJCF written to:\n  {out_xml}")
        print(f"{'='*60}")
        print("\nNext steps:")
        print("  1. Add a ModuleAttachment(mjcf=...) for it in your robot's scene.")
        print("  2. Fill in the `sim:` block of the module's conf yaml (joint/actuator + radian range).")
        print("  See the 'Onshape -> Sim Models' docs page.\n")
