"""Onshape → MJCF export command.

Turns an Onshape assembly into an attachable MuJoCo (MJCF) model that a
[Module] can contribute to a simulated [Robot]'s scene. See the reusable engine
in :mod:`solarian.sim.onshape` and the docs page "Onshape → Sim Models".

Usage:
    solarian onshape-export <assembly_dir> [--clean-only] [--keys-file PATH]
                             [--update-yaml PATH]

``<assembly_dir>`` is a directory containing a ``config.json`` with the
onshape-to-robot fields (``document_id``/``workspace_id``/``element_id`` or
``url``, ``output_format: mujoco``, ``output_filename``, ``assets_directory``,
``robot_name``, ``joint_properties``) plus a ``solarian`` block describing how to
turn the raw export into an attachable Solarian model
(:class:`solarian.sim.onshape.SolarianExportSpec`).

Credentials are read from the environment by onshape-to-robot
(``ONSHAPE_API``, ``ONSHAPE_ACCESS_KEY``, ``ONSHAPE_SECRET_KEY``). If the keys
are not already in the environment, this command loads them from a keys file
(``--keys-file``, or by default ``onshape_keys.txt`` then ``.env`` in the
current directory) that contains ``KEY=VALUE`` lines.

Joint limits always flow **CAD → MJCF → config.json**: Onshape mate limits win
over any stale ``joint_properties.*.limits`` during the pull, and the cleaned
MJCF ranges are written back into ``config.json``. With
``--update-yaml conf/<module>.yaml``, those same ranges are also written into
the module conf's ``sim_range_min`` / ``sim_range_max``.

Examples:
    solarian onshape-export assets/so106_description
    solarian onshape-export assets/so106_description --update-yaml conf/right_arm.yaml
    solarian onshape-export assets/umi_slim_gripper_description --clean-only
"""

import argparse
import os
import sys
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils.env_file import load_env_file

#: Public Onshape stack; overridden if ``ONSHAPE_API`` is already set.
DEFAULT_ONSHAPE_API = "https://cad.onshape.com"

#: Default files searched for credentials when they are not in the environment.
#: First match wins. ``.env`` is included so project-local secrets work without
#: a separate onshape_keys.txt.
DEFAULT_KEYS_FILES = ("onshape_keys.txt", ".env")


def _ensure_credentials(keys_file: str | None) -> None:
    """Make sure the Onshape API env vars onshape-to-robot expects are set."""
    os.environ.setdefault("ONSHAPE_API", DEFAULT_ONSHAPE_API)

    have_keys = os.environ.get("ONSHAPE_ACCESS_KEY") and os.environ.get("ONSHAPE_SECRET_KEY")
    if not have_keys:
        candidates = (
            [Path(keys_file)] if keys_file else [Path(p) for p in DEFAULT_KEYS_FILES]
        )
        for candidate in candidates:
            if candidate.exists():
                load_env_file(candidate)
                print(f"Loaded Onshape credentials from {candidate}")
                break

    if not (os.environ.get("ONSHAPE_ACCESS_KEY") and os.environ.get("ONSHAPE_SECRET_KEY")):
        searched = keys_file or ", ".join(DEFAULT_KEYS_FILES)
        raise RuntimeError(
            "Onshape API credentials not found. Set ONSHAPE_ACCESS_KEY and "
            "ONSHAPE_SECRET_KEY in the environment, or put them in a keys file "
            f"({searched!r}, or pass --keys-file). Create keys at "
            "https://dev-portal.onshape.com/keys."
        )


@CommandRegistry.register("onshape-export")
class OnshapeExportCommand(BaseCommand):
    category = "Utilities"
    uses_config = False
    help = "Export an Onshape assembly into an attachable MuJoCo sim model"
    description = """
    Pulls the assembly named by the directory's config.json, then cleans the raw
    export into an MJCF a Module can attach to a simulated Robot's scene.

    Joint limits flow CAD -> MJCF -> config.json: the Onshape mate limits win over
    stale joint_properties.*.limits, and the cleaned ranges are written back.

    Credentials come from ONSHAPE_ACCESS_KEY / ONSHAPE_SECRET_KEY, or from a
    KEY=VALUE file (onshape_keys.txt, then .env, unless --keys-file says
    otherwise). Create keys at https://dev-portal.onshape.com/keys.
    """
    usage = "<assembly_dir> [--clean-only] [--keys-file PATH] [--update-yaml PATH]"
    options = {
        "<assembly_dir>": "Directory holding the assembly's config.json.",
        "--clean-only": "Re-clean a previous raw export; no network and no "
                        "credentials needed.",
        "--keys-file PATH": "File with the ONSHAPE_* KEY=VALUE lines (default: "
                            "onshape_keys.txt, then .env).",
        "--update-yaml PATH": "Also write the exported joint ranges into this module "
                              "conf's sim_range_min / sim_range_max.",
    }
    examples = (
        "solarian onshape-export assets/so106_description",
        "solarian onshape-export assets/so106_description --update-yaml conf/right_arm.yaml",
        "solarian onshape-export assets/umi_slim_gripper_description --clean-only",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian onshape-export")
        parser.add_argument("assembly_dir", help="Directory containing the assembly config.json.")
        parser.add_argument("--clean-only", action="store_true",
                            help="Re-run only the clean stage on a previous raw export (no network).")
        parser.add_argument("--keys-file", default=None,
                            help="File with ONSHAPE_ACCESS_KEY/ONSHAPE_SECRET_KEY (default: onshape_keys.txt).")
        parser.add_argument("--update-yaml", default=None, metavar="PATH",
                            help="After export, sync MJCF joint ranges into this module conf "
                                 "(sim_range_min/max) and config.json joint_properties.limits.")
        ns = parser.parse_args(argv)
        self.assembly_dir = ns.assembly_dir
        self.clean_only = ns.clean_only
        self.keys_file = ns.keys_file
        self.update_yaml = ns.update_yaml
        return []

    def handle(self, cfg: DictConfig) -> None:
        assembly_dir = self.assembly_dir
        if not assembly_dir:
            print("Error: onshape-export requires an assembly directory argument")
            print(
                "Usage: solarian onshape-export <assembly_dir> "
                "[--clean-only] [--keys-file PATH] [--update-yaml PATH]"
            )
            sys.exit(1)

        clean_only = self.clean_only
        keys_file = self.keys_file or None
        update_yaml = self.update_yaml or None

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
        if update_yaml:
            print(f"Update yaml: {update_yaml}")
        print(f"{'='*60}\n")

        out_xml = export(
            str(assembly_path), clean_only=clean_only, update_yaml=update_yaml
        )

        print(f"\n{'='*60}")
        print(f"Done. Attachable MJCF written to:\n  {out_xml}")
        print(f"{'='*60}")
        print("\nNext steps:")
        print("  1. Add a ModuleAttachment(mjcf=...) for it in your robot's scene.")
        print(
            "  2. config.json joint_properties.*.limits were synced from the "
            "Onshape mate limits in the MJCF."
        )
        if update_yaml:
            print(
                "  3. `sim_range_min`/`sim_range_max` were synced too; verify "
                "attach prefixes / joint names."
            )
        else:
            print(
                "  3. Fill in `sim_joint` / `sim_range_*` on the module conf, "
                "or re-run with `--update-yaml conf/<module>.yaml` to sync "
                "ranges from the MJCF."
            )
        print("  See the 'Onshape -> Sim Models' docs page.\n")
