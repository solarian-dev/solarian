"""CAD URDF → MJCF import command.

Turns a URDF and meshes downloaded from a CAD tool into an attachable MuJoCo
(MJCF) model that a [Module] can contribute to a simulated [Robot]'s scene.
See :mod:`solarian.sim.cad` and the docs page "Models from CAD".

Usage:
    solarian cad-import <assembly_dir> [--urdf PATH] [--clean-only]
                         [--density KG/M3] [--update-yaml PATH]

``<assembly_dir>`` holds the downloaded files. The model name is that directory
with ``_description`` removed (``gripper1_description`` → ``gripper1``)::

    <assembly_dir>/
      robot.urdf          # or urdf/robot.urdf
      meshes/*.stl
      <name>_raw.xml      # written by the command

The attachable model is ``../<name>.xml``. The first import writes
``config.yaml`` beside the URDF (density, ``kp``, friction loss, and the other
clean-stage values). Later imports read that file and do not change it.
``--density`` overrides the file's density for that run only.

Name parts, frames, and mates with the same prefixes the old Onshape export
used. The prefix is stripped in the MJCF:

    dof_<name>       actuated joint <name>     (dof_jaw → jaw)
    dof_<name>_inv   same joint, axis flipped
    frame_<name>     site <name>               (frame_tcp → tcp)
    fix_<name>       fastened; merged into the parent link
    link_<name>      body <name>
    closing_<name>   loop closure, not a driven joint

Put ``wheel`` or ``continuous`` in a mate name for a joint with no limits.
``frame_flange``, ``frame_tcp``, and ``frame_camera`` are the sites Solarian
looks up. A mate relation exported as ``<mimic>`` is a follower with no actuator.

Examples:
    solarian cad-import assets/gripper1_description
    solarian cad-import assets/gripper1_description --urdf ~/Downloads/gripper.zip
    solarian cad-import assets/gripper1_description --density 522
    solarian cad-import assets/gripper1_description --clean-only
"""

import argparse
import sys
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register("cad-import")
class CadImportCommand(BaseCommand):
    category = "Utilities"
    uses_config = False
    help = "Import a downloaded CAD URDF and meshes into an attachable MuJoCo sim model"
    description = """
    Reads a URDF plus the STL or OBJ meshes it references and cleans it into an
    MJCF a Module can attach to a simulated Robot's scene. Nothing is fetched
    from a CAD server.

    Put the .urdf (or urdf/*.urdf) and meshes/ in <assembly_dir>. The model name
    is that directory with _description removed, so gripper1_description writes
    gripper1_raw.xml, copies meshes into meshes/, and writes ../gripper1.xml.
    The first run writes config.yaml (density, kp, frictionloss, dampratio,
    armature). Later imports keep that file. --density overrides its density
    for one run. --urdf takes a .urdf or the .zip when the download is not in
    that layout yet.

    Onshape: right-click the assembly, Export, URDF, geometry STL, and check
    "Export rigid subassemblies as a single link", then unzip into <assembly_dir>.
    FreeCAD: the RobotCAD workbench, meshes as STL in meters. Blender: the Phobos
    add-on, STL. SolidWorks: SW2URDF writes urdf/<robot>.urdf and meshes/; set
    the mesh format to STL. The package folder can be the assembly directory.

    Name mates dof_<name>, frames frame_<name> (frame_tcp, frame_flange,
    frame_camera), fastened parts fix_<name>, link overrides link_<name>, and
    loop closures closing_<name>. The prefix is removed in the MJCF. See the
    'Models from CAD' docs page for the click-by-click export and the full table.
    """
    usage = "<assembly_dir> [--urdf PATH] [--clean-only] [--density KG/M3] [--update-yaml PATH]"
    options = {
        "<assembly_dir>": "Directory holding the URDF and meshes/. "
                          "gripper1_description writes ../gripper1.xml.",
        "--urdf PATH": "A .urdf, or the .zip a CAD tool downloaded. "
                       "Omit when the URDF is already in the assembly directory.",
        "--clean-only": "Re-clean the raw MJCF already written. Does not read the URDF.",
        "--density KG/M3": "Mass density in kg/m³ for this run. Overrides config.yaml "
                           "and does not rewrite that file. Default 1400.",
        "--update-yaml PATH": "Also write the imported joint ranges into this module "
                              "conf's sim_range_min / sim_range_max.",
    }
    examples = (
        "solarian cad-import assets/gripper1_description",
        "solarian cad-import assets/gripper1_description --urdf ~/Downloads/gripper.zip",
        "solarian cad-import assets/gripper1_description --density 522",
        "solarian cad-import assets/gripper --update-yaml conf/right_gripper.yaml",
        "solarian cad-import assets/gripper1_description --clean-only",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian cad-import")
        parser.add_argument("assembly_dir", help="Directory containing config.json, the URDF, and meshes.")
        source = parser.add_mutually_exclusive_group()
        source.add_argument("--urdf", default=None, metavar="PATH",
                            help="A .urdf file or a .zip of the URDF and its meshes.")
        source.add_argument("--clean-only", action="store_true",
                            help="Re-run only the clean stage on the raw MJCF (does not read the URDF).")
        parser.add_argument("--density", default=None, type=float, metavar="KG/M3",
                            help="Mass density in kg/m³ for inferred link mass (default 1400).")
        parser.add_argument("--update-yaml", default=None, metavar="PATH",
                            help="After import, sync MJCF joint ranges into this module conf "
                                 "(sim_range_min/max) and config.json joint_properties.limits.")
        ns = parser.parse_args(argv)
        self.assembly_dir = ns.assembly_dir
        self.urdf = ns.urdf
        self.clean_only = ns.clean_only
        self.density = ns.density
        self.update_yaml = ns.update_yaml
        return []

    def handle(self, cfg: DictConfig) -> None:
        assembly_dir = self.assembly_dir
        if not assembly_dir:
            print("Error: cad-import requires an assembly directory argument")
            print(
                "Usage: solarian cad-import <assembly_dir> "
                "[--urdf PATH] [--clean-only] [--density KG/M3] [--update-yaml PATH]"
            )
            sys.exit(1)

        clean_only = self.clean_only
        urdf = self.urdf or None
        update_yaml = self.update_yaml or None
        density = self.density

        assembly_path = Path(assembly_dir)
        if not assembly_path.is_dir():
            raise FileNotFoundError(f"Assembly directory not found: {assembly_path}")

        from solarian.sim.cad import export

        print(f"\n{'='*60}")
        print(f"Importing CAD model: {assembly_path}")
        if clean_only:
            print("Mode: clean-only")
        elif urdf:
            print(f"URDF: {urdf}")
        else:
            print("Mode: URDF in the assembly directory")
        if density is not None:
            print(f"Density: {density} kg/m³")
        if update_yaml:
            print(f"Update yaml: {update_yaml}")
        print(f"{'='*60}\n")

        out_xml = export(
            str(assembly_path),
            clean_only=clean_only,
            update_yaml=update_yaml,
            urdf=urdf,
            density=density,
        )

        print(f"\n{'='*60}")
        print(f"Done. Attachable MJCF written to:\n  {out_xml}")
        print(f"{'='*60}")
        print("\nNext steps:")
        print("  1. Add a ModuleAttachment(mjcf=...) for it in your robot's scene.")
        if (assembly_path / "config.json").is_file():
            print("  2. config.json joint_properties.*.limits were synced from the MJCF.")
        else:
            print("  2. Joint ranges are the ranges in the MJCF.")
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
        print("  See the 'Models from CAD' docs page.\n")
