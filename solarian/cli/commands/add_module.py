"""
Add Module Command

Installs a module plugin package and copies its default configuration file
to the project's conf/ directory.

Supports either a PyPI package name (e.g. ur5) or a local path
(e.g. ./plugins/solarian-ur5).

Usage:
    solarian add-module <module_name_or_path> <config_name>

Examples:
    solarian add-module ur5 arm
    solarian add-module umi-gripper gripper
    solarian add-module ./plugins/solarian-ur5 arm
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
import importlib.metadata
import importlib.util

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry


def _looks_like_path(arg: str) -> bool:
    """Return True if the argument looks like a filesystem path rather than a package name."""
    return "/" in arg or arg.startswith(".") or arg.startswith("~")


def _parse_module_arg(module_name: str) -> Tuple[str, str, Optional[Path]]:
    """
    Parse the first argument: either a short name (ur5) or a local path (./plugins/solarian-ur5).
    Returns (module_name, plugin_package, local_path_or_none).
    """
    if _looks_like_path(module_name):
        # Treat as a filesystem path
        path = Path(module_name).resolve()
        if not path.exists():
            print(f"Error: Path '{module_name}' does not exist (resolved to {path}).")
            print(f"  Check the relative path from your current directory.")
            sys.exit(1)
        if not path.is_dir():
            print(f"Error: '{path}' is not a directory.")
            sys.exit(1)
        if not (path / "pyproject.toml").exists() and not (path / "setup.py").exists():
            print(f"Error: '{path}' does not contain a pyproject.toml or setup.py.")
            sys.exit(1)
        dir_name = path.name
        # e.g. solarian-ur5 -> module_name ur5, package solarian-ur5
        if dir_name.startswith("solarian-"):
            short = dir_name[len("solarian-"):]
        else:
            short = dir_name
        return short, dir_name, path
    else:
        # Treat as a short package name (ur5 -> solarian-ur5)
        name = module_name.strip()
        pkg = f"solarian-{name}"
        return name, pkg, None


@CommandRegistry.register('add-module')
class AddModuleCommand(BaseCommand):
    help = 'Install a module plugin and copy its default configuration to the project'
    description = """
    Runs pip install for the plugin, copies the module's shipped default yaml to
    conf/<config_name>.yaml, and adds the package to the project's
    pyproject.toml dependencies.

    The config name is how the module is addressed from then on: the yaml the
    module reads, the attribute to declare on the Robot or Controller class, and
    the name every other command lists it under. Run this once per physical
    component, so two arms are two names (left_arm, right_arm) off one plugin.
    """
    usage = "<module_name_or_path> <config_name>"
    options = {
        "<module_name_or_path>": "Short plugin name (ur5 installs solarian-ur5) or a "
                                 "local path to a plugin directory for an editable "
                                 "install.",
        "<config_name>": "Name for this instance of the module; becomes "
                         "conf/<config_name>.yaml.",
    }
    examples = (
        "solarian add-module ur5 arm",
        "solarian add-module umi-gripper gripper",
        "solarian add-module ./plugins/solarian-ur5 right_arm",
    )

    def handle(self, cfg: DictConfig) -> None:
        """Install the plugin package and copy its conf.yaml to the project."""
        # Get positional arguments from environment
        module_arg = os.environ.get("SOLARIAN_ADD_MODULE_NAME")
        config_name = os.environ.get("SOLARIAN_ADD_MODULE_ID")
        
        if not module_arg or not config_name:
            print("Error: module_name (or path) and config_name are required")
            print("Usage: solarian add-module <module_name_or_path> <config_name>")
            print("  e.g. solarian add-module ur5 arm")
            print("  e.g. solarian add-module ./plugins/solarian-ur5 arm")
            sys.exit(1)
        
        module_name, plugin_package, local_path = _parse_module_arg(module_arg)
        
        print(f"\n{'='*60}")
        print(f"Adding module: {plugin_package}")
        print(f"Config name: {config_name}")
        if local_path:
            print(f"Local path: {local_path}")
        print(f"{'='*60}\n")
        
        # Step 1: Install plugin package
        self._install_plugin(plugin_package, local_path, config_name)
        
        # Step 2: Find and copy conf.yaml
        conf_yaml_path = self._find_and_copy_conf(module_name, plugin_package, local_path, config_name)
        
        # Step 3: Add dependency to project's pyproject.toml
        project_path = os.environ.get("SOLARIAN_PROJECT_PATH", os.getcwd())
        self._add_dependency_to_pyproject(
            project_path=Path(project_path),
            plugin_package=plugin_package,
        )
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Done! Module '{plugin_package}' added successfully.")
        print(f"{'='*60}")
        print(f"\nConfiguration file: conf/{config_name}.yaml")
        print(f"\nNext steps:")
        print(f"  1. Edit conf/{config_name}.yaml to match your hardware setup")
        print(f"  2. Import the module class directly in your robot.py:")
        
        # Try to show the actual import path
        import_module = f"solarian_{module_name.replace('-', '_')}"
        print(f"     from {import_module}.module import <ModuleClass>")
        print()
    
    def _install_plugin(self, plugin_package: str, local_path: Optional[Path], config_name: str) -> None:
        """Install the plugin package if not already installed."""
        try:
            importlib.metadata.distribution(plugin_package)
            print(f"Step 1: Plugin '{plugin_package}' is already installed (skipping pip install).\n")
        except importlib.metadata.PackageNotFoundError:
            print(f"Step 1: Installing '{plugin_package}'...")
            try:
                if local_path is not None:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "-e", str(local_path)],
                        stdout=sys.stdout,
                        stderr=sys.stderr
                    )
                else:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", plugin_package],
                        stdout=sys.stdout,
                        stderr=sys.stderr
                    )
                print(f"  Installed {plugin_package}\n")
            except subprocess.CalledProcessError as e:
                print(f"\n  Failed to install {plugin_package}")
                print(f"  Error: {e}")
                if local_path is None:
                    print(f"\n  Make sure the package name is correct.")
                    print(f"  For local development use a path: solarian add-module ./plugins/{plugin_package} {config_name}")
                sys.exit(1)
    
    def _find_and_copy_conf(
        self,
        module_name: str,
        plugin_package: str,
        local_path: Optional[Path],
        config_name: str,
    ) -> Optional[Path]:
        """Find the conf.yaml in the plugin and copy it to the project's conf/ directory."""
        print(f"Step 2: Copying default configuration...")
        
        conf_yaml_path = self._find_conf_yaml(module_name, plugin_package, local_path=local_path)
        
        if not conf_yaml_path:
            print(f"  Could not find conf.yaml in {plugin_package}")
            print(f"  The package may not include a default conf.yaml.")
            print(f"  You will need to create conf/{config_name}.yaml manually.")
            return None
        
        # Get project conf directory
        project_path = os.environ.get("SOLARIAN_PROJECT_PATH", os.getcwd())
        conf_dir = Path(project_path) / "conf"
        conf_dir.mkdir(exist_ok=True)
        
        dest_path = conf_dir / f"{config_name}.yaml"
        
        if dest_path.exists():
            print(f"  conf/{config_name}.yaml already exists. Skipping copy.")
            print(f"  (source was: {conf_yaml_path})")
            return dest_path
        
        try:
            shutil.copy2(conf_yaml_path, dest_path)
            print(f"  Copied default config to conf/{config_name}.yaml\n")
        except Exception as e:
            print(f"\n  Failed to copy configuration file: {e}")
            sys.exit(1)
        
        return dest_path
    
    def _add_dependency_to_pyproject(
        self,
        project_path: Path,
        plugin_package: str,
    ) -> None:
        """Add the plugin package to the project's pyproject.toml dependencies."""
        pyproject = project_path / "pyproject.toml"
        if not pyproject.exists():
            print(
                f"Step 3: No pyproject.toml found. "
                f"Add '{plugin_package}' to your project dependencies manually.\n"
            )
            return
        text = pyproject.read_text(encoding="utf-8")
        dep_line = f'    "{plugin_package}>=0.1.0",'
        # Check uncommented lines only to avoid matching commented-out examples
        uncommented_lines = [
            l for l in text.split("\n") if not l.strip().startswith("#")
        ]
        uncommented_text = "\n".join(uncommented_lines)
        if f'"{plugin_package}' in uncommented_text:
            print(f"Step 3: '{plugin_package}' already in pyproject.toml dependencies.\n")
            return
        # Find [project] dependencies = [ ... ] and insert before the closing ]
        lines = text.split("\n")
        in_project = False
        in_deps = False
        insert_at = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("[project]"):
                in_project = True
                in_deps = False
                continue
            if line.strip().startswith("[") and not line.strip().startswith("[project]"):
                in_project = False
                in_deps = False
                continue
            if in_project and re.match(r"^\s*dependencies\s*=\s*\[\s*$", line):
                in_deps = True
                continue
            if in_project and re.match(r"^\s*dependencies\s*=\s*\[", line):
                in_deps = True
                if line.rstrip().endswith("]"):
                    insert_at = i
                    break
                continue
            if in_deps:
                stripped = line.strip()
                if stripped == "]":
                    insert_at = i
                    break
        if insert_at >= 0:
            lines.insert(insert_at, dep_line)
            pyproject.write_text("\n".join(lines), encoding="utf-8")
            print(f"Step 3: Added '{plugin_package}' to pyproject.toml dependencies.\n")
        else:
            print(
                f"Step 3: Could not find [project] dependencies in pyproject.toml. "
                f"Add '{plugin_package}>=0.1.0' to your dependencies manually.\n"
            )
    
    def _find_conf_yaml(
        self,
        module_name: str,
        plugin_package: Optional[str] = None,
        local_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Find the conf.yaml file in an installed plugin package.
        
        Looks for conf.yaml in the package directory (e.g., solarian_ur5/conf.yaml).
        For local/editable installs, searches the source directory directly first,
        since the editable install's .pth file may not be active in the current process.
        """
        if plugin_package is None:
            plugin_package = f"solarian-{module_name}"
        
        # For local installs, search the source directory directly first.
        if local_path is not None:
            possible_package_names = [
                f"solarian_{module_name.replace('-', '_')}",
                module_name.replace('-', '_'),
                f"solarian-{module_name}",
                "src",
            ]
            for pkg_name in possible_package_names:
                conf_path = local_path / pkg_name / "conf.yaml"
                if conf_path.exists() and conf_path.is_file():
                    return conf_path
            conf_path = local_path / "conf.yaml"
            if conf_path.exists() and conf_path.is_file():
                return conf_path
            # Walk subdirectories as a last resort
            for child in local_path.rglob("conf.yaml"):
                if child.is_file() and ".egg-info" not in str(child):
                    return child
        
        # Try to find the package via importlib
        try:
            possible_package_names = [
                f"solarian_{module_name.replace('-', '_')}",
                module_name.replace('-', '_'),
                f"solarian-{module_name}",
            ]
            
            for pkg_name in possible_package_names:
                try:
                    spec = importlib.util.find_spec(pkg_name)
                    if spec and spec.origin:
                        package_dir = Path(spec.origin).parent
                        conf_path = package_dir / "conf.yaml"
                        if conf_path.exists() and conf_path.is_file():
                            return conf_path
                except Exception:
                    continue
            
            # Fallback: search via distribution metadata
            dist = importlib.metadata.distribution(plugin_package)
            dist_files = dist.files
            if dist_files:
                dist_location = Path(dist_files[0].locate()).parent
                for parent in dist_location.parents:
                    if parent.name == "site-packages" or parent.name == "src":
                        dist_location = parent
                        break
                
                search_paths = [
                    dist_location / pkg_name / "conf.yaml"
                    for pkg_name in possible_package_names
                ]
                search_paths.append(dist_location / "conf.yaml")
                
                for conf_path in search_paths:
                    if conf_path.exists() and conf_path.is_file():
                        return conf_path
            
        except importlib.metadata.PackageNotFoundError:
            print(f"  Package {plugin_package} not found. Make sure it's installed.")
            return None
        except Exception as e:
            print(f"  Error finding conf.yaml: {e}")
            return None
        
        return None
