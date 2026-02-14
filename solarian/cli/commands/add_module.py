"""
Add Module Command

Installs a module plugin package and copies its configuration file to the project.
Supports either a PyPI package name (e.g. ur5) or a local path (e.g. ./plugins/solarian-ur5)
so that entry points from the plugin's pyproject.toml are registered.
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
    help = 'Install a module plugin and add its configuration to the project'

    def handle(self, cfg: DictConfig) -> None:
        """Install the plugin and copy its conf.yaml file."""
        # Get positional arguments from environment
        module_arg = os.environ.get("SOLARIAN_ADD_MODULE_NAME")
        module_id = os.environ.get("SOLARIAN_ADD_MODULE_ID")
        
        if not module_arg or not module_id:
            print("Error: module_name (or path) and module_id are required")
            print("Usage: solarian add-module <module_name_or_path> <module_id>")
            print("  e.g. solarian add-module ur5 my_ur5")
            print("  e.g. solarian add-module ./plugins/solarian-ur5 my_ur5")
            sys.exit(1)
        
        module_name, plugin_package, local_path = _parse_module_arg(module_arg)
        
        print(f"\n{'='*60}")
        print(f"Adding module: {module_id}")
        print(f"Plugin package: {plugin_package}")
        if local_path:
            print(f"Local path: {local_path}")
        print(f"{'='*60}\n")
        
        # Check if plugin is already installed (e.g. dev/editable mode)
        try:
            importlib.metadata.distribution(plugin_package)
            print(f"Step 1: Plugin package '{plugin_package}' is already installed (skipping pip install).\n")
        except importlib.metadata.PackageNotFoundError:
            print(f"Step 1: Installing plugin package '{plugin_package}'...")
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
                print(f"✓ Successfully installed {plugin_package}\n")
            except subprocess.CalledProcessError as e:
                print(f"\n✗ Failed to install {plugin_package}")
                print(f"  Error: {e}")
                if local_path is None:
                    print(f"\n  Make sure the package name is correct.")
                    print(f"  For local development use a path: solarian add-module ./plugins/solarian-ur5 {module_id}")
                sys.exit(1)
        
        # Verify entry points and get module entry point info
        module_entry_point = self._get_module_entry_point(plugin_package)
        self._verify_entry_points(plugin_package)
        
        # Find the conf.yaml file in the installed package
        print(f"Step 2: Finding configuration file in {plugin_package}...")
        conf_yaml_path = self._find_conf_yaml(module_name, plugin_package, local_path=local_path)
        
        if not conf_yaml_path:
            print(f"\n✗ Could not find conf.yaml in {plugin_package}")
            print(f"  The package may not include a conf.yaml file.")
            print(f"  Please check the plugin documentation.")
            sys.exit(1)
        
        print(f"✓ Found configuration at: {conf_yaml_path}\n")
        
        # Get project conf directory
        project_path = os.environ.get("SOLARIAN_PROJECT_PATH", os.getcwd())
        conf_dir = Path(project_path) / "conf"
        conf_dir.mkdir(exist_ok=True)
        
        # Copy conf.yaml to project conf directory with module_id name
        dest_path = conf_dir / f"{module_id}.yaml"
        
        print(f"Step 3: Copying configuration to project...")
        try:
            shutil.copy2(conf_yaml_path, dest_path)
            print(f"✓ Copied configuration to {dest_path}\n")
        except Exception as e:
            print(f"\n✗ Failed to copy configuration file: {e}")
            sys.exit(1)
        
        # Step 4: Add plugin to project's pyproject.toml (dependency + optional entry-points reference)
        self._add_plugin_to_pyproject(
            project_path=Path(project_path),
            plugin_package=plugin_package,
            local_path=local_path,
        )
        
        # Step 5: Add module entry point to project's pyproject.toml
        if module_entry_point:
            ep_name, ep_value = module_entry_point
            self._add_entry_point_to_pyproject(
                project_path=Path(project_path),
                entry_point_name=ep_name,
                entry_point_value=ep_value,
            )
        else:
            print(
                f"Step 5: Skipping entry point (no solarian.modules entry point in {plugin_package}).\n"
            )
        
        print(f"{'='*60}")
        print(f"✓ Module '{module_id}' has been successfully added!")
        print(f"{'='*60}")
        print(f"\nConfiguration file: {dest_path}")
        print(f"Module type: {module_name}")
        if module_entry_point:
            print(f"Entry point: {module_entry_point[0]}")
        print(f"\nYou can now use this module in your robot configuration.")
    
    def _get_module_entry_point(self, plugin_package: str) -> Optional[Tuple[str, str]]:
        """Return the first solarian.modules entry point (name, value) from the package, or None."""
        try:
            from importlib.metadata import distribution
            dist = distribution(plugin_package)
            for ep in dist.entry_points:
                if ep.group == "solarian.modules":
                    return ep.name, ep.value
        except Exception:
            pass
        return None
    
    def _verify_entry_points(self, plugin_package: str) -> None:
        """Check that the installed package registered Solarian entry points."""
        try:
            from importlib.metadata import distribution
            dist = distribution(plugin_package)
            groups = ("solarian.modules", "solarian.actuators", "solarian.actuator_buses")
            registered = [ep for ep in dist.entry_points if ep.group in groups]
            if registered:
                print(f"✓ Entry points registered ({len(registered)}). Run 'solarian list-plugins' to see the plugin.\n")
            else:
                print(
                    f"⚠ No Solarian entry points found in '{plugin_package}'.\n"
                    f"  The plugin may not be discoverable. Ensure its pyproject.toml declares:\n"
                    f"  [project.entry-points.\"solarian.modules\"] or solarian.actuators / actuator_buses.\n"
                    f"  If developing locally, use: solarian add-module ./plugins/{plugin_package} <module_id>\n"
                )
        except Exception as e:
            print(f"⚠ Could not verify entry points: {e}\n")
    
    def _add_plugin_to_pyproject(
        self,
        project_path: Path,
        plugin_package: str,
        local_path: Optional[Path],
    ) -> None:
        """Add the plugin package to the project's pyproject.toml dependencies."""
        pyproject = project_path / "pyproject.toml"
        if not pyproject.exists():
            print(
                f"Step 4: No pyproject.toml found in {project_path}. "
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
            print(f"Step 4: pyproject.toml already references '{plugin_package}'. No change.\n")
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
            print(f"Step 4: Added '{plugin_package}' to pyproject.toml dependencies.\n")
        else:
            print(
                f"Step 4: Could not find [project] dependencies in pyproject.toml. "
                f"Add '{plugin_package}>=0.1.0' to your dependencies manually.\n"
            )
    
    def _add_entry_point_to_pyproject(
        self,
        project_path: Path,
        entry_point_name: str,
        entry_point_value: str,
    ) -> None:
        """Add the module entry point to the project's pyproject.toml."""
        pyproject = project_path / "pyproject.toml"
        if not pyproject.exists():
            print(
                f"Step 5: No pyproject.toml found in {project_path}. "
                f"Add the entry point manually.\n"
            )
            return

        text = pyproject.read_text(encoding="utf-8")
        entry_line = f'{entry_point_name} = "{entry_point_value}"'

        # Check if entry point already exists (uncommented)
        uncommented_lines = [
            l for l in text.split("\n") if not l.strip().startswith("#")
        ]
        uncommented_text = "\n".join(uncommented_lines)
        if entry_point_name in uncommented_text and entry_point_value in uncommented_text:
            print(f"Step 5: Entry point '{entry_point_name}' already in pyproject.toml. No change.\n")
            return

        section_header = '[project.entry-points."solarian.modules"]'

        if section_header in text:
            # Section exists – append the entry point line after the header
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if line.strip() == section_header:
                    lines.insert(i + 1, entry_line)
                    break
            text = "\n".join(lines)
        else:
            # Section doesn't exist – add it before [build-system] or at the end
            if "[build-system]" in text:
                text = text.replace(
                    "[build-system]",
                    f'{section_header}\n{entry_line}\n\n[build-system]',
                )
            else:
                text = text.rstrip() + f"\n\n{section_header}\n{entry_line}\n"

        pyproject.write_text(text, encoding="utf-8")
        print(f"Step 5: Added entry point '{entry_point_name}' to pyproject.toml.\n")
    
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
        # This is necessary because editable installs register a .pth file that
        # is only processed on Python startup, so importlib.util.find_spec() may
        # not find the package within the same process that ran pip install -e.
        if local_path is not None:
            possible_package_names = [
                f"solarian_{module_name.replace('-', '_')}",
                module_name.replace('-', '_'),
                f"solarian-{module_name}",
                "src",
            ]
            # Search for conf.yaml in subdirectories that match package names
            for pkg_name in possible_package_names:
                conf_path = local_path / pkg_name / "conf.yaml"
                if conf_path.exists() and conf_path.is_file():
                    return conf_path
            # Also check if conf.yaml is directly in the root
            conf_path = local_path / "conf.yaml"
            if conf_path.exists() and conf_path.is_file():
                return conf_path
            # Walk all subdirectories for conf.yaml as a last resort
            for child in local_path.rglob("conf.yaml"):
                if child.is_file() and ".egg-info" not in str(child):
                    return child
        
        # Try to get the package location using importlib.metadata
        try:
            dist = importlib.metadata.distribution(plugin_package)
            # Get the package location from the distribution files
            dist_files = dist.files
            if dist_files:
                dist_location = Path(dist_files[0].locate()).parent
                # Walk up to find the site-packages root
                for parent in dist_location.parents:
                    if parent.name == "site-packages" or parent.name == "src":
                        dist_location = parent
                        break
            else:
                dist_location = Path(dist.locate_file(""))
            
            # Try different possible package directory names
            # e.g., solarian-ur5 -> solarian_ur5, solarian-koch -> solarian_koch
            possible_package_names = [
                f"solarian_{module_name.replace('-', '_')}",
                module_name.replace('-', '_'),
                f"solarian-{module_name}",
            ]
            
            # First, try to find the package by importing it
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
            
            # Fallback: search in the distribution location
            # Look for conf.yaml in common locations
            search_paths = [
                dist_location / pkg_name / "conf.yaml"
                for pkg_name in possible_package_names
            ]
            # Also check if conf.yaml is directly in the distribution root
            search_paths.append(dist_location / "conf.yaml")
            
            for conf_path in search_paths:
                if conf_path.exists() and conf_path.is_file():
                    return conf_path
            
        except importlib.metadata.PackageNotFoundError:
            print(f"Package {plugin_package} not found. Make sure it's installed.")
            return None
        except Exception as e:
            print(f"Error finding conf.yaml: {e}")
            return None
        
        return None

