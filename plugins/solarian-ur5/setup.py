"""
Minimal setup.py so we can run a post-install notice on Darwin/arm64.
All package metadata is in pyproject.toml.
"""
import platform

from setuptools import setup
from setuptools.command.install import install


_UR_RTDE_DARWIN_ARM64_MSG = """
----------------------------------------------------------------------
solarian-ur5 on macOS (Darwin) arm64: ur-rtde must be installed from source.

This plugin does not install ur-rtde on this platform (no pre-built wheel).
To use the UR5 plugin, install ur-rtde from source:

  1. Install Boost:  brew install boost
  2. Clone and build: https://sdurobotics.gitlab.io/ur_rtde/installation/installation.html

Full instructions: https://gitlab.com/sdurobotics/ur_rtde
----------------------------------------------------------------------
"""


class InstallWithNotice(install):
    def run(self):
        install.run(self)
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print(_UR_RTDE_DARWIN_ARM64_MSG)


setup(cmdclass={"install": InstallWithNotice})
