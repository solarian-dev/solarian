import os
import sys

from solarian._platform_deps import ensure_atomics, ensure_xformers

# MuJoCo picks its GL backend once, when first imported. GLFW needs a usable
# display, which a Pi over SSH lacks (or forwards to a host that cannot make
# its GL context); EGL renders offscreen on the GPU either way. Set here so it
# precedes any project module that imports mujoco.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

ensure_atomics()
ensure_xformers()
