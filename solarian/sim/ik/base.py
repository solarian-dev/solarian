"""Backwards-compatible re-export.

The differential IK interface now lives in :mod:`solarian.core.ik.base` (the IK
infrastructure is part of Solarian core so it can drive real hardware as well as
simulation). This module re-exports it for any existing imports.
"""

from solarian.core.ik.base import DifferentialIKBackend

__all__ = ["DifferentialIKBackend"]
