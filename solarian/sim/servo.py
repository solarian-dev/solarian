"""Simulated servo actuator.

Drop-in replacement for a hardware servo (e.g. ``FeetechServo``) that reads /
writes joint state in a MuJoCo world instead of over a serial bus. It exposes
the same ``pos`` / ``speed`` data descriptors so that observation/action
schemas are identical between real and simulated robots.
"""

from __future__ import annotations

from solarian.core.servo import Servo
from solarian.core.data import PosData, SpeedData

from solarian.sim.mujoco_bus import MujocoBus


class SimServo(Servo):
    """Position/velocity servo backed by a MuJoCo joint + actuator.

    The mapping from this component to a concrete MuJoCo joint/actuator (and
    the radian range that the normalized ``[-100, 100]`` command space maps to)
    is declared in the *same* module yaml used by the real hardware, under a
    ``sim:`` block on the actuator (consumed only when running under a
    ``SimRobot``)::

        shoulder_pan:
            id: 1
            model: sts3250        # real hardware fields, ignored in sim
            port: ${actuator_port}
            type: feetech
            mode: range_m100_100
            sim:
                joint: "1"        # MJCF joint name (post-prefix)
                actuator: "1"     # MJCF actuator name (post-prefix)
                range_min: -1.91986   # radians mapped to -100
                range_max: 1.91986    # radians mapped to +100

    Omitting ``joint``/``actuator`` makes the servo *virtual*: it holds the last
    commanded value and echoes it back on read (useful when the robot schema
    has a joint that the chosen MJCF model does not physically simulate).
    """

    bus_class = MujocoBus

    pos = PosData()
    speed = SpeedData()
