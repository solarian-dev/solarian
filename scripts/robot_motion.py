#!/usr/bin/env python3
"""Drive robot arms around a small Cartesian circle through the relay, as teleop does.

A repeatable load for profiling the robot. Every tick of every arm then runs
IK against a moving target. Each arm starts and ends at its measured pose, and
the circle passes through that pose: offsets are ``radius * (sin, 1 - cos)``
in the arm's base x and z.

Run it on the controller machine, not the robot, so its own CPU is not part of
what is measured:

    python scripts/robot_motion.py --relay turing.local --arms right_arm left_arm
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
import zlib

import msgpack
import zmq

AXES = ("x", "y", "z", "rx", "ry", "rz")
MODULE_ACTION = 1
SERVO_L = 12


def _scalar(entry) -> float:
    value = entry["value"] if isinstance(entry, dict) else entry
    if isinstance(value, (bytes, bytearray)):
        value = pickle.loads(zlib.decompress(value))
    return float(value)


def read_poses(sock: zmq.Socket, arms: list[str], timeout_s: float) -> dict[str, list[float]]:
    poses: dict[str, list[float]] = {}
    deadline = time.monotonic() + timeout_s
    while len(poses) < len(arms) and time.monotonic() < deadline:
        if not sock.poll(200):
            continue
        msg = msgpack.unpackb(sock.recv())
        observation = (msg.get("data") or {}).get("observation") or {}
        for arm in arms:
            fields = observation.get(arm) or {}
            if all(axis in fields for axis in AXES):
                poses[arm] = [_scalar(fields[axis]) for axis in AXES]
    return poses


def servo_message(poses: dict[str, list[float]]) -> bytes:
    now = time.time()
    action = {
        "action_type": MODULE_ACTION,
        "data": {
            arm: {
                "action_type": SERVO_L,
                "data": {axis: {"value": v, "timestamp": now} for axis, v in zip(AXES, pose)},
                "params": {},
            }
            for arm, pose in poses.items()
        },
    }
    return msgpack.packb(
        {
            "robot_timestamp": 0,
            "server_timestamp": 0,
            "controller_timestamp": time.time_ns(),
            "data": {"action": action},
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--relay", default="turing.local")
    parser.add_argument("--action-port", type=int, default=5556, help="relay port controllers send actions to")
    parser.add_argument("--state-port", type=int, default=5557, help="relay port controllers read state from")
    parser.add_argument("--arms", nargs="+", default=["right_arm", "left_arm"])
    parser.add_argument("--radius", type=float, default=0.015, help="m")
    parser.add_argument("--period", type=float, default=4.0, help="s per lap")
    parser.add_argument("--rate", type=float, default=30.0, help="Hz, the controller's send rate")
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()
    if not 0.0 < args.radius <= 0.05:
        parser.error("--radius must be in (0, 0.05] m")

    ctx = zmq.Context()
    state = ctx.socket(zmq.SUB)
    state.setsockopt(zmq.SUBSCRIBE, b"")
    state.setsockopt(zmq.LINGER, 0)
    state.connect(f"tcp://{args.relay}:{args.state_port}")
    actions = ctx.socket(zmq.PUSH)
    actions.setsockopt(zmq.LINGER, 1000)
    actions.connect(f"tcp://{args.relay}:{args.action_port}")

    start = read_poses(state, args.arms, timeout_s=10.0)
    missing = [arm for arm in args.arms if arm not in start]
    if missing:
        print(f"No pose from {missing} within 10 s; is the robot running?", file=sys.stderr)
        return 1
    for arm, pose in start.items():
        print(f"{arm} start pose {[round(v, 4) for v in pose]}")

    period = 1.0 / args.rate
    t0 = time.monotonic()
    tick = 0
    try:
        while True:
            t = time.monotonic() - t0
            if t >= args.seconds:
                break
            theta = 2.0 * math.pi * t / args.period
            dx = args.radius * math.sin(theta)
            dz = args.radius * (1.0 - math.cos(theta))
            targets = {}
            for arm, pose in start.items():
                target = list(pose)
                target[0] += dx
                target[2] += dz
                targets[arm] = target
            actions.send(servo_message(targets))
            tick += 1
            time.sleep(max(0.0, t0 + tick * period - time.monotonic()))
    finally:
        actions.send(servo_message(start))
        time.sleep(0.5)
        actions.close()
        state.close()
        ctx.term()
    print(f"Sent {tick} waypoints over {args.seconds:.0f} s, back at the start poses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
