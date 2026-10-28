"""Clearance between an arm, the tool it carries, and the world, from a robot scene.

The IK model of an arm is the arm alone: it can drive the gripper straight into
the forearm and report success. The robot's composed scene has the gripper's
meshes mated on the flange, so it is the place to ask how close a joint
configuration brings any two parts that are not bolted together.
"""

from __future__ import annotations

import numpy as np

from solarian.sim.configs import SceneConfig


class CollisionGuard:
    """Nearest approach over every pair of collision geoms that can meet.

    MuJoCo measures mesh geoms by their convex hulls, which only ever makes a
    part larger than it is, so a clearance here errs on the safe side. Geoms on
    bodies within *adjacent_links* steps of each other in the kinematic tree are
    never compared: neighbouring links, and the gripper and the horn it is
    screwed to, touch by construction. World geoms (the floor, a table) are
    compared with everything on the arm's chain past that same distance.

    The tool's own joints (gripper fingers on slides) are not commanded by the
    sweep and may sit anywhere in their travel, so every pair they move is
    measured at each combination of their limits and the worst one counts.
    """

    def __init__(
        self,
        scene: SceneConfig,
        arm_module: str,
        joint_names: list[str],
        adjacent_links: int = 2,
    ):
        import mujoco

        from solarian.sim.scene import compose_scene

        self._mujoco = mujoco
        model = compose_scene(scene)
        self._model = model
        self._data = mujoco.MjData(model)
        prefix = next(a.prefix for a in scene.attachments if a.module == arm_module)
        joints = [model.joint(prefix + name) for name in joint_names]
        self._qpos = np.array([int(j.qposadr[0]) for j in joints])

        parent = model.body_parentid
        depth = np.zeros(model.nbody, dtype=int)
        for body in range(1, model.nbody):
            depth[body] = depth[parent[body]] + 1

        def tree_steps(a: int, b: int) -> int:
            steps = 0
            while a != b:
                if depth[a] < depth[b]:
                    a, b = b, a
                a = parent[a]
                steps += 1
            return steps

        base = int(joints[0].bodyid[0])
        while parent[base] != 0:
            base = int(parent[base])

        def on_chain(body: int) -> bool:
            while body != 0:
                if body == base:
                    return True
                body = int(parent[body])
            return False

        geoms = [
            g
            for g in range(model.ngeom)
            if (model.geom_contype[g] or model.geom_conaffinity[g])
            and (model.geom_bodyid[g] == 0 or on_chain(int(model.geom_bodyid[g])))
        ]
        body_of = model.geom_bodyid
        pairs = [
            (g1, g2)
            for i, g1 in enumerate(geoms)
            for g2 in geoms[i + 1:]
            if tree_steps(int(body_of[g1]), int(body_of[g2])) > adjacent_links
        ]

        arm_joints = {int(j.id) for j in joints}
        tool = [
            j
            for j in range(model.njnt)
            if j not in arm_joints
            and model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            and model.jnt_limited[j]
            and on_chain(int(model.jnt_bodyid[j]))
        ]
        self._tool_qpos = np.array([int(model.jnt_qposadr[j]) for j in tool], dtype=int)
        grid = np.meshgrid(*[model.jnt_range[j] for j in tool], indexing="ij")
        self._tool_poses = np.stack([g.ravel() for g in grid], axis=-1) if tool else np.zeros((1, 0))

        def moved_by_tool(body: int) -> bool:
            while body != 0:
                if any(int(model.jnt_bodyid[j]) == body for j in tool):
                    return True
                body = int(parent[body])
            return False

        self._fixed_pairs = [
            p for p in pairs if not (moved_by_tool(int(body_of[p[0]])) or moved_by_tool(int(body_of[p[1]])))
        ]
        self._tool_pairs = [p for p in pairs if p not in self._fixed_pairs]

    def clearance(self, q, horizon: float = 0.1) -> tuple[float, str]:
        """``(metres, "body / body")`` of the closest pair at joint angles *q*.

        Distances past *horizon* are not resolved -- the answer is then just
        *horizon* -- which keeps each pair's test cheap.
        """
        mujoco, model, data = self._mujoco, self._model, self._data
        data.qpos[self._qpos] = np.asarray(q, dtype=np.float64)
        best, where = float(horizon), ""
        for k, tool_q in enumerate(self._tool_poses):
            data.qpos[self._tool_qpos] = tool_q
            mujoco.mj_kinematics(model, data)
            for g1, g2 in self._tool_pairs if k else self._fixed_pairs + self._tool_pairs:
                dist = mujoco.mj_geomDistance(model, data, g1, g2, best, None)
                if dist < best:
                    best = float(dist)
                    where = (
                        f"{model.body(model.geom_bodyid[g1]).name} / "
                        f"{model.body(model.geom_bodyid[g2]).name}"
                    )
        return best, where
