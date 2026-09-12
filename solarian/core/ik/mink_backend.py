"""mink / MuJoCo differential IK backend.

Wraps `mink <https://github.com/kevinzakka/mink>`_ (a MuJoCo-based differential
IK library) behind the framework-agnostic
:class:`~solarian.core.ik.base.DifferentialIKBackend` interface. It loads a
MuJoCo model of the arm (shared with the simulator), builds a Cartesian frame
task for the end-effector plus a posture task for regularization, and solves one
velocity-level IK step per call.

``mink`` and ``mujoco`` are imported lazily so importing :mod:`solarian` (or even
:mod:`solarian.core.ik`) never requires the optional ``[sim]`` extras; they are
only needed once a :class:`MinkBackend` is actually instantiated.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from solarian.core.ik.base import DifferentialIKBackend

import logging

logger = logging.getLogger(__name__)


class MinkBackend(DifferentialIKBackend):
    """Differential IK for an arm described by a MuJoCo (MJCF) model.

    Args:
        mjcf: MJCF reference resolved via
            :func:`solarian.sim.scene.resolve_mjcf` (an ``.xml`` path or a
            ``robot_descriptions`` module name, e.g. ``"so_arm101_mj_description"``).
        joint_actuator_names: Ordered names of the module's joint *actuators*
            that IK controls. Defines the ordering of ``q`` in :meth:`set_q` /
            :meth:`integrate`.
        joint_model_names: The MuJoCo joint names corresponding, in the same
            order, to ``joint_actuator_names``.
        frame_name: TCP frame (site/body/geom) in the model.
        frame_type: ``"site"`` (default), ``"body"`` or ``"geom"``.
        position_cost / orientation_cost: Frame-task weights. For an arm with
            fewer than 6 DOF, keep ``position_cost`` above ``orientation_cost`` so
            reachable positions are prioritized over exact orientation.
        posture_cost: Weight of the posture (regularization) task that resolves
            redundancy / DOF-deficiency by staying near the home configuration.
        lm_damping: Levenberg-Marquardt damping on the frame task (robust near
            singularities / for over-constrained tasks).
        solver: QP solver name understood by ``qpsolvers`` (``"daqp"`` default).
        damping: Tikhonov damping passed to ``mink.solve_ik``.
        model: Optional pre-composed ``MjModel``. When set, ``mjcf`` is only
            used in error messages (the backend does not reload it).
    """

    def __init__(
        self,
        mjcf: str,
        joint_actuator_names: Sequence[str],
        joint_model_names: Sequence[str],
        frame_name: str,
        frame_type: str = "site",
        position_cost: float = 1.0,
        orientation_cost: float = 0.5,
        posture_cost: float = 1e-2,
        lm_damping: float = 1.0,
        solver: str = "daqp",
        damping: float = 1e-12,
        model=None,
    ):
        import mujoco
        import mink

        from solarian.sim.scene import resolve_mjcf

        if len(joint_actuator_names) != len(joint_model_names):
            raise ValueError(
                "joint_actuator_names and joint_model_names must be the same length"
            )

        self._joint_actuator_names = list(joint_actuator_names)
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.solver = solver
        self.damping = damping

        self.model = model if model is not None else mujoco.MjModel.from_xml_path(
            resolve_mjcf(mjcf)
        )

        # Resolve qpos / dof addresses and radian limits of the controlled joints.
        qpos_ids: list[int] = []
        dof_ids: list[int] = []
        limits: dict[str, tuple[float, float]] = {}
        for act_name, model_name in zip(self._joint_actuator_names, joint_model_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, model_name)
            if jid < 0:
                raise ValueError(
                    f"Joint '{model_name}' not found in model for MJCF '{mjcf}'"
                )
            qpos_ids.append(int(self.model.jnt_qposadr[jid]))
            dof_ids.append(int(self.model.jnt_dofadr[jid]))
            lo, hi = self.model.jnt_range[jid]
            limits[act_name] = (float(lo), float(hi))
        self._qpos_ids = np.asarray(qpos_ids, dtype=int)
        self._dof_ids = np.asarray(dof_ids, dtype=int)
        self._limits = limits

        self.configuration = mink.Configuration(self.model)
        self.configuration.update(self.configuration.q)
        self._qfull = np.asarray(self.configuration.q, dtype=float).copy()

        self.frame_task = mink.FrameTask(
            frame_name=frame_name,
            frame_type=frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=lm_damping,
        )
        self.posture_task = mink.PostureTask(model=self.model, cost=posture_cost)
        # Regularize toward the model's default (home) configuration.
        self.posture_task.set_target_from_configuration(self.configuration)
        self.tasks = [self.frame_task, self.posture_task]

    # -- DifferentialIKBackend interface -----------------------------------
    def set_q(self, q) -> None:
        self._qfull[self._qpos_ids] = np.asarray(q, dtype=float)
        self.configuration.update(self._qfull)

    def get_pose(self) -> np.ndarray:
        transform = self.configuration.get_transform_frame_to_world(
            self.frame_name, self.frame_type
        )
        translation = np.asarray(transform.translation(), dtype=float)
        rotvec = np.asarray(transform.rotation().log(), dtype=float)
        return np.concatenate([translation, rotvec])

    def solve(self, target_pose, dt: float):
        import mink

        target = np.asarray(target_pose, dtype=float)
        rotation = mink.SO3.exp(target[3:6])
        transform = mink.SE3.from_rotation_and_translation(rotation, target[:3])
        self.frame_task.set_target(transform)
        return mink.solve_ik(
            self.configuration, self.tasks, dt, self.solver, self.damping
        )

    def integrate(self, v, dt: float) -> np.ndarray:
        q_next = np.asarray(self.configuration.integrate(v, dt), dtype=float)
        return q_next[self._qpos_ids]

    @property
    def joint_actuator_names(self) -> list[str]:
        return list(self._joint_actuator_names)

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        return dict(self._limits)
