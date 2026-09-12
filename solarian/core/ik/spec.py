"""Declarative IK backend specifications.

A module that wants Cartesian (IK) control declares an ``ik_backend`` attribute
holding one of these specs instead of building a backend by hand. The core
:class:`~solarian.core.arm_module.ArmModule` resolves the spec (calling ``.build()``)
the first time it connects a port with Cartesian IK actuators, so modules using a
built-in backend like mink never have to write backend-construction code.

Example::

    from solarian.core.arm_module import ArmModule

    class MyArm(ArmModule):
        ik_backend = MinkIK(
            mjcf="so_arm101_mj_description",
            joints={"shoulder_pan": "1", "shoulder_lift": "2", ...},
            frame="tcp",
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field

from solarian.core.ik.base import DifferentialIKBackend


@dataclass
class MinkIK:
    """Parameters for a mink (MuJoCo) differential IK backend.

    Attributes:
        mjcf: Robot description name (``robot_descriptions``) or path to an MJCF.
        joints: Ordered mapping of joint-space actuator name -> MuJoCo joint
            name. The order defines the joint vector used by the solver.
        frame: Name of the TCP frame (site/body/geom) in the (possibly
            composed) model.
        frame_type: Kind of ``frame`` ("site", "body" or "geom").
        position_cost / orientation_cost: Frame-task tracking weights.
        posture_cost: Regularization pulling toward the rest posture (helps a
            5-DOF arm resolve a 6D target gracefully).
        lm_damping / damping / solver: QP damping and solver knobs.
        attachments: Optional ``ModuleAttachment`` list composed onto ``mjcf``
            as the world before the backend loads. Use this when the TCP site
            lives on a tool model mated to the arm (e.g. gripper ``tcp``).
    """

    mjcf: str
    joints: dict[str, str]
    frame: str
    frame_type: str = "site"
    position_cost: float = 1.0
    orientation_cost: float = 0.5
    posture_cost: float = 1e-2
    lm_damping: float = 1.0
    solver: str = "daqp"
    damping: float = 1e-12
    attachments: list = field(default_factory=list)

    def build(self) -> DifferentialIKBackend:
        # Imported lazily so `import solarian` never pulls in mink/mujoco.
        from solarian.core.ik.mink_backend import MinkBackend

        model = None
        if self.attachments:
            from solarian.sim.configs import SceneConfig
            from solarian.sim.scene import compose_scene

            model = compose_scene(SceneConfig(
                world_mjcf=self.mjcf,
                attachments=list(self.attachments),
            ))

        return MinkBackend(
            mjcf=self.mjcf,
            joint_actuator_names=list(self.joints.keys()),
            joint_model_names=list(self.joints.values()),
            frame_name=self.frame,
            frame_type=self.frame_type,
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
            posture_cost=self.posture_cost,
            lm_damping=self.lm_damping,
            solver=self.solver,
            damping=self.damping,
            model=model,
        )
