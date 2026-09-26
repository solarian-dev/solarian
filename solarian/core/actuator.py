from solarian.core.component import Component
from solarian.core.configs import ActuatorConfig


class Actuator(Component):
    """
    Base class for actuators.

    Each actuator subclass must define `bus_class` pointing to an
    ActuatorBus subclass that handles its hardware communication.

    Structural facts belong on the class-level declaration::

        shoulder_pan = FeetechServo(model="sts3250")
        wrist_roll = FeetechServo(model="sts3215", is_continuous=True)
    """

    kind: str = "actuator"

    cfg: ActuatorConfig | None
    default_model: str | None = None
    default_mode: str | None = None
    #: False for virtual pose axes that never get a MotorCalibration row.
    uses_motor_calibration = True

    @classmethod
    def validate_act(cls, data: dict):
        for data_name, data_value in data.items():
            if data_name not in cls.data_names():
                raise ValueError(f"Invalid data name: '{data_name}' for actuator '{cls.__name__}'")
            data_class = getattr(cls, data_name)
            data_class.model_validate({"value": data_value})

    def __init__(
        self,
        cfg: ActuatorConfig | None = None,
        *,
        model: str | None = None,
        is_continuous: bool = False,
        mode: str | None = None,
        id: int | None = None,
    ):
        super().__init__(cfg)
        self.is_continuous = bool(is_continuous)
        self._declared_model = model
        self._declared_mode = mode
        self._declared_id = id

    def structural_defaults(self, attr_name: str) -> dict:
        """Class-declaration defaults to overlay onto yaml (yaml wins when set)."""
        from solarian.core.ik.actuator import CARTESIAN_AXIS_IDS, CartesianIKActuator

        model = (
            self._declared_model
            if self._declared_model is not None
            else type(self).default_model
        )
        mode = (
            self._declared_mode
            if self._declared_mode is not None
            else type(self).default_mode
        )
        axis_id = self._declared_id
        if axis_id is None and isinstance(self, CartesianIKActuator):
            axis_id = CARTESIAN_AXIS_IDS.get(attr_name)
        return {"model": model, "mode": mode, "id": axis_id}


def merge_actuator_config(
    marker: Actuator,
    name: str,
    yaml_cfg: ActuatorConfig | None,
    hardware_port: str | None,
) -> ActuatorConfig:
    """Yaml deployment fields over class-level structural defaults."""
    data = yaml_cfg.model_dump() if yaml_cfg is not None else {}
    defaults = marker.structural_defaults(name)
    if data.get("model") in (None, ""):
        data["model"] = defaults["model"]
    if data.get("mode") in (None, ""):
        data["mode"] = defaults["mode"]
    if yaml_cfg is None and defaults["id"] is not None:
        data["id"] = defaults["id"]
    if not data.get("port"):
        data["port"] = hardware_port or "ik"
    return ActuatorConfig.model_validate(data)
