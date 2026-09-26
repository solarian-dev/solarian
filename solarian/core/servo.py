from solarian.core.actuator import Actuator
from solarian.core.sensor import Sensor


class Servo(Actuator, Sensor):
    # A servo reports position in its calibrated normalized space, not in
    # degrees or radians; which normalization is in the actuator's config, and
    # the session's robot structure records it per component.
    units = {"pos": "normalized"}
