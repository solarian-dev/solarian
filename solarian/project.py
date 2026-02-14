from dataclasses import dataclass, field
from solarian.utils import get_class


@dataclass(frozen=True, slots=True)
class Project:
    """Project configuration that references classes by dotted string paths.

    Classes are lazily resolved on first access, avoiding circular imports
    between the settings module and the framework.

    Example usage in settings.py::

        PROJECT = Project(
            robot_class='my_project.robot.MyRobot',
            controller_class='my_project.controller.MyController',
            observation_class='solarian.core.observation.Observation',
            action_class='solarian.core.action.Action',
        )
    """
    robot_class: str
    controller_class: str
    observation_class: str
    action_class: str

    # resolved cache (not part of __init__, not compared)
    _resolved: dict = field(default_factory=dict, init=False, repr=False, compare=False, hash=False)

    def _resolve(self, attr: str):
        if attr not in self._resolved:
            path = object.__getattribute__(self, attr)
            self._resolved[attr] = get_class(path)
        return self._resolved[attr]

    def __getattr__(self, name: str):
        if name in ('robot_class', 'controller_class', 'observation_class', 'action_class'):
            return self._resolve(name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
