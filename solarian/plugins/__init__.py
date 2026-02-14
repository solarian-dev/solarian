"""
Solarian Plugin System

Provides automatic discovery and registration of modules and actuator buses
via Python entry points. Similar to Django's app loading and database backend system.

Third-party packages can register plugins by declaring entry points in their
pyproject.toml:

    [project.entry-points."solarian.modules"]
    my_arm = "my_package.modules:MyArmModule"

    [project.entry-points."solarian.actuator_buses"]
    my_bus = "my_package.buses:MyActuatorBus"

Usage:
    from solarian.plugins import modules, actuator_buses
    
    # Get a module class
    SO100Arm = modules.get('so100_follower_arm')
    
    # Get an actuator bus class
    FeetechBus = actuator_buses.get('feetech')
    
    # List available plugins
    print(modules.available())
    print(actuator_buses.available())
"""

from solarian.plugins.registry import (
    ModuleRegistry,
    ActuatorBusRegistry,
    ActuatorRegistry,
    modules,
    actuator_buses,
    actuators,
)
from solarian.plugins.loader import (
    get_module_class,
    create_module,
    list_available_plugins,
)

__all__ = [
    # Registry classes
    'ModuleRegistry',
    'ActuatorBusRegistry',
    'ActuatorRegistry',
    # Singleton instances
    'modules',
    'actuator_buses',
    'actuators',
    # Loader utilities
    'get_module_class',
    'create_module',
    'list_available_plugins',
]

