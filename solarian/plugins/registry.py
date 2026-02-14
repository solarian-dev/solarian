"""
Plugin Registry System

Automatic discovery and registration of Solarian plugins using Python entry points.
"""

import sys
from typing import Dict, Type, Optional, Any, Callable, TypeVar, cast, TYPE_CHECKING
from importlib.metadata import entry_points

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

# Import base classes for type hints (TYPE_CHECKING to avoid circular imports)
if TYPE_CHECKING:
    from solarian.core.actuator import Actuator
    from solarian.core.actuator_bus import ActuatorBus
    from solarian.core.module import Module


class PluginRegistry:
    """
    Base registry for discovering and managing plugins via entry points.
    
    Plugins are discovered lazily on first access and cached.
    
    Example:
        registry = PluginRegistry('solarian.modules')
        
        # Get a specific plugin class
        MyModule = registry.get('my_module')
        
        # List all available plugins
        print(registry.available())
    """
    
    def __init__(self, group: str, base_type: Optional[Type[Any]] = None):
        """
        Initialize the registry for a specific entry point group.
        
        Args:
            group: The entry point group name (e.g., 'solarian.modules')
            base_type: The base class type for type hints (e.g., Actuator, Module)
        """
        self.group = group
        self.base_type = base_type
        self._plugins: Dict[str, Type] = {}
        self._discovered = False
    
    def _discover(self) -> None:
        """Discover all plugins from entry points."""
        if self._discovered:
            return
            
        eps = entry_points()
        
        # Python 3.10+ returns SelectableGroups, earlier versions return dict
        if hasattr(eps, 'select'):
            group_eps = eps.select(group=self.group)
        else:
            group_eps = eps.get(self.group, [])
        
        for ep in group_eps:
            try:
                self._plugins[ep.name] = ep.load()
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Failed to load plugin '{ep.name}' from {self.group}: {e}"
                )
        
        self._discovered = True
    
    def register(self, name: str, cls: Type) -> None:
        """
        Manually register a plugin class.
        
        Useful for testing or programmatic registration without entry points.
        
        Args:
            name: The plugin name/identifier
            cls: The plugin class
        """
        self._plugins[name] = cls
    
    def get(self, name: str) -> Optional[Type[Any]]:
        """
        Get a plugin class by name.
        
        Args:
            name: The plugin name
            
        Returns:
            The plugin class (subclass of base_type), or None if not found
            
        Note:
            The return type is typed as the base class (e.g., Actuator, Module),
            so IDEs will provide autocomplete for base class methods and properties.
            The actual returned class will be a subclass (e.g., FeetechServo).
        """
        self._discover()
        cls = self._plugins.get(name)
        if cls is None:
            return None
        # Return with proper type hint if base_type is specified
        if self.base_type is not None:
            return cast(Type[self.base_type], cls)
        return cls
    
    def get_or_raise(self, name: str) -> Type[Any]:
        """
        Get a plugin class by name, raising if not found.
        
        Args:
            name: The plugin name
            
        Returns:
            The plugin class (subclass of base_type)
            
        Raises:
            KeyError: If plugin not found
        """
        cls = self.get(name)
        if cls is None:
            available = ', '.join(self.available())
            raise KeyError(
                f"Plugin '{name}' not found in {self.group}. "
                f"Available: [{available}]"
            )
        return cls
    
    def available(self) -> list[str]:
        """
        List all available plugin names.
        
        Returns:
            List of registered plugin names
        """
        self._discover()
        return list(self._plugins.keys())
    
    def all(self) -> Dict[str, Type]:
        """
        Get all registered plugins.
        
        Returns:
            Dict mapping names to plugin classes
        """
        self._discover()
        return dict(self._plugins)
    
    def __contains__(self, name: str) -> bool:
        """Check if a plugin is registered."""
        self._discover()
        return name in self._plugins
    
    def __getitem__(self, name: str) -> Type:
        """Get a plugin by name using bracket notation."""
        return self.get_or_raise(name)
    
    def __iter__(self):
        """Iterate over plugin names."""
        self._discover()
        return iter(self._plugins)
    
    def __len__(self) -> int:
        """Return the number of registered plugins."""
        self._discover()
        return len(self._plugins)


class ModuleRegistry(PluginRegistry):
    """
    Registry for robot modules (arms, grippers, cameras, etc.)
    
    Modules are the main building blocks of a robot system, similar to
    Django apps. They encapsulate actuators, sensors, and control logic.
    
    Example:
        # Get a module class - IDE knows it's a Module subclass
        SO100Arm = modules.get('so100_follower_arm')
        # ↑ IDE autocomplete works for Module methods!
        
        # Create instance
        arm = SO100Arm(name='arm', cfg=cfg, component_bus=bus, shm=shm)
    """
    
    def __init__(self):
        if TYPE_CHECKING:
            from solarian.core.module import Module
            base_type = Module
        else:
            base_type = None
        super().__init__('solarian.modules', base_type=base_type)
    
    def get(self, name: str) -> Optional["Type[Module]"]:
        """Get a module class by name. Returns Type[Module] for IDE autocomplete."""
        return super().get(name)
    
    def get_or_raise(self, name: str) -> "Type[Module]":
        """Get a module class by name, raising if not found."""
        return super().get_or_raise(name)


class ActuatorBusRegistry(PluginRegistry):
    """
    Registry for actuator bus implementations.
    
    Actuator buses handle communication with specific hardware types
    (Feetech servos, Dynamixel servos, UR5 robots, etc.)
    
    Third-party packages can provide custom bus implementations for
    new hardware types.
    
    Example:
        # Get a bus class - IDE knows it's an ActuatorBus subclass
        FeetechBus = actuator_buses.get('feetech')
        # ↑ IDE autocomplete works for ActuatorBus methods!
        
        # Create instance
        bus = FeetechBus.create(port='/dev/ttyUSB0', components=..., calibration=...)
    """
    
    def __init__(self):
        if TYPE_CHECKING:
            from solarian.core.actuator_bus import ActuatorBus
            base_type = ActuatorBus
        else:
            base_type = None
        super().__init__('solarian.actuator_buses', base_type=base_type)
    
    def get(self, name: str) -> Optional["Type[ActuatorBus]"]:
        """Get an actuator bus class by name. Returns Type[ActuatorBus] for IDE autocomplete."""
        return super().get(name)
    
    def get_or_raise(self, name: str) -> "Type[ActuatorBus]":
        """Get an actuator bus class by name, raising if not found."""
        return super().get_or_raise(name)


class ActuatorRegistry(PluginRegistry):
    """
    Registry for actuator classes.
    
    Each actuator class is paired with an actuator bus. When you use an
    actuator class in a module, the corresponding bus is automatically used.
    
    Example:
        # Get an actuator class - IDE knows it's an Actuator subclass
        FeetechServo = actuators.get('feetech')
        # ↑ IDE autocomplete works for Actuator methods and properties!
        
        class MyModule(Module):
            motor1: FeetechServo = FeetechServo(id=1, model="sts3215")
    """
    
    def __init__(self):
        if TYPE_CHECKING:
            from solarian.core.actuator import Actuator
            base_type = Actuator
        else:
            base_type = None
        super().__init__('solarian.actuators', base_type=base_type)
    
    def get(self, name: str) -> Optional["Type[Actuator]"]:
        """Get an actuator class by name. Returns Type[Actuator] for IDE autocomplete."""
        return super().get(name)
    
    def get_or_raise(self, name: str) -> "Type[Actuator]":
        """Get an actuator class by name, raising if not found."""
        return super().get_or_raise(name)


# Singleton instances - import these to use
modules = ModuleRegistry()
actuator_buses = ActuatorBusRegistry()
actuators = ActuatorRegistry()

