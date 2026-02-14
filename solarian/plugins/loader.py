"""
Plugin Loader Utilities

Helpers for loading modules from settings configuration.
"""

from __future__ import annotations

from typing import Dict, Type, Any, Optional, TYPE_CHECKING
from multiprocessing.managers import SharedMemoryManager

from solarian.plugins.registry import modules, actuator_buses
from solarian.core.configs import ModuleConfig

if TYPE_CHECKING:
    from solarian.core.component_bus import ComponentBus


def get_module_class(module_name: str, module_config: ModuleConfig) -> Type:
    """
    Get a module class by looking up in the registry.
    
    First checks if the module type is registered via entry points.
    Falls back to MODULE_CLASSES in settings for backward compatibility.
    
    Args:
        module_name: Name of the module instance
        module_config: The module's configuration
        
    Returns:
        The module class
        
    Raises:
        ValueError: If module class not found
    """
    module_type = module_config.type
    
    # First, try the entry points registry
    module_cls = modules.get(module_type)
    if module_cls is not None:
        return module_cls
    
    # Fallback: try settings.MODULE_CLASSES for backward compatibility
    try:
        from solarian.conf import settings
        from solarian.utils import get_class
        
        if hasattr(settings, 'MODULE_CLASSES') and module_name in settings.MODULE_CLASSES:
            class_path = settings.MODULE_CLASSES[module_name]
            return get_class(class_path)
    except Exception:
        pass
    
    # Not found anywhere
    available = ', '.join(modules.available())
    raise ValueError(
        f"Module type '{module_type}' not found. "
        f"Available modules: [{available}]. "
        f"Register modules via entry points or settings.MODULE_CLASSES."
    )


def create_module(
    module_name: str,
    module_config: ModuleConfig,
    component_bus: ComponentBus,
    shm_manager: SharedMemoryManager
):
    """
    Create a module instance from configuration.
    
    Args:
        module_name: Name of the module instance
        module_config: The module's configuration
        component_bus: The component bus for hardware communication
        shm_manager: Shared memory manager for IPC
        
    Returns:
        An instance of the module
    """
    module_cls = get_module_class(module_name, module_config)
    return module_cls(
        name=module_name,
        cfg=module_config,
        component_bus=component_bus,
        shm_manager=shm_manager
    )


def list_available_plugins() -> Dict[str, Dict[str, str]]:
    """
    List all available plugins organized by type.
    
    Returns:
        Dict with 'modules' and 'actuator_buses' keys, each containing
        a dict mapping plugin names to their class paths.
    """
    result = {
        'modules': {},
        'actuator_buses': {}
    }
    
    for name in modules.available():
        cls = modules.get(name)
        if cls:
            result['modules'][name] = f"{cls.__module__}:{cls.__name__}"
    
    for name in actuator_buses.available():
        cls = actuator_buses.get(name)
        if cls:
            result['actuator_buses'][name] = f"{cls.__module__}:{cls.__name__}"
    
    return result

