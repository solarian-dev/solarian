import importlib.util
import os
import sys


class SettingsLoader:
    _instance = None

    @classmethod
    def get_settings(cls):
        if cls._instance is None:
            # dynamically load settings from environment variable
            settings_module_name = os.getenv("SOLARIAN_SETTINGS")
            if settings_module_name is None:
                raise ValueError("SOLARIAN_SETTINGS environment variable not set")

            # ensure project root is on path so e.g. "from ur5.robot import ..." works
            settings_path = os.path.abspath(settings_module_name)
            project_root = os.path.dirname(os.path.dirname(settings_path))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # load settings
            spec = importlib.util.spec_from_file_location("project_settings", settings_module_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # cls._instance = module.ProjectSettings()
            cls._instance = module
        return cls._instance


class LazySettings:
    """
    A lazy proxy for the settings module. Delays loading until an attribute
    is actually accessed, so that importing solarian.conf doesn't require
    the SOLARIAN_SETTINGS environment variable to be set yet.
    """

    def __getattr__(self, name):
        # Load on first attribute access and cache the real module
        real_settings = SettingsLoader.get_settings()
        # Replace this proxy with the real module on the parent so future
        # accesses bypass __getattr__ entirely.
        object.__setattr__(self, "_wrapped", real_settings)
        return getattr(real_settings, name)

    def __repr__(self):
        try:
            real_settings = SettingsLoader.get_settings()
            return repr(real_settings)
        except ValueError:
            return "<LazySettings [not configured]>"


# Create lazy singleton — no settings are loaded until first attribute access
settings = LazySettings()
