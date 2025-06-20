import importlib.util
import os


class SettingsLoader:
    _instance = None

    @classmethod
    def get_settings(cls):
        if cls._instance is None:
            # dynamically load settings from environment variable
            settings_module_name = os.getenv("PROTO_SETTINGS")
            if settings_module_name is None:
                raise ValueError("PROTO_SETTINGS environment variable not set")

            # load settings
            spec = importlib.util.spec_from_file_location("project_settings", settings_module_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls._instance = module.ProjectSettings()
        return cls._instance

# Create singleton instance
settings = SettingsLoader.get_settings()
