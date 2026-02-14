"""
Management command to list all installed Solarian plugins.
"""

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register('list-plugins', None)
class ListPluginsCommand(BaseCommand):
    help = 'List all installed Solarian plugins (modules, actuators, and buses).'

    def get_config_class(self):
        """This command doesn't require configuration."""
        return None

    def handle(self, cfg=None) -> None:
        from solarian.plugins import modules, actuator_buses, actuators
        
        print("\n" + "=" * 60)
        print("INSTALLED SOLARIAN PLUGINS")
        print("=" * 60)
        
        # List modules
        print("\n📦 MODULES")
        print("-" * 40)
        module_names = modules.available()
        if module_names:
            for name in sorted(module_names):
                cls = modules.get(name)
                if cls:
                    print(f"  • {name}")
                    print(f"    └─ {cls.__module__}:{cls.__name__}")
        else:
            print("  (no modules installed)")
        
        # List actuators
        print("\n🦾 ACTUATORS")
        print("-" * 40)
        actuator_names = actuators.available()
        if actuator_names:
            for name in sorted(actuator_names):
                cls = actuators.get(name)
                if cls:
                    bus_type = getattr(cls, '_bus_type', '?')
                    print(f"  • {name}")
                    print(f"    ├─ {cls.__module__}:{cls.__name__}")
                    print(f"    └─ bus: {bus_type}")
        else:
            print("  (no actuators installed)")
        
        # List actuator buses
        print("\n🔌 ACTUATOR BUSES")
        print("-" * 40)
        bus_names = actuator_buses.available()
        if bus_names:
            for name in sorted(bus_names):
                cls = actuator_buses.get(name)
                if cls:
                    print(f"  • {name}")
                    print(f"    └─ {cls.__module__}:{cls.__name__}")
        else:
            print("  (no actuator buses installed)")
        
        print("\n" + "=" * 60)
        print(f"Total: {len(module_names)} modules, {len(actuator_names)} actuators, {len(bus_names)} buses")
        print("=" * 60 + "\n")
        
        # Show how to add more
        print("💡 To add third-party plugins, install packages that declare")
        print("   entry points for 'solarian.modules', 'solarian.actuators', or 'solarian.actuator_buses'.")
        print()

