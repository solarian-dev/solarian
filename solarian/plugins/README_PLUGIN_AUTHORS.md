# Guide for Plugin Authors

If you're creating a third-party Solarian plugin, here's how to ensure users get the best IDE experience.

## Plugin Structure

Your plugin should follow this structure:

```
solarian-mycustom/
├── pyproject.toml
├── solarian_mycustom/
│   ├── __init__.py
│   ├── actuator.py      # Your actuator class
│   └── bus.py            # Your bus class (if needed)
└── README.md
```

## Entry Points

Register your plugin in `pyproject.toml`:

```toml
[project.entry-points."solarian.actuators"]
my_custom_servo = "solarian_mycustom.actuator:MyCustomServo"

[project.entry-points."solarian.actuator_buses"]
my_custom_servo = "solarian_mycustom.bus:MyCustomBus"
```

## User Experience

### ✅ Recommended: Direct Import (Best IDE Support)

Users should import directly from your package:

```python
from solarian_mycustom.actuator import MyCustomServo

class MyModule(Module):
    motor: MyCustomServo = MyCustomServo(id=1)
    # ↑ Full IDE autocomplete!
```

**Why?** This gives users perfect IDE support - autocomplete, type checking, go-to-definition.

### ⚠️ Alternative: Registry Pattern (Limited IDE Support)

Users can also use the registry, but IDE support will be limited:

```python
from solarian.plugins import actuators
MyCustomServo = actuators.get("my_custom_servo")

class MyModule(Module):
    motor: MyCustomServo = MyCustomServo(id=1)
    # ↑ Limited autocomplete - IDE doesn't know the specific type
```

**When to use:** Only when you need dynamic plugin discovery at runtime.

## Documentation

In your plugin's README, recommend the direct import pattern:

```markdown
## Usage

```python
from solarian.core.module import Module
from solarian_mycustom.actuator import MyCustomServo

class MyModule(Module):
    motor: MyCustomServo = MyCustomServo(id=1, model="custom-001")
```
```

## Why Not Add to solarian.plugins.actuators?

The `solarian.plugins.actuators` module only includes **official Solarian plugins** (feetech, dynamixel, ur5). We can't know all future third-party plugins in advance, so they must be imported directly from their packages.

This is actually a **good thing** because:
1. **Explicit is better than implicit** - Users know exactly which plugins they're using
2. **No namespace pollution** - Official Solarian namespace stays clean
3. **Version control** - Each plugin can be versioned independently
4. **IDE support** - Direct imports give perfect autocomplete

## Example Plugin

See `solarian/templates/third_party_module_example/` for a complete example.
