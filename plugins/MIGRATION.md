# Migration Guide: Built-in Actuators to Plugin Packages

This guide explains how to migrate from the built-in actuator/bus implementations to the new plugin-based architecture.

## Overview

The actuator and bus implementations (Feetech, Dynamixel, UR5) have been extracted from the core Solarian framework into separate installable plugin packages. This makes the framework more modular and allows independent versioning of hardware support.

## What Changed

### Before (Built-in)
- Actuators and buses were in `solarian/core/`
- Entry points were in main `pyproject.toml`
- All code shipped with Solarian

### After (Plugins)
- Each actuator/bus is a separate package
- Plugins register themselves via entry points
- Install only what you need: `pip install solarian-feetech`

## Migration Steps

### 1. Install Plugin Packages

```bash
# Install all official plugins
pip install solarian-feetech solarian-dynamixel solarian-ur5

# Or install only what you need
pip install solarian-feetech
```

### 2. Update Your Code

**Old way (still works, but deprecated):**
```python
from solarian.core.feetech_servo import FeetechServo
```

**New way (recommended):**
```python
from solarian.plugins import actuators
FeetechServo = actuators.get("feetech")
```

### 3. Verify Installation

```bash
solarian list-plugins
```

You should see:
```
📦 MODULES
  ...

🦾 ACTUATORS
  • feetech
    ├─ solarian_feetech.actuator:FeetechServo
    └─ bus: feetech
  • dynamixel
    ...
  • ur5_joint
    ...
  • ur5_cartesian
    ...

🔌 ACTUATOR BUSES
  • feetech
    └─ solarian_feetech.bus:FeetechBus
  ...
```

## Plugin Package Structure

Each plugin follows this structure:

```
solarian-feetech/
├── pyproject.toml          # Package config with entry points
├── README.md              # Plugin documentation
├── solarian_feetech/
│   ├── __init__.py
│   ├── actuator.py        # Actuator class
│   └── bus.py             # Bus implementation
└── .gitignore
```

## Setting Up Plugin Repositories

Each plugin should be moved to its own repository:

1. **Create repository:**
   ```bash
   # On GitHub/GitLab, create: solarian-feetech
   ```

2. **Move plugin code:**
   ```bash
   cd plugins/solarian-feetech
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourorg/solarian-feetech.git
   git push -u origin main
   ```

3. **Set up CI/CD** (optional, for PyPI publishing):
   - GitHub Actions workflow
   - Automated testing
   - Version tagging

4. **Publish to PyPI:**
   ```bash
   python -m build
   twine upload dist/*
   ```

## Benefits

1. **Modularity:** Install only what you need
2. **Independent versioning:** Each plugin can be updated separately
3. **Community contributions:** Easier to add new hardware support
4. **Smaller core:** Solarian framework stays lightweight
5. **Better testing:** Each plugin can have its own test suite

## Backward Compatibility

The old import paths still work during a transition period, but they're deprecated:

```python
# Still works, but will be removed in future versions
from solarian.core.feetech_servo import FeetechServo
```

**Recommendation:** Update to the plugin registry pattern as soon as possible.

## Creating Your Own Plugin

See `solarian/templates/third_party_module_example/` for a complete example of creating a plugin package.

## Questions?

- Check each plugin's README.md
- See `solarian/plugins/README.md` for general plugin information
- Review `solarian/templates/` for examples
