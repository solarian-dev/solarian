# Solarian Plugin Packages

This directory contains separate plugin packages that were extracted from the core Solarian framework. Each plugin should be maintained in its own repository and can be published to PyPI.

## Actuator/Bus Plugins

### solarian-feetech
Feetech servo actuator and bus implementation.

**Location:** `plugins/solarian-feetech/`  
**Repository:** Should be moved to `https://github.com/yourorg/solarian-feetech`  
**Installation:** `pip install solarian-feetech`

### solarian-dynamixel
Dynamixel servo actuator and bus implementation.

**Location:** `plugins/solarian-dynamixel/`  
**Repository:** Should be moved to `https://github.com/yourorg/solarian-dynamixel`  
**Installation:** `pip install solarian-dynamixel`

### solarian-ur5
Universal Robots UR5 actuator, bus, and arm module implementation.

**Location:** `plugins/solarian-ur5/`  
**Repository:** Should be moved to `https://github.com/yourorg/solarian-ur5`  
**Installation:** `pip install solarian-ur5`

**Includes:**
- UR5JointActuator and UR5CartesianActuator
- UR5Bus
- UR5ArmModule (complete arm module)

## Module Plugins

### Arm Modules

#### solarian-koch
Koch leader arm module for teleoperation.

**Location:** `plugins/solarian-koch/`  
**Repository:** Should be moved to `https://github.com/yourorg/solarian-koch`  
**Installation:** `pip install solarian-koch`

#### solarian-so100
SO100 follower arm module for teleoperation.

**Location:** `plugins/solarian-so100/`  
**Repository:** Should be moved to `https://github.com/yourorg/solarian-so100`  
**Installation:** `pip install solarian-so100`


### Gripper Modules

#### solarian-umi-evo
UMI Evo gripper module.

**Location:** `plugins/solarian-umi-evo/`  
**Repository:** Should be moved to `https://github.com/yourorg/solarian-umi-evo`  
**Installation:** `pip install solarian-umi-evo`

### Mobile Base Modules

#### solarian-lekiwi
LeKiwi mobile base module with 3-wheel omni-directional drive.

**Location:** `plugins/solarian-lekiwi/`  
**Repository:** Should be moved to `https://github.com/yourorg/solarian-lekiwi`  
**Installation:** `pip install solarian-lekiwi`

## Migration from Built-in to Plugins

The module and actuator/bus implementations have been moved from `solarian/modules/` and `solarian/core/` to separate plugin packages. To use them:

1. **Install the plugins:**
   ```bash
   # Arm modules
   pip install solarian-koch solarian-so100 solarian-ur5
   
   # Gripper modules
   pip install solarian-umi-evo
   
   # Mobile base modules
   pip install solarian-lekiwi
   
   # Actuator/bus plugins
   pip install solarian-feetech solarian-dynamixel solarian-ur5
   ```

2. **Use from plugin registry:**
   ```python
   from solarian.plugins import modules
   SO100Arm = modules.get("so100_follower_arm")
   ```

3. **Or import directly (best IDE support):**
   ```python
   from solarian_so100.module import SO100FollowerArmModule
   ```

## Setting Up Plugin Repositories

Each plugin should be moved to its own repository:

1. Create a new repository (e.g., `solarian-koch`)
2. Copy the plugin directory contents
3. Initialize git and push
4. Set up CI/CD for PyPI publishing
5. Update documentation to point to the new repository

## Development

To develop plugins locally:

```bash
# Install in development mode
cd plugins/solarian-koch
pip install -e .

# Or install all plugins
cd plugins
for dir in solarian-*; do
    cd "$dir" && pip install -e . && cd ..
done
```

## Publishing to PyPI

Each plugin can be published independently:

```bash
cd plugins/solarian-koch
python -m build
twine upload dist/*
```
