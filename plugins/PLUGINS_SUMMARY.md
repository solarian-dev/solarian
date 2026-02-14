# Solarian Plugin Packages Summary

Complete list of all plugin packages for the Solarian robotics framework.

## Actuator/Bus Plugins

These plugins provide actuator classes and bus implementations for hardware communication.

| Plugin | Package Name | Entry Points | Dependencies |
|--------|--------------|--------------|--------------|
| **Feetech** | `solarian-feetech` | `feetech` (actuator, bus) | feetech-servo-sdk, lerobot |
| **Dynamixel** | `solarian-dynamixel` | `dynamixel` (actuator, bus) | dynamixel-sdk, lerobot |
| **UR5** | `solarian-ur5` | `ur5_joint`, `ur5_cartesian` (actuators), `ur5` (bus), `ur5_arm` (module) | urx, ur-rtde, lerobot |

## Module Plugins

These plugins provide complete robot modules (arms, grippers, mobile bases).

### Arm Modules

| Plugin | Package Name | Entry Point | Actuators Used |
|--------|--------------|-------------|----------------|
| **Koch Leader** | `solarian-koch` | `koch_leader_arm` | Dynamixel XL330-M077 |
| **SO100 Follower** | `solarian-so100` | `so100_follower_arm` | Feetech STS3215 |
| **UR5 Arm** | `solarian-ur5` | `ur5_arm` | UR5 Joint/Cartesian |

### Gripper Modules

| Plugin | Package Name | Entry Point | Actuators Used |
|--------|--------------|-------------|----------------|
| **UMI Evo** | `solarian-umi-evo` | `umi_evo_gripper` | Feetech STS3215 |

### Mobile Base Modules

| Plugin | Package Name | Entry Point | Actuators Used |
|--------|--------------|-------------|----------------|
| **LeKiwi** | `solarian-lekiwi` | `lekiwi_base` | Feetech STS3215 |

## Installation

Install all plugins:
```bash
pip install \
  solarian-feetech \
  solarian-dynamixel \
  solarian-ur5 \
  solarian-koch \
  solarian-so100 \
  solarian-umi-evo \
  solarian-lekiwi
```

Or install only what you need:
```bash
pip install solarian-feetech solarian-so100
```

## Usage

### Using the Registry

```python
from solarian.plugins import modules, actuators

# Get modules
SO100Arm = modules.get("so100_follower_arm")
UMIGripper = modules.get("umi_evo_gripper")

# Get actuators
FeetechServo = actuators.get("feetech")
```

### Direct Import (Best IDE Support)

```python
from solarian_so100.module import SO100FollowerArmModule
from solarian_umi_evo.module import UMIEvoGripperModule
from solarian_feetech.actuator import FeetechServo
```

## Repository Structure

Each plugin should be in its own repository:

```
solarian-feetech/          → github.com/yourorg/solarian-feetech
solarian-dynamixel/        → github.com/yourorg/solarian-dynamixel
solarian-ur5/              → github.com/yourorg/solarian-ur5
solarian-koch/             → github.com/yourorg/solarian-koch
solarian-so100/            → github.com/yourorg/solarian-so100
solarian-ur5-arm/          → github.com/yourorg/solarian-ur5-arm
solarian-umi-evo/          → github.com/yourorg/solarian-umi-evo
solarian-lekiwi/           → github.com/yourorg/solarian-lekiwi
```

## Development

To develop plugins locally:

```bash
# Install a plugin in development mode
cd plugins/solarian-feetech
pip install -e .

# Install all plugins
cd plugins
for dir in solarian-*; do
    cd "$dir" && pip install -e . && cd ..
done
```

## Publishing

Each plugin can be published independently to PyPI:

```bash
cd plugins/solarian-feetech
python -m build
twine upload dist/*
```

