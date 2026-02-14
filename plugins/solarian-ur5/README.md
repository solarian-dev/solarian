# Solarian-UR5

Universal Robots UR5 actuator, bus, and arm module plugin for the Solarian robotics framework.

This package provides everything needed to use UR5 robots:
- **Actuators**: UR5JointActuator and UR5CartesianActuator
- **Bus**: UR5Bus for hardware communication
- **Module**: UR5ArmModule - ready-to-use arm module

## Installation

```bash
pip install solarian-ur5
```

Or install from source:

```bash
cd solarian-ur5
pip install -e .
```

## Usage

### Option 1: Use the Pre-built Module (Recommended)

After installation, the module is automatically discovered:

```python
from solarian.plugins import modules

# Get the UR5 arm module
UR5Arm = modules.get("ur5_arm")

# Or import directly for best IDE support
from solarian_ur5.module import UR5ArmModule

class MyRobot:
    arm: UR5ArmModule = ...
```

### Option 2: Build Your Own Module

Use the actuators to build a custom module:

```python
from solarian.core.module import Module
from solarian_ur5.actuator import UR5JointActuator, UR5CartesianActuator

class MyCustomUR5Module(Module):
    # Joint-space control
    shoulder_pan: UR5JointActuator = UR5JointActuator(id=0, model="ur5")
    shoulder_lift: UR5JointActuator = UR5JointActuator(id=1, model="ur5")
    elbow: UR5JointActuator = UR5JointActuator(id=2, model="ur5")
    
    # Cartesian-space control (optional)
    x: UR5CartesianActuator = UR5CartesianActuator(id=0, model="ur5")
    y: UR5CartesianActuator = UR5CartesianActuator(id=1, model="ur5")
    z: UR5CartesianActuator = UR5CartesianActuator(id=2, model="ur5")
```

The `UR5Bus` is automatically used for communication - no configuration needed!

## Components

This package includes:

1. **Actuators** (`solarian_ur5.actuator`):
   - `UR5JointActuator` - For joint-space control
   - `UR5CartesianActuator` - For Cartesian-space control

2. **Bus** (`solarian_ur5.bus`):
   - `UR5Bus` - Handles communication with UR5 robots

3. **Module** (`solarian_ur5.module`):
   - `UR5ArmModule` - Complete 6-DOF arm module

## Supported Models

- UR5 (CB3 and e-Series)
- Other Universal Robots models (with minor modifications)

## Requirements

- Solarian framework (>=0.1.0)
- ur-rtde (>=1.5.6)

### macOS (Darwin) on Apple Silicon (arm64)

There are no pre-built wheels for **ur-rtde** on Darwin/arm64. You must install it from source:

1. Install Boost: `brew install boost`
2. Clone and build ur_rtde: see [Installation — ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/installation/installation.html) (clone from [GitLab](https://gitlab.com/sdurobotics/ur_rtde), then `git submodule update --init --recursive`, build with cmake/make and install).

git clone git@gitlab.com:sdurobotics/ur_rtde.git

cd ur_rtde

Activate your virtual environment, eg:

conda activate solarian

# Conda’s python must be the one CMake finds
export PATH="$CONDA_PREFIX/bin:$PATH"

mkdir build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DPYTHON_BINDINGS=ON \
  -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
make install


If you use the plugin without ur-rtde on this platform, you will get an `ImportError` with the same instructions.

## Repository

This plugin should be maintained in its own repository:
- GitHub: `https://github.com/yourorg/solarian-ur5`
- Can be published to PyPI for easy installation
