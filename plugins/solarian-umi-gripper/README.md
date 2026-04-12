# Solarian-UMI-Evo

UMI Evo gripper module plugin for the Solarian robotics framework.

## Installation

```bash
pip install solarian-umi-evo
```

Or install from source:

```bash
cd solarian-umi-evo
pip install -e .
```

## Usage

After installation, the module is automatically discovered by Solarian:

```python
from solarian.core.module import Module
from solarian.plugins import modules

# Get the module class
UMIEvoGripper = modules.get("umi_evo_gripper")

# Or import directly for best IDE support
from solarian_umi_evo.module import UMIEvoGripperModule

class MyRobot:
    gripper: UMIEvoGripperModule = ...
```

## Configuration

The module uses Feetech STS3215 servos. See `conf.yaml` for the complete configuration.

## Requirements

- Solarian framework (>=0.1.0)
- solarian-feetech (>=0.1.0) - for Feetech servo support

## Repository

This plugin should be maintained in its own repository:
- GitHub: `https://github.com/yourorg/solarian-umi-evo`
- Can be published to PyPI for easy installation

