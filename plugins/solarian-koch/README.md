# Solarian-Koch

Koch leader arm module plugin for the Solarian robotics framework.

## Installation

```bash
pip install solarian-koch
```

Or install from source:

```bash
cd solarian-koch
pip install -e .
```

## Usage

After installation, the module is automatically discovered by Solarian:

```python
from solarian.core.module import Module
from solarian.plugins import modules

# Get the module class
KochLeaderArm = modules.get("koch_leader_arm")

# Or import directly for best IDE support
from solarian_koch.module import KochLeaderArmModule

class MyRobot:
    leader_arm: KochLeaderArmModule = ...
```

## Configuration

The module uses Dynamixel XL330-M077 servos. See `conf.yaml` for the complete configuration.

## Requirements

- Solarian framework (>=0.1.0)
- solarian-dynamixel (>=0.1.0) - for Dynamixel servo support

## Repository

This plugin should be maintained in its own repository:
- GitHub: `https://github.com/yourorg/solarian-koch`
- Can be published to PyPI for easy installation

