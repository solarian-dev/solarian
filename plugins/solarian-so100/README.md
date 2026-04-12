# Solarian-SO100

SO100 follower arm module plugin for the Solarian robotics framework.

## Installation

```bash
pip install solarian-so100
```

Or install from source:

```bash
cd solarian-so100
pip install -e .
```

## Usage

After installation, the module is automatically discovered by Solarian:

```python
from solarian.core.module import Module
from solarian.plugins import modules

# Get the module class
SO100Arm = modules.get("so100_follower_arm")

# Or import directly for best IDE support
from solarian_so100.module import SO100FollowerArmModule

class MyRobot:
    follower_arm: SO100FollowerArmModule = ...
```

## Configuration

The module uses Feetech STS3215 servos. See `conf.yaml` for the complete configuration.

## Requirements

- Solarian framework (>=0.1.0)
- solarian-feetech (>=0.1.0) - for Feetech servo support

## Repository

This plugin should be maintained in its own repository:
- GitHub: `https://github.com/yourorg/solarian-so100`
- Can be published to PyPI for easy installation

