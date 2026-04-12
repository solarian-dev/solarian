# Solarian-LeKiwi

LeKiwi mobile base module plugin for the Solarian robotics framework.

## Installation

```bash
pip install solarian-lekiwi
```

Or install from source:

```bash
cd solarian-lekiwi
pip install -e .
```

## Usage

After installation, the module is automatically discovered by Solarian:

```python
from solarian.core.module import Module
from solarian.plugins import modules

# Get the module class
LeKiwiBase = modules.get("lekiwi_base")

# Or import directly for best IDE support
from solarian_lekiwi.module import LekiwiBaseModule

class MyRobot:
    base: LekiwiBaseModule = ...
```

## Configuration

The module uses Feetech STS3215 servos for wheel control and includes a front camera sensor.
See `conf.yaml` for the complete configuration.

## Requirements

- Solarian framework (>=0.1.0)
- solarian-feetech (>=0.1.0) - for Feetech servo support

## Repository

This plugin should be maintained in its own repository:
- GitHub: `https://github.com/yourorg/solarian-lekiwi`
- Can be published to PyPI for easy installation

