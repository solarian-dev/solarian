---
name: add-module
description: Add a hardware module (arm, gripper, camera, any actuator or sensor group) to this Solarian robot, from an installed plugin or as a local Module class, and wire its config, observation and action fields, deployment ports and simulation model. Use when the user wants the robot to have a new device.
---

# Add a module

Docs: `solarian docs show modules`, `robots`, `observations`, `actions`, `simulation`, `project-configs`.

Pick the module's **name** first, e.g. `gripper`. It is the robot attribute, the yaml filename, the key in observations and actions, and the name skills use. Keep it identical everywhere.

## 1. Get the module class

- **Published hardware**: `solarian install <plugin>` (e.g. `solarian install gripper1`). It pip-installs `solarian-<plugin>`, adds it to `pyproject.toml`, copies the plugin's `templates/**/*.yaml` into `conf/` when absent, and adds the plugin's section to `AGENTS.md` (re-read it). That section, or the plugin's README, names the module class to import and, when install says the plugin "ships no config template", the default yaml to copy by hand.
- **Your own device**: a `Module` subclass (or `ArmModule`, `GripperModule`) in `{{ package }}/modules/<file>.py`, declaring its actuators and sensors as class attributes (e.g. `shoulder_pan = FeetechServo()`). For a driver other projects will reuse, make a plugin instead: `solarian create-plugin module <name>`.

## 2. Attach it to the robot

```python
# {{ package }}/robot.py
class MyRobot(Robot):
    arm = MyArmModule()
    gripper = Gripper1FeetechModule()
```

Declare modules on the robot class itself: modules on a base class are ignored.

## 3. Its config

`conf/modules/<name>.yaml` is required and validated against the module's `config_class`. Start from the plugin's default yaml rather than writing one; `frequency` (Hz) is required. Device paths come from the deployment, never literally, and each actuator reads the module's port:

```yaml
# conf/modules/gripper.yaml
actuator_port: ${deployment.actuator_port_gripper}
frequency: 30
actuators:
    gripper:
        id: 1
        port: ${actuator_port}
        sim_joint: "gripper/left_manifold_to_rail"    # prefixed as in the scene
```

Add the key (`actuator_port_<name>`, or `camera_port_<name>` for a camera) to **every** `conf/deployment/*.yaml`. Modules on one servo bus share a port and need distinct ids. If the yaml was seeded or copied under another filename, rename it to `<name>.yaml`. `solarian doctor` reports a missing yaml or deployment key.

## 4. Observation and action fields

```python
# {{ package }}/observation.py
class GripperObs(ModuleObs):
    gripper = SensorField(MyRobot.gripper.gripper.pos)    # normalized: -100 closed, 100 open

class MyObservation(Observation):
    # ... existing modules ...
    gripper = GripperObs()
```

```python
# {{ package }}/action.py
from solarian.core.gripper_module import APERTURE_UNIT

class GripperAction(ModuleAction):
    gripper = ActuatorField(MyRobot.gripper.gripper.pos, unit=APERTURE_UNIT)    # commanded in mm

class MyAction(Action):
    # ... existing modules ...
    gripper = GripperAction()
```

Every module needs both; a camera gets an empty `class CameraAction(ModuleAction): pass`. Name a field after its actuator or sensor: an action's `data` is keyed by these names (`{"gripper": {"value": 40.0}}`). Imports: `solarian.core.observation` (`SensorField`, `CalculatedField`, `ModuleObs`, `Observation`), `solarian.core.action` (`ActuatorField`, `ModuleAction`, `Action`). Units: arm joints are normalized to [-100, 100]; Cartesian `x, y, z` in metres; a gripper is commanded in mm of aperture and observed normalized.

## 5. Simulation model

On the robot class, `scene = SceneConfig(...)` (from `solarian.sim`) attaches each module's MuJoCo model with `ModuleAttachment(module="<name>", mjcf=..., parent_body="world" or parent_site=..., prefix="<name>/")`, and cameras with `CameraMount(...)`. Each servo's `sim_joint` in the module yaml uses the prefixed joint name; a servo with none is virtual. See `solarian docs show simulation`.

## 6. Check it

```bash
solarian doctor
solarian run-server                  # background
solarian run-robot --sim --view      # the log shows "<name> started"
solarian run-controller              # background; stop all three when done
```

Nothing moves a new module until something sends it actions: a key handler in the controller or a skill (see the `add-skill` skill).

On hardware, only when the user asks: `solarian detect-device` finds the serial port, `solarian setup-module` and `solarian calibrate` prepare the device, then `solarian run-robot -d <profile>`.
