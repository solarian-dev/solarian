# IDE Autocomplete Support

Solarian provides multiple ways to import actuators, each with different IDE autocomplete support.

**Key Insight:** The plugin registry pattern (`actuators.get()`) is inherently dynamic, so IDE support is limited. For best IDE experience, use direct imports from plugin packages.

## Option 1: Direct Import (Best IDE Support) ⭐

**Full autocomplete, type checking, and IDE navigation:**

```python
from solarian_feetech.actuator import FeetechServo
from solarian_dynamixel.actuator import DynamixelServo

class MyModule(Module):
    shoulder: FeetechServo = FeetechServo(id=1, model="sts3215")
    # ↑ Full autocomplete here! IDE knows it's FeetechServo
```

**Benefits:**
- ✅ Perfect IDE autocomplete (including subclass-specific methods)
- ✅ Type checking works
- ✅ Go-to-definition works
- ✅ Refactoring support

**When to use:** When you know which plugins you're using (most common case).

## Option 2: Plugin Registry (Dynamic Discovery)

**Good for dynamic loading with base class autocomplete:**

```python
from solarian.plugins import actuators

FeetechServo = actuators.get("feetech")
# ↑ IDE knows it's Type[Actuator] - autocomplete works for Actuator methods!

class MyModule(Module):
    shoulder: FeetechServo = FeetechServo(id=1, model="sts3215")
    # ↑ Autocomplete works for Actuator base class methods and properties
```

**Benefits:**
- ✅ Dynamic plugin discovery
- ✅ Works with third-party plugins
- ✅ IDE autocomplete for base class (Actuator) methods
- ✅ Type checking works (knows it's an Actuator subclass)
- ⚠️ IDE doesn't know the specific subclass (e.g., FeetechServo-specific methods)

**When to use:** When you need to dynamically discover plugins at runtime, or when base class methods are sufficient.

## Option 3: Hybrid Approach (Best of Both Worlds)

**Use direct import for type hints, registry for runtime:**

```python
from solarian.plugins import actuators
from solarian.plugins.actuators import FeetechServo  # For type hints

# Get from registry (dynamic)
FeetechServo = actuators.get("feetech")  # type: ignore[assignment]

class MyModule(Module):
    shoulder: FeetechServo = FeetechServo(id=1, model="sts3215")
    # ↑ Type hint gives IDE autocomplete, runtime uses registry
```

**Benefits:**
- ✅ Full IDE autocomplete (from type hint)
- ✅ Dynamic plugin discovery (from registry)
- ⚠️ Requires type: ignore comment

**When to use:** When you need both dynamic discovery and good IDE support.

## Option 4: Direct Plugin Package Import

**Import directly from plugin package:**

```python
from solarian_feetech.actuator import FeetechServo

class MyModule(Module):
    shoulder: FeetechServo = FeetechServo(id=1, model="sts3215")
    # ↑ Perfect autocomplete!
```

**Benefits:**
- ✅ Perfect IDE autocomplete
- ✅ Direct access to plugin code
- ❌ Requires knowing plugin package name
- ❌ Less portable (tied to specific package)

**When to use:** When you know the plugin package and want direct access.

## Recommended Patterns

### For All Plugins (Official and Third-Party)

Import directly from the plugin package:

```python
from solarian.core.module import Module
from solarian_feetech.actuator import FeetechServo
from solarian_dynamixel.actuator import DynamixelServo
from solarian_mycustom.actuator import MyCustomServo  # Third-party too!

class MyArmModule(Module):
    # Full IDE support - autocomplete, type checking, navigation
    shoulder: FeetechServo = FeetechServo(id=1, model="sts3215")
    elbow: FeetechServo = FeetechServo(id=2, model="sts3215")
    gripper: DynamixelServo = DynamixelServo(id=1, model="xl330-m077")
    custom_motor: MyCustomServo = MyCustomServo(id=1, model="custom-001")
```

**Why?** Direct imports give you the best IDE experience - full autocomplete for both base class and subclass-specific methods. This works the same for official and third-party plugins.

## Understanding the Trade-offs

### Plugin Registry Pattern
```python
from solarian.plugins import actuators
MyServo = actuators.get("my_servo")
# ↑ Returns Type[Actuator] - IDE knows it's an Actuator subclass
```

**Pros:**
- ✅ Dynamic discovery - works with any installed plugin
- ✅ No need to know plugin package names
- ✅ Great for runtime plugin loading
- ✅ IDE autocomplete for base class (Actuator) methods and properties
- ✅ Type checking works (knows it's an Actuator subclass)

**Cons:**
- ⚠️ IDE doesn't know the specific subclass (e.g., FeetechServo-specific methods)
- ⚠️ Limited go-to-definition (goes to base class, not subclass)

**Use when:** You need dynamic plugin discovery at runtime, or when base class methods are sufficient.

### Direct Import Pattern
```python
from solarian_mycustom.actuator import MyCustomServo
```

**Pros:**
- ✅ Perfect IDE autocomplete
- ✅ Full type checking
- ✅ Go-to-definition works
- ✅ Scales to any plugin

**Cons:**
- ❌ Requires knowing the plugin package name
- ❌ Less "magical" - explicit imports

**Use when:** You know which plugins you're using (most common case).

## Best Practice

**For development with known plugins:** Use direct imports from plugin packages for the best IDE experience (autocomplete for subclass-specific methods).

**For dynamic discovery:** Use the registry pattern - it now provides good IDE support for base class methods!

**Example - Registry with base class autocomplete:**
```python
from solarian.plugins import actuators

# IDE knows this is Type[Actuator] - autocomplete works for base class!
FeetechServo = actuators.get("feetech")

class MyModule(Module):
    # Can use all Actuator base class methods with autocomplete
    motor: FeetechServo = FeetechServo(id=1, model="sts3215")
    # motor.get_bus_type()  # ← IDE autocomplete works!
    # motor.get_bus_class()  # ← IDE autocomplete works!
```

**Example - Direct import for full autocomplete:**
```python
from solarian_feetech.actuator import FeetechServo

class MyModule(Module):
    motor: FeetechServo = FeetechServo(id=1, model="sts3215")
    # Can use both base class AND FeetechServo-specific methods
    # Full IDE autocomplete for everything!
```

**Recommendation:** Use direct imports for development (best IDE experience). Use the registry for dynamic discovery at runtime.

