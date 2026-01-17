# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Qudi-core** is a Python framework for modular multi-instrument measurement applications, primarily used for quantum diamond sensor magnetometry and ODMR (Optically Detected Magnetic Resonance) experiments. It provides a plugin-based architecture for hardware control, measurement logic, and GUI interfaces with automatic threading, configuration management, and state persistence.

- **License**: LGPL v3
- **Python**: 3.8-3.10 (installed dependencies require these specific versions)
- **Repository**: https://github.com/Ulm-IQO/qudi-core
- **Documentation**: https://ulm-iqo.github.io/qudi-core/

## Common Commands

### Running Qudi

```bash
# Activate Python environment first (example for venv)
cd qudi-env\Scripts
.\activate

# Start qudi (three equivalent methods)
qudi
python -m qudi.core
python src\qudi\runnable.py

# Command line options
qudi -h                           # Show help
qudi -c path\to\config.cfg       # Use specific config file
qudi -d                          # Debug mode (verbose logging)
qudi -g                          # Headless mode (no GUI)
qudi -l path\to\logdir           # Custom log directory
```

### Installation & Development

```bash
# Install in development mode (from repository root)
python -m pip install -e .

# Install from PyPI
python -m pip install qudi-core

# Register IPython kernel (needed for interactive console)
qudi-install-kernel

# Unregister IPython kernel
qudi-uninstall-kernel

# Launch config editor
qudi-config-editor
```

## Architecture

### Module Types

Qudi applications are composed of three types of modules, all inheriting from `qudi.core.module.Base`:

1. **Hardware Modules** (`qudi.core.module.HardwareBase`)
   - Device drivers with abstracted interfaces
   - Run on main thread by default (configurable)
   - Define interface contracts as abstract base classes

2. **Logic Modules** (`qudi.core.module.LogicBase`)
   - Application orchestration and measurement control
   - Run on dedicated threads by default
   - Connect to hardware via `Connector` objects

3. **GUI Modules** (`qudi.core.module.GuiBase`)
   - Optional Qt/PySide2 user interfaces
   - Run on Qt event loop
   - Connect to logic modules via `Connector` objects

### Module Lifecycle

All modules follow a state machine pattern (managed by fysom):

```
deactivated ↔ idle ↔ locked
```

- **on_activate()**: Initialize module (connect hardware, load state, etc.)
- **on_deactivate()**: Cleanup module (disconnect hardware, save state, etc.)

StatusVar attributes are automatically saved on deactivation and restored on activation.

### Key Framework Components

**ConfigOption** - Static parameters from YAML configuration:
```python
class MyLogic(LogicBase):
    _frequency = ConfigOption(name='center_frequency', default=2.87e9,
                              checker=lambda x: x > 0)
```

**StatusVar** - Persistent module state (auto-saved to JSON):
```python
class MyLogic(LogicBase):
    _count = StatusVar(name='count', default=0)
```

**Connector** - Type-safe inter-module connections:
```python
class MyLogic(LogicBase):
    _hardware = Connector(interface='MyHardwareInterface', name='hw')

    def measure(self):
        hw = self._hardware()  # Get connected hardware module
        return hw.read_value()
```

**Qt Signals** - Thread-safe asynchronous communication:
```python
class MyLogic(LogicBase):
    sigResultReady = QtCore.Signal(dict)

    def process(self, data):
        result = compute(data)
        self.sigResultReady.emit(result)  # Qt handles threading
```

### Configuration System

Applications are configured via YAML `.cfg` files with three main sections:

```yaml
global:
    startup_modules: [my_gui]
    default_data_dir: 'C:\Data\'
    stylesheet: 'qdark.qss'

hardware:
    my_device:
        module.Class: 'my_pkg.qudi.hardware.device.MyHardware'
        device_address: 'COM1'

logic:
    my_logic:
        module.Class: 'my_pkg.qudi.logic.logic.MyLogic'
        param: 42
        connectors:
            device: my_device

gui:
    my_gui:
        module.Class: 'my_pkg.qudi.gui.gui.MyGui'
        connectors:
            logic: my_logic
```

The `module.Class` entry uses dot-separated paths to locate the class to instantiate.

## Hardware Abstraction Pattern

Qudi uses interface-based hardware abstraction to keep logic independent of specific hardware implementations:

**1. Define Interface** (abstract base class):
```python
class MyDeviceInterface:
    def read(self) -> float: raise NotImplementedError()
    def write(self, value: float): raise NotImplementedError()
```

**2. Implement Hardware**:
```python
class MyDeviceHardware(HardwareBase, MyDeviceInterface):
    _port = ConfigOption(name='port', default='COM1')

    def on_activate(self):
        self._dev = connect(self._port)

    def on_deactivate(self):
        disconnect(self._dev)

    def read(self):
        return self._dev.read()

    def write(self, value):
        self._dev.write(value)
```

**3. Use in Logic**:
```python
class MyLogic(LogicBase):
    _device = Connector(interface='MyDeviceInterface')

    def measure(self):
        return self._device().read()
```

**Handling Multiple Interfaces**: When a hardware module implements multiple interfaces with overlapping method names, use `qudi.util.overload.OverloadedAttribute`:

```python
from qudi.util.overload import OverloadedAttribute

class MyHardware(InterfaceA, InterfaceB):
    start = OverloadedAttribute()

    @start.overload('InterfaceA')
    def start(self):
        # Start functionality for InterfaceA
        pass

    @start.overload('InterfaceB')
    def start(self):
        # Start functionality for InterfaceB
        pass
```

The `Connector` object automatically handles overloaded attributes transparently based on the specified interface type.

## Threading Model

- **LogicBase**: Dedicated QThread by default (configurable via `_threaded = True/False`)
- **HardwareBase**: Main thread by default (configurable)
- **GuiBase**: Qt event loop by default (configurable)

Thread-safe communication is handled automatically:
- Qt Signals marshal data between threads
- Connector proxies methods across thread boundaries
- Use `RecursiveMutex` for protecting shared data structures

## Data Persistence

**Module State** (StatusVar):
- Auto-saves to JSON on module deactivation
- Auto-restores on module activation
- Location: `<user_home>/qudi/<module_name>/`
- Supports complex types

**Measurement Data** (`qudi.util.datastorage`):
- CSV format with metadata headers
- NumPy `.npy` binary format
- PDF figure export via matplotlib
- Timestamped filenames: `YYYYMMDD-HHMM-SS_nametag`
- Auto-creates daily subdirectories
- Custom metadata support

## Directory Structure

```
qudi-core/
├── src/qudi/               # Core framework (accessible to Claude Code)
│   ├── core/              # Module management, threading, servers
│   ├── logic/             # TaskRunnerLogic for automation
│   ├── gui/               # Qt GUI frameworks
│   ├── util/              # Utilities (datastorage, fitting, constraints)
│   ├── tools/             # Config editor
│   ├── artwork/           # Icons, stylesheets, logos
│   └── runnable.py        # CLI entry point
├── docs/                  # Documentation
│   ├── design_concepts/   # Architecture patterns
│   ├── setup/             # Installation guides
│   └── programming_guidelines/
├── qudi-iqo-modules/      # External addon modules
├── my_software/           # Custom experiment code
│   └── tools/             # Shared utilities (fitting, analysis)
├── tests/                 # Test suite (RESTRICTED)
├── venv/                  # Virtual environment (RESTRICTED)
├── setup.py               # Package installation config
└── VERSION                # Version string
```

**Note**: Claude Code has restricted access to `tests/` and `venv/` directories per `.claude/settings.json`. The `my_software/tools/` directory is accessible for shared utilities like fitting functions.

## Important Development Patterns

### Module Development Workflow

1. Define interface (abstract base class)
2. Implement hardware (HardwareBase subclass)
3. Create logic (LogicBase with Connectors)
4. Optional GUI (GuiBase with Qt interface)
5. Write YAML config linking modules

### Best Practices

- Use `Connector` objects for module dependencies, not direct imports
- Implement interface contracts in hardware modules
- Keep GUIs logic-free (thin wrappers around logic modules)
- Log key events and errors using `self.log.debug/info/warning/error`
- Use `StatusVar` for user-important state that should persist
- Validate `ConfigOption` values with checker functions
- Communicate between threads using Qt Signals, not direct calls
- Module behavior flows from YAML configuration, not hardcoded values

### Remote Module Access

Qudi supports network-based module access via RPC (rpyc):
- Remote Modules Server for network access (configurable in YAML `global.remote_modules_server`)
- Namespace Server for local Jupyter/IPython access
- Optional SSL encryption
- Allows distributed measurement setups across multiple computers

### Logging

Per-module loggers accessible via `self.log`:
```python
class MyModule(Base):
    def on_activate(self):
        self.log.debug('Debug message')
        self.log.info('Info message')
        self.log.warning('Warning message')
        self.log.error('Error message')
```

Logs are written to rotating daily files in `<user_home>/qudi/log/`.

## Debugging - Accessing Qudi Logs

When debugging issues, check the qudi log files:

**Log Location (this user):**
```
C:\Users\aj92uwef\qudi\log\
```

**Log Files:**
- `qudi.log` - Current/most recent session (typically 10-50 KB)
- `qudi.log.1` through `qudi.log.5` - Rotated older logs (can be 100KB - 1MB+)

**Reading Strategies:**
```bash
# List log files with sizes
ls -la "C:\Users\aj92uwef\qudi\log"

# Read current session log (usually small enough to read fully)
# Use Read tool on: C:\Users\aj92uwef\qudi\log\qudi.log

# For large log files, read last N lines
tail -n 100 "C:\Users\aj92uwef\qudi\log\qudi.log"

# Search for errors in logs
grep -i "error\|warning\|exception" "C:\Users\aj92uwef\qudi\log\qudi.log"
```

**What to Look For:**
- Module activation errors during startup
- Hardware connection failures
- Qt threading warnings (e.g., `QObject::startTimer`)
- Timeout errors from hardware modules
- Exception tracebacks

**PyRPL Logs:**
PyRPL logs are integrated into the qudi log (prefixed with `pyrpl.`). Look for:
- `pyrpl.redpitaya` - Connection status
- Scan module warnings about "busy" or "already running"
- Timeout messages from `wait_done()`

## Resources

- Documentation: https://ulm-iqo.github.io/qudi-core/
- GitHub Repository: https://github.com/Ulm-IQO/qudi-core/
- Discussions/Forum: https://github.com/Ulm-IQO/qudi-core/discussions
- Issues: https://github.com/Ulm-IQO/qudi-core/issues
- Citation: [Qudi: A modular python suite for experiment control and data processing (SoftwareX 2017)](http://doi.org/10.1016/j.softx.2017.02.001)
