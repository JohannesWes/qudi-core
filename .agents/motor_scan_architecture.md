# Motor XY Scan Module Architecture

## Overview

The Motor XY Scan module provides motorized XY scanning functionality with synchronous data acquisition for quantum diamond sensor magnetometry. It supports two acquisition modes:

1. **STEP_ODMR**: Motor stops at each grid position, executes an ODMR scan, fits the data, then proceeds
2. **CONTINUOUS_STREAM**: Motors move continuously while streaming data is binned to grid positions

This document describes the architecture, module interactions, data flow, and implementation details.

**Last Updated:** December 2024

---

## Module Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                        motor_scan_gui.py                            │
│                         (MotorScanGui)                              │
│                     GUI Layer - GuiBase                             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ Connector: motor_scan_logic
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      motor_scan_logic.py                            │
│                       (MotorScanLogic)                              │
│                    Logic Layer - LogicBase                          │
└───────┬───────────────────────┬───────────────────────┬─────────────┘
        │                       │                       │
        │ Connector:            │ Connector:            │ Connector:
        │ motor_hardware        │ odmr_logic            │ time_series_logic
        │ (required)            │ (optional)            │ (optional)
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌────────────────────┐
│ MotorInterface│      │  OdmrLogic    │      │ TimeSeriesReader   │
│   Hardware    │      │    Logic      │      │      Logic         │
└───────────────┘      └───────────────┘      └────────────────────┘
```

---

## File Locations

| File | Path | Description |
|------|------|-------------|
| Logic Module | `qudi-iqo-modules/src/qudi/logic/motor_scan_logic.py` | Main scan orchestration logic |
| GUI Module | `qudi-iqo-modules/src/qudi/gui/motor_scan/motor_scan_gui.py` | User interface |
| Motor Interface | `qudi-iqo-modules/src/qudi/interface/motor_interface.py` | Hardware abstraction |
| Motor Dummy | `qudi-iqo-modules/src/qudi/hardware/dummy/motor_dummy.py` | Simulated hardware |
| Thorlabs KDC101 | `qudi-iqo-modules/src/qudi/hardware/motor/thorlabs_kdc101_kinesis.py` | Real Thorlabs MTS50-Z8 stages via KDC101 controllers |
| Fit Functions | `my_software/tools/fitting.py` | ODMR hyperfine fitting (`fit_hyperfine`) |

---

## Configuration

### Example Config File

```yaml
global:
    startup_modules: [motor_scan_gui]

hardware:
    # For real Thorlabs hardware
    thorlabs_xy_stage:
        module.Class: 'motor.thorlabs_kdc101_kinesis.ThorlabsKDC101Kinesis'
        options:
            axis_config:
                x:
                    serial: '27267130'  # X-axis KDC101 serial number
                    pos_min: 0
                    pos_max: 0.05  # 50mm in meters
                y:
                    serial: '27601623'  # Y-axis KDC101 serial number
                    pos_min: 0
                    pos_max: 0.05
            default_velocity: 2.0e-3  # m/s
            settle_time: 0.01         # seconds
            auto_home: false          # Manual homing recommended

    # OR for testing with dummy hardware
    motor_dummy:
        module.Class: 'dummy.motor_dummy.MotorDummy'

logic:
    motor_scan_logic:
        module.Class: 'motor_scan_logic.MotorScanLogic'
        connect:
            motor_hardware: thorlabs_xy_stage
            odmr_logic: odmr_logic           # Optional, for STEP_ODMR mode
            time_series_logic: time_series_reader_logic  # Optional, for CONTINUOUS_STREAM
        options:
            default_scan_mode: 'STEP_ODMR'
            position_poll_interval: 0.05
            odmr_fit_function: 'fit_hyperfine'
            save_thumbnails: True
            save_odmr_fit_plots: True    # Save per-pixel hyperfine fit plots
            home_before_scan: True       # Home stages before each scan

gui:
    motor_scan_gui:
        module.Class: 'motor_scan.motor_scan_gui.MotorScanGui'
        connect:
            motor_scan_logic: motor_scan_logic
```

---

## Core Data Structures

### ScanMode (Enum)

```python
class ScanMode(Enum):
    CONTINUOUS_STREAM = 0  # Motors move continuously, data binned by position
    STEP_ODMR = 1          # Motors stop at each point, ODMR scan taken
```

### ScanPattern (Enum)

```python
class ScanPattern(Enum):
    LINE_BY_LINE_X = 0  # Scan X lines, return to x_start for each Y step
    SNAKE_X = 1         # Scan X lines, alternate direction (boustrophedon)
    LINE_BY_LINE_Y = 2  # Scan Y lines, return to y_start for each X step
    SNAKE_Y = 3         # Scan Y lines, alternate direction
```

### ScanState (Enum)

```python
class ScanState(Enum):
    IDLE = 0
    RUNNING = 1
    PAUSED = 2
    STOPPING = 3
```

### MotorScanData (Dataclass)

Central data container holding all scan configuration and results:

```python
@dataclass
class MotorScanData:
    # Configuration
    scan_axes: Tuple[str, ...]           # e.g., ('x', 'y')
    scan_range: Tuple[Tuple[float, float], ...]  # [(x_start, x_stop), (y_start, y_stop)]
    scan_resolution: Tuple[int, ...]     # (nx, ny)
    scan_mode: ScanMode
    scan_pattern: ScanPattern
    
    # Timing
    timestamp_start: datetime
    timestamp_end: datetime
    scan_duration: float
    
    # Positions
    target_positions: np.ndarray         # Shape: (n_points, n_axes)
    actual_positions: np.ndarray         # From encoder feedback
    
    # CONTINUOUS_STREAM mode data
    stream_data_mean: Dict[str, np.ndarray]   # channel -> 2D array (nx, ny)
    stream_data_raw: Dict[str, List[List]]    # Raw samples per point
    
    # STEP_ODMR mode data
    odmr_frequency_data: np.ndarray
    odmr_signal_data: Dict[str, np.ndarray]
    odmr_fit_results: List[List[Dict]]
    
    # Derived quantities (from ODMR fits)
    center_frequency: np.ndarray         # 2D array (nx, ny)
    linewidth: np.ndarray                # 2D array (nx, ny)
    splitting: np.ndarray                # 2D array (nx, ny)
    fit_quality: np.ndarray              # Number of features found
    
    # Progress tracking
    current_point_index: int
    total_points: int
    completed: bool
```

**Key Methods:**
- `get_flat_target_positions()`: Generate positions array accounting for scan pattern
- `point_index_to_grid_index()`: Convert linear index to (ix, iy) grid indices
- `initialize_data_arrays()`: Set up storage arrays before scan
- `to_dict()` / `from_dict()`: Serialization for saving/loading

---

## Signal Flow

### Logic Module Signals

```python
# Emitted when scan state changes (IDLE, RUNNING, PAUSED, STOPPING)
sigScanStateChanged = QtCore.Signal(object)  # ScanState

# Emitted when new data is available for display
sigScanDataUpdated = QtCore.Signal()

# Emitted when a single scan point completes
sigScanPointCompleted = QtCore.Signal(int, dict)  # point_index, result_dict

# Emitted when entire scan completes
sigScanCompleted = QtCore.Signal(object)  # MotorScanData

# Emitted when motor position changes
sigPositionUpdated = QtCore.Signal(dict)  # {'x': pos_x, 'y': pos_y}

# Emitted when settings change
sigScanSettingsChanged = QtCore.Signal(dict)

# Emitted during save operations
sigSaveStateChanged = QtCore.Signal(bool)  # True=saving, False=done

# Internal signal for non-blocking scan loop
_sigNextPoint = QtCore.Signal()
```

### GUI Signal Connections

```
┌─────────────────────────────────────────────────────────────────┐
│                    GUI → Logic (Control)                        │
├─────────────────────────────────────────────────────────────────┤
│ action_start_scan.triggered → _toggle_scan() → logic.start_scan│
│ action_pause_scan.triggered → _toggle_pause() → logic.pause    │
│ action_save.triggered → _save_data() → logic.save_scan_data    │
│ apply_settings_button.clicked → _apply_settings()              │
│ mode_combo.changed → _mode_changed() → logic.set_scan_mode     │
│ pattern_combo.changed → _pattern_changed() → logic.set_pattern │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Logic → GUI (Feedback)                       │
├─────────────────────────────────────────────────────────────────┤
│ sigScanStateChanged → _on_scan_state_changed()                  │
│ sigScanDataUpdated → _on_scan_data_updated() → _update_display()│
│ sigPositionUpdated → _on_position_updated()                     │
│ sigScanSettingsChanged → _on_settings_changed()                 │
│ sigScanCompleted → _on_scan_completed()                         │
│ sigSaveStateChanged → _on_save_state_changed()                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Scan Execution Flow

### Non-Blocking Architecture

The scan uses a **non-blocking event-driven architecture** to keep the GUI responsive:

```
┌────────────┐     ┌──────────────┐     ┌────────────────┐
│ start_scan │────▶│ _sigNextPoint│────▶│_process_next_  │
│   (Slot)   │     │   (Signal)   │     │    point       │
└────────────┘     └──────────────┘     └───────┬────────┘
                                                │
                   ┌────────────────────────────┘
                   ▼
    ┌──────────────────────────────┐
    │   motor.move_abs(position)   │ ◀─── Non-blocking motor command
    │   _motor_poll_timer.start()  │
    └──────────────┬───────────────┘
                   │
    ┌──────────────┼────────────────────────────────────┐
    │              ▼                                     │
    │   _on_motor_poll_timeout()                        │
    │   ├─ If still moving: restart timer               │
    │   └─ If stopped:                                  │
    │      ├─ STEP_ODMR: _start_odmr_scan_async()       │
    │      └─ CONTINUOUS: _collect_data_and_advance()   │
    └───────────────────────────────────────────────────┘
```

### STEP_ODMR Mode Sequence

```
For each grid point (ix, iy):
    1. Move motor to target position (non-blocking)
    2. Poll motor status until idle
    3. Trigger ODMR scan via odmr_logic.start_odmr_scan()
    4. Wait for sigScanStateUpdated(False) from ODMR logic
    5. Extract ODMR data and perform fitting:
       - Get frequency_data, signal_data from odmr_logic
       - Call fit_hyperfine() to extract features
       - Store center_frequency, linewidth, splitting
    6. Update scan_data arrays
    7. Emit sigScanDataUpdated, sigScanPointCompleted
    8. Advance to next point via _sigNextPoint
```

### CONTINUOUS_STREAM Mode Sequence

```
For each grid point:
    1. Move motor to target position (non-blocking)
    2. Poll motor status until idle
    3. Collect buffered streaming data:
       - Data arrives continuously via sigNewRawData from time_series_logic
       - Data is buffered in _ts_raw_data_buffer during movement
    4. Process and store data for current position:
       - Calculate mean values
       - Store in stream_data_mean arrays
    5. Clear buffer for next position
    6. Emit sigScanDataUpdated
    7. Advance to next point
```

---

## Timer Architecture

### Motor Poll Timer

```python
self._motor_poll_timer = QtCore.QTimer()
self._motor_poll_timer.setSingleShot(True)
self._motor_poll_timer.timeout.connect(self._on_motor_poll_timeout)
```

- **Purpose**: Check if motor has finished moving
- **Interval**: 50ms (configurable)
- **Behavior**: Restarts until motor is idle, then triggers data acquisition

### Position Timer

```python
self._position_timer = QtCore.QTimer()
self._position_timer.setSingleShot(True)
self._position_timer.timeout.connect(self._on_position_poll_timeout)
```

- **Purpose**: Update position display during continuous scans
- **Interval**: Configured via `position_poll_interval` (default 50ms)

---

## Thread Safety

### RecursiveMutex

```python
self._thread_lock = RecursiveMutex()

# Usage pattern:
with self._thread_lock:
    # Protected operations
```

All public methods and signal handlers acquire the lock before modifying state.

### Module State Machine

```python
# Lock module during scan
self.module_state.lock()

# Unlock on completion
self.module_state.unlock()
```

### Saving During Paused Scans

Saving is allowed when:
- Scan state is **IDLE** (no scan in progress)
- Scan state is **PAUSED** (data is stable, safe to save)

Saving is blocked when:
- Scan state is **RUNNING** (data actively being modified)
- Scan state is **STOPPING** (transitional cleanup state)

```python
# Check scan state, not just module_state
if self._scan_state == ScanState.RUNNING:
    self.log.error('Unable to save. Pause the scan first.')
    return

# Handle already-locked state (e.g., paused scan)
already_locked = (self.module_state() == 'locked')
if not already_locked:
    self.module_state.lock()
try:
    # ... save operations ...
finally:
    if not already_locked:
        self.module_state.unlock()
```

---

## Data Storage

### File Organization

Data is organized in scan-specific folders within daily directories:

```
<data_root>/YYYY/MM/YYYY-MM-DD/motor_scan_logic/
    YYYYMMDD-HHMM-SS_<nametag>_motor_scan_STEP_ODMR/
        ├── center_frequency.dat          # 2D center frequency data
        ├── center_frequency.pdf          # Plot thumbnail
        ├── linewidth.dat                 # 2D linewidth data  
        ├── linewidth.pdf                 # Plot thumbnail
        ├── fit_quality.dat               # Number of features found per pixel
        ├── fit_quality.pdf               # Plot thumbnail
        ├── odmr_fits/                    # Per-pixel hyperfine fit plots
        │   ├── pixel_000_000_fit.png
        │   ├── pixel_000_001_fit.png
        │   └── ...
        └── odmr_raw_per_pixel/           # Raw ODMR data per pixel
            ├── pixel_000_000_odmr.dat
            ├── pixel_000_001_odmr.dat
            └── ...
```

### Scan Folder Naming

- **During scan**: Folder created at scan start with timestamp
- **On save**: Folder renamed to include user nametag if provided
- **Format**: `YYYYMMDD-HHMM-SS_<nametag>_motor_scan_<MODE>`

### Per-Pixel ODMR Data (STEP_ODMR mode)

Each pixel's ODMR scan is saved with:
- Frequency array (Hz)
- Signal data (V or counts)
- Actual measured position
- Grid indices (ix, iy)

```
# Pixel ODMR Data
# Grid Index: (0, 0)
# Actual Position: x=0.000mm, y=0.000mm
# 
# Frequency (Hz)    Signal (V)
2.85e9              0.0012
2.851e9             0.0011
...
```

### ODMR Fit Plots

When `save_odmr_fit_plots: True`, hyperfine fit plots are saved for each pixel showing:
- Raw ODMR spectrum
- Fitted hyperfine model
- Extracted center frequency and linewidth
- Fit quality indicator

### Metadata Format

```
# [General]
# timestamp=2024-12-18T10:30:00
# ...
# 
# [Metadata]
# Scan Mode=STEP_ODMR
# Scan Pattern=SNAKE_X
# Scan Axes=('x', 'y')
# Scan Range=((0.0, 0.01), (0.0, 0.01))
# Scan Resolution=(20, 20)
# Total Points=400
# Completed Points=400
# Scan Duration (s)=1234.5
# Home Before Scan=True
# x axis min=0.0
# x axis max=0.01
# ...
# 
# ---- END HEADER ----
# <2D data array>
```

### PDF Thumbnails

Generated via `_draw_figure()` using matplotlib:
- 2D color map with 'inferno' colormap
- SI-scaled axes (mm, µm, etc.)
- Colorbar with proper units
- Scan metadata annotation

---

## Hardware Interface

### MotorInterface (Abstract Base Class)

Required methods that hardware modules must implement:

```python
class MotorInterface(Base):
    @abstractmethod
    def get_constraints(self) -> dict:
        """Return dict of axis constraints."""
        pass
    
    @abstractmethod
    def move_abs(self, param_dict: dict):
        """Move to absolute position. Non-blocking."""
        pass
    
    @abstractmethod
    def move_rel(self, param_dict: dict):
        """Move relative amount."""
        pass
    
    @abstractmethod
    def get_pos(self, param_list=None) -> dict:
        """Get current position."""
        pass
    
    @abstractmethod
    def get_status(self, param_list=None) -> dict:
        """Get movement status (0=idle, non-zero=moving)."""
        pass
    
    @abstractmethod
    def abort(self):
        """Emergency stop."""
        pass
    
    @abstractmethod
    def get_velocity(self, param_list=None) -> dict:
        """Get current velocity settings."""
        pass
    
    @abstractmethod
    def set_velocity(self, param_dict: dict):
        """Set velocity."""
        pass
    
    @abstractmethod
    def calibrate(self, param_list=None):
        """Home/calibrate axes."""
        pass
```

### ThorlabsKDC101Kinesis Implementation

The `ThorlabsKDC101Kinesis` class provides support for Thorlabs MTS50-Z8 stages via KDC101 controllers using pylablib.

**Extended Methods (beyond MotorInterface):**

```python
def move_abs_sync(self, param_dict, timeout=30.0, position_tolerance=50e-6):
    """Move to position and wait with verification."""
    pass

def wait_for_idle(self, timeout=30.0) -> bool:
    """Wait for all movement to complete."""
    pass

def is_moving(self) -> bool:
    """Check if any axis is moving."""
    pass

@staticmethod
def list_devices() -> List[Tuple[str, str]]:
    """List all Thorlabs Kinesis devices."""
    pass

@staticmethod
def find_kdc101_devices() -> List[Tuple[str, str]]:
    """Find KDC101 controllers specifically."""
    pass
```

**Configuration Example:**

```yaml
thorlabs_xy_stage:
    module.Class: 'motor.thorlabs_kdc101_kinesis.ThorlabsKDC101Kinesis'
    options:
        axis_config:
            x:
                serial: '27267130'
                pos_min: 0
                pos_max: 0.05  # 50mm
            y:
                serial: '27601623'
                pos_min: 0
                pos_max: 0.05
        default_velocity: 2.0e-3  # 2 mm/s
        settle_time: 0.01         # 10 ms
        auto_home: false
```

**Finding Device Serial Numbers:**

```python
from qudi.hardware.motor.thorlabs_kdc101_kinesis import ThorlabsKDC101Kinesis

# List all Kinesis devices
print(ThorlabsKDC101Kinesis.list_devices())

# Find KDC101 controllers
print(ThorlabsKDC101Kinesis.find_kdc101_devices())
```

### Constraints Format

```python
constraints = {
    'x': {
        'label': 'x',
        'unit': 'm',
        'pos_min': 0.0,
        'pos_max': 0.050,  # 50mm travel
        'pos_step': 0.8e-6,  # 0.8 µm resolution
        'vel_min': 0.0,
        'vel_max': 2.4e-3,  # 2.4 mm/s max
        'vel_step': 1e-6,
        'acc_min': 0.0,
        'acc_max': 4.5e-3,  # 4.5 mm/s²
        'acc_step': 1e-6,
        'ramp': ['Trapez'],
    },
    'y': {...}
}
```

---

## Scan Pattern Implementation

### Grid Index Mapping

The scan visits grid points in order determined by `ScanPattern`. The `point_index_to_grid_index()` method converts linear index to array indices:

**SNAKE_X Pattern (default):**
```
Y
▲
│  ←←←←←←
│  │
│  ──────▶
│  │
│  ←←←←←←
│  │
│  ──────▶ Start
└──────────────▶ X
```

```python
# For SNAKE_X: fast axis is X, step in Y
iy = flat_index // nx
ix_in_line = flat_index % nx
if iy % 2 == 1:  # Odd lines go backwards
    ix = nx - 1 - ix_in_line
else:
    ix = ix_in_line
return (ix, iy)
```

### Data Array Indexing

Data arrays use **matrix indexing (ij)** convention:
- `array[ix, iy]` where first index is X, second is Y
- Shape: `(nx, ny)`

For display with `imshow()`, data is transposed: `data.T`

---

## ConfigOptions & StatusVariables

### ConfigOptions (Static Configuration)

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `default_scan_mode` | str | 'STEP_ODMR' | Initial scan mode |
| `position_poll_interval` | float | 0.05 | Position update interval (s) |
| `odmr_fit_function` | str | 'fit_hyperfine' | Name of fit function to use |
| `require_fit_function` | bool | True | Warn if fit function unavailable |
| `save_thumbnails` | bool | True | Save PDF plots with data |
| `save_odmr_fit_plots` | bool | True | Save per-pixel hyperfine fit plots |
| `home_before_scan` | bool | True | Home stages before each scan |
| `position_tolerance` | float | 100e-6 | Position verification tolerance (m) |

### StatusVariables (Persistent State)

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `scan_ranges` | dict | {'x': (0, 0.01), 'y': (0, 0.01)} | Scan range per axis (m) |
| `scan_resolution` | dict | {'x': 10, 'y': 10} | Points per axis |
| `active_scan_mode` | int | None | Current mode (ScanMode value) |
| `active_scan_pattern` | str | 'SNAKE_X' | Current pattern name |

---

## Error Handling

### Connector Validation

```python
def on_activate(self):
    motor = self._motor_hardware()
    if motor is None:
        self.log.error("Motor hardware connector not available!")
        return
    
    # Mode-specific validation at scan start
    if mode == ScanMode.STEP_ODMR and self._odmr_logic() is None:
        self.log.error("ODMR logic not connected.")
        return
```

### Graceful Degradation

- If fit function unavailable: Fall back to finding minimum in ODMR spectrum
- If time_series already running: Subscribe to existing data stream
- If motor reports error: Log and continue to next point

---

## Stage Positioning & Homing

### Overview

Accurate stage positioning is critical for XY scanning. The system uses Thorlabs MTS50-Z8 stages with KDC101 controllers, which have specific positioning characteristics that must be handled correctly.

### Hardware Specifications (MTS50-Z8)

| Specification | Value | Notes |
|---------------|-------|-------|
| Travel Range | 50 mm | Configurable via `pos_min`/`pos_max` |
| Encoder Resolution | 29 nm | 34,555 counts/mm |
| Min Repeatable Increment | 0.8 µm | Practical positioning limit |
| Home Position Accuracy | ±4 µm | After proper homing |
| Backlash | <6 µm | Consider for bidirectional scans |
| Max Velocity | 2.4 mm/s | Default: 2.0 mm/s (safe) |

### Homing Implementation

**Why Homing Matters:**
- Establishes absolute position reference (encoder zero)
- Eliminates accumulated positioning errors
- Required before first scan for accurate positioning

**Homing Sequence:**
1. Stage moves to negative limit switch (reverse direction)
2. Stage contacts limit switch and stops
3. Stage moves forward by configured offset (~1 mm)
4. Encoder zero reference is established at this position
5. Reported position will be near 0 (typically ±1 µm)

**Implementation Details (ThorlabsKDC101Kinesis):**

```python
# Homing uses force=True to override "already homed" state
stage.home(sync=False, force=True)

# Poll until complete with timeout
while time.time() - start_time < timeout:
    if not stage.is_moving():
        break
    time.sleep(0.5)
```

**Known Issues with pylablib:**
- The `stage.home(sync=True)` may return immediately without actually homing if the device reports "already homed"
- Solution: Use `force=True` parameter to ensure homing occurs
- After homing, position may show as slightly negative (~-1mm) due to home offset - this is normal

### Homing in GUI

The GUI provides a "Home Stages" button that:
1. Shows confirmation dialog (homing takes 30-60s per axis)
2. Is disabled during active scans
3. Calls `motor_scan_logic.home_stages()`
4. Executes homing sequentially on X then Y axis

### Pre-Scan Homing

Configurable via `home_before_scan` option (default: `True`):
- When enabled, stages are homed before each scan starts
- Ensures consistent absolute positioning across scans
- Adds ~30-60 seconds to scan initialization

### Position Verification During Scans

For STEP_ODMR mode, each measurement point includes position verification:

```python
# After move_abs() completes, verify position
actual_pos = motor.get_pos()
for axis, target in target_position.items():
    error = abs(actual_pos[axis] - target)
    if error > position_tolerance:  # Default: 100 µm
        self.log.warning(f"Position error: target={target}, actual={actual_pos[axis]}")
```

**Position Data Stored Per Point:**
- `actual_position`: Dict of measured positions after move
- Logged at info level: `"Point N at x=X.XXmm, y=Y.YYmm"`

### Synchronous vs Asynchronous Moves

**Synchronous Moves (move_abs_sync):**
- Used for discrete positioning in STEP_ODMR mode
- Blocks until movement complete and position verified
- Includes configurable position tolerance (default 50 µm)

```python
motor.move_abs_sync(
    param_dict={'x': 0.025, 'y': 0.010},
    timeout=30.0,
    position_tolerance=50e-6
)
```

**Asynchronous Moves (move_abs):**
- Used for continuous scanning modes
- Returns immediately after command sent
- Position verification done separately via polling

### Common Positioning Problems & Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| Stages not homing | Position unchanged after home | Use `force=True` in home command |
| Position shows ~-1mm after homing | Normal | This is the home offset from limit switch |
| First point position error | Large error on point 0 | Ensure proper homing before scan |
| Position drift during long scans | Gradual offset increase | Re-home between scans; check stage temperature |
| Inconsistent positioning | Random errors | Check USB connection stability; increase settle time |

### Settle Time Configuration

After each move, a settle time allows mechanical vibrations to decay:

```yaml
options:
    settle_time: 0.01  # 10 ms default, increase for sensitive measurements
```

For high-precision measurements, consider increasing to 50-100 ms.

### Backlash Compensation

The MTS50-Z8 has <6 µm backlash. For SNAKE pattern scans (alternating direction), this can cause systematic positioning errors. Mitigation strategies:

1. **Use LINE_BY_LINE pattern**: Always approach from same direction
2. **Increase position tolerance**: Accept small errors
3. **Add backlash compensation**: (Not currently implemented)

---

## Extension Points

### Adding New Scan Modes

1. Add mode to `ScanMode` enum
2. Add branch in `start_scan()` for initialization
3. Add data collection logic in `_on_motor_poll_timeout()`
4. Add result processing in `_advance_to_next_point()`
5. Update `save_scan_data()` to handle new data types

### Adding New Fit Functions

1. Add ConfigOption for function selection
2. Load in `_load_fit_function()` based on config
3. Ensure fit function returns dict with standard keys:
   - `zero_crossing_frequencies [Hz]`
   - `linewidths [Hz]`
   - `n_features_found`

### Adding New Display Channels

1. Add data storage in `MotorScanData`
2. Initialize in `initialize_data_arrays()`
3. Populate in result processing
4. Add to GUI display_combo
5. Add to `save_scan_data()` if needed

---

## Testing

### Dummy Hardware

Use `motor_dummy.py` for testing without real hardware:

```yaml
hardware:
    motor_dummy:
        module.Class: 'dummy.motor_dummy.MotorDummy'
```

### Available Config Files

| Config File | Description |
|-------------|-------------|
| `motor_scan_dummy.cfg` | Full simulation - dummy stages and dummy sensor |
| `motor_scan_real_data.cfg` | Simulated stages with real sensor data |
| `motor_scan_real_hardware.cfg` | Real Thorlabs stages and real sensor |

### Key Test Scenarios

1. **Basic STEP_ODMR scan**: Verify ODMR triggering and data extraction
2. **CONTINUOUS_STREAM scan**: Verify data binning and buffering
3. **Pause/Resume**: Verify state machine transitions
4. **Stop during scan**: Verify clean shutdown
5. **Save with nametag**: Verify file/folder naming
6. **Pattern variations**: Verify index mapping for all patterns
7. **Homing**: Verify stages reach home position (position ~0 after homing)
8. **Position verification**: Check actual vs target positions in logs
9. **ODMR fit plots**: Verify fit plots generated in odmr_fits folder

### Device Discovery Script

```python
# Run to find connected Thorlabs devices
from pylablib.devices import Thorlabs

print("Kinesis devices:")
print(Thorlabs.list_kinesis_devices())

# Alternative via FTDI
print("\nFTDI devices:")
from pylablib.core.utils import general as general_utils
print(general_utils.backend.find_devices())
```

---

## Related Modules

- `odmr_logic.py`: ODMR measurement logic
- `time_series_reader_logic.py`: Continuous data streaming
- `scanning_probe_logic.py`: Similar architecture for piezo scanning
- `scanning_data_logic.py`: Reference for data saving patterns

---

## Version History

| Date | Author | Changes |
|------|--------|---------|
| 2024-12 | - | Initial implementation with STEP_ODMR and CONTINUOUS_STREAM modes |
| 2024-12 | - | Added scan pattern support (SNAKE_X, LINE_BY_LINE_X, etc.) |
| 2024-12 | - | Added PDF thumbnail saving |
| 2024-12 | - | Added save nametag support in GUI |
| 2024-12 | - | Added ThorlabsKDC101Kinesis hardware driver using pylablib |
| 2024-12 | - | Added per-scan folder organization for data storage |
| 2024-12 | - | Added raw ODMR data saving per pixel |
| 2024-12 | - | Added ODMR hyperfine fit plot saving per pixel |
| 2024-12 | - | Added Home Stages button in GUI with confirmation dialog |
| 2024-12 | - | Added home_before_scan ConfigOption |
| 2024-12 | - | Fixed homing to use force=True for reliable operation |
| 2024-12 | - | Added position verification and logging at each scan point |
| 2024-12 | - | Added move_abs_sync with position tolerance checking |
| 2024-12 | - | Fixed linewidth calculation to be average (not sum) |
| 2024-12 | - | Fixed fit_hyperfine import path |

---

## Troubleshooting

### Common Issues

**"Could not load fit function: No module named 'my_software'"**
- The fit function is loaded from `my_software.tools.fitting`
- Ensure the `my_software` package is in PYTHONPATH or installed
- Check that `fit_hyperfine` function exists in `fitting.py`

**Stages not homing / homing completes instantly**
- Symptom: Position unchanged after homing, completes in <1 second
- Cause: pylablib's `home()` returns immediately if device reports "already homed"
- Solution: Use `force=True` parameter in home command
- Fixed in current ThorlabsKDC101Kinesis implementation

**Position shows ~-1mm after homing**
- This is normal! The stage homes to limit switch then moves by offset
- The home offset is configured in the stage firmware (~1mm)
- Position 0 is the reference point, actual physical home is at ~-1mm

**Large position error on first scan point**
- Cause: Stage was not properly homed before scan
- Solution: Enable `home_before_scan: true` in config
- Or manually home via GUI button before starting scan

**USB communication errors**
- Ensure FTDI drivers are installed
- Check USB cable and connection
- Only one application can access a stage at a time
- Close Kinesis software if running

**Scan takes too long**
- Each ODMR measurement adds ~5-10 seconds per point
- Homing adds ~30-60 seconds per axis at scan start
- Consider: fewer points, faster velocity, skip homing if position is known good

### Debug Logging

Enable verbose logging in config:
```yaml
global:
    log_level: DEBUG
```

Key log messages to watch:
- `"Homing x-axis..."` - Homing started
- `"x-axis homed in X.Xs"` - Homing complete
- `"Point N at x=X.XXmm, y=Y.YYmm"` - Each measurement point position
- `"Position error..."` - Position verification failed
