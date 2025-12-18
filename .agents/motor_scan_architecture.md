# Motor XY Scan Module Architecture

## Overview

The Motor XY Scan module provides motorized XY scanning functionality with synchronous data acquisition for quantum diamond sensor magnetometry. It supports two acquisition modes:

1. **STEP_ODMR**: Motor stops at each grid position, executes an ODMR scan, fits the data, then proceeds
2. **CONTINUOUS_STREAM**: Motors move continuously while streaming data is binned to grid positions

This document describes the architecture, module interactions, data flow, and implementation details.

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

---

## Configuration

### Example Config File

```yaml
global:
    startup_modules: [motor_scan_gui]

hardware:
    thorlabs_xy_stage:
        module.Class: 'motor.aptmotor.APTMotor'
        options:
            # Hardware-specific options

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

# Check before save
if self.module_state() == 'locked':
    self.log.error('Cannot save during scan')
    return
```

---

## Data Storage

### File Format

Data is saved using `TextDataStorage` with automatic daily directories:

```
<data_root>/YYYY/MM/YYYYMMDD/<module_name>/
    YYYYMMDD-HHMM-SS_<nametag>_motor_scan_STEP_ODMR_center_frequency.dat
    YYYYMMDD-HHMM-SS_<nametag>_motor_scan_STEP_ODMR_center_frequency.pdf
    YYYYMMDD-HHMM-SS_<nametag>_motor_scan_STEP_ODMR_linewidth.dat
    ...
```

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

### Constraints Format

```python
constraints = {
    'x': {
        'label': 'x',
        'unit': 'm',
        'pos_min': 0.0,
        'pos_max': 0.050,  # 50mm travel
        'pos_step': 0.000001,  # 1µm resolution
        'vel_min': 0.0001,
        'vel_max': 0.010,
        'vel_step': 0.0001,
        'ramp': ['Linear', 'Sinus'],
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

### Key Test Scenarios

1. **Basic STEP_ODMR scan**: Verify ODMR triggering and data extraction
2. **CONTINUOUS_STREAM scan**: Verify data binning and buffering
3. **Pause/Resume**: Verify state machine transitions
4. **Stop during scan**: Verify clean shutdown
5. **Save with nametag**: Verify file naming
6. **Pattern variations**: Verify index mapping for all patterns

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
