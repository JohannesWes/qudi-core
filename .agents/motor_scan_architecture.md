# Motor XY Scan Module Architecture

## Overview

The Motor XY Scan module provides motorized XY scanning with synchronous data acquisition. It supports four acquisition modes:

| Mode | Description |
|------|-------------|
| `STEP_ODMR` | Stop at each point, run ODMR scan, fit, proceed |
| `CONTINUOUS_STREAM` | Move continuously, bin streaming data by position |
| `CONTINUOUS_FREQ_TRACK` | Move continuously, track absolute frequency from lock |
| `POSITION_ONLY` | Stage movement only (debugging/alignment) |

---

## Module Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                     motor_scan_gui.py (MotorScanGui)                │
│                     pixel_odmr_widget.py (PixelOdmrWidget)          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ Connector: motor_scan_logic
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    motor_scan/scan_logic.py                         │
│                       (MotorScanLogic)                              │
│         Mixins: ContinuousLineScanMixin, MotorControlMixin,         │
│                 DataProcessingMixin, DataSavingMixin                │
└───────┬───────────────────┬──────────────────┬─────────────────┬────┘
        │                   │                  │                 │
        │ motor_hardware    │ odmr_logic       │ time_series     │ odmr_freq_tracking
        │ (required)        │ (optional)       │ (optional)      │ (optional)
        ▼                   ▼                  ▼                 ▼
┌───────────────┐  ┌───────────────┐  ┌────────────────────┐  ┌──────────────────────────┐
│ MotorInterface│  │  OdmrLogic    │  │ TimeSeriesReader   │  │  OdmrFrequencyTracking   │
└───────────────┘  └───────────────┘  └────────────────────┘  └──────────────────────────┘
```

---

## File Locations

| Component | Path |
|-----------|------|
| **Logic Package** | `qudi-iqo-modules/src/qudi/logic/motor_scan/` |
| **GUI** | `qudi-iqo-modules/src/qudi/gui/motor_scan/` |
| **Motor Interface** | `qudi-iqo-modules/src/qudi/interface/motor_interface.py` |
| **Hardware** | `qudi-iqo-modules/src/qudi/hardware/motor/` (Thorlabs, Newport, PI, Zaber, Micos) |
| **Dummy** | `qudi-iqo-modules/src/qudi/hardware/dummy/motor_dummy.py` |
| **Fit Functions** | `my_software/tools/fitting.py` |

### Logic Package Structure

```
motor_scan/
├── __init__.py
├── scan_logic.py          # Main class, orchestrates all modes
├── data_structures.py     # ScanMode, ScanState, ScanPattern, MotorScanData
├── continuous_line_scan.py # ContinuousLineScanMixin - line-by-line scanning
├── motor_control.py       # MotorControlMixin - movement, homing, position sampling
├── data_processing.py     # DataProcessingMixin - ODMR fitting, data binning
└── data_saving.py         # DataSavingMixin - file I/O, figure generation
```

---

## Key Design Decisions

### Non-Blocking Architecture

All scan operations are non-blocking to keep the GUI responsive:

```
User calls start_scan()
    ↓
Emits _sigDoStartScan (queued to logic thread)
    ↓
Logic thread: _do_start_scan_async()
    ↓
Motor movement via poll timer (50ms interval)
    ↓
Data acquisition triggers _sigNextPoint or line completion
    ↓
Repeat until scan complete
```

**Why:** Qt requires GUI thread to remain responsive. Long-running operations block the event loop.

### Mixin Architecture

The logic is split into mixins to:
1. Keep files manageable (~150 KB total across 6 files)
2. Separate concerns (motor control vs data processing vs saving)
3. Allow testing individual components

**Inheritance order matters:** `ContinuousLineScanMixin` must come before `MotorControlMixin` to override certain methods.

### Position-Based Data Binning (Continuous Modes)

For `CONTINUOUS_STREAM` and `CONTINUOUS_FREQ_TRACK`, data samples arrive at ~30 kHz but motor position is sampled at 20 Hz. The binning algorithm:

1. Records timestamped positions during line movement
2. Interpolates crossing times at bin boundaries using `scipy.interpolate.interp1d`
3. Uses `np.searchsorted` to assign data samples to bins
4. Calculates mean per bin

See `data_processing.py:_bin_line_data()` for implementation.

**Why position-based (not time-based):** Motor velocity varies (acceleration/deceleration), so equal time bins would give unequal spatial resolution.

### State Machine

```
IDLE ←→ INITIALIZING → RUNNING ←→ PAUSED
              ↓            ↓
           STOPPING ←──────┘
              ↓
            IDLE
```

See `data_structures.py:ScanState` for the enum definition.

---

## Thread Safety

| Mechanism | Purpose |
|-----------|---------|
| `RecursiveMutex` (`_thread_lock`) | Protects all state modifications |
| Qt `QueuedConnection` | All cross-thread signal/slot connections |
| `module_state.lock()` | Qudi framework lock during scan |

**Rule:** All public `@Slot` methods acquire `_thread_lock` at entry.

---

## Scan Execution Flows

### STEP_ODMR (Point-by-Point)

```
start_scan() → _sigDoStartScan
    ↓
Initialize MotorScanData, create scan folder
    ↓
For each point:
    motor.move_abs(target) → poll until arrived
        ↓
    odmr_logic.start_odmr_scan() → wait for sigScanStateUpdated
        ↓
    fit_hyperfine() → store results
        ↓
    Save fit plot (optional)
    ↓
Emit sigScanCompleted
```

### CONTINUOUS_STREAM / CONTINUOUS_FREQ_TRACK (Line-by-Line)

```
start_scan() → _sigDoStartScan
    ↓
Connect to time_series sigNewRawData
    ↓
For each line:
    motor.move_abs(line_start) → poll until arrived
        ↓
    Start position sampling (20 Hz)
    motor.move_abs(line_end)
        ↓
    While moving: buffer incoming data
        ↓
    On arrival: _bin_line_data()
    ↓
Disconnect signals, emit sigScanCompleted
```

### Lock Loss Handling (CONTINUOUS_FREQ_TRACK)

Lock status polled every 500ms. On lock loss:
1. Emit `sigLockLostDuringScan`
2. Transition to `PAUSED`
3. User re-establishes lock, calls `resume_scan()`
4. Zero-crossing history updated, scan continues

---

## Configuration

### Key ConfigOptions

| Option | Default | Notes |
|--------|---------|-------|
| `default_scan_mode` | `'STEP_ODMR'` | |
| `continuous_line_mode` | `True` | Enable line-by-line (vs point-by-point) for continuous modes |
| `position_sample_interval` | `0.05` | 20 Hz position sampling |
| `position_poll_interval` | `0.05` | 50ms motor polling |
| `odmr_fit_function` | `'fit_hyperfine'` | From `my_software.tools.fitting` |

See `scan_logic.py` for full list with defaults.

### Example Configuration

```yaml
hardware:
    thorlabs_xy_stage:
        module.Class: 'motor.thorlabs_kdc101_kinesis.ThorlabsKDC101Kinesis'
        options:
            axis_config:
                x: {serial: 27XXXXXX, pos_min: 0.0, pos_max: 0.050}
                y: {serial: 27YYYYYY, pos_min: 0.0, pos_max: 0.050}

logic:
    motor_scan_logic:
        module.Class: 'motor_scan.scan_logic.MotorScanLogic'
        connect:
            motor_hardware: thorlabs_xy_stage
            odmr_logic: odmr_logic           # For STEP_ODMR
            time_series_logic: time_series   # For CONTINUOUS_*
        options:
            default_scan_mode: 'STEP_ODMR'
            continuous_line_mode: true

gui:
    motor_scan_gui:
        module.Class: 'motor_scan.motor_scan_gui.MotorScanGui'
        connect:
            motor_scan_logic: motor_scan_logic
```

---

## Data Storage

### Output Structure

```
<data_dir>/YYYYMMDD-HHMM-SS_<tag>_motor_scan_<MODE>/
├── center_frequency.dat     # 2D data + metadata header
├── center_frequency.pdf     # Thumbnail
├── linewidth.dat/pdf
├── splitting.dat/pdf
├── fit_quality.dat/pdf
├── positions_target.dat
├── positions_actual.dat
├── odmr_raw_per_pixel/      # STEP_ODMR only
│   └── pixel_x{ix:03d}_y{iy:03d}_odmr.dat
└── odmr_fits/               # If save_odmr_fit_plots=True
    └── pixel_x{ix:03d}_y{iy:03d}.pdf
```

### File Format

`.dat` files: Tab-separated with `# key: value` metadata header. Load with `np.loadtxt(file, comments='#')`.

---

## Hardware Interface

The `MotorInterface` (in `motor_interface.py`) defines the contract. Key methods:

- `move_abs(param_dict)` - Non-blocking absolute move
- `get_pos()` - Current position
- `get_status()` - Movement status (0=idle)
- `calibrate()` - Home/calibrate
- `get_constraints()` - Axis limits and capabilities

Optional methods (checked with `hasattr`): `is_moving()`, `move_abs_sync()`, `wait_for_idle()`

---

## Extension Points

### Adding a New Scan Mode

1. Add to `ScanMode` enum in `data_structures.py`
2. Add initialization in `scan_logic.py:_do_start_scan_async()`
3. Add data collection (new mixin or extend existing)
4. Update `data_saving.py:save_scan_data()` for new data types
5. Update GUI if needed

### Adding a New Fit Function

Fit function must return dict with keys:
- `zero_crossing_frequencies` (Hz)
- `linewidths` (Hz)
- `n_features_found`

See `my_software/tools/fitting.py:fit_hyperfine()` for reference.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Could not load fit function" | Add `my_software` to PYTHONPATH |
| Position ~-1mm after homing | Normal - home offset |
| USB communication errors | Close Kinesis software, check cable |
| Lock lost during scan | Re-establish lock, then `resume_scan()` |

Enable debug logging:
```yaml
global:
    log_level: DEBUG
```

---

## Known Issues / Fixes

Documented in code comments:

1. **Race condition in `_resume_continuous_line()`** - Buffer must be prepared before timer starts
2. **Time alignment in binning** - Both position and data samples aligned to `_line_scan_start_time`
3. **Motor may stop short** - 1mm threshold validation, warns if short
4. **Stale position data** - Detects identical consecutive samples

---

## Related Documentation

- Motor interface: `motor_interface.py` docstrings
- Data structures: `data_structures.py` (MotorScanData dataclass)
- Fit functions: `my_software/tools/fitting.py`
- Similar architecture: `scanning_probe_logic.py` (piezo scanning)
