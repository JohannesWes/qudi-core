# Motor XY Scan Module Architecture

## Overview

The Motor XY Scan module provides motorized XY scanning functionality with synchronous data acquisition. It supports three acquisition modes:

1. **STEP_ODMR**: Motor stops at each grid position, executes an ODMR scan, fits the data, then proceeds
2. **CONTINUOUS_STREAM**: Motors move continuously while streaming data is binned to grid positions
3. **CONTINUOUS_FREQ_TRACK**: Motors move continuously, absolute frequency measured from frequency lock

---

## Module Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                        motor_scan_gui.py                            │
│                         (MotorScanGui)                              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ Connector: motor_scan_logic
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      motor_scan_logic.py                            │
│                       (MotorScanLogic)                              │
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
| Logic | `qudi-iqo-modules/src/qudi/logic/motor_scan_logic.py` |
| GUI | `qudi-iqo-modules/src/qudi/gui/motor_scan/motor_scan_gui.py` |
| Motor Interface | `qudi-iqo-modules/src/qudi/interface/motor_interface.py` |
| Thorlabs Hardware | `qudi-iqo-modules/src/qudi/hardware/motor/thorlabs_kdc101_kinesis.py` |
| Motor Dummy | `qudi-iqo-modules/src/qudi/hardware/dummy/motor_dummy.py` |
| Fit Functions | `my_software/tools/fitting.py` |
| Example Config | `C:\Users\aj92uwef\qudi\config\motor_scan_freq_tracking.cfg` |

---

## Core Data Structures

Defined in `motor_scan_logic.py`:

| Class | Purpose |
|-------|---------|
| `ScanMode` | Enum: `CONTINUOUS_STREAM`, `STEP_ODMR`, `CONTINUOUS_FREQ_TRACK` |
| `ScanPattern` | Enum: `LINE_BY_LINE_X`, `SNAKE_X`, `LINE_BY_LINE_Y`, `SNAKE_Y` |
| `ScanState` | Enum: `IDLE`, `RUNNING`, `PAUSED`, `STOPPING` |
| `MotorScanData` | Dataclass holding scan config, positions, and all result arrays |

**MotorScanData key fields:**
- `scan_axes`, `scan_range`, `scan_resolution` - Configuration
- `target_positions`, `actual_positions` - Position arrays
- `stream_data_mean` - For CONTINUOUS modes (channel → 2D array)
- `odmr_frequency_data`, `odmr_signal_data`, `odmr_fit_results` - For STEP_ODMR
- `center_frequency`, `linewidth`, `splitting`, `fit_quality` - Derived from fits

---

## Signal Flow

### Key Logic Signals

| Signal | Payload | Purpose |
|--------|---------|---------|
| `sigScanStateChanged` | `ScanState` | State machine transitions |
| `sigScanDataUpdated` | (none) | Trigger GUI refresh |
| `sigScanPointCompleted` | `int, dict` | Per-point progress |
| `sigScanCompleted` | `MotorScanData` | Scan finished |
| `sigPositionUpdated` | `dict` | Motor position feedback |
| `sigLockLostDuringScan` | (none) | Lock lost in FREQ_TRACK mode |
| `sigHomingStateChanged` | `bool` | Homing started/finished |
| `_sigNextPoint` | (none) | Internal: advance scan loop |

### GUI ↔ Logic Connection Pattern

```
GUI → Logic:  action.triggered → logic.start_scan/pause/save
Logic → GUI:  sigScanDataUpdated → _update_display()
```

---

## Scan Execution Flow

### Non-Blocking Architecture

The scan uses event-driven, non-blocking execution to keep GUI responsive:

```
start_scan() → _sigNextPoint → _process_next_point()
                                      │
                               motor.move_abs()
                               _motor_poll_timer.start()
                                      │
                               _on_motor_poll_timeout()
                               ├─ moving: restart timer
                               └─ idle: acquire data → _sigNextPoint
```

### Mode-Specific Acquisition

| Mode | After motor stops... |
|------|---------------------|
| STEP_ODMR | Trigger `odmr_logic.start_odmr_scan()`, wait for completion, fit data |
| CONTINUOUS_STREAM | Collect buffered `time_series_logic` data, compute mean |
| CONTINUOUS_FREQ_TRACK | Convert FTW→frequency, add to baseline, check lock status |

---

## Thread Safety

- **RecursiveMutex**: All public methods acquire `_thread_lock` before modifying state
- **Module state machine**: `module_state.lock()` during scan, `unlock()` on completion
- **Save allowed**: Only in `IDLE` or `PAUSED` states (not `RUNNING` or `STOPPING`)

---

## Data Storage

### File Organization

```
<data_root>/YYYY/MM/YYYY-MM-DD/motor_scan_logic/
    YYYYMMDD-HHMM-SS_<nametag>_motor_scan_<MODE>/
        ├── center_frequency.dat          # 2D data + metadata header
        ├── center_frequency.pdf          # Plot thumbnail
        ├── linewidth.dat
        ├── linewidth.pdf
        ├── fit_quality.dat
        ├── odmr_fits/                    # Per-pixel fit plots (if enabled)
        │   └── pixel_XXX_YYY_fit.png
        └── odmr_raw_per_pixel/           # Raw ODMR data per pixel
            └── pixel_XXX_YYY_odmr.dat
```

### Data File Format

Files use qudi's standard format: metadata header followed by 2D array data. See `qudi.util.datastorage` for details.

---

## Hardware Interface

The `MotorInterface` (in `motor_interface.py`) defines required methods:

| Method | Purpose |
|--------|---------|
| `get_constraints()` | Return axis limits, velocity ranges |
| `move_abs(param_dict)` | Move to absolute position (non-blocking) |
| `move_rel(param_dict)` | Move relative amount |
| `get_pos()` | Get current position dict |
| `get_status()` | Get movement status (0=idle, non-zero=moving) |
| `abort()` | Emergency stop |
| `calibrate()` | Home/calibrate axes |
| `get_velocity()` / `set_velocity()` | Velocity control |

### ThorlabsKDC101Kinesis Extensions

Additional methods beyond interface (see hardware file for details):
- `move_abs_sync()` - Blocking move with position verification
- `wait_for_idle()` - Wait for movement completion
- `is_moving()` - Check if any axis moving
- `list_devices()` / `find_kdc101_devices()` - Device discovery

### Hardware Specifications (MTS50-Z8)

| Spec | Value |
|------|-------|
| Travel Range | 50 mm |
| Encoder Resolution | 29 nm |
| Min Repeatable Increment | 0.8 µm |
| Home Position Accuracy | ±4 µm |
| Backlash | <6 µm |
| Max Velocity | 2.4 mm/s |

---

## Scan Pattern Implementation

### SNAKE_X Pattern (default)

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

The `point_index_to_grid_index()` method in `MotorScanData` handles conversion from linear index to (ix, iy) grid coordinates, accounting for snake direction reversal on odd rows.

**Data array convention**: `array[ix, iy]` (matrix indexing). For `imshow()` display, transpose: `data.T`

---

## ConfigOptions & StatusVariables

### ConfigOptions (in config file)

| Name | Default | Description |
|------|---------|-------------|
| `default_scan_mode` | `'STEP_ODMR'` | Initial scan mode |
| `position_poll_interval` | `0.05` | Motor poll interval (s) |
| `require_fit_function` | `True` | Warn if fit unavailable |
| `save_thumbnails` | `True` | Save PDF plots |
| `save_odmr_fit_plots` | `True` | Save per-pixel fit plots |
| `home_before_scan` | `False` | Home stages before each scan |
| `lock_status_poll_interval` | `0.5` | Lock poll for FREQ_TRACK mode (s) |
| `fit_feature_prominence` | `0.02` | Peak detection threshold |
| `fit_n_most_prominent_peaks` | `5` | Max peaks to fit |

### StatusVariables (persisted)

| Name | Default | Description |
|------|---------|-------------|
| `scan_ranges` | `{'x': (0, 0.01), 'y': (0, 0.01)}` | Scan range per axis (m) |
| `scan_resolution` | `{'x': 10, 'y': 10}` | Points per axis |
| `active_scan_mode` | `None` | Current mode |
| `active_scan_pattern` | `'SNAKE_X'` | Current pattern |

---

## Homing

### Why Homing Matters
- Establishes absolute position reference (encoder zero)
- Eliminates accumulated positioning errors
- Required before first scan for accurate positioning

### Homing Behavior
1. Stage moves to negative limit switch
2. Moves forward by configured offset (~1 mm)
3. Encoder zero reference established
4. Position after homing is near 0 (may show ~-1mm due to offset - this is normal)

### GUI Integration
- "Home Stages" button with confirmation dialog
- Disabled during active scans
- Executes sequentially (X then Y)
- Takes ~30-60s per axis

---

## Extension Points

### Adding New Scan Modes

1. Add to `ScanMode` enum
2. Add initialization branch in `start_scan()`
3. Add data collection in `_on_motor_poll_timeout()`
4. Add result processing in `_advance_to_next_point()`
5. Update `save_scan_data()` for new data types

### Adding New Fit Functions

Fit function must return dict with keys:
- `zero_crossing_frequencies` (Hz)
- `linewidths` (Hz)
- `n_features_found`

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "Could not load fit function" | Ensure `my_software` package in PYTHONPATH |
| Stages not homing / instant completion | Fixed: implementation uses `force=True` |
| Position ~-1mm after homing | Normal - this is the home offset |
| Large error on first point | Enable `home_before_scan: true` |
| USB communication errors | Close Kinesis software, check USB cable |
| Scan too slow | Reduce points, increase velocity |
| Homing fails after abort | Fixed: abort now calls `motor.abort()` |

### Debug Logging

Enable in config:
```yaml
global:
    log_level: DEBUG
```

Key log messages:
- `"Homing x-axis..."` / `"x-axis homed in X.Xs"`
- `"Point N at x=X.XXmm, y=Y.YYmm"`
- `"Position error..."` (verification failed)

---

## Related Modules

- `odmr_logic.py` - ODMR measurement
- `time_series_reader_logic.py` - Continuous streaming
- `scanning_probe_logic.py` - Similar architecture for piezo scanning

---

## Version History

| Date | Changes |
|------|---------|
| 2024-12 | Initial: STEP_ODMR, CONTINUOUS_STREAM, scan patterns, Thorlabs driver |
| 2024-12 | Added: PDF thumbnails, per-pixel ODMR saving, homing support |
| 2025-01 | Fixed: Abort not stopping motors, homing verification |
| 2026-01 | Added: CONTINUOUS_FREQ_TRACK mode, lock monitoring |
