# ODMR Frequency Tracking Refactor - Session Summary

**Date:** 2025-12-18
**Status:** Implementation Complete (pending hardware testing)
**Related Plan:** [odmr_tracking_refactor_plan.md](odmr_tracking_refactor_plan.md)

---

## Overview

This session implemented the refactoring plan to make `OdmrFrequencyTrackingLogic` delegate data streaming to `TimeSeriesReaderLogic` (TSR), following the same architectural pattern as the working `SensitivitySweepLogic`.

### Problem Statement

The original `OdmrFrequencyTrackingLogic` had several issues:
- Custom circular buffer implementation duplicated TSR functionality
- QTimer polling at 100ms blocked the Qt main thread during FPGA reads
- Race conditions in buffer management
- Inconsistent architecture compared to `SensitivitySweepLogic`

### Solution

Delegate all streaming data acquisition to `TimeSeriesReaderLogic`, which provides:
- Thread-safe buffered data acquisition
- Proper Qt signal integration (no GUI blocking)
- Proven implementation already used by sensitivity measurements

---

## Changes Made

### 1. OdmrFrequencyTrackingLogic (`odmr_frequency_tracking_logic.py`)

#### Connector Changes
```python
# BEFORE
_error_streamer = Connector(name='error_streamer', interface='DataInStreamInterface')

# AFTER
_time_series_logic = Connector(name='time_series_logic', interface='TimeSeriesReaderLogic')
```

#### Removed Code
- `_error_buffer`, `_error_times`, `_error_write_pos` state variables
- `_error_buffer_size` ConfigOption
- `_add_to_error_buffer()` method (~30 lines)
- Stream data reading from `_poll_lock_status()`

#### Added Code
- `_apply_stream_mode_to_hardware()` - Helper to set hardware input mode via TSR
- `_on_tsr_raw_data()` - Signal handler for raw data from TSR
- `_on_tsr_data_changed()` - Signal handler for processed trace data
- `_on_tsr_status_changed()` - Synchronizes stream state with TSR

#### Modified Methods
| Method | Change |
|--------|--------|
| `__init__` | Removed buffer initialization |
| `on_activate` | Connect to TSR signals, apply initial stream mode |
| `on_deactivate` | Disconnect from TSR signals |
| `set_stream_mode` | Use `_apply_stream_mode_to_hardware()` |
| `start_error_stream` | Start TSR instead of direct hardware |
| `stop_error_stream` | Stop TSR instead of direct hardware |
| `_poll_lock_status` | Only polls lock status (no streaming) |

#### Config Option Changes
| Option | Before | After |
|--------|--------|-------|
| `error_buffer_size` | 10000 | **REMOVED** |
| `status_poll_interval` | 0.1s | 0.5s |

### 2. SensitivitySweepLogic (`sensitivity_sweep_logic.py`)

Added safeguard in `_configure_lock_in_filters()` to ensure stream input is set to `'demod'` before sensitivity measurements. This protects against mode changes from the tracking module.

```python
# CRITICAL: Ensure stream input is set to 'demod' for sensitivity measurements
if hasattr(streamer, 'set_stream_input') and hasattr(streamer, 'stream_input'):
    current_mode = streamer.stream_input
    if current_mode != 'demod':
        self.log.warning(f'Stream input was "{current_mode}", switching to "demod"')
        if ts_logic.module_state() == 'locked':
            ts_logic.stop_reading()
        streamer.set_stream_input('demod')
```

### 3. Configuration Files

#### New Config File Created
`C:\Users\aj92uwef\qudi\config\rp_windfreak_odmr_scan_via_FPGA_with_sensitivity_refactored.cfg`

Key changes in `odmr_frequency_tracking_logic` section:
```yaml
# BEFORE
connect:
    odmr_lock_hw: 'redpitaya_odmr_lock'
    error_streamer: 'redpitaya_stream'

# AFTER
connect:
    odmr_lock_hw: 'redpitaya_odmr_lock'
    time_series_logic: 'time_series_reader_logic'
```

#### Updated Example Config
`qudi-iqo-modules/config_examples/odmr_tracking_config_example.cfg`

### 4. Documentation

Updated `odmr_tracking_refactor_plan.md` with implementation notes section.

---

## Architecture Comparison

### Before (Direct Hardware Access)
```
┌─────────────────────────────────────────┐
│     OdmrFrequencyTrackingLogic          │
│  - Custom _error_buffer (circular)      │
│  - QTimer polling @ 100ms               │
│  - _add_to_error_buffer() Python loop   │
└──────────────────┬──────────────────────┘
                   │ Direct access
                   ▼
┌─────────────────────────────────────────┐
│       RedPitayaDataInStream             │
│           (Hardware)                    │
└─────────────────────────────────────────┘
```

### After (TSR Pattern)
```
┌─────────────────────────────────────────┐
│     OdmrFrequencyTrackingLogic          │
│  - Forwards TSR signals to GUI          │
│  - Controls stream mode                 │
│  - Lock status polling only (0.5s)      │
└──────────────────┬──────────────────────┘
                   │ Qt Signals
                   ▼
┌─────────────────────────────────────────┐
│       TimeSeriesReaderLogic             │
│  - Thread-safe buffering                │
│  - Signal-based data delivery           │
│  - Proper Qt integration                │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│       RedPitayaDataInStream             │
│           (Hardware)                    │
└─────────────────────────────────────────┘
```

---

## Signal Flow

### Data Streaming
```
RedPitayaDataInStream
    │
    ▼ (hardware polling)
TimeSeriesReaderLogic
    │
    ├──► sigNewRawData ──► TrackingLogic._on_tsr_raw_data()
    │                              │
    └──► sigDataChanged ──► TrackingLogic._on_tsr_data_changed()
                                   │
                                   ▼
                           sigErrorDataUpdated
                                   │
                                   ▼
                           TrackingGui._update_error_plot()
```

### Stream Mode Switching
```
User clicks "Frequency Correction" in GUI
    │
    ▼
TrackingGui.sigSetStreamMode.emit('correction')
    │
    ▼
TrackingLogic._handle_set_stream_mode('correction')
    │
    ├── Check stream not active (raise error if active)
    │
    ▼
TrackingLogic.set_stream_mode('correction')
    │
    ▼
TrackingLogic._apply_stream_mode_to_hardware()
    │
    ▼
ts_logic._streamer().set_stream_input('ftw_corr')
```

---

## Module Interaction: Shared Resources

Both `SensitivitySweepLogic` and `OdmrFrequencyTrackingLogic` share:
- Same `TimeSeriesReaderLogic` instance
- Same `RedPitayaDataInStream` hardware

### Potential Conflict
If tracking switches to `'ftw_corr'` mode, sensitivity would get wrong data.

### Resolution
Sensitivity logic now explicitly ensures `'demod'` mode before measurements:
- Checks current mode via `streamer.stream_input`
- If not `'demod'`, stops TSR, switches mode, logs warning
- Sensitivity proceeds with correct data

---

## Files Modified

| File | Type | Changes |
|------|------|---------|
| `qudi-iqo-modules/src/qudi/logic/odmr_frequency_tracking_logic.py` | Logic | Core refactoring |
| `qudi-iqo-modules/src/qudi/logic/sensitivity_sweep_logic.py` | Logic | Added mode safeguard |
| `qudi-iqo-modules/config_examples/odmr_tracking_config_example.cfg` | Config | Updated for TSR pattern |
| `C:\Users\aj92uwef\qudi\config\rp_windfreak_odmr_scan_via_FPGA_with_sensitivity_refactored.cfg` | Config | New user config |
| `.agents/odmr_tracking_refactor_plan.md` | Docs | Added implementation notes |

---

## Usage

### Starting with New Config
```bash
qudi -c rp_windfreak_odmr_scan_via_FPGA_with_sensitivity_refactored.cfg
```

### Stream Mode Switching (Programmatic)
```python
# Must stop streaming before changing mode
tracking_logic.stop_error_stream()
tracking_logic.set_stream_mode('correction')  # or 'error'
tracking_logic.start_error_stream()
```

### GUI Usage
- Stream mode can be selected in the Time Series dock
- Mode selection is disabled while streaming is active
- Must stop stream, change mode, then restart

---

## Remaining Work

- [ ] Integration testing with Red Pitaya hardware
- [ ] Verify GUI displays data correctly from TSR signals
- [ ] Performance testing (verify no GUI lag)
- [ ] Long-duration stability test (memory leaks, buffer handling)

---

## Benefits Achieved

1. **Thread-safe streaming** - No more Qt main thread blocking
2. **Consistent architecture** - Same pattern as SensitivitySweepLogic
3. **Reduced code complexity** - ~60 lines of custom buffer code removed
4. **Better maintainability** - Single source of truth for streaming logic
5. **Future-proof** - TSR improvements automatically benefit tracking
6. **Resource protection** - Sensitivity ensures correct mode before measurements
