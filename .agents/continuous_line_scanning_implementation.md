# Continuous Line Scanning Implementation

**Date:** 2026-01-06  
**Status:** Implementation complete, pending hardware testing

## Background

### Original Problem

The motor XY scan module supports three scan modes:
- `STEP_ODMR`: Stop at each grid point, run ODMR measurement
- `CONTINUOUS_STREAM`: Continuous movement with streaming data
- `CONTINUOUS_FREQ_TRACK`: Continuous movement with frequency tracking

**Issue discovered:** Despite the naming, `CONTINUOUS_STREAM` and `CONTINUOUS_FREQ_TRACK` modes were **not actually moving continuously**. The scan loop was stopping at each grid point:

```
move_abs(point_N) → poll until motor stops → collect data → move_abs(point_N+1) → ...
```

This resulted in an effective scan speed of ~0.25 mm/s instead of the hardware's 2 mm/s capability—an **8x slowdown**.

### Root Cause

In [scan_logic.py](file:///c:/Users/aj92uwef/PycharmProjects/qudi-core/qudi-iqo-modules/src/qudi/logic/motor_scan/scan_logic.py), the `_on_motor_poll_timeout()` method waits for `is_moving == False` before triggering data collection via `_collect_continuous_data_and_advance()`. This is appropriate for `STEP_ODMR` but defeats the purpose of continuous modes.

### Reference Documentation

- [Motor Scan Architecture](file:///c:/Users/aj92uwef/PycharmProjects/qudi-core/.agents/motor_scan_architecture.md) - Original module documentation

---

## Implementation Plan

### Design Decisions

1. **Line-by-line continuous scanning** - Move continuously across each row/column, stop only between lines
2. **Position sampling at 20 Hz** - Record `(time, position)` pairs during movement for later binning
3. **Post-line data binning** - After line completes, bin time-series data to grid points using interpolated positions
4. **Preserve STEP_ODMR behavior** - All changes scoped to `CONTINUOUS_*` modes only
5. **Configurable fallback** - Add `continuous_line_mode: false` config option to disable if issues arise

### Binning Algorithm

For grid points spaced distance `d` apart:
- Bin boundary at midpoint between adjacent points (±d/2)
- Interpolate position vs time to find when motor crossed each boundary
- Split time-series data at boundary crossing times
- Store mean and raw samples per bin

---

## Files Changed

### New File

#### [continuous_line_scan.py](file:///c:/Users/aj92uwef/PycharmProjects/qudi-core/qudi-iqo-modules/src/qudi/logic/motor_scan/continuous_line_scan.py)

New mixin class `ContinuousLineScanMixin` containing:
- `_start_continuous_line_scan(line_index)` - Initiate line-by-line scanning
- `_on_line_motor_poll_timeout()` - Two-phase state machine (wait for line start, then line end)
- `_resume_continuous_line()` - Resume from mid-line pause preserving position data

**Design note:** Created as a separate mixin to avoid bloating `scan_logic.py` (already ~1000 lines).

---

### Modified Files

#### [data_structures.py](file:///c:/Users/aj92uwef/PycharmProjects/qudi-core/qudi-iqo-modules/src/qudi/logic/motor_scan/data_structures.py)

Added helper methods to `MotorScanData` class:
- `get_fast_axis()` / `get_slow_axis()` - Determine scan direction from pattern
- `get_num_lines()` / `get_points_per_line()` - Line geometry
- `get_line_point_indices(line_index)` - Flat point indices for a line
- `get_line_grid_positions(line_index)` - Target positions array
- `get_line_start_end_positions(line_index)` - Start/end position dicts
- `get_bin_boundaries(line_index)` - Midpoint boundaries for binning

---

#### [motor_control.py](file:///c:/Users/aj92uwef/PycharmProjects/qudi-core/qudi-iqo-modules/src/qudi/logic/motor_scan/motor_control.py)

Added position sampling functionality:
- `_init_position_sampling_state()` - Initialize buffer and timer
- `_start_position_sampling(sample_interval_ms)` - Begin recording positions
- `_stop_position_sampling()` - Stop and return recorded buffer
- `_record_position_sample()` - Record single `(timestamp, position)` tuple
- `_on_position_sample_timeout()` - Timer callback

---

#### [data_processing.py](file:///c:/Users/aj92uwef/PycharmProjects/qudi-core/qudi-iqo-modules/src/qudi/logic/motor_scan/data_processing.py)

Added data binning implementation:
- `_bin_line_data(line_index, position_time_buffer, raw_data_buffer, data_start_time)` - Main binning algorithm
- `_get_line_raw_data_buffer()` - Get copy of current data buffer
- `_clear_line_raw_data_buffer()` - Clear buffer for next line

Uses `scipy.interpolate.interp1d` to map position→time for boundary crossing detection.

---

#### [scan_logic.py](file:///c:/Users/aj92uwef/PycharmProjects/qudi-core/qudi-iqo-modules/src/qudi/logic/motor_scan/scan_logic.py)

Integration changes:
- Added `ContinuousLineScanMixin` to class inheritance
- Added config options: `continuous_line_mode` (default: True), `position_sample_interval` (default: 0.05s)
- Modified `_do_start_scan_async()` to choose between continuous and point-by-point modes
- Modified `resume_scan()` to handle mid-line resume
- Added mixin initialization calls in `__init__()`

---

## Bugs Found and Fixed During Review

### 1. Missing State Variable Initializations

**File:** `continuous_line_scan.py`  
**Issue:** `_waiting_for_line_start` and `_line_end_position` used but not initialized in `_init_continuous_line_state()`  
**Fix:** Added initialization to `__init__`

### 2. Pause/Resume Discarding Position Data

**File:** `continuous_line_scan.py`  
**Issue:** When pausing mid-line, position samples collected so far were lost  
**Fix:** Added `_line_pause_position_buffer` to preserve samples, restore on resume

### 3. Unused Variable

**File:** `data_processing.py`  
**Issue:** Created `pos_interp` interpolation but never used it  
**Fix:** Removed unused code, kept only the position→time interpolation needed

### 4. Logging Format Error

**File:** `data_processing.py`  
**Issue:** `{boundary_times[[0, -1]]:.3f}` would crash (can't format array with `.3f`)  
**Fix:** Changed to `{boundary_times[0]:.3f}s to {boundary_times[-1]:.3f}s`

---

## Configuration

New config options for `motor_scan_logic`:

```yaml
motor_scan_logic:
    options:
        # Enable continuous line scanning (default: true)
        continuous_line_mode: true
        
        # Position sampling interval in seconds (default: 0.05 = 20 Hz)
        position_sample_interval: 0.05
```

Set `continuous_line_mode: false` to fall back to original point-by-point behavior.

---

## Testing Instructions

1. Start qudi with motor scan module configured
2. Run a `CONTINUOUS_STREAM` scan (e.g., 10×10 grid)
3. Verify in logs:
   - "Using continuous line scanning mode" appears
   - Line-by-line progress messages with binning statistics
4. Verify timing:
   - Scan should complete ~8x faster than before
5. Verify data:
   - Compare spatial pattern to `STEP_ODMR` reference scan
6. Test pause/resume:
   - Pause mid-scan, wait, resume
   - Verify data is correctly binned despite pause

---

## Open Questions for Reviewer

1. **Sample rate assumption:** The binning algorithm assumes time-series data arrives at `ts_logic.data_rate`. Is this reliable, or should we include explicit timestamps from the time-series logic?

2. **Edge case handling:** What should happen if the motor stops mid-line unexpectedly (e.g., limit switch)? Currently this would result in partial line data.

3. **Acceleration/deceleration zones:** Data collected during motor acceleration at line start and deceleration at line end is included in the first/last bins. Should we discard data from accel/decel zones?

---

## Post-Review Fixes (2026-01-06)

Based on external code review, the following issues were addressed:

| Finding | Severity | Status | Description |
|---------|----------|--------|-------------|
| #1 | Major | ✅ Fixed | Race condition in pause/resume position buffer - reordered operations |
| #2 | Critical | ✅ Fixed | Timestamp alignment - added offset calculation and validation |
| #3 | Minor | ✅ Fixed | Added assertion for boundary count clarity |
| #4 | Major | ✅ Fixed | Motor stop detection - validates motor reached line end |
| #5 | Minor | ✅ Fixed | Jitter detection for stale position data |
| #6 | Minor | Skipped | scipy.interpolate performance (not a bottleneck) |
| #7 | Minor | ✅ Fixed | Buffer cleanup on binning failure |
| #8 | Minor | Skipped | Mixin docstring (current pattern correct) |
| Q3 | - | Deferred | Acceleration zone handling (future enhancement) |
