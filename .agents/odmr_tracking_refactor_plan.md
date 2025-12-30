# ODMR Frequency Tracking Module Refactor Plan

**Date:** 2025-12-18  
**Status:** Planning  
**Author:** Claude (AI Assistant)  

---

## Executive Summary

This document outlines a plan to refactor the `OdmrFrequencyTrackingLogic` module to follow the same architectural pattern as the working `SensitivitySweepLogic` module. The key insight is that the sensitivity module works well because it **delegates data streaming to existing, battle-tested modules** rather than implementing custom streaming logic.

---

## Table of Contents

1. [Problem Analysis](#problem-analysis)
2. [Why Sensitivity Module Works Well](#why-sensitivity-module-works-well)
3. [Current vs Proposed Architecture](#current-vs-proposed-architecture)
4. [Detailed Implementation Plan](#detailed-implementation-plan)
5. [Risk Assessment & Mitigations](#risk-assessment--mitigations)
6. [Testing Strategy](#testing-strategy)
7. [Implementation Order](#implementation-order)

---

## Problem Analysis

### Current Issues in `OdmrFrequencyTrackingLogic`

#### 1. **Streaming Architecture Mismatch** ⚠️ MAJOR

The tracking logic expects `error_streamer` to implement `DataInStreamInterface` but relies on implementation-specific methods:

```python
# Line 369-373 in odmr_frequency_tracking_logic.py
streamer = self._error_streamer()
input_mode = 'demod' if self._stream_mode == 'error' else 'ftw_corr'
streamer.set_stream_input(input_mode)  # ← Not in DataInStreamInterface!
```

#### 2. **Polling-Based Data Acquisition is Inefficient** ⚠️ MODERATE

Uses `QTimer` polling at 100ms intervals:
- Creates latency (~3050 samples per poll at 30.5 kHz)
- CPU overhead from constant timer wakeups
- Runs in Qt main thread, blocking GUI updates

#### 3. **Circular Buffer Race Conditions** ⚠️ MODERATE

```python
# Sample-by-sample write with no lock protection
for i in range(n_samples):
    self._error_buffer[self._error_write_pos] = data[i]
    self._error_times[self._error_write_pos] = times[i]
    self._error_write_pos = (self._error_write_pos + 1) % buffer_size
```

Issues:
- No lock protection during concurrent access
- Full buffer copy on every signal emit (GC pressure)
- Python loop is ~1000x slower than vectorized numpy

#### 4. **Threading Model Inconsistency** ⚠️ MAJOR

`_poll_lock_status()` calls `streamer.read_data()` which triggers synchronous FPGA polling over RPyC (5-50ms blocking on network latency), freezing the GUI.

#### 5. **Duplicated Functionality**

The tracking logic reimplements what `TimeSeriesReaderLogic` already provides:
- Circular buffer management
- Thread-safe data handling
- Signal-based updates
- Proper async streaming

---

## Why Sensitivity Module Works Well

### Key Insight: Module Reuse vs. Direct Hardware Access

The sensitivity module **doesn't implement its own data acquisition** - it orchestrates existing, well-tested modules:

```python
# sensitivity_sweep_logic.py - Uses existing logic modules
_odmr_logic = Connector(name='odmr_logic', interface='OdmrLogic')
_time_series_logic = Connector(name='time_series_logic', interface='TimeSeriesReaderLogic')
```

### Data Flow in Sensitivity Sweep

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         sensitivity_sweep_logic                              │
│  (Orchestrator - no direct hardware access)                                  │
└──────────────────────┬────────────────────────────┬─────────────────────────┘
                       │                            │
                       ▼                            ▼
              ┌────────────────┐           ┌────────────────────┐
              │   odmr_logic   │           │ time_series_reader │
              │                │           │      _logic        │
              │ - scan control │           │ - buffer mgmt      │
              │ - fitting      │           │ - thread safety    │
              │ - data storage │           │ - signal emission  │
              └───────┬────────┘           └─────────┬──────────┘
                      │                              │
                      ▼                              ▼
              ┌────────────────┐           ┌────────────────────┐
              │ redpitaya_     │           │ redpitaya_data_    │
              │ finite_sampling│           │    instream        │
              │   (hardware)   │           │   (hardware)       │
              └───────┬────────┘           └─────────┬──────────┘
                      │                              │
                      └──────────────┬───────────────┘
                                     ▼
                              ┌─────────────┐
                              │   PyRPL     │
                              │  (shared)   │
                              └─────────────┘
```

### Why This Works

1. **OdmrLogic** already handles ODMR scanning, data buffering, signal emission
2. **TimeSeriesReaderLogic** already handles streaming, circular buffers, thread safety
3. **GUIs automatically update** because they're connected to these same logic modules
4. **No duplicate state** - one source of truth for each data type

---

## Current vs Proposed Architecture

### Current Architecture (Problematic)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OdmrFrequencyTrackingLogic                                │
│  (Inherits OdmrLogic + implements own streaming)                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ INHERITED from OdmrLogic:                                            │    │
│  │  - ODMR scanning ✓                                                   │    │
│  │  - Data storage ✓                                                    │    │
│  │  - Signal emission ✓                                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ CUSTOM (duplicates TimeSeriesReaderLogic):                           │    │
│  │  - _error_buffer circular buffer                                     │    │
│  │  - _poll_lock_status() timer                                         │    │
│  │  - _add_to_error_buffer() manual management                          │    │
│  │  - sigErrorDataUpdated emission                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────┬────────────────────────────┬─────────────────────────┘
                       │                            │
                       ▼                            ▼
              ┌────────────────┐           ┌────────────────────┐
              │ redpitaya_     │           │ redpitaya_data_    │
              │  odmr_lock     │           │    instream        │
              │ (lock control) │           │ (error streaming)  │
              └───────┬────────┘           └─────────┬──────────┘
                      │                              │
                      └──────────────┬───────────────┘
                                     ▼
                              ┌─────────────┐
                              │   PyRPL     │
                              └─────────────┘
```

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OdmrFrequencyTrackingLogic                                │
│  (Orchestrator for tracking operations)                                      │
│                                                                              │
│  Responsibilities:                                                           │
│   - Coordinate ODMR scanning (via inherited OdmrLogic)                       │
│   - Linear fitting for slope extraction                                      │
│   - Lock configuration (bandwidth, gains)                                    │
│   - Lock enable/disable control                                              │
│   - Mode management (open-loop vs closed-loop)                               │
│                                                                              │
│  NOT responsible for:                                                        │
│   - Streaming data buffering (delegate to TimeSeriesReaderLogic)             │
│   - Circular buffer management (delegate to TimeSeriesReaderLogic)           │
│   - Polling hardware (delegate to TimeSeriesReaderLogic)                     │
└──────────────────────┬────────────────────────┬─────────────────────────────┘
                       │                        │
                       ▼                        ▼
              ┌────────────────┐       ┌────────────────────┐
              │ redpitaya_     │       │ time_series_reader │
              │  odmr_lock     │       │      _logic        │
              │                │       │                    │
              │ - set_bandwidth│       │ - start_reading()  │
              │ - enable_lock  │       │ - stop_reading()   │
              │ - get_status   │       │ - trace_data       │
              │ - clear        │       │ - sigDataChanged   │
              └───────┬────────┘       └─────────┬──────────┘
                      │                          │
                      │                          ▼
                      │                ┌────────────────────┐
                      │                │ redpitaya_data_    │
                      │                │    instream        │
                      │                │                    │
                      │                │ - stream demod     │
                      │                │ - stream ftw_corr  │
                      │                └─────────┬──────────┘
                      │                          │
                      └──────────────┬───────────┘
                                     ▼
                              ┌─────────────┐
                              │   PyRPL     │
                              └─────────────┘
```

### Key Changes Summary

| Aspect | Current | Proposed |
|--------|---------|----------|
| Error signal streaming | Custom `_error_buffer` + `QTimer` polling | `TimeSeriesReaderLogic` connector |
| Stream mode switching | Direct call to `error_streamer.set_stream_input()` | Configure `RedPitayaDataInStream` via config or method |
| Data buffering | Manual circular buffer in tracking logic | Handled by `TimeSeriesReaderLogic` |
| GUI updates | Custom `sigErrorDataUpdated` | Reuse `TimeSeriesReaderLogic.sigDataChanged` |
| Thread safety | Ad-hoc (buggy) | Proven implementation in TSR |

---

## Detailed Implementation Plan

### Phase 1: Preparation & Interface Alignment

#### 1.1 Extend `RedPitayaDataInStream` for Stream Mode Switching

**Why:** Currently `set_stream_input()` exists but isn't in the interface. We need a clean way to switch between `demod` and `ftw_corr` modes.

**What to do:**
- Add `stream_input` property to `DataInStreamInterface` (or create specialized interface)
- OR: Make stream mode a configuration option that can be changed at runtime
- Ensure `TimeSeriesReaderLogic` can work with mode-switchable streamers

**Files affected:**
- `qudi/interface/data_instream_interface.py` - Add optional `stream_input` property
- `qudi/hardware/redpitaya/redpitaya_data_instream.py` - Already has implementation

#### 1.2 Verify `TimeSeriesReaderLogic` Compatibility

**Why:** Need to confirm TSR can handle our use case (continuous streaming with mode switching).

**What to check:**
- Can TSR handle indefinite continuous streaming? (Yes, has `StreamingMode.CONTINUOUS`)
- Can we access raw data for real-time display? (Yes, `sigNewRawData` signal)
- Can we switch stream modes without full restart? (Need to verify)

**Potential issue:** TSR may need stream stop/restart to change input mode. This is acceptable for our use case (mode change is infrequent).

### Phase 2: Refactor `OdmrFrequencyTrackingLogic`

#### 2.1 Add `TimeSeriesReaderLogic` Connector

**Current connectors:**
```python
_odmr_lock_hw = Connector(name='odmr_lock_hw', interface='OdmrFreqLockInterface')
_error_streamer = Connector(name='error_streamer', interface='DataInStreamInterface')
```

**Proposed connectors:**
```python
_odmr_lock_hw = Connector(name='odmr_lock_hw', interface='OdmrFreqLockInterface')
_time_series_logic = Connector(name='time_series_logic', interface='TimeSeriesReaderLogic')
```

**Why:** Replace direct hardware access with logic-layer access, matching sensitivity pattern.

#### 2.2 Remove Custom Streaming Implementation

**Delete:**
- `_error_buffer`, `_error_times`, `_error_write_pos` state variables
- `_status_timer` QTimer and `_poll_lock_status()` method
- `_add_to_error_buffer()` method
- Custom circular buffer logic

**Keep:**
- Lock status polling (separate from data streaming) - but simplify
- Signal emissions for GUI (adapt to forward TSR signals)

#### 2.3 Implement Stream Control via TSR

**New implementation pattern:**
```python
def start_error_stream(self):
    """Start error signal streaming via TimeSeriesReaderLogic."""
    ts_logic = self._time_series_logic()
    
    # Configure stream mode on underlying hardware
    streamer = ts_logic._streamer()  # Access hardware via TSR
    input_mode = 'demod' if self._stream_mode == 'error' else 'ftw_corr'
    streamer.set_stream_input(input_mode)
    
    # Start TSR streaming (handles buffering, thread safety, etc.)
    ts_logic.start_reading()
    
    self._stream_active = True
    self.sigStreamStateChanged.emit(True)

def stop_error_stream(self):
    """Stop error signal streaming."""
    ts_logic = self._time_series_logic()
    ts_logic.stop_reading()
    
    self._stream_active = False
    self.sigStreamStateChanged.emit(False)
```

#### 2.4 Forward TSR Signals to Tracking GUI

**Connect TSR signals to tracking-specific handlers:**
```python
def on_activate(self):
    super().on_activate()
    
    # Connect to TimeSeriesReaderLogic signals
    ts_logic = self._time_series_logic()
    ts_logic.sigDataChanged.connect(self._on_stream_data_changed)
    ts_logic.sigNewRawData.connect(self._on_raw_data)

def _on_stream_data_changed(self, data_x, data_y, ...):
    """Forward TSR data updates to tracking GUI."""
    # Transform data if needed (e.g., apply calibration)
    # Emit tracking-specific signal
    self.sigErrorDataUpdated.emit(data_x, data_y)
```

#### 2.5 Separate Lock Status Polling from Data Streaming

**Current problem:** `_poll_lock_status()` does both:
1. Poll lock hardware status (locked, saturated, correction_hz)
2. Read and buffer stream data

**Proposed separation:**
- **Lock status:** Keep simple QTimer polling (low frequency, ~1-10 Hz)
- **Stream data:** Fully delegated to TSR (high frequency, handled properly)

```python
def _poll_lock_status(self):
    """Poll lock status only (not stream data)."""
    if self._lock_enabled:
        lock_hw = self._odmr_lock_hw()
        status = lock_hw.get_status()
        self.sigLockStatusUpdated.emit(status)
    # NO stream data reading here - TSR handles that
```

### Phase 3: Update GUI

#### 3.1 Option A: Reuse Time Series GUI (Recommended)

**Concept:** Don't create custom time series plots in tracking GUI. Instead:
- Time Series GUI shows the streaming data (already works with TSR)
- Tracking GUI focuses on: fit controls, lock controls, status display

**Advantage:** Zero custom plotting code, automatic compatibility with TSR improvements.

**Implementation:** Remove `_create_time_series_dock()` from tracking GUI, instruct users to open Time Series GUI alongside.

#### 3.2 Option B: Embed TSR Data in Tracking GUI

**Concept:** Keep tracking GUI self-contained but connect to TSR signals.

**Implementation:**
```python
def _connect_tracking_signals(self):
    # Connect to TSR for data display
    ts_logic = self._odmr_logic()._time_series_logic()  # Access via tracking logic
    ts_logic.sigDataChanged.connect(self._update_error_plot)
```

### Phase 4: Configuration Updates

#### 4.1 Update Config Example

**Current config:**
```yaml
odmr_frequency_tracking_logic:
    module.Class: 'odmr_frequency_tracking_logic.OdmrFrequencyTrackingLogic'
    connect:
        microwave: 'mw_source'
        data_scanner: 'redpitaya_finite_sampling'
        odmr_lock_hw: 'redpitaya_odmr_lock'
        error_streamer: 'redpitaya_stream'  # Direct hardware
```

**Proposed config:**
```yaml
# Time series reader (reusable)
time_series_reader_logic:
    module.Class: 'time_series_reader_logic.TimeSeriesReaderLogic'
    connect:
        streamer: 'redpitaya_stream'

# Tracking logic (uses TSR)
odmr_frequency_tracking_logic:
    module.Class: 'odmr_frequency_tracking_logic.OdmrFrequencyTrackingLogic'
    connect:
        microwave: 'mw_source'
        data_scanner: 'redpitaya_finite_sampling'
        odmr_lock_hw: 'redpitaya_odmr_lock'
        time_series_logic: 'time_series_reader_logic'  # Logic layer!
```

---

## Risk Assessment & Mitigations

### Risk 1: TSR May Not Support Dynamic Mode Switching

**Concern:** `TimeSeriesReaderLogic` may require stop/reconfigure/start cycle to change stream input.

**Mitigation:** This is acceptable - mode switching (error ↔ correction) is an infrequent operation. We can:
1. Stop TSR
2. Change hardware input mode
3. Restart TSR

**Verification:** Test this sequence before full implementation.

### Risk 2: TSR Signals May Not Match Tracking Needs

**Concern:** `sigDataChanged` signature may not provide exactly what tracking GUI needs.

**Mitigation:** 
- Use `sigNewRawData` for raw samples if `sigDataChanged` is processed/decimated
- Add thin transformation layer in tracking logic if needed

### Risk 3: Breaking Existing Functionality

**Concern:** Refactor may break working ODMR scanning (inherited from OdmrLogic).

**Mitigation:**
- ODMR scanning is inherited and unchanged
- Only streaming-related code is modified
- Incremental testing at each phase

### Risk 4: Hardware Resource Conflicts

**Concern:** TSR and tracking logic accessing same PyRPL instance.

**Mitigation:** 
- Already handled by `resource_manager.py` (shared PyRPL instance)
- Same pattern works in sensitivity module

---

## Testing Strategy

### Unit Tests
1. `RedPitayaDataInStream.set_stream_input()` - mode switching works
2. TSR with mode-switchable streamer - data flows correctly
3. Lock status polling independent of streaming

### Integration Tests
1. **Sensitivity-style workflow:**
   - Start time series GUI
   - Start tracking GUI
   - Enable streaming → data appears in time series GUI
   - Switch modes → data type changes
   - Enable lock → status updates in tracking GUI

2. **ODMR scan + streaming concurrent operation:**
   - Stream error signal
   - Run ODMR scan (should work independently)
   - Verify no interference

3. **Long-duration stability:**
   - Stream for 10+ minutes
   - Verify no memory leaks, buffer overflows, GUI lag

---

## Implementation Order

1. **Phase 1.2** - Verify TSR compatibility (low effort, high information value)
2. **Phase 1.1** - Interface alignment if needed
3. **Phase 2.1-2.3** - Core refactor (main work)
4. **Phase 4** - Config updates
5. **Phase 2.4-2.5** - Signal forwarding and status polling cleanup
6. **Phase 3** - GUI updates (Option A recommended)
7. **Testing** - Full integration testing

---

## Files Affected

### Modified Files
- `qudi/logic/odmr_frequency_tracking_logic.py` - Main refactor
- `qudi/gui/odmr_tracking/odmr_tracking_gui.py` - GUI simplification
- `qudi/gui/odmr_tracking/config_example.yaml` - Config update
- `qudi/interface/data_instream_interface.py` - Optional interface extension

### Potentially Modified
- `qudi/hardware/redpitaya/redpitaya_data_instream.py` - If interface changes needed

### Unchanged
- `qudi/hardware/redpitaya/redpitaya_odmr_lock.py` - Lock hardware interface
- `qudi/interface/odmr_freq_lock_interface.py` - Lock interface
- `qudi/logic/time_series_reader_logic.py` - Reused as-is
- PyRPL modules - No changes needed

---

## Expected Benefits

After refactoring:

1. **Thread-safe streaming** - Proven implementation in TSR
2. **Automatic GUI updates** - Via existing time series infrastructure
3. **Reduced code complexity** - ~100 lines deleted from tracking logic
4. **Future-proof** - Improvements to TSR automatically benefit tracking
5. **Consistent architecture** - Same pattern as sensitivity module
6. **Better maintainability** - Single source of truth for streaming logic

---

## Open Questions

1. Should we keep the custom time series dock in tracking GUI (Option B) or rely on separate Time Series GUI (Option A)?
2. Do we need to add `set_stream_input()` to `DataInStreamInterface` or use a separate specialized interface?
3. What polling frequency is appropriate for lock status (independent of data streaming)?

---

## References

- `qudi/logic/sensitivity_sweep_logic.py` - Working reference implementation
- `qudi/logic/time_series_reader_logic.py` - Streaming logic to reuse
- `pyrpl/hardware_modules/scan.py` - FPGA streaming implementation
- `pyrpl/docs/developer_guide/odmr_freq_lock_implementation.md` - Lock theory

---

## Implementation Notes (2025-12-18)

**Status:** Core refactoring COMPLETED

### Changes Made

#### Phase 1: Analysis
- Verified TSR supports all required features (continuous streaming, raw data signals, stop/restart)
- Confirmed `set_stream_input()` can be called when TSR is stopped (via `ts_logic._streamer()`)
- No interface changes required - existing pattern from SensitivitySweepLogic works

#### Phase 2: Core Refactoring

1. **Connector Changes** (`odmr_frequency_tracking_logic.py`):
   - Replaced `_error_streamer` with `_time_series_logic` connector
   - Kept `_odmr_lock_hw` for lock control

2. **Removed Custom Streaming Code**:
   - Removed `_error_buffer`, `_error_times`, `_error_write_pos` state variables
   - Removed `_error_buffer_size` ConfigOption
   - Removed `_add_to_error_buffer()` method

3. **Added TSR Signal Handlers**:
   - `_on_tsr_raw_data()` - Handles raw data from TSR
   - `_on_tsr_data_changed()` - Handles processed trace data
   - `_on_tsr_status_changed()` - Synchronizes stream state with TSR

4. **Updated Stream Control Methods**:
   - `_apply_stream_mode_to_hardware()` - New helper to set hardware input mode via TSR
   - `set_stream_mode()` - Now uses `_apply_stream_mode_to_hardware()`
   - `start_error_stream()` - Now starts TSR instead of direct hardware
   - `stop_error_stream()` - Now stops TSR instead of direct hardware

5. **Simplified Lock Status Polling**:
   - `_poll_lock_status()` now ONLY polls lock status (0.5s default interval)
   - All streaming data acquisition delegated to TSR signals

#### Phase 4: Configuration Updates
- Updated `odmr_tracking_config_example.cfg` with new TSR-based configuration
- Added `time_series_reader_logic` section
- Updated connectors and usage instructions

### Open Questions Resolved

1. **GUI Option**: Currently using Option B (embedded via sigErrorDataUpdated). GUI can connect to TSR signals for raw data.
2. **Interface Changes**: Not needed - accessing via `ts_logic._streamer()` works (same as SensitivitySweepLogic)
3. **Lock Polling Frequency**: Changed to 0.5s (from 0.1s) since streaming is handled separately

### Remaining Work

- [ ] Integration testing with actual hardware
- [ ] GUI testing to verify data display works correctly
- [ ] Performance testing for streaming latency

### Files Modified

- `qudi-iqo-modules/src/qudi/logic/odmr_frequency_tracking_logic.py` - Core refactoring
- `qudi-iqo-modules/config_examples/odmr_tracking_config_example.cfg` - Updated config
