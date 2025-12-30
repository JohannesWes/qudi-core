# PyRPL Thread Safety Solution for Qudi Integration

**Date**: 2025-12-30
**Status**: Implemented and tested
**Supersedes**: `odmr_tracking_tcp_race_condition_fix.md`, `pyrpl_monitorclient_thread_safety_fix.md`

---

## Executive Summary

When using PyRPL from Qudi's multi-threaded environment, concurrent TCP socket access caused crashes with "Wrong control sequence from server" errors. This document describes the root cause, the temporary workaround that was used, and the permanent fix now implemented.

**The Fix**: Added `threading.RLock()` to PyRPL's `MonitorClient` class to serialize all TCP socket operations. This is a 24-line change that makes PyRPL inherently thread-safe.

**Result**: Qudi modules can now safely access the Red Pitaya concurrently without workarounds, stop/restart patterns, or timing hacks.

---

## Table of Contents

1. [Background: Threading Concepts](#1-background-threading-concepts)
2. [The Problem: TCP Protocol Desynchronization](#2-the-problem-tcp-protocol-desynchronization)
3. [Historical Workaround (Now Removed)](#3-historical-workaround-now-removed)
4. [The Solution: RLock in MonitorClient](#4-the-solution-rlock-in-monitorclient)
5. [Implementation Details](#5-implementation-details)
6. [Modules Affected](#6-modules-affected)
7. [Testing Procedure](#7-testing-procedure)
8. [Future Considerations](#8-future-considerations)

---

## 1. Background: Threading Concepts

### What is Multi-threading?

Multi-threading allows a program to execute multiple code sequences "simultaneously." In Qudi:

```
┌─────────────────────────────────────────────────────────────┐
│                      Qudi Application                        │
├─────────────────────────────────────────────────────────────┤
│  Main Thread (Qt GUI)     Logic Threads      Hardware I/O   │
│  ┌────────────────┐      ┌─────────────┐    ┌───────────┐  │
│  │ Button clicks  │      │ ODMR scans  │    │ Streaming │  │
│  │ Plot updates   │ ◄──► │ Data proc.  │ ◄─►│ Polling   │  │
│  │ User input     │      │ Fitting     │    │ Register  │  │
│  └────────────────┘      └─────────────┘    └───────────┘  │
│                                 │                  │        │
│                                 └────────┬─────────┘        │
│                                          ▼                  │
│                              PyRPL TCP Socket               │
│                              (SHARED RESOURCE)              │
└─────────────────────────────────────────────────────────────┘
```

Qudi's `LogicBase` modules run on dedicated QThreads by default, while `GuiBase` runs on the main Qt thread. This enables responsive UIs during long measurements.

### What is Thread Safety?

Code is **thread-safe** if it behaves correctly when accessed by multiple threads simultaneously. Without protection, shared resources can become corrupted.

### What is a Mutex/Lock?

A **mutex** (mutual exclusion lock) ensures only one thread executes a critical section at a time:

```python
import threading
lock = threading.Lock()

def safe_operation():
    with lock:  # Only one thread can be here at a time
        do_something_with_shared_resource()
```

### What is an RLock (Reentrant Lock)?

An **RLock** allows the same thread to acquire the lock multiple times without deadlocking:

```python
lock = threading.RLock()

def outer():
    with lock:        # Acquire (count=1)
        inner()       # Same thread can re-acquire

def inner():
    with lock:        # Acquire again (count=2) - OK with RLock!
        do_work()
    # Release (count=1)
# Release (count=0)
```

This is necessary when function call chains need the lock at multiple levels.

### What is Serialization (in Threading)?

**Serialization** forces operations to execute one-after-another, even if requested concurrently:

```
Without serialization (UNSAFE):
Thread A: ████████████████████████████
Thread B:     ████████████████████████  ← Overlap causes corruption

With serialization (SAFE):
Thread A: ████████████████████████████
Thread B:                              ████████████████████████
                                       ↑ Waits for A to finish
```

---

## 2. The Problem: TCP Protocol Desynchronization

### PyRPL's TCP Architecture

PyRPL communicates with the Red Pitaya via a TCP socket on port 2222. The protocol is request-response:

```
Python                          Red Pitaya
──────                          ──────────
send(read_header)  ──────────►
                   ◄──────────  send(data)
```

Each transaction must complete atomically: send request, then receive response.

### The Race Condition

When multiple threads access the socket simultaneously, the protocol becomes desynchronized:

```
Thread A (streaming):    socket.send(header_A)
Thread B (GUI click):    socket.send(header_B)   ← INTERLEAVED!
Thread A:                socket.recv() → Gets response_B  ← WRONG!
Thread B:                socket.recv() → Gets response_A  ← WRONG!
```

### Symptoms

- `"Wrong control sequence from server"` - Header mismatch detected
- `TimeoutError: timed out` - Response never arrives (already consumed)
- `OSError: socket not connected` - Connection corrupted
- Data corruption in acquired signals
- Application crashes

### When This Occurred in Qudi

Typical failure scenario:
1. `TimeSeriesReaderLogic` continuously polls `stream_read()` in a QThread
2. User clicks "Enable Lock" in GUI, triggering a register write
3. Both operations use the same TCP socket simultaneously
4. Protocol desync → crash

---

## 3. Historical Workaround (Now Removed)

### The Stop-Modify-Restart Pattern

Before the proper fix, a workaround was implemented in `odmr_frequency_tracking_logic.py`:

```python
# WORKAROUND (now removed)
def start_tracking(self):
    ts_logic = self._time_series_logic()
    was_streaming = self._stream_active

    # 1. Stop streaming to release TCP socket
    if was_streaming:
        ts_logic.stop_reading()
        time.sleep(0.5)  # Allow thread to fully stop
        self._stream_active = False

    try:
        # 2. Perform register operation (exclusive socket access)
        lock_hw.enable_lock(True)
    finally:
        # 3. Restart streaming
        if was_streaming:
            ts_logic.start_reading()
            time.sleep(0.2)
            self._stream_active = True
```

### Problems with the Workaround

| Issue | Impact |
|-------|--------|
| ~700ms interruption per operation | Visible gaps in streaming data |
| Complex defensive code | Hard to maintain, easy to forget |
| Not a real fix | Root cause remained in PyRPL |
| Every new method needed the pattern | Code duplication |
| Status polling had to be skipped | Missing GUI updates during streaming |

---

## 4. The Solution: RLock in MonitorClient

### Core Insight

The fix belongs in PyRPL, not in every Qudi module that uses it. By making `MonitorClient` thread-safe, all callers automatically benefit.

### Why RLock Instead of Lock?

The `MonitorClient` has nested call chains that require the lock at multiple levels:

```
reads()
  └─► try_n_times()
        └─► _reads()
              └─► [on error] emptybuffer()
                    └─► restart()
                          └─► close()  ← Also needs lock!
```

With a regular `Lock`, this would deadlock. With `RLock`, the same thread can re-acquire.

### The Fix (24 lines added to PyRPL)

```python
# In MonitorClient.__init__():
self._socket_lock = threading.RLock()

# In reads(), writes(), close():
def reads(self, addr, length):
    with self._socket_lock:
        # ... existing code
```

### How Interleaving Now Works

```
Time ──────────────────────────────────────────────────────────────►

Thread A (streaming):
  │ stream_read() │ Python │ stream_read() │ Python │
  │   ┌──┬────┐   │  ...   │   ┌──┬────┐   │  ...   │
  │   │S │DATA│   │        │   │S │DATA│   │        │
  │   └──┴────┘   │        │   └──┴────┘   │        │
      ▲    ▲                    ▲    ▲
      │    │                    │    │
   Lock acquired             Lock acquired
   and released              and released

Thread B (GUI: enable_lock):
                                 │enable_lock()│
                                 │   ┌─┐       │
                                 │   │W│ ← Waits only for current
                                 │   └─┘   TCP transaction (~3ms)
                                 │         │

S = status read (~0.3ms), DATA = BRAM read (~3-10ms), W = register write (~0.3ms)
```

**Key insight**: The lock is held only during individual TCP transactions, not entire high-level operations. Other threads wait at most for one transaction to complete.

### Performance Impact

| Operation | Duration |
|-----------|----------|
| `RLock.acquire()` | ~0.5 µs |
| `RLock.release()` | ~0.5 µs |
| Network round-trip | ~100-500 µs |
| Bulk read (2000 samples) | ~3-10 ms |

**Total overhead: <0.1%** - Lock operations are 1000x faster than network I/O.

---

## 5. Implementation Details

### Changes to PyRPL

**File**: `pyrpl_new/pyrpl/redpitaya_client.py`

**Commit**: `87fef4b2` - "Add thread safety to MonitorClient using RLock"

```diff
+import threading

 class MonitorClient(object):
+    """TCP client for communication with Red Pitaya monitor_server.
+
+    Thread Safety:
+        This class is thread-safe. All socket operations are protected by a
+        reentrant lock (RLock), allowing safe concurrent access from multiple
+        threads (e.g., Qt QThreads in qudi, Python threading, or asyncio).
+    """
+
     def __init__(self, ...):
         self.logger = logging.getLogger(name=__name__)
+        self._socket_lock = threading.RLock()
         # ... rest unchanged

     def close(self):
+        with self._socket_lock:
             # ... existing code

     def reads(self, addr, length):
+        with self._socket_lock:
             # ... existing code

     def writes(self, addr, values):
+        with self._socket_lock:
             # ... existing code
```

### Changes to Qudi

**File**: `qudi-iqo-modules/src/qudi/logic/odmr_frequency_tracking_logic.py`

Removed the Stop-Modify-Restart workaround from:
- `configure_lock()` - Now direct hardware call
- `configure_lock_pi()` - Now direct hardware call
- `start_tracking()` - Now direct hardware call
- `stop_tracking()` - Now direct hardware call
- `clear_integrator()` - Now direct hardware call
- `_poll_lock_status()` - Removed "skip if streaming" logic

**Example (before → after)**:

```python
# BEFORE (with workaround): ~50 lines
def start_tracking(self):
    ts_logic = self._time_series_logic()
    was_streaming = self._stream_active
    if was_streaming:
        ts_logic.stop_reading()
        time.sleep(0.5)
        self._stream_active = False
    try:
        lock_hw.enable_lock(True)
        self._lock_enabled = True
        # ... restart logic
    except Exception as e:
        # ... error recovery

# AFTER (with RLock fix): ~20 lines
def start_tracking(self):
    lock_hw = self._odmr_lock_hw()
    lock_hw.enable_lock(True)  # Thread-safe via MonitorClient RLock
    self._lock_enabled = True
    # ... simple flow
```

---

## 6. Modules Affected

### Modules Updated (Workaround Removed)

| Module | Changes |
|--------|---------|
| `odmr_frequency_tracking_logic.py` | Removed stop/restart pattern from 5 methods |

### Modules That Benefit (No Changes Needed)

These modules now work correctly without modification:

| Module | Benefit |
|--------|---------|
| `time_series_reader_logic.py` | Can stream while other modules access FPGA |
| `redpitaya_data_instream.py` | Streaming is thread-safe |
| `redpitaya_finite_sampling_input.py` | ODMR scans are thread-safe |
| `sensitivity_sweep_logic.py` | Existing stop/restart pattern still works but is now optional |

### Modules to Review (Optional Cleanup)

The following module has the Stop-Modify-Restart pattern that could be simplified:

| Module | Location | Action |
|--------|----------|--------|
| `sensitivity_sweep_logic.py` | Lines ~1127-1135 | Optional: simplify if pattern exists |

**Note**: The workaround pattern is not harmful - it just adds unnecessary delays. Removal is optional and low priority.

---

## 7. Testing Procedure

### Test 1: Basic PyRPL Functionality

```python
from pyrpl import Pyrpl
p = Pyrpl('test_config', hostname='your_rp_ip')

# Basic operations should work
p.rp.scope.duration = 0.1
data = p.rp.scope.single()
print(f"Got {len(data[0])} samples")
```

### Test 2: Concurrent Thread Access

```python
import threading
import time
from pyrpl import Pyrpl

p = Pyrpl('test_config', hostname='your_rp_ip')
errors = []

def stream_reader():
    for i in range(100):
        try:
            data = p.rp.scan.stream_read()
        except Exception as e:
            errors.append(f"Stream error: {e}")
        time.sleep(0.01)

def register_writer():
    for i in range(50):
        try:
            p.rp.pid0.p = i * 0.1
        except Exception as e:
            errors.append(f"Write error: {e}")
        time.sleep(0.02)

p.rp.scan.stream_start()
t1 = threading.Thread(target=stream_reader)
t2 = threading.Thread(target=register_writer)
t1.start(); t2.start()
t1.join(); t2.join()
p.rp.scan.stream_stop()

print("ERRORS:", errors if errors else "None - SUCCESS!")
```

### Test 3: Qudi ODMR Tracking

1. Start Qudi with ODMR tracking GUI
2. Take ODMR scan and fit slope
3. Click "Configure" (should work instantly)
4. Click "Enable Lock" during streaming (should NOT crash)
5. Verify streaming continues smoothly
6. Change bandwidth during lock (should work)
7. Clear integrator during lock (should work)
8. Click "Disable Lock" (should work cleanly)

**Expected**: No errors, no ~700ms delays, smooth operation.

### Test 4: Log Verification

Check logs for absence of:
- `"Wrong control sequence from server"`
- `TimeoutError`
- `OSError: socket not connected`

---

## 8. Future Considerations

### Completed Work

- [x] RLock added to `MonitorClient` in PyRPL
- [x] Workaround removed from `odmr_frequency_tracking_logic.py`
- [x] Documentation consolidated

### Optional Future Cleanup

| Task | Priority | Effort |
|------|----------|--------|
| Review `sensitivity_sweep_logic.py` for unnecessary stop/restart patterns | Low | ~1 hour |
| Add thread-safety tests to PyRPL test suite | Low | ~2 hours |

### No Action Required

- Other Qudi hardware modules automatically benefit from the fix
- No changes needed to `redpitaya_data_instream.py` or `redpitaya_finite_sampling_input.py`
- The fix is backward-compatible with single-threaded PyRPL usage

---

## References

- **PyRPL Commit**: `87fef4b2` - "Add thread safety to MonitorClient using RLock"
- **PyRPL File**: `pyrpl_new/pyrpl/redpitaya_client.py`
- **Qudi File**: `qudi-iqo-modules/src/qudi/logic/odmr_frequency_tracking_logic.py`
- **PyRPL Architecture**: `pyrpl_new/CLAUDE.md`
- **Qudi Threading Model**: `qudi-core/CLAUDE.md`
