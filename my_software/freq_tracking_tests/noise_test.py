# -*- coding: utf-8 -*-
"""Automated single-resonance frequency-tracking noise test (Test 1).

Given the RF scan range of one 5-hyperfine ODMR feature, this driver:

  1. acquires an LO-swept ODMR spectrum (reusing the calibrated qudi modules),
  2. fits the central zero-crossing + discriminator slope (``fit_hyperfine``),
  3. parks the CW at that frequency (``-10 dBm``),
  4. records the high-rate (~30.5 kHz) error AND correction simultaneously
     (dual stream, single slot) open-loop and closed-loop,
  5. converts error/correction -> frequency -> magnetic field and computes the
     noise amplitude spectral density (ASD),
  6. sweeps loop settings (bandwidth / PI / deadband) to find the best ones.

Design notes (see plan ``resilient-whistling-newt.md``):

* Units: ``redpitaya_finite_sampling`` (ODMR scan) and the high-rate ``demod``
  stream both use ``signal_scale=calibration_factor=1.0`` and the SAME scan-module
  demod source, so the ODMR-scan slope (V/Hz) is numerically LSB/Hz and applies
  directly to the stream -- no LSB<->V conversion needed.
* Field conversion matches
  ``my_software/sensitivity_msmt/auswertung/sensitivity_auswertung_modular.py``
  (``magnetic_field_from_voltages``): B[nT] = (x - mean) / slope / 28.024 .
* Execution: attaches to the running qudi via the namespace server (rpyc, the
  same path the qudi jupyter kernel uses) and drives the already-activated,
  calibrated modules. The pure-math functions need no hardware (dry-testable).

The pure-analysis functions (``select_central_zero_crossing``,
``fit_central_zero_crossing``, ``error_to_field``, ``correction_to_field``,
``reconstruct_dual``, ``compute_asd``, ``asd_noise_floor``, ``analyze_asds``)
import nothing hardware-related and are unit-tested in ``__main__`` with
``--dry``.
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# --- constants ------------------------------------------------------------
FTW_PER_HZ = (2.0 ** 32) / 125e6              # ~34.359738 FTW/Hz (Red Pitaya @125 MHz)
STREAM_SAMPLE_RATE_HZ = 125e6 / 4096          # ~30517.578 Hz demod/triplet rate
GYROMAGNETIC_RATIO_NT_PER_HZ = 28.024         # Hz/nT (matches sensitivity pipeline)
HYPERFINE_SPACING_HZ = 2.158e6                # N14 hyperfine spacing


# =========================================================================
# Pure analysis (no hardware, no Qt) -- dry-testable
# =========================================================================
def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs (zero-order hold). Leading NaNs stay NaN.

    Mirrors ``pyrpl.hardware_modules.scan.Scan._ffill``.
    """
    a = np.asarray(a, dtype=np.float64)
    valid = ~np.isnan(a)
    if not valid.any():
        return a.copy()
    idx = np.where(valid, np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    out[:int(np.argmax(valid))] = np.nan
    return out


def reconstruct_dual(words: np.ndarray, nslots: int = 1,
                     to_hz_corr: bool = True) -> Dict[str, np.ndarray]:
    """Local port of ``Scan.reconstruct_dual_hop_series`` (self-describing triplets).

    ``words`` is the contiguous int32-as-float64 stream ([err, corr, step] per
    demod strobe; NaN = transport loss). Returns ``{'err': (N,T) LSB,
    'corr': (N,T) Hz}`` with fresh-while-live + zero-order-hold semantics. For the
    single-resonance test ``nslots=1`` (every sample is live, no hold needed), but
    the general path is kept so the same code serves Test 2.
    """
    words = np.asarray(words, dtype=np.float64).ravel()
    n3 = (words.size // 3) * 3
    if n3 == 0:
        z = np.full((int(nslots), 0), np.nan)
        return {'err': z, 'corr': z.copy()}
    trip = words[:n3].reshape(-1, 3)
    err_col, corr_col, step_raw = trip[:, 0], trip[:, 1], trip[:, 2]
    T = trip.shape[0]
    step_filled = _ffill(step_raw)
    finite = np.isfinite(step_filled)
    step_int = np.where(finite, np.rint(step_filled), -1).astype(np.int64)
    nslots = int(nslots)
    out_err = np.full((nslots, T), np.nan)
    out_corr = np.full((nslots, T), np.nan)
    for r in range(nslots):
        live = (step_int == r)
        e = np.full(T, np.nan); e[live] = err_col[live]
        c = np.full(T, np.nan); c[live] = corr_col[live]
        ef = _ffill(e); cf = _ffill(c)
        ef[live & np.isnan(err_col)] = np.nan
        cf[live & np.isnan(corr_col)] = np.nan
        out_err[r] = ef
        out_corr[r] = cf
    if to_hz_corr:
        out_corr = out_corr / FTW_PER_HZ
    return {'err': out_err, 'corr': out_corr}


def reconstruct_dual_fresh(words: np.ndarray, nslots: int,
                           dead_time_s: float = 0.0) -> Dict[str, Any]:
    """Decode a dual-quantity stream into FRESH-only per-resonance samples.

    IMPORTANT (timing): the FPGA streams a triplet ONLY on a demod strobe, and the
    active lock-in chain is frozen (no strobe) during the per-hop settle/trigger
    dead-time (``lock_in.v`` ``valid & aclken``; ``odmr_multitrack.v`` settle
    window). So the settle/trigger dead-time is ZERO-WIDTH in the word stream: the
    last sample of one dwell and the first of the next are adjacent triplets even
    though real time elapsed between them. ``dead_time_s`` is that missing wall-clock
    time per FULL visit cycle (= N*(settle+trigger)); pass it so the per-resonance
    ``eff_rate_hz`` (used for the ASD frequency axis) is the TRUE visit rate
    ``1/(live_period + dead_time_s)`` rather than the live-only rate (which
    overestimates by ~20% and pushes real lines, e.g. 50 Hz mains, to ~60 Hz).

    Unlike :func:`reconstruct_dual` (which zero-order-holds each resonance across
    the intervals it is parked, giving a (N,T) staircase at the full ~30.5 kHz
    aggregate rate), this returns only the samples that are genuinely *live* for
    each resonance, i.e. acquired during that resonance's own dwell. This is the
    key to a HONEST 2-resonance ASD: the ZOH staircase injects a sample-and-hold
    rolloff plus a comb at the hop cadence (1/(N*(dwell+settle))) that has nothing
    to do with the tracking noise -- the genuine per-resonance update happens only
    once per hop cycle.

    Two products are returned per resonance:

    * ``fresh_*`` -- every live sample (bursty: ~dwell*fs samples per visit, then a
      gap while parked). Useful to inspect within-dwell transients.
    * ``visit_*`` -- ONE value per dwell visit (the last live sample of each visit =
      the settled loop estimate for that visit), uniformly spaced at the hop cadence
      -> the series to ASD for genuine per-resonance tracking noise. ``visit_idx``
      are the global triplet indices (``/fs`` -> seconds) and ``eff_rate_hz`` is the
      measured per-resonance visit rate (aggregate_fs / median visit spacing).

    Correction columns are converted FTW -> Hz; error columns stay raw LSB.
    """
    words = np.asarray(words, dtype=np.float64).ravel()
    n3 = (words.size // 3) * 3
    nslots = int(nslots)
    empty = {'fresh_err': [np.array([]) for _ in range(nslots)],
             'fresh_corr': [np.array([]) for _ in range(nslots)],
             'fresh_idx': [np.array([], dtype=np.int64) for _ in range(nslots)],
             'visit_err': [np.array([]) for _ in range(nslots)],
             'visit_corr': [np.array([]) for _ in range(nslots)],
             'visit_idx': [np.array([], dtype=np.int64) for _ in range(nslots)],
             'eff_rate_hz': [float('nan')] * nslots,
             'eff_rate_live_hz': [float('nan')] * nslots,
             'visit_jitter': [float('nan')] * nslots}
    if n3 == 0:
        return empty
    trip = words[:n3].reshape(-1, 3)
    err_col, corr_col, step_raw = trip[:, 0], trip[:, 1], trip[:, 2]
    step_filled = _ffill(step_raw)
    finite_step = np.isfinite(step_filled)
    step_int = np.where(finite_step, np.rint(step_filled), -1).astype(np.int64)

    fresh_err, fresh_corr, fresh_idx = [], [], []
    visit_err, visit_corr, visit_idx = [], [], []
    eff_rate, eff_rate_live, jitter = [], [], []
    for r in range(nslots):
        live = (step_int == r)
        fresh_mask = live & np.isfinite(err_col)
        fi = np.where(fresh_mask)[0]
        fresh_idx.append(fi)
        fresh_err.append(err_col[fi])
        fresh_corr.append(corr_col[fi] / FTW_PER_HZ)
        # one value per contiguous live run (= one dwell visit): take its last
        # live sample (the settled loop output for that visit).
        live_i = live.astype(np.int8)
        edges = np.diff(np.r_[np.int8(0), live_i, np.int8(0)])
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1   # inclusive last index of each run
        vix, ve, vc = [], [], []
        for s, e in zip(starts, ends):
            seg_fresh = fresh_mask[s:e + 1]
            if seg_fresh.any():
                j = s + int(np.where(seg_fresh)[0][-1])
                vix.append(j); ve.append(err_col[j]); vc.append(corr_col[j] / FTW_PER_HZ)
        vix = np.asarray(vix, dtype=np.int64)
        visit_idx.append(vix)
        visit_err.append(np.asarray(ve, dtype=np.float64))
        visit_corr.append(np.asarray(vc, dtype=np.float64))
        if vix.size >= 3:
            d = np.diff(vix.astype(np.float64))
            med = float(np.median(d))
            live_period = med / STREAM_SAMPLE_RATE_HZ if med > 0 else float('nan')
            # TRUE visit period adds back the per-cycle dead-time absent from the stream
            true_period = live_period + float(dead_time_s)
            eff_rate.append(1.0 / true_period if true_period > 0 else float('nan'))
            eff_rate_live.append(STREAM_SAMPLE_RATE_HZ / med if med > 0 else float('nan'))
            jitter.append(float(np.std(d) / med) if med > 0 else float('nan'))
        else:
            eff_rate.append(float('nan')); eff_rate_live.append(float('nan'))
            jitter.append(float('nan'))
    return {'fresh_err': fresh_err, 'fresh_corr': fresh_corr, 'fresh_idx': fresh_idx,
            'visit_err': visit_err, 'visit_corr': visit_corr, 'visit_idx': visit_idx,
            'eff_rate_hz': eff_rate, 'eff_rate_live_hz': eff_rate_live,
            'visit_jitter': jitter}


def select_central_zero_crossing(zc_freqs: np.ndarray, zc_slopes: np.ndarray,
                                 envelope_center: Optional[float] = None
                                 ) -> Tuple[float, float]:
    """Pick the central zero-crossing (the m_I=0 line) of a 5-hyperfine feature.

    Central = the crossing nearest the envelope center (mean of valid crossings if
    not given). Returns (frequency_hz, slope). NaNs are dropped first.
    """
    zc_freqs = np.asarray(zc_freqs, dtype=np.float64)
    zc_slopes = np.asarray(zc_slopes, dtype=np.float64)
    good = np.isfinite(zc_freqs) & np.isfinite(zc_slopes) & (np.abs(zc_slopes) > 0)
    zc_freqs, zc_slopes = zc_freqs[good], zc_slopes[good]
    if zc_freqs.size == 0:
        raise ValueError('No valid zero-crossings to select from.')
    order = np.argsort(zc_freqs)
    zc_freqs, zc_slopes = zc_freqs[order], zc_slopes[order]
    center = float(np.mean(zc_freqs)) if envelope_center is None else float(envelope_center)
    k = int(np.argmin(np.abs(zc_freqs - center)))
    return float(zc_freqs[k]), float(zc_slopes[k])


def fit_central_zero_crossing(rf: np.ndarray, demod: np.ndarray,
                              n_features: int = 5,
                              envelope_center: Optional[float] = None,
                              slope_fit_halfwidth_hz: float = 0.12e6) -> Dict[str, Any]:
    """Find the 5-hyperfine feature's central zero-crossing + discriminator slope.

    Primary method is ``find_zero_crossings_direct`` (RELATIVE thresholds: it
    baseline-subtracts, detects the signal envelope, and keeps sign-changes by slope
    + min spacing). This is robust on raw LSB-scale demod data, unlike
    ``fit_hyperfine`` whose default prominence/height are absolute (0.02) and only
    worked here because ``n_most_prominent_peaks`` grabbed the biggest 5. The slope
    at the central crossing is then REFINED by a local linear fit over
    +/- ``slope_fit_halfwidth_hz`` of the raw data (needs adequate resolution -> use a
    feature-sized scan window, not the wide search range). Falls back to
    ``fit_hyperfine`` (normalized) then ``fit_odmr_robust``.

    Returns the central crossing, slope (LSB/Hz), all detected crossings, and the
    detected envelope bounds (so callers can size a tight re-scan).
    """
    from my_software.tools.fitting import (find_zero_crossings_direct,
                                           fit_hyperfine, fit_odmr_robust)
    rf = np.asarray(rf, dtype=np.float64)
    demod = np.asarray(demod, dtype=np.float64)

    method = 'direct'
    env = (float('nan'), float('nan'))
    zc, sl, info = find_zero_crossings_direct(
        rf, demod, smooth_window_hz=0.3e6,
        min_crossing_spacing_hz=0.75 * HYPERFINE_SPACING_HZ,
        signal_envelope_threshold=0.12)
    zc = np.asarray(zc, dtype=np.float64); sl = np.asarray(sl, dtype=np.float64)
    if zc.size:
        env = (float(info.get('envelope_start_hz', np.nan)),
               float(info.get('envelope_end_hz', np.nan)))
    if zc.size < 1:
        # fallback 1: fit_hyperfine on amplitude-normalized data (so 0.02 thresholds
        # become ~2% of full scale); rescale slope back to LSB/Hz afterwards.
        scale = float(np.nanmax(demod) - np.nanmin(demod)) or 1.0
        yn = (demod - np.nanmedian(demod)) / scale
        res = fit_hyperfine(rf, yn, n_most_prominent_peaks=n_features,
                            max_pair_distance_hz=0.75 * HYPERFINE_SPACING_HZ)
        zc = np.asarray(res.get('zero_crossing_frequencies [Hz]', [np.nan]), float)
        sl = np.asarray(res.get('zero_crossing_slopes [V/Hz]', [np.nan]), float) * scale
        method = 'fit_hyperfine_norm'
        if np.isfinite(zc).sum() < 2:
            rob = fit_odmr_robust(rf, demod, n_expected_features=n_features,
                                  hyperfine_spacing_hz=HYPERFINE_SPACING_HZ)
            zc = np.asarray(rob.get('zero_crossing_frequencies [Hz]', [np.nan]), float)
            sl = np.asarray(rob.get('zero_crossing_slopes [V/Hz]', [np.nan]), float)
            method = 'fit_odmr_robust:' + str(rob.get('method_used', '?'))

    if envelope_center is None and np.all(np.isfinite(env)):
        envelope_center = 0.5 * (env[0] + env[1])
    f_zc, slope = select_central_zero_crossing(zc, sl, envelope_center)

    # refine the discriminator slope with a local linear fit on the RAW data
    m = np.isfinite(demod) & (np.abs(rf - f_zc) <= slope_fit_halfwidth_hz)
    if m.sum() >= 4:
        slope = float(np.polyfit(rf[m], demod[m], 1)[0])
    return {'f_zc_rf': f_zc, 'slope_lsb_per_hz': slope, 'method': method,
            'n_crossings': int(np.isfinite(zc).sum()),
            'all_zc': zc.tolist(), 'all_slopes': sl.tolist(),
            'envelope_hz': list(env), 'slope_fit_n': int(m.sum())}


def error_to_field(err_lsb: np.ndarray, slope_lsb_per_hz: float) -> np.ndarray:
    """error[LSB] -> magnetic field [nT] (mean-subtracted).

    Matches ``magnetic_field_from_voltages``: B = (x - mean)/slope/gyromag.
    """
    x = np.asarray(err_lsb, dtype=np.float64)
    x = x - np.nanmean(x)
    return x / slope_lsb_per_hz / GYROMAGNETIC_RATIO_NT_PER_HZ


def correction_to_field(corr_hz: np.ndarray) -> np.ndarray:
    """correction[Hz] -> magnetic field [nT] (mean-subtracted)."""
    x = np.asarray(corr_hz, dtype=np.float64)
    x = x - np.nanmean(x)
    return x / GYROMAGNETIC_RATIO_NT_PER_HZ


def compute_asd(samples_B: np.ndarray, sample_rate: float,
                window: str = 'hann', nperseg: Optional[int] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Amplitude spectral density (nT/sqrt(Hz)) via Welch. NaNs are interpolated.

    Default ``nperseg = sample_rate`` gives 1 Hz bins (matches the sensitivity
    pipeline). Returns (freqs, asd).
    """
    from scipy.signal import welch
    x = np.asarray(samples_B, dtype=np.float64)
    if np.any(~np.isfinite(x)):  # bridge transport-loss gaps for the PSD
        idx = np.arange(x.size)
        good = np.isfinite(x)
        x = np.interp(idx, idx[good], x[good]) if good.any() else np.zeros_like(x)
    if nperseg is None:
        nperseg = int(sample_rate)
    nperseg = max(16, min(nperseg, x.size))
    f, pxx = welch(x, fs=sample_rate, nperseg=nperseg, noverlap=0, window=window)
    return f, np.sqrt(pxx)


def asd_noise_floor(freqs: np.ndarray, asd: np.ndarray,
                    f1: float, f2: float) -> float:
    """Mean ASD in [f1, f2] (the sensitivity-band noise floor)."""
    freqs = np.asarray(freqs); asd = np.asarray(asd)
    m = (freqs >= f1) & (freqs <= f2)
    return float(np.mean(asd[m])) if m.any() else float('nan')


def analyze_asds(freqs: np.ndarray, asd_open_err: np.ndarray,
                 asd_closed_corr: np.ndarray, asd_closed_err: np.ndarray,
                 floor_band: Tuple[float, float] = (200.0, 1400.0)) -> Dict[str, Any]:
    """Extract servo metrics from the three field ASDs (all share ``freqs``).

    * noise floors in ``floor_band`` for each trace,
    * servo bandwidth estimate: the frequency where the closed-loop residual
      error ASD rises back to ~1/sqrt(2) of the open-loop error ASD (the loop
      stops suppressing noise above its bandwidth),
    * servo bump: peak of the closed-loop residual/open-loop ratio above ~50 Hz
      (a >1 ratio indicates noise amplification / a control resonance).
    """
    freqs = np.asarray(freqs)
    out: Dict[str, Any] = {
        'floor_band_hz': floor_band,
        'floor_open_err_nT_sqrtHz': asd_noise_floor(freqs, asd_open_err, *floor_band),
        'floor_closed_corr_nT_sqrtHz': asd_noise_floor(freqs, asd_closed_corr, *floor_band),
        'floor_closed_err_nT_sqrtHz': asd_noise_floor(freqs, asd_closed_err, *floor_band),
    }
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.asarray(asd_closed_err) / np.asarray(asd_open_err)
    band = freqs > 5.0
    # servo bandwidth: lowest freq (in band) where suppression ratio crosses 1/sqrt(2)
    thr = 1.0 / np.sqrt(2.0)
    sb = np.nan
    fb = freqs[band]; rb = ratio[band]
    crossing = np.where(np.isfinite(rb) & (rb >= thr))[0]
    if crossing.size:
        sb = float(fb[crossing[0]])
    out['servo_bandwidth_hz'] = sb
    bump = freqs > 50.0
    rbump = ratio[bump]
    if np.any(np.isfinite(rbump)):
        kmax = int(np.nanargmax(rbump))
        out['servo_bump_ratio'] = float(rbump[kmax])
        out['servo_bump_freq_hz'] = float(freqs[bump][kmax])
    else:
        out['servo_bump_ratio'] = float('nan')
        out['servo_bump_freq_hz'] = float('nan')
    return out


# =========================================================================
# Loop-setting description
# =========================================================================
@dataclass
class LockSettings:
    bandwidth_hz: float = 150.0
    pi: bool = False
    zero_ratio: float = 3.0
    deadband_lsb: Optional[int] = None       # None -> deadband off
    max_correction_hz: float = 1e6

    def label(self) -> str:
        mode = f'PI(a={self.zero_ratio:g})' if self.pi else 'I'
        db = 'off' if self.deadband_lsb is None else f'{self.deadband_lsb}'
        return f'BW{self.bandwidth_hz:g}_{mode}_db{db}'


# =========================================================================
# qudi attach (rpyc namespace -- same path the jupyter kernel uses)
# =========================================================================
class QudiHandles:
    """Live handles to the running, calibrated qudi modules + pyrpl sub-modules."""

    def __init__(self):
        from qudi.core.qudikernel import QudiKernelClient
        from rpyc.utils.classic import obtain
        self._obtain = obtain
        self._client = QudiKernelClient()
        self._client.connect()
        mods = self._client.get_active_modules()
        avail = list(mods.keys())

        def _pick(*candidates):
            for c in candidates:
                if c in mods:
                    return c, mods[c]
            return None, None

        mw_name, self.mw = _pick('mw_source_synthnv', 'mw_source_rp_windfreak')
        # any active OdmrLogic-derived module can drive the LO-swept ODMR scan
        # (all share the same microwave + data_scanner hardware)
        odmr_name, self.odmr = _pick('odmr_frequency_tracking_logic',
                                     'multi_resonance_odmr_tracking_logic', 'odmr_logic')
        lock_name, self.lock_hw = _pick('redpitaya_odmr_lock')
        missing = [n for n, m in (('microwave', self.mw), ('odmr-scan logic', self.odmr),
                                  ('redpitaya_odmr_lock', self.lock_hw)) if m is None]
        if missing:
            raise RuntimeError(
                f'Required qudi modules not active: {missing}. Available: {avail}. '
                f'Activate them (e.g. via the GUI) and retry.')
        self.names = {'mw': mw_name, 'odmr': odmr_name, 'lock_hw': lock_name}
        # pyrpl sub-modules held by the lock hardware module
        self.scan = self.lock_hw._scan
        self.lock = self.lock_hw._lock
        self.fgen3 = self.lock_hw._fgen3

    def obtain(self, x):
        """Bring an rpyc netref fully local (e.g. numpy arrays)."""
        try:
            return self._obtain(x)
        except Exception:
            return x

    def close(self):
        try:
            self._client.disconnect()
        except Exception:
            pass


# =========================================================================
# Hardware-driving steps
# =========================================================================
def acquire_spectrum(h: QudiHandles, rf_min: float, rf_max: float, points: int = 1000,
                     power: float = -10.0, runtime: float = 6.0,
                     data_rate: float = 1000.0,
                     timeout: float = 180.0) -> Tuple[np.ndarray, np.ndarray]:
    """Run an LO-swept ODMR scan over [rf_min, rf_max] and return (rf, demod_LSB).

    Uses ``points`` frequency points at ``data_rate`` Hz (1 ms/point at 1 kHz).
    """
    odmr = h.odmr
    if odmr.module_state() != 'idle':
        odmr.stop_odmr_scan()
        _wait_idle(odmr, 10.0)
    odmr.set_scan_power(power)
    try:
        odmr.set_data_rate(float(data_rate))
    except Exception as e:
        print(f'[scan] set_data_rate({data_rate}) skipped: {e}')
    odmr.set_frequency_range(float(rf_min), float(rf_max), int(points), 0)
    odmr.set_runtime(float(runtime))
    odmr.start_odmr_scan()
    t0 = time.time()
    time.sleep(min(runtime, 1.0))
    while odmr.module_state() != 'idle' and (time.time() - t0) < timeout:
        time.sleep(0.2)
    if odmr.module_state() != 'idle':
        odmr.stop_odmr_scan()
        _wait_idle(odmr, 10.0)
    rf = np.asarray(h.obtain(odmr.frequency_data[0]), dtype=np.float64)
    sig = odmr.signal_data
    ch = list(sig.keys())[0]
    demod = np.asarray(h.obtain(sig[ch][0]), dtype=np.float64)
    return rf, demod


def scan_and_fit_feature(h: QudiHandles, rf_min: float, rf_max: float, *,
                         points: int = 1000, power: float = -10.0,
                         data_rate: float = 1000.0,
                         coarse_runtime: float = 4.0, fine_runtime: float = 6.0,
                         min_halfwidth_hz: float = 6.0e6,
                         max_halfwidth_hz: float = 12.0e6,
                         margin_hz: float = 2.0e6
                         ) -> Dict[str, Any]:
    """Two-stage acquisition: locate the 5-hyperfine feature in a wide SEARCH range,
    then re-scan a TIGHT, feature-sized window and fit it.

    The user-supplied [rf_min, rf_max] is a coarse search band (tens of MHz) that is
    far wider than one feature (~9 MHz of hyperfine lines). Scanning it directly gives
    poor resolution (e.g. 40 MHz / 1000 pts = 40 kHz/pt -> only ~5 points across the
    +/-0.1 MHz slope-fit window) and lets baseline/other structure confuse the fit.
    So: (1) coarse scan the search band and find the feature ENVELOPE
    (``find_zero_crossings_direct`` via ``fit_central_zero_crossing``); (2) re-scan
    center +/- half_width (sized from the envelope: ``env_width/2 + margin``, clamped to
    [min,max]) at full ``points`` -> ~12-15 kHz/pt; (3) fit the tight scan. This is
    what the GUI fit window should also look like.

    Returns ``{rf, demod, fit, center_hz, halfwidth_hz, coarse_fit}`` where rf/demod are
    the TIGHT scan and fit is from the tight scan.
    """
    # --- stage 1: coarse locate over the search band ---
    rf_c, demod_c = acquire_spectrum(h, rf_min, rf_max, points, power,
                                     runtime=coarse_runtime, data_rate=data_rate)
    coarse = fit_central_zero_crossing(rf_c, demod_c)
    center = float(coarse['f_zc_rf'])
    env = coarse.get('envelope_hz', [np.nan, np.nan])
    if np.all(np.isfinite(env)) and (env[1] > env[0]):
        half = 0.5 * (env[1] - env[0]) + margin_hz
    else:
        half = min_halfwidth_hz
    half = float(np.clip(half, min_halfwidth_hz, max_halfwidth_hz))
    lo = max(float(rf_min) - margin_hz, center - half)
    hi = min(float(rf_max) + margin_hz, center + half)
    print(f'[locate] coarse center={center/1e9:.6f} GHz, envelope={[round(e/1e6,2) for e in env]} MHz '
          f'-> tight scan [{lo/1e9:.5f}, {hi/1e9:.5f}] GHz ({(hi-lo)/1e6:.1f} MHz, '
          f'{(hi-lo)/points/1e3:.1f} kHz/pt)')

    # --- stage 2: tight re-scan + fit ---
    rf_t, demod_t = acquire_spectrum(h, lo, hi, points, power,
                                     runtime=fine_runtime, data_rate=data_rate)
    fit = fit_central_zero_crossing(rf_t, demod_t, envelope_center=center)
    return {'rf': rf_t, 'demod': demod_t, 'fit': fit, 'center_hz': center,
            'halfwidth_hz': half, 'coarse_fit': coarse}


def park_cw(h: QudiHandles, f_zc_rf: float, power: float = -10.0) -> None:
    """Park the CW microwave at the central zero-crossing RF frequency."""
    mw = h.mw
    if mw.module_state() != 'idle':
        mw.off()
        time.sleep(0.1)
    mw.set_cw(float(f_zc_rf), float(power))
    mw.cw_on()
    time.sleep(0.3)


def setup_single_resonance(h: QudiHandles) -> None:
    """Put the hardware in the clean single-resonance signal path.

    Disables the multi-resonance freeze/settle oscillator (so the FPGA reverts to
    the legacy single-resonance demod path) and ensures the lock is off + the hop
    loop is stopped. Safe no-ops if those sub-modules are absent.
    """
    try:
        h.lock_hw.enable_oscillator(False)
    except Exception as e:
        print(f'[setup] enable_oscillator(False) skipped: {e}')
    try:
        h.lock.enable = False
    except Exception:
        pass
    try:
        h.scan.continuous_hop_stop()
    except Exception:
        pass


def _probe_lock(h: QudiHandles, invert: bool, slope: float, settings: 'LockSettings',
                probe: float = 1.5) -> Dict[str, Any]:
    """Enable the lock briefly with a given invert flag; report acquisition quality."""
    lock, lock_hw = h.lock, h.lock_hw
    lock.enable = False
    lock_hw.set_invert(bool(invert))
    lock_hw.set_max_correction_hz(settings.max_correction_hz)
    lock_hw.set_bandwidth(settings.bandwidth_hz, slope, pi=settings.pi,
                          zero_ratio=settings.zero_ratio)
    lock_hw.clear()
    lock.enable = True
    time.sleep(probe)
    errs, sat, locked = [], False, False
    for _ in range(10):
        try:
            st = dict(h.obtain(lock_hw.get_status()))
            errs.append(float(st.get('error_lsb', float(lock.error_lsb))))
            sat = sat or bool(st.get('saturated', False))
            locked = locked or bool(st.get('locked', False))
        except Exception:
            errs.append(float(lock.error_lsb))
        time.sleep(0.03)
    lock.enable = False
    lock_hw.clear()
    return {'invert': invert, 'abs_err': float(np.mean(np.abs(errs))),
            'saturated': sat, 'locked': locked}


def ensure_lock_polarity(h: QudiHandles, slope: float, settings: 'LockSettings',
                         default_invert: bool = False) -> Dict[str, Any]:
    """Choose the ``invert`` flag that actually acquires lock (closed-loop probe).

    Tries the default sign (USB->False) then the opposite; keeps whichever gives the
    smaller residual error without saturating. Wrong polarity just runs the
    correction to the (±1 MHz) saturation limit briefly, then we flip -- harmless.
    """
    a = _probe_lock(h, default_invert, slope, settings)
    b = _probe_lock(h, not default_invert, slope, settings)
    def score(r):  # prefer not-saturated, then smallest |error|
        return (1 if r['saturated'] else 0, r['abs_err'])
    best = min((a, b), key=score)
    h.lock_hw.set_invert(bool(best['invert']))
    return {'chosen': best, 'default': a, 'flipped': b}


def record_traces(h: QudiHandles, duration: float, closed_loop: bool,
                  slope_lsb_per_hz: float, settings: Optional[LockSettings] = None,
                  poll: float = 0.05) -> Dict[str, Any]:
    """Record simultaneous error+correction via the single-slot dual stream.

    Opens the dual stream (nslots=1, no hop loop), optionally enables the lock with
    ``settings``, drains for ``duration`` s, and reconstructs err (LSB) + corr (Hz).
    """
    scan, lock, lock_hw = h.scan, h.lock, h.lock_hw
    # ensure clean state
    try:
        scan.hop_stream_stop()
    except Exception:
        pass
    lock.enable = False

    if closed_loop:
        s = settings or LockSettings()
        lock_hw.set_max_correction_hz(s.max_correction_hz)
        lock_hw.set_bandwidth(s.bandwidth_hz, slope_lsb_per_hz,
                              pi=s.pi, zero_ratio=s.zero_ratio)
        if s.deadband_lsb is not None:
            lock.deadband_lsb = int(s.deadband_lsb)
            lock.deadband_enable = True
        else:
            lock.deadband_enable = False
        lock_hw.clear()

    scan.hop_stream_start(input_source='dual')
    if closed_loop:
        lock.enable = True
        time.sleep(0.2)  # let the loop settle before we keep the analysis window

    words: List[np.ndarray] = []
    t0 = time.time()
    while (time.time() - t0) < duration:
        w = h.obtain(scan.hop_stream_read())
        if w is not None and len(w):
            words.append(np.asarray(w, dtype=np.float64))
        time.sleep(poll)
    stats = {}
    try:
        stats = dict(h.obtain(scan.push_stream_stats()))
    except Exception:
        pass
    if closed_loop:
        lock.enable = False
    scan.hop_stream_stop()

    allw = np.concatenate(words) if words else np.array([], dtype=np.float64)
    rec = reconstruct_dual(allw, nslots=1, to_hz_corr=True)
    err = rec['err'][0]
    corr = rec['corr'][0]
    loss = float(np.mean(np.isnan(allw))) if allw.size else 1.0
    times = np.arange(err.size, dtype=np.float64) / STREAM_SAMPLE_RATE_HZ
    return {'times': times, 'err_lsb': err, 'corr_hz': corr,
            'sample_rate': STREAM_SAMPLE_RATE_HZ, 'loss_frac': loss,
            'n_words': int(allw.size), 'stream_stats': stats}


# =========================================================================
# High-level Test 1 orchestration
# =========================================================================
def run_test1(h: QudiHandles, rf_min: float, rf_max: float, *,
              points: int = 1000, power: float = -10.0, data_rate: float = 1000.0,
              open_duration: float = 30.0, closed_duration: float = 30.0,
              bandwidth_hz: float = 150.0, pi: bool = False,
              floor_band: Tuple[float, float] = (200.0, 1400.0),
              outdir: Optional[str] = None) -> Dict[str, Any]:
    """Single-resonance magnetometry comparison: open-loop (error->B, CW on at the
    zero-crossing, lock off) vs closed-loop (correction->B). Both at ``power`` dBm.

    Saves traces, ASDs, a summary, AND plots (spectrum+fit, time traces, ASD
    comparison). Returns the summary incl. the open-loop error->B ASD so a
    subsequent ``sweep_settings`` can use it as the servo-bump reference.
    """
    outdir = _make_outdir(outdir)
    setup_single_resonance(h)
    sf = scan_and_fit_feature(h, rf_min, rf_max, points=points, power=power,
                              data_rate=data_rate)
    rf, demod, fit = sf['rf'], sf['demod'], sf['fit']
    f_zc, slope = fit['f_zc_rf'], abs(fit['slope_lsb_per_hz'])
    print(f'[fit] central zero-crossing = {f_zc/1e9:.6f} GHz, slope = {slope:.4g} LSB/Hz '
          f'({fit["method"]}, {fit["n_crossings"]} crossings)')
    _plot_spectrum_fit(outdir, rf, demod, fit)

    # park CW at the zero-crossing (CW on, -10 dBm) for BOTH measurements
    park_cw(h, f_zc, power)

    # --- open-loop magnetometry: lock OFF, error -> B ---
    open_rec = record_traces(h, open_duration, closed_loop=False, slope_lsb_per_hz=slope)
    print(f'[open]   {open_rec["n_words"]} words, loss={open_rec["loss_frac"]*100:.3f}%')

    # --- closed-loop magnetometry: lock ON, correction -> B ---
    settings = LockSettings(bandwidth_hz=bandwidth_hz, pi=pi)
    pol = ensure_lock_polarity(h, slope, settings)
    print(f'[polarity] invert={pol["chosen"]["invert"]} '
          f'(|err|={pol["chosen"]["abs_err"]:.1f} LSB, locked={pol["chosen"]["locked"]}, '
          f'sat={pol["chosen"]["saturated"]})')
    closed_rec = record_traces(h, closed_duration, closed_loop=True,
                               slope_lsb_per_hz=slope, settings=settings)
    print(f'[closed] {closed_rec["n_words"]} words, loss={closed_rec["loss_frac"]*100:.3f}%')

    fs = STREAM_SAMPLE_RATE_HZ
    B_open_err = error_to_field(open_rec['err_lsb'], slope)        # open-loop measurement
    B_closed_corr = correction_to_field(closed_rec['corr_hz'])     # closed-loop measurement
    B_closed_err = error_to_field(closed_rec['err_lsb'], slope)    # in-loop residual (diagnostic)
    f_oe, a_oe = compute_asd(B_open_err, fs)
    f_cc, a_cc = compute_asd(B_closed_corr, fs)
    f_ce, a_ce = compute_asd(B_closed_err, fs)
    metrics = analyze_asds(f_oe, a_oe, a_cc, a_ce, floor_band=floor_band)

    summary = {'rf_range_hz': [rf_min, rf_max], 'power_dbm': power,
               'scan_points': points, 'scan_data_rate_hz': data_rate, 'fit': fit,
               'polarity': pol, 'settings': asdict(settings), 'metrics': metrics,
               'open_loss_frac': open_rec['loss_frac'],
               'closed_loss_frac': closed_rec['loss_frac'], 'outdir': outdir}
    _save_run(outdir, rf, demod, open_rec, closed_rec,
              (f_oe, a_oe), (f_ce, a_ce), (f_cc, a_cc), summary)
    _plot_timetraces(outdir, open_rec, closed_rec, slope)
    _plot_asd_compare(outdir, f_oe, a_oe, a_cc, a_ce, settings.label())
    # stash the open-loop reference for the sweep
    summary['_open_ref'] = (f_oe.tolist(), a_oe.tolist())
    print(f'[done] open-loop(error) floor={metrics["floor_open_err_nT_sqrtHz"]:.3g} | '
          f'closed-loop(correction) floor={metrics["floor_closed_corr_nT_sqrtHz"]:.3g} nT/sqrtHz '
          f'({floor_band[0]:.0f}-{floor_band[1]:.0f} Hz); '
          f'servo BW~{metrics["servo_bandwidth_hz"]:.0f} Hz; '
          f'bump x{metrics["servo_bump_ratio"]:.2f}@{metrics["servo_bump_freq_hz"]:.0f}Hz')
    print(f'[out] {outdir}')
    return summary


def sweep_settings(h: QudiHandles, slope_lsb_per_hz: float, *,
                   open_ref: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                   bandwidths: Tuple[float, ...] = (50, 100, 150, 200, 300),
                   modes: Tuple[Tuple[bool, float], ...] = ((False, 3.0), (True, 3.0)),
                   deadbands: Tuple[Optional[int], ...] = (None,),
                   duration: float = 12.0,
                   floor_band: Tuple[float, float] = (1.0, 50.0),
                   err_std_gate_lsb: float = 5e5,
                   outdir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Per-setting closed-loop measurement; rank stable settings by cleanliness.

    For each setting records the closed-loop correction->B ASD (the magnetometer
    output) plus the in-loop residual error->B ASD. Metrics:

    * ``stable``: not saturated and broadband ``err_std`` below ``err_std_gate_lsb``
      (PI oscillation / runaway is rejected),
    * ``servo_bump``: peak of (residual-error ASD / open-loop-error ASD) for
      f>50 Hz -- how much the loop AMPLIFIES noise (needs ``open_ref``); >1 is bad,
    * ``track_floor``: correction->B ASD floor in ``floor_band`` (low band where the
      loop tracks; the achieved magnetometry noise),
    * ``track_bw``: -3 dB rolloff of the correction ASD (tracking bandwidth).

    Ranking: stable first, then smallest servo bump, then highest tracking
    bandwidth. Saves a JSON table and an overlay plot of all correction ASDs.
    Call after ``run_test1`` (CW parked); pass ``open_ref=summary['_open_ref']``.
    """
    outdir = _make_outdir(outdir)
    fs = STREAM_SAMPLE_RATE_HZ
    f_ref = a_ref = None
    if open_ref is not None:
        f_ref = np.asarray(open_ref[0], dtype=np.float64)
        a_ref = np.asarray(open_ref[1], dtype=np.float64)
    rows: List[Dict[str, Any]] = []
    curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for bw in bandwidths:
        for pi, zr in modes:
            for db in deadbands:
                s = LockSettings(bandwidth_hz=bw, pi=pi, zero_ratio=zr, deadband_lsb=db)
                rec = record_traces(h, duration, closed_loop=True,
                                    slope_lsb_per_hz=slope_lsb_per_hz, settings=s)
                B_corr = correction_to_field(rec['corr_hz'])
                B_err = error_to_field(rec['err_lsb'], slope_lsb_per_hz)
                f, a_corr = compute_asd(B_corr, fs)
                _, a_err = compute_asd(B_err, fs)
                curves.append((s.label(), f, a_corr))
                err_std = float(np.nanstd(rec['err_lsb']))
                stable = (err_std < err_std_gate_lsb) and not rec.get('saturated', False)
                bump = float('nan')
                if a_ref is not None and np.array_equal(f, f_ref):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = a_err / a_ref
                    m = f > 50.0
                    bump = float(np.nanmax(ratio[m])) if np.any(np.isfinite(ratio[m])) else float('nan')
                track_floor = asd_noise_floor(f, a_corr, *floor_band)
                track_bw = _minus3db_bw(f, a_corr, floor_band)
                row = {'label': s.label(), 'settings': asdict(s), 'stable': bool(stable),
                       'servo_bump': bump, 'track_floor_nT_sqrtHz': track_floor,
                       'track_bw_hz': track_bw, 'err_std_lsb': err_std,
                       'loss_frac': rec['loss_frac']}
                rows.append(row)
                print(f'  {s.label():>22}: stable={stable!s:>5} bump={bump:>6.2f} '
                      f'track_floor={track_floor:.3g} track_bw={track_bw:>6.1f}Hz '
                      f'err_std={err_std:.3g} loss={rec["loss_frac"]*100:.2f}%')

    def rank(r):
        return (0 if r['stable'] else 1,
                np.inf if not np.isfinite(r['servo_bump']) else r['servo_bump'],
                -(r['track_bw_hz'] if np.isfinite(r['track_bw_hz']) else 0.0))
    rows.sort(key=rank)
    with open(os.path.join(outdir, 'sweep_summary.json'), 'w') as fh:
        json.dump(rows, fh, indent=2, default=float)
    _plot_sweep(outdir, f_ref, a_ref, curves)
    if rows:
        b = rows[0]
        print(f'[best] {b["label"]}: bump={b["servo_bump"]:.2f}, '
              f'track_bw={b["track_bw_hz"]:.0f} Hz, floor={b["track_floor_nT_sqrtHz"]:.3g} nT/rtHz')
    print(f'[out] {outdir}')
    return rows


def _minus3db_bw(freqs: np.ndarray, asd: np.ndarray,
                 baseline_band: Tuple[float, float]) -> float:
    """Tracking bandwidth = highest freq where the correction ASD is within 3 dB
    (1/sqrt(2)) of its low-frequency baseline before rolling off."""
    freqs = np.asarray(freqs); asd = np.asarray(asd)
    m = (freqs >= baseline_band[0]) & (freqs <= baseline_band[1])
    if not m.any():
        return float('nan')
    base = np.median(asd[m])
    if not np.isfinite(base) or base <= 0:
        return float('nan')
    above = freqs > baseline_band[0]
    ok = above & (asd >= base / np.sqrt(2))
    return float(np.max(freqs[ok])) if ok.any() else float('nan')


# =========================================================================
# Test 2: two-resonance vs one-resonance
# =========================================================================
def record_traces_multi(h: QudiHandles, duration: float, nslots: int,
                        settle_drop_s: float = 2.0, poll: float = 0.05,
                        dead_time_s: float = 0.0) -> Dict[str, Any]:
    """Drain the running multi-resonance dual stream and reconstruct it.

    Assumes multi-resonance tracking is ALREADY started (``start_multi_tracking``
    -> ``continuous_hop_start(input_source='dual')``; the dual push stream is
    running). We drain it ourselves via ``scan.hop_stream_read()`` -- exactly the
    Test-1 path. NOTE: when this driver is invoked over the rpyc namespace, the
    logic's ``_multi_status_timer`` runs on the rpyc service thread, NOT the logic's
    QThread, so ``QTimer.start`` silently fails ("Timers cannot be started from
    another thread") and the logic NEVER drains the stream -- so there is no race
    and we are the sole drainer. We reconstruct both the ZOH (N,T) staircase AND the
    fresh per-visit series; the first ``settle_drop_s`` of reconstructed time is
    dropped (loop acquisition transient).
    """
    scan = h.scan
    words_list: List[np.ndarray] = []
    t0 = time.time()
    while (time.time() - t0) < (duration + settle_drop_s):
        w = h.obtain(scan.hop_stream_read())
        if w is not None and len(w):
            words_list.append(np.asarray(w, dtype=np.float64))
        time.sleep(poll)
    words = np.concatenate(words_list) if words_list else np.array([], dtype=np.float64)
    loss = float(np.mean(np.isnan(words))) if words.size else 1.0

    zoh = reconstruct_dual(words, nslots=nslots, to_hz_corr=True)   # (N,T) ZOH
    fresh = reconstruct_dual_fresh(words, nslots=nslots, dead_time_s=dead_time_s)  # per-visit
    T = zoh['err'].shape[1]
    drop = int(settle_drop_s * STREAM_SAMPLE_RATE_HZ)
    drop = min(drop, max(0, T - 1))
    times = np.arange(T - drop, dtype=np.float64) / STREAM_SAMPLE_RATE_HZ
    err_zoh = zoh['err'][:, drop:]
    corr_zoh = zoh['corr'][:, drop:]
    # drop the settle window from the per-visit series too
    for r in range(nslots):
        keep = fresh['visit_idx'][r] >= drop
        fresh['visit_idx'][r] = fresh['visit_idx'][r][keep] - drop
        fresh['visit_err'][r] = fresh['visit_err'][r][keep]
        fresh['visit_corr'][r] = fresh['visit_corr'][r][keep]
    return {'times': times, 'err_zoh': err_zoh, 'corr_zoh': corr_zoh,
            'fresh': fresh, 'sample_rate': STREAM_SAMPLE_RATE_HZ,
            'loss_frac': loss, 'n_words': int(words.size), '_words': words}


def run_test2(h: QudiHandles, ranges: List[Tuple[float, float]], *,
              points: int = 1000, power: float = -10.0, data_rate: float = 1000.0,
              baseline_duration: float = 30.0, two_res_duration: float = 18.0,
              bandwidth_hz: float = 150.0, dwell_time_s: float = 1.0e-3,
              scan_settle_s: float = 100e-6, osc_settle_s: float = 200e-6,
              trigger_s: float = 50e-6, demod_phase_deg: Optional[float] = None,
              linear_halfwidth_hz: float = 0.4e6, settle_drop_s: float = 2.0,
              run_baselines: bool = True, outdir: Optional[str] = None
              ) -> Dict[str, Any]:
    """Compare 2-resonance tracking to the 1-resonance baseline for each resonance.

    Steps: (1) for each feature, run the single-resonance ``run_test1`` (1-res
    baseline: open-loop error->B and closed-loop correction->B) AND get its central
    zero-crossing + slope via ``fit_hyperfine``; (2) configure the live multi-res
    logic to track those SAME central crossings (narrow linear detail scan +
    ``fit_resonance_n`` per resonance, then ``configure_multi_tracking``); (3) start
    hopping and record the dual stream; (4) compute, per resonance, the naive ZOH
    correction->B ASD (with the hop comb annotated) AND the fresh per-visit
    correction->B ASD, overlaid on the 1-res closed-loop baseline; (5) save + plot.

    The fresh per-visit ASD is the honest 2-res tracking noise; the naive ZOH ASD
    shows the hop-schedule artifacts the user sees as 'periodic noise'.
    """
    outdir = _make_outdir(outdir)
    odmr = h.odmr
    n = len(ranges)
    try:
        odmr.stop_multi_tracking()
    except Exception:
        pass

    # --- 1-res baselines + central crossings -----------------------------
    baselines: List[Optional[Dict[str, Any]]] = []
    zcs: List[float] = []
    slopes: List[float] = []
    for i, (fmin, fmax) in enumerate(ranges):
        if run_baselines:
            b = run_test1(h, fmin, fmax, points=points, power=power,
                          data_rate=data_rate, open_duration=baseline_duration,
                          closed_duration=baseline_duration, bandwidth_hz=bandwidth_hz,
                          outdir=os.path.join(outdir, f'baseline_res{i}'))
            baselines.append(b)
            zcs.append(float(b['fit']['f_zc_rf']))
            slopes.append(abs(float(b['fit']['slope_lsb_per_hz'])))
        else:
            sf = scan_and_fit_feature(h, fmin, fmax, points=points, power=power,
                                      data_rate=data_rate)
            rf, demod, fit = sf['rf'], sf['demod'], sf['fit']
            baselines.append(None)
            zcs.append(float(fit['f_zc_rf']))
            slopes.append(abs(float(fit['slope_lsb_per_hz'])))
        print(f'[2res] res{i} central zc = {zcs[i]/1e9:.6f} GHz, slope = {slopes[i]:.4g} LSB/Hz')

    # --- configure the live multi-res logic onto the SAME crossings ------
    odmr.set_scan_power(power)
    try:
        odmr.set_data_rate(float(data_rate))
    except Exception:
        pass
    fit2: List[Dict[str, Any]] = []
    for i, zc in enumerate(zcs):
        nmin, nmax = zc - linear_halfwidth_hz, zc + linear_halfwidth_hz
        odmr.set_scan_region(float(nmin), float(nmax), int(points))
        odmr.set_runtime(6.0)
        odmr.start_odmr_scan()
        _wait_idle(odmr, 180.0)
        fr = dict(h.obtain(odmr.fit_resonance_n(i, float(nmin), float(nmax))))
        fit2.append(fr)
        print(f'[2res] logic fit res{i}: zc={fr["zero_crossing_freq"]/1e9:.6f} GHz, '
              f'slope={fr["slope"]:.3g} LSB/Hz, R2={fr.get("r_squared", float("nan")):.4f}')

    if demod_phase_deg is not None:
        odmr.set_demod_phase(float(demod_phase_deg))
    odmr.lock_bandwidth = float(bandwidth_hz)
    odmr.set_dwell_time(float(dwell_time_s))
    odmr.set_settle_time(float(osc_settle_s))
    try:                       # LO-settle + hop-trigger are plain config options
        odmr._scan_settling_time = float(scan_settle_s)
        odmr._trigger_length = float(trigger_s)
    except Exception as e:
        print(f'[2res] could not set scan_settle/trigger: {e}')

    odmr.configure_multi_tracking()
    odmr.start_multi_tracking()
    try:
        rec = record_traces_multi(h, two_res_duration, nslots=n, settle_drop_s=settle_drop_s)
    finally:
        odmr.stop_multi_tracking()
    print(f'[2res] stream: {rec["n_words"]} words, loss={rec["loss_frac"]*100:.3f}%')

    # --- per-resonance ASD analysis --------------------------------------
    fs = STREAM_SAMPLE_RATE_HZ
    per_res: List[Dict[str, Any]] = []
    for i in range(n):
        slope = abs(float(fit2[i]['slope']))
        # naive ZOH correction -> B ASD at the full aggregate rate
        B_zoh = correction_to_field(rec['corr_zoh'][i])
        f_zoh, a_zoh = compute_asd(B_zoh, fs)
        # honest fresh per-visit correction -> B ASD at the per-resonance hop rate
        vcorr = rec['fresh']['visit_corr'][i]
        eff = float(rec['fresh']['eff_rate_hz'][i])
        if vcorr.size >= 16 and np.isfinite(eff) and eff > 0:
            B_fresh = correction_to_field(vcorr)
            f_fresh, a_fresh = compute_asd(B_fresh, eff, nperseg=min(vcorr.size, int(eff)))
        else:
            f_fresh = np.array([]); a_fresh = np.array([])
        # 1-res closed-loop baseline (correction -> B) for overlay
        base_f = base_a = None
        if baselines[i] is not None:
            ba = np.load(os.path.join(baselines[i]['outdir'], 'asd.npz'))
            base_f, base_a = ba['freqs'], ba['asd_closed_corr']
        per_res.append({'slope_lsb_per_hz': slope,
                        'eff_rate_hz': eff, 'visit_jitter': float(rec['fresh']['visit_jitter'][i]),
                        'n_visits': int(vcorr.size),
                        'naive_floor_nT_sqrtHz': asd_noise_floor(f_zoh, a_zoh, 1.0, 50.0),
                        'fresh_floor_nT_sqrtHz': (asd_noise_floor(f_fresh, a_fresh, 1.0, 50.0)
                                                  if a_fresh.size else float('nan')),
                        '_asd_zoh': (f_zoh, a_zoh), '_asd_fresh': (f_fresh, a_fresh),
                        '_asd_base': (base_f, base_a)})
        print(f'[2res] res{i}: eff_rate={eff:.1f} Hz (jitter {per_res[i]["visit_jitter"]*100:.1f}%), '
              f'{per_res[i]["n_visits"]} visits; floor naive={per_res[i]["naive_floor_nT_sqrtHz"]:.3g} '
              f'fresh={per_res[i]["fresh_floor_nT_sqrtHz"]:.3g} nT/rtHz (1-50 Hz)')

    summary = {'ranges_hz': [list(r) for r in ranges], 'power_dbm': power,
               'scan_points': points, 'scan_data_rate_hz': data_rate,
               'bandwidth_hz': bandwidth_hz, 'dwell_time_s': dwell_time_s,
               'scan_settle_s': scan_settle_s, 'osc_settle_s': osc_settle_s,
               'central_zc_hz': zcs, 'logic_fit_zc_hz': [f['zero_crossing_freq'] for f in fit2],
               'loss_frac': rec['loss_frac'], 'n_words': rec['n_words'],
               'per_resonance': [{k: v for k, v in p.items() if not k.startswith('_')}
                                 for p in per_res],
               'baseline_dirs': [b['outdir'] if b else None for b in baselines],
               'outdir': outdir}
    np.savez_compressed(os.path.join(outdir, 'traces_2res.npz'),
                        times=rec['times'], err_zoh=rec['err_zoh'], corr_zoh=rec['corr_zoh'])
    with open(os.path.join(outdir, 'summary_2res.json'), 'w') as fh:
        json.dump(summary, fh, indent=2, default=float)
    _plot_2res_timetraces(outdir, rec, [p['slope_lsb_per_hz'] for p in per_res])
    _plot_2res_asd(outdir, per_res)
    print(f'[out] {outdir}')
    summary['_per_res'] = per_res
    return summary


def _fresh_metrics(f: np.ndarray, a: np.ndarray, eff_rate: float) -> Dict[str, float]:
    """Servo metrics of a fresh per-visit correction ASD (rate = eff_rate)."""
    f = np.asarray(f); a = np.asarray(a)
    nyq = eff_rate / 2.0
    if f.size < 4 or not np.isfinite(nyq) or nyq <= 0:
        return {'baseline': float('nan'), 'bump': float('nan'),
                'floor': float('nan'), 'nyquist_hz': nyq}
    low = (f >= 2.0) & (f <= 20.0)
    base = float(np.median(a[low])) if low.any() else float('nan')
    bb = (f >= 0.4 * nyq) & (f <= 0.98 * nyq)
    bump = (float(np.nanmax(a[bb])) / base) if (bb.any() and base and base > 0) else float('nan')
    floor = asd_noise_floor(f, a, 5.0, min(50.0, 0.5 * nyq))
    return {'baseline': base, 'bump': bump, 'floor': floor, 'nyquist_hz': nyq}


def within_dwell_profile(words: np.ndarray, nslots: int) -> Dict[int, np.ndarray]:
    """Mean error (raw LSB) vs sample-index-since-hop, per resonance.

    Averaging the error over many dwell visits as a function of the sample index
    within the dwell reveals the post-hop SETTLE TRANSIENT (LO + demod settling):
    a large value at small index that decays to ~0 means the early dwell samples
    are corrupted and the integrator is being kicked every visit (a periodic hop
    artifact). A flat ~0 profile means the dwell is clean.
    """
    words = np.asarray(words, dtype=np.float64).ravel()
    n3 = (words.size // 3) * 3
    if n3 == 0:
        return {r: np.array([]) for r in range(nslots)}
    trip = words[:n3].reshape(-1, 3)
    err_col, step_raw = trip[:, 0], trip[:, 2]
    step_int = np.where(np.isfinite(_ffill(step_raw)),
                        np.rint(_ffill(step_raw)), -1).astype(np.int64)
    profiles: Dict[int, np.ndarray] = {}
    for r in range(nslots):
        live = (step_int == r).astype(np.int8)
        edges = np.diff(np.r_[np.int8(0), live, np.int8(0)])
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        if starts.size == 0:
            profiles[r] = np.array([]); continue
        maxL = int(np.max(ends - starts + 1))
        acc = np.zeros(maxL); cnt = np.zeros(maxL)
        for s, e in zip(starts, ends):
            seg = err_col[s:e + 1]
            k = np.arange(seg.size)
            good = np.isfinite(seg)
            acc[k[good]] += seg[good]
            cnt[k[good]] += 1.0
        prof = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
        profiles[r] = prof
    return profiles


def _crosstalk(fresh: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Descriptive cross-resonance coupling of the fresh per-visit corrections.

    Resamples both per-visit correction series to a common uniform grid and reports
    the Pearson correlation of (a) the raw corrections -- dominated by COMMON-MODE
    lab field drift, expected positive -- and (b) their high-passed residuals, a
    rough proxy for genuine slot crosstalk after removing the shared field. Purely
    diagnostic; not a clean separation.
    """
    if n < 2:
        return {}
    c0, c1 = fresh['visit_corr'][0], fresh['visit_corr'][1]
    i0, i1 = fresh['visit_idx'][0], fresh['visit_idx'][1]
    if c0.size < 16 or c1.size < 16:
        return {'corr_raw': float('nan'), 'corr_resid': float('nan')}
    lo = max(i0.min(), i1.min()); hi = min(i0.max(), i1.max())
    grid = np.linspace(lo, hi, min(c0.size, c1.size))
    g0 = np.interp(grid, i0, c0); g1 = np.interp(grid, i1, c1)
    raw = float(np.corrcoef(g0, g1)[0, 1])
    # high-pass = subtract a moving average (~64-visit window)
    w = max(8, min(64, grid.size // 8))
    ker = np.ones(w) / w
    r0 = g0 - np.convolve(g0, ker, mode='same')
    r1 = g1 - np.convolve(g1, ker, mode='same')
    resid = float(np.corrcoef(r0[w:-w], r1[w:-w])[0, 1]) if grid.size > 2 * w else float('nan')
    return {'corr_raw': raw, 'corr_resid': resid}


def _2res_grid() -> List[Dict[str, Any]]:
    """Parameter grid for the 2-res sweep (deduped).

    Three axes crossing at the reference point (BW=100, dwell=1 ms, settle 100/200 us):
    bandwidth, dwell (sets the per-resonance update rate / Nyquist), and LO-settle.
    """
    pts: List[Dict[str, Any]] = []
    ref = dict(bw=100.0, dwell=1.0e-3, scan_settle=100e-6, osc_settle=200e-6, pi=False)
    for bw in (50.0, 100.0, 150.0, 300.0):
        pts.append({**ref, 'bw': bw, 'axis': 'bw'})
    for dw in (0.3e-3, 0.5e-3, 1.0e-3, 2.0e-3, 5.0e-3):
        pts.append({**ref, 'dwell': dw, 'axis': 'dwell'})
    for st in (50e-6, 200e-6, 500e-6):
        pts.append({**ref, 'scan_settle': st, 'osc_settle': max(2 * st, 200e-6), 'axis': 'settle'})
    pts.append({**ref, 'pi': True, 'axis': 'mode'})
    # dedupe by the hardware-relevant tuple, keep first axis tag
    seen: Dict[Tuple, Dict[str, Any]] = {}
    for p in pts:
        key = (p['bw'], p['dwell'], p['scan_settle'], p['osc_settle'], p['pi'])
        if key not in seen:
            seen[key] = p
    return list(seen.values())


def _2res_grid_lowbw() -> List[Dict[str, Any]]:
    """Focused low-bandwidth grid: find where the 2-res loop becomes STABLE.

    The full sweep showed servo bump rising monotonically with BW (>=x13 even at
    50 Hz), so the stable optimum is below 50 Hz. This grid sweeps BW {10..50} at
    dwell 1 ms, plus a fast-hop branch (dwell {0.3, 0.5} ms at BW 20) where the
    higher per-resonance update rate buys more phase margin.
    """
    pts: List[Dict[str, Any]] = []
    ref = dict(scan_settle=100e-6, osc_settle=200e-6, pi=False)
    for bw in (10.0, 20.0, 30.0, 40.0, 50.0):
        pts.append({**ref, 'bw': bw, 'dwell': 1.0e-3, 'axis': 'bw'})
    for dw in (0.3e-3, 0.5e-3, 1.0e-3):
        pts.append({**ref, 'bw': 20.0, 'dwell': dw, 'axis': 'dwell'})
    seen: Dict[Tuple, Dict[str, Any]] = {}
    for p in pts:
        key = (p['bw'], p['dwell'], p['scan_settle'], p['osc_settle'], p['pi'])
        seen.setdefault(key, p)
    return list(seen.values())


def sweep_2res(h: QudiHandles, slopes: List[float], n: int, *,
               grid: Optional[List[Dict[str, Any]]] = None,
               duration: float = 10.0, settle_drop_s: float = 1.5) -> Dict[str, Any]:
    """Sweep 2-res tracking params; per point compute the fresh per-visit corr ASD.

    Assumes the logic is configured on the two crossings (targets + power set). For
    each grid point: set BW/dwell/settle/mode, ``configure_multi_tracking`` (cheap,
    no rescan), ``start_multi_tracking``, record, ``stop_multi_tracking``; then
    fresh-reconstruct and score (servo bump near Nyquist, mid-band floor, eff rate,
    duty, hop-comb metric). Keeps the raw words of the ``diag_key`` (bw, dwell) point
    for the within-dwell settle profile + crosstalk diagnostics.
    """
    odmr = h.odmr
    grid = grid or _2res_grid()
    rep_slope = float(np.mean([s for s in slopes if s])) if any(slopes) else 1.1
    fs = STREAM_SAMPLE_RATE_HZ
    try:
        trigger_s = float(h.obtain(odmr._trigger_length))
    except Exception:
        trigger_s = 50e-6
    rows: List[Dict[str, Any]] = []
    for p in grid:
        try:
            # configure_multi_tracking -> mw.set_cw requires the MW output OFF; after
            # the previous point the Windfreak jump-list output is still active, so
            # turn it off first (stop_multi_tracking does not).
            try:
                h.mw.off()
                time.sleep(0.1)
            except Exception:
                pass
            odmr.set_dwell_time(float(p['dwell']))
            odmr.set_settle_time(float(p['osc_settle']))
            odmr._scan_settling_time = float(p['scan_settle'])
            odmr.lock_bandwidth = float(p['bw'])
            odmr.configure_multi_tracking()
            # apply the per-point loop mode/BW after configure (configure uses I-only)
            h.lock_hw.set_bandwidth(float(p['bw']), rep_slope, pi=bool(p['pi']), zero_ratio=3.0)
            odmr.start_multi_tracking()
        except Exception as e:
            print(f'  [skip] {p}: configure/start failed: {e}')
            try:
                odmr.stop_multi_tracking()
            except Exception:
                pass
            continue
        # Marked-continuous stream (STREAM_CONTROL[5]) emits dead-time as explicit
        # sample slots, so the per-resonance visit spacing already reflects TRUE wall
        # time -> dead_time_s=0 (no config correction needed; the FPGA fix replaces the
        # old PC-side dead-time hack used for the legacy 'dual' stream).
        dead = 0.0
        try:
            rec = record_traces_multi(h, duration, nslots=n, settle_drop_s=settle_drop_s,
                                      dead_time_s=dead)
        finally:
            odmr.stop_multi_tracking()

        fresh = rec['fresh']
        per = []
        for i in range(n):
            vc = fresh['visit_corr'][i]
            eff = float(fresh['eff_rate_hz'][i])
            if vc.size >= 16 and np.isfinite(eff) and eff > 0:
                f, a = compute_asd(correction_to_field(vc), eff,
                                   nperseg=min(vc.size, int(eff)))
                m = _fresh_metrics(f, a, eff)
            else:
                f = a = np.array([]); m = _fresh_metrics(np.array([]), np.array([]), eff)
            per.append({'eff_rate_hz': eff, **m, '_asd': (f, a)})
        duty = p['dwell'] / (p['dwell'] + p['scan_settle'])
        row = {'bw': p['bw'], 'dwell': p['dwell'], 'scan_settle': p['scan_settle'],
               'osc_settle': p['osc_settle'], 'pi': p['pi'], 'axis': p['axis'],
               'duty': duty, 'loss_frac': rec['loss_frac'],
               'eff_rate_hz': float(np.nanmean([pr['eff_rate_hz'] for pr in per])),
               'bump': float(np.nanmean([pr['bump'] for pr in per])),
               'floor_nT_sqrtHz': float(np.nanmean([pr['floor'] for pr in per])),
               '_per': per, '_words': rec.get('_words'), '_fresh': fresh}
        rows.append(row)
        mode = 'PI' if p['pi'] else 'I'
        print(f'  bw={p["bw"]:>5.0f} dwell={p["dwell"]*1e3:>4.1f}ms settle={p["scan_settle"]*1e6:>4.0f}us '
              f'{mode:>2} | eff={row["eff_rate_hz"]:>5.0f}Hz duty={duty:.2f} '
              f'bump=x{row["bump"]:>5.2f} floor={row["floor_nT_sqrtHz"]:.3g} loss={rec["loss_frac"]*100:.2f}%')
    # pick the diagnostic point: the most STABLE recorded point (lowest servo bump
    # among those that locked), so the within-dwell profile shows a genuine settle
    # transient rather than a limit-cycle.
    cand = [r for r in rows if r['_words'] is not None and np.isfinite(r['bump'])
            and r['loss_frac'] < 0.05 and r['floor_nT_sqrtHz'] < 100]
    # prefer a dwell >= 0.8 ms (clean step labeling) then lowest servo bump, so the
    # within-dwell settle profile is meaningful (short dwells mislabel/merge runs).
    dpick = (min(cand, key=lambda r: (r['dwell'] < 0.8e-3, r['bump']))
             if cand else (rows[0] if rows else None))
    diag = ({'words': dpick['_words'], 'fresh': dpick['_fresh'], 'point': dict(dpick)}
            if dpick else {'words': None, 'fresh': None, 'point': None})
    return {'rows': rows, 'diag': diag}


def _plot_2res_timetraces(outdir, rec, slopes) -> None:
    plt = _mpl()
    n = rec['corr_zoh'].shape[0]
    fig, axs = plt.subplots(n, 1, figsize=(11, 3 * n), squeeze=False)
    t = rec['times']
    for i in range(n):
        ax = axs[i][0]
        ax.plot(t, correction_to_field(rec['corr_zoh'][i]), lw=0.4, color=f'C{i}',
                label='ZOH staircase')
        fr = rec['fresh']
        vix = fr['visit_idx'][i]
        if vix.size:
            vt = vix.astype(np.float64) / rec['sample_rate']
            ax.plot(vt, correction_to_field(fr['visit_corr'][i]), '.', ms=2.5,
                    color='k', alpha=0.6, label='fresh per-visit')
        ax.set_ylabel('B [nT]'); ax.grid(alpha=0.3)
        ax.set_title(f'Resonance {i}: closed-loop correction -> B  '
                     f'(eff visit rate {fr["eff_rate_hz"][i]:.0f} Hz)')
        ax.legend(fontsize=8)
    axs[-1][0].set_xlabel('Time [s]')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'timetraces_2res.png'), dpi=140)
    plt.close(fig)


def _plot_2res_asd(outdir, per_res) -> None:
    plt = _mpl()
    n = len(per_res)
    fig, axs = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)
    for i, p in enumerate(per_res):
        ax = axs[0][i]
        f_zoh, a_zoh = p['_asd_zoh']
        f_fresh, a_fresh = p['_asd_fresh']
        base_f, base_a = p['_asd_base']
        if base_f is not None:
            ax.loglog(base_f[1:], base_a[1:], color='k', lw=1.4, ls='--',
                      label='1-res closed-loop corr -> B')
        if f_zoh.size:
            ax.loglog(f_zoh[1:], a_zoh[1:], color='C3', lw=0.8, alpha=0.6,
                      label='2-res naive ZOH corr -> B')
        if f_fresh.size:
            ax.loglog(f_fresh[1:], a_fresh[1:], color=f'C{i}', lw=1.2,
                      label='2-res fresh per-visit corr -> B')
        # annotate the hop comb (fundamental = per-resonance visit rate + harmonics)
        eff = p['eff_rate_hz']
        if np.isfinite(eff) and eff > 0:
            for k in range(1, 6):
                fk = k * eff / 2.0  # ZOH staircase harmonics fold at visit-rate/2 too
                if fk < (f_zoh[-1] if f_zoh.size else 0):
                    ax.axvline(fk, color='gray', ls=':', alpha=0.4)
            ax.axvline(eff / 2.0, color='C2', ls='-.', alpha=0.6,
                       label=f'visit Nyquist {eff/2:.0f} Hz')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('B ASD [nT/' + r'$\sqrt{\mathrm{Hz}}$' + ']')
        ax.set_title(f'Resonance {i}: 2-res vs 1-res tracking ASD')
        ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'asd_2res_compare.png'), dpi=140)
    plt.close(fig)


def _oneres_ref(h: QudiHandles, zc: float, slope: float, power: float, bw: float,
                open_dur: float, closed_dur: float, do_open: bool) -> Dict[str, Any]:
    """One 1-resonance reference: optional open-loop error->B + closed-loop corr->B,
    both at the full ~30.5 kHz (no hopping). Floors in 5-50 Hz."""
    setup_single_resonance(h)
    park_cw(h, zc, power)
    out: Dict[str, Any] = {'bw': bw, 'zc': zc, 'slope': slope}
    if do_open and open_dur > 0:
        orec = record_traces(h, open_dur, closed_loop=False, slope_lsb_per_hz=slope)
        fo, ao = compute_asd(error_to_field(orec['err_lsb'], slope), STREAM_SAMPLE_RATE_HZ)
        out['open'] = (fo, ao); out['open_floor'] = asd_noise_floor(fo, ao, 5.0, 50.0)
    else:
        out['open'] = (None, None); out['open_floor'] = float('nan')
    ensure_lock_polarity(h, slope, LockSettings(bandwidth_hz=bw))
    crec = record_traces(h, closed_dur, closed_loop=True, slope_lsb_per_hz=slope,
                         settings=LockSettings(bandwidth_hz=bw))
    fc, ac = compute_asd(correction_to_field(crec['corr_hz']), STREAM_SAMPLE_RATE_HZ)
    out['closed_corr'] = (fc, ac)
    out['closed_floor'] = asd_noise_floor(fc, ac, 5.0, 50.0)
    out['loss'] = crec['loss_frac']
    return out


def run_test2_full(h: QudiHandles, ranges: List[Tuple[float, float]], *,
                   points: int = 1000, power: float = -10.0, data_rate: float = 1000.0,
                   ref_open_duration: float = 15.0, ref_closed_duration: float = 12.0,
                   ref_bandwidth: float = 100.0, sweep_duration: float = 10.0,
                   grid: Optional[List[Dict[str, Any]]] = None,
                   outdir: Optional[str] = None) -> Dict[str, Any]:
    """Comprehensive 2-res vs 1-res study with a drift-bracketed protocol.

    Phases (kept within a few minutes so setup drift cannot confound the compare):
      A. fit both features (wide scan + fit_hyperfine -> central zc + slope);
      B. 1-res references per resonance (open-loop error->B + closed-loop corr->B at
         full rate) -- the START bracket;
      C. configure the live multi-res logic on those SAME crossings and SWEEP the
         2-res-only params (bandwidth, dwell=update rate, LO-settle, I vs PI);
      D. repeat the 1-res closed reference per resonance -- the END bracket (drift).

    Then: optimal-2res vs optimal-1res ASD overlay, per-axis sweep overlays + metric
    trends, within-dwell settle transient, and cross-resonance coupling.
    """
    outdir = _make_outdir(outdir)
    odmr = h.odmr; n = len(ranges)
    try:
        odmr.stop_multi_tracking()
    except Exception:
        pass
    t0 = time.time()

    # --- A: fit both resonances ---
    zcs: List[float] = []; slopes: List[float] = []; specs = []
    for i, (fmin, fmax) in enumerate(ranges):
        setup_single_resonance(h)
        sf = scan_and_fit_feature(h, fmin, fmax, points=points, power=power,
                                  data_rate=data_rate)
        rf, demod, fit = sf['rf'], sf['demod'], sf['fit']
        zcs.append(float(fit['f_zc_rf'])); slopes.append(abs(float(fit['slope_lsb_per_hz'])))
        specs.append((rf, demod, fit))
        print(f'[A] res{i} zc={zcs[i]/1e9:.6f} GHz, slope={slopes[i]:.3g} LSB/Hz '
              f'({fit["method"]}, {fit["n_crossings"]} xings, '
              f'slope-fit n={fit.get("slope_fit_n","?")})')
        _plot_spectrum_fit(outdir, rf, demod, fit, fname=f'spectrum_fit_res{i}.png')

    # --- B: 1-res references (start bracket) ---
    refs_start: List[Dict[str, Any]] = []
    for i in range(n):
        r = _oneres_ref(h, zcs[i], slopes[i], power, ref_bandwidth,
                        ref_open_duration, ref_closed_duration, do_open=True)
        refs_start.append(r)
        print(f'[B] res{i} 1-res: open={r["open_floor"]:.3g} closed={r["closed_floor"]:.3g} '
              f'nT/rtHz (5-50 Hz)')
    t_B = time.time()

    # --- C: configure live logic on the crossings + sweep 2-res params ---
    odmr.set_scan_power(power)
    try:
        odmr.set_data_rate(float(data_rate))
    except Exception:
        pass
    odmr._resonance_freqs = [float(z) for z in zcs]
    odmr._resonance_slopes = [float(s) for s in slopes]
    try:
        odmr.set_max_correction_hz(1.0e6)
    except Exception:
        pass
    print('[C] 2-res parameter sweep:')
    sweep = sweep_2res(h, slopes, n, grid=grid, duration=sweep_duration)
    t_C = time.time()

    # --- D: 1-res references (end bracket, closed only) ---
    refs_end: List[Dict[str, Any]] = []
    for i in range(n):
        r = _oneres_ref(h, zcs[i], slopes[i], power, ref_bandwidth,
                        0.0, ref_closed_duration, do_open=False)
        refs_end.append(r)
    t_end = time.time()
    print(f'[timing] phase B@{t_B-t0:.0f}s, C done@{t_C-t0:.0f}s, end@{t_end-t0:.0f}s '
          f'(B->C span {t_C-t_B:.0f}s)')

    # --- drift bracket ---
    drift = []
    for i in range(n):
        a, b = refs_start[i]['closed_floor'], refs_end[i]['closed_floor']
        drift.append(abs(b - a) / a if a else float('nan'))
    print(f'[drift] closed-floor change start->end: '
          f'{[f"{d*100:.0f}%" for d in drift]}')

    # --- pick the best (stable) 2-res point per the mean floor ---
    rows = sweep['rows']
    def _ok(r):
        return (r['loss_frac'] < 0.05 and np.isfinite(r['floor_nT_sqrtHz'])
                and np.isfinite(r['bump']) and r['bump'] < 5.0)
    stable = [r for r in rows if _ok(r)]
    best = (min(stable, key=lambda r: r['floor_nT_sqrtHz']) if stable
            else (min(rows, key=lambda r: r['floor_nT_sqrtHz']) if rows else None))
    if best:
        print(f'[best 2-res] bw={best["bw"]:.0f} dwell={best["dwell"]*1e3:.1f}ms '
              f'settle={best["scan_settle"]*1e6:.0f}us {"PI" if best["pi"] else "I"}: '
              f'floor={best["floor_nT_sqrtHz"]:.3g} nT/rtHz bump=x{best["bump"]:.2f}')

    # --- diagnostics from the reference 2-res point ---
    diag = sweep['diag']
    profiles = (within_dwell_profile(diag['words'], n)
                if diag.get('words') is not None else {})
    xtalk = _crosstalk(diag['fresh'], n) if diag.get('fresh') is not None else {}
    if xtalk:
        print(f'[crosstalk] corr_raw={xtalk.get("corr_raw"):.3f} '
              f'corr_resid={xtalk.get("corr_resid"):.3f}')

    # --- save + plots ---
    summary = {'ranges_hz': [list(r) for r in ranges], 'power_dbm': power,
               'central_zc_hz': zcs, 'slopes_lsb_per_hz': slopes,
               'ref_bandwidth_hz': ref_bandwidth, 'drift_closedfloor_frac': drift,
               'timing_s': {'phaseB': t_B - t0, 'phaseC': t_C - t0, 'end': t_end - t0,
                            'B_to_C_span': t_C - t_B},
               'refs_start': [{k: v for k, v in r.items() if not isinstance(v, tuple)}
                              for r in refs_start],
               'refs_end': [{k: v for k, v in r.items() if not isinstance(v, tuple)}
                            for r in refs_end],
               'sweep': [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
               'best_2res': ({k: v for k, v in best.items() if not k.startswith('_')}
                             if best else None),
               'crosstalk': xtalk, 'outdir': outdir}
    with open(os.path.join(outdir, 'summary_2res_full.json'), 'w') as fh:
        json.dump(summary, fh, indent=2, default=float)
    # save the raw diagnostic word stream for offline re-analysis (timing checks etc.)
    if diag.get('words') is not None:
        np.savez_compressed(os.path.join(outdir, 'diag_words.npz'),
                            words=np.asarray(diag['words'], dtype=np.float64),
                            nslots=n, point=json.dumps(
                                {k: v for k, v in (diag.get('point') or {}).items()
                                 if not k.startswith('_')}, default=float))
    _plot_bracket_compare(outdir, refs_start, refs_end, best, n)
    _plot_sweep_axes(outdir, rows, n)
    _plot_sweep_metrics(outdir, rows)
    if profiles:
        _plot_within_dwell(outdir, profiles, slopes)
    print(f'[out] {outdir}')
    summary['_refs_start'] = refs_start; summary['_refs_end'] = refs_end
    summary['_sweep'] = sweep
    return summary


def _plot_bracket_compare(outdir, refs_start, refs_end, best, n) -> None:
    plt = _mpl()
    fig, axs = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)
    for i in range(n):
        ax = axs[0][i]
        fo, ao = refs_start[i]['open']
        if fo is not None:
            ax.loglog(fo[1:], ao[1:], color='gray', lw=0.9, alpha=0.8,
                      label='1-res open-loop err->B')
        fc, ac = refs_start[i]['closed_corr']
        ax.loglog(fc[1:], ac[1:], color='C0', lw=1.2, label='1-res closed corr->B (start)')
        fe, ae = refs_end[i]['closed_corr']
        ax.loglog(fe[1:], ae[1:], color='C0', lw=0.9, ls=':', alpha=0.7,
                  label='1-res closed corr->B (end / drift)')
        if best is not None:
            f2, a2 = best['_per'][i]['_asd']
            if f2.size:
                ax.loglog(f2[1:], a2[1:], color='C3', lw=1.4,
                          label=f'best 2-res fresh (bw{best["bw"]:.0f}/{best["dwell"]*1e3:.1f}ms)')
                ax.axvline(best['_per'][i]['nyquist_hz'], color='C2', ls='-.', alpha=0.5)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('B ASD [nT/' + r'$\sqrt{\mathrm{Hz}}$' + ']')
        ax.set_title(f'Resonance {i}: optimal 2-res vs 1-res')
        ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'bracket_compare.png'), dpi=140)
    plt.close(fig)


def _plot_sweep_axes(outdir, rows, n) -> None:
    plt = _mpl()
    axes = [('bw', 'bw', 'Bandwidth [Hz]'),
            ('dwell', 'dwell', 'Dwell [s]'),
            ('settle', 'scan_settle', 'LO settle [s]')]
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for j, (axis, key, lbl) in enumerate(axes):
        ax = axs[j]
        pts = [r for r in rows if r['axis'] == axis] or [r for r in rows]
        pts = sorted(pts, key=lambda r: r[key])
        for r in pts:
            f, a = r['_per'][0]['_asd']     # show resonance 0
            if f.size:
                ax.loglog(f[1:], a[1:], lw=1.0, alpha=0.85,
                          label=f'{r[key]*1e3:.2f}ms' if key != 'bw' else f'{r[key]:.0f}Hz')
        ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('res0 fresh corr->B ASD [nT/rtHz]')
        ax.set_title(f'2-res sweep: {lbl}'); ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8, title=lbl)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'sweep_axes.png'), dpi=140)
    plt.close(fig)


def _plot_sweep_metrics(outdir, rows) -> None:
    plt = _mpl()
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    specs = [('bw', 'bw', 'Bandwidth [Hz]', False),
             ('dwell', 'dwell', 'Dwell [ms]', True),
             ('settle', 'scan_settle', 'LO settle [us]', True)]
    for j, (axis, key, lbl, _) in enumerate(specs):
        ax = axs[j]
        pts = sorted([r for r in rows if r['axis'] == axis], key=lambda r: r[key])
        if not pts:
            continue
        x = np.array([r[key] for r in pts])
        xs = x * (1e3 if key == 'dwell' else (1e6 if key == 'scan_settle' else 1))
        ax.plot(xs, [r['bump'] for r in pts], 'o-', color='C3', label='servo bump (x)')
        ax.set_xlabel(lbl); ax.set_ylabel('servo bump (x)', color='C3')
        ax.tick_params(axis='y', labelcolor='C3'); ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(xs, [r['floor_nT_sqrtHz'] for r in pts], 's--', color='C0',
                 label='floor [nT/rtHz]')
        ax2.set_ylabel('floor [nT/rtHz]', color='C0'); ax2.tick_params(axis='y', labelcolor='C0')
        ax.set_title(f'metrics vs {lbl}')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'sweep_metrics.png'), dpi=140)
    plt.close(fig)


def _plot_within_dwell(outdir, profiles, slopes) -> None:
    plt = _mpl()
    n = len(profiles)
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in range(n):
        prof = profiles.get(r, np.array([]))
        if prof.size == 0:
            continue
        t_us = np.arange(prof.size) / STREAM_SAMPLE_RATE_HZ * 1e6
        # convert error LSB -> Hz via the per-res slope for interpretability
        sl = slopes[r] if r < len(slopes) and slopes[r] else 1.0
        ax.plot(t_us, prof / sl, lw=1.2, label=f'resonance {r}')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Time since hop [us]')
    ax.set_ylabel('mean error [Hz] (settle transient)')
    ax.set_title('Within-dwell settle transient (averaged over visits)')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'within_dwell.png'), dpi=140)
    plt.close(fig)


# =========================================================================
# small utilities
# =========================================================================
def _wait_idle(module, timeout: float) -> bool:
    t0 = time.time()
    while module.module_state() != 'idle' and (time.time() - t0) < timeout:
        time.sleep(0.1)
    return module.module_state() == 'idle'


def _make_outdir(outdir: Optional[str]) -> str:
    if outdir is None:
        base = os.path.join(os.path.expanduser('~'), 'qudi', 'Data', 'freq_tracking_tests')
        outdir = os.path.join(base, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _save_run(outdir, rf, demod, open_rec, closed_rec,
              oe, ce, cc, summary) -> None:
    np.savez_compressed(os.path.join(outdir, 'traces.npz'),
                        rf=rf, demod=demod,
                        open_err=open_rec['err_lsb'], open_corr=open_rec['corr_hz'],
                        closed_err=closed_rec['err_lsb'], closed_corr=closed_rec['corr_hz'])
    np.savez_compressed(os.path.join(outdir, 'asd.npz'),
                        freqs=oe[0], asd_open_err=oe[1],
                        asd_closed_err=ce[1], asd_closed_corr=cc[1])
    save_summary = {k: v for k, v in summary.items() if not k.startswith('_')}
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(save_summary, f, indent=2, default=float)


# --- plotting (matplotlib Agg, headless-safe) ----------------------------
def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _plot_spectrum_fit(outdir, rf, demod, fit, fname='spectrum_fit.png') -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rf / 1e9, demod, color='gray', lw=0.8, label='ODMR demod')
    for z in fit['all_zc']:
        if np.isfinite(z):
            ax.axvline(z / 1e9, color='C0', ls=':', alpha=0.5)
    env = fit.get('envelope_hz', [np.nan, np.nan])
    if np.all(np.isfinite(env)):
        ax.axvspan(env[0] / 1e9, env[1] / 1e9, color='C2', alpha=0.08,
                   label='feature envelope')
    # mark the local slope-fit window used for the discriminator
    fz = fit['f_zc_rf']
    ax.axvline(fz / 1e9, color='C3', ls='--', lw=1.5,
               label=f'central zc {fz/1e9:.6f} GHz')
    ax.set_xlabel('RF [GHz]'); ax.set_ylabel('demod [LSB]')
    ax.set_title(f'ODMR scan + fit ({fit.get("method","?")}, {fit.get("n_crossings","?")} '
                 f'crossings, slope {fit["slope_lsb_per_hz"]:.3g} LSB/Hz)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, fname), dpi=140)
    plt.close(fig)


def _plot_timetraces(outdir, open_rec, closed_rec, slope) -> None:
    plt = _mpl()
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    to = open_rec['times']; tc = closed_rec['times']
    axs[0].plot(to, error_to_field(open_rec['err_lsb'], slope), lw=0.4, color='gray')
    axs[0].set_title('Open-loop: error -> B (CW on at zero-crossing, lock off)')
    axs[0].set_ylabel('B [nT]'); axs[0].grid(alpha=0.3)
    axs[1].plot(tc, correction_to_field(closed_rec['corr_hz']), lw=0.4, color='C0')
    axs[1].set_title('Closed-loop: correction -> B')
    axs[1].set_xlabel('Time [s]'); axs[1].set_ylabel('B [nT]'); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'timetraces.png'), dpi=140)
    plt.close(fig)


def _plot_asd_compare(outdir, f, a_open_err, a_closed_corr, a_closed_err, label) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(f[1:], a_open_err[1:], color='gray', lw=1.0,
              label='open-loop: error -> B')
    ax.loglog(f[1:], a_closed_corr[1:], color='C0', lw=1.0,
              label='closed-loop: correction -> B')
    ax.loglog(f[1:], a_closed_err[1:], color='C3', lw=0.8, alpha=0.7,
              label='closed-loop: residual error -> B (diagnostic)')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('B ASD [nT/' + r'$\sqrt{\mathrm{Hz}}$' + ']')
    ax.set_title(f'Open-loop vs closed-loop magnetic-field ASD  ({label})')
    ax.grid(True, which='both', alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'asd_compare.png'), dpi=140)
    plt.close(fig)


def _plot_sweep(outdir, f_ref, a_ref, curves) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(10, 6))
    if f_ref is not None and a_ref is not None:
        ax.loglog(f_ref[1:], a_ref[1:], color='k', lw=1.5, ls='--',
                  label='open-loop error -> B (reference)')
    for label, f, a in curves:
        ax.loglog(f[1:], a[1:], lw=0.9, alpha=0.8, label=label)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('correction -> B ASD [nT/' + r'$\sqrt{\mathrm{Hz}}$' + ']')
    ax.set_title('Closed-loop correction ASD per loop setting')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'sweep_asds.png'), dpi=140)
    plt.close(fig)


# =========================================================================
# Dry self-test (no hardware) + CLI
# =========================================================================
def _dry_test() -> int:
    """Validate the fit + convert + ASD math on synthetic data."""
    ok = True
    # synthetic 5-hyperfine dispersion comb centered at 2.87 GHz
    f0 = 2.87e9
    rf = np.linspace(f0 - 8e6, f0 + 8e6, 801)
    gamma = 0.4e6
    sig = np.zeros_like(rf)
    for k in range(-2, 3):
        x = (rf - (f0 + k * HYPERFINE_SPACING_HZ)) / gamma
        sig += -2 * 1.0 * x / (1 + x ** 2) ** 2  # Lorentzian derivative
    sig += 0.001 * np.random.default_rng(0).standard_normal(rf.size)
    fit = fit_central_zero_crossing(rf, sig)
    err_zc = abs(fit['f_zc_rf'] - f0)
    print(f'[dry] central zc = {fit["f_zc_rf"]/1e9:.6f} GHz (truth {f0/1e9:.6f}), '
          f'err = {err_zc/1e3:.1f} kHz, slope = {fit["slope_lsb_per_hz"]:.3g}, {fit["method"]}')
    ok &= err_zc < 0.3e6  # within 300 kHz of true center
    ok &= np.isfinite(fit['slope_lsb_per_hz']) and fit['slope_lsb_per_hz'] != 0

    # synthetic dual-word stream: nslots=1, white error, drifting correction
    rng = np.random.default_rng(1)
    T = 60000
    slope = abs(fit['slope_lsb_per_hz'])
    err = (rng.standard_normal(T) * 50.0)               # LSB
    corr = np.cumsum(rng.standard_normal(T) * 1.0)      # FTW random walk
    step = np.zeros(T)
    words = np.empty(3 * T)
    words[0::3], words[1::3], words[2::3] = err, corr, step
    rec = reconstruct_dual(words, nslots=1)
    ok &= np.allclose(rec['err'][0], err, equal_nan=True)
    ok &= np.allclose(rec['corr'][0], corr / FTW_PER_HZ, equal_nan=True)
    B = error_to_field(rec['err'][0], slope)
    f, a = compute_asd(B, STREAM_SAMPLE_RATE_HZ)
    floor = asd_noise_floor(f, a, 200, 1400)
    # white error -> flat ASD; expected level = (50 LSB/slope/gyromag) / sqrt(fs/2)
    expect = (50.0 / slope / GYROMAGNETIC_RATIO_NT_PER_HZ) / np.sqrt(STREAM_SAMPLE_RATE_HZ / 2)
    print(f'[dry] ASD floor = {floor:.4g} nT/rtHz, expected ~{expect:.4g} (ratio {floor/expect:.2f})')
    ok &= 0.5 < (floor / expect) < 2.0
    # loss handling: inject NaNs, ensure reshape/interp survive
    words2 = words.copy(); words2[::997] = np.nan
    rec2 = reconstruct_dual(words2, nslots=1)
    _f2, a2 = compute_asd(error_to_field(rec2['err'][0], slope), STREAM_SAMPLE_RATE_HZ)
    ok &= np.all(np.isfinite(a2))

    # --- Test 2: 2-slot fresh reconstruction --------------------------------
    # synthetic 2-resonance hop: dwell=30 samples/slot, 200 cycles. Each slot's
    # correction is a slow ramp + the slot index so visits are identifiable.
    N, dwell, cycles = 2, 30, 200
    steps = np.tile(np.repeat(np.arange(N), dwell), cycles)
    Tm = steps.size
    errm = np.zeros(Tm)
    corrm = np.zeros(Tm)
    for r in range(N):
        liv = steps == r
        # settled value per visit = slot r baseline + slow ramp; within-dwell wiggle
        corrm[liv] = (1000.0 * (r + 1) + np.linspace(0, 50, liv.sum())) * FTW_PER_HZ
        errm[liv] = 10.0 * (r + 1)
    wm = np.empty(3 * Tm)
    wm[0::3], wm[1::3], wm[2::3] = errm, corrm, steps
    fr = reconstruct_dual_fresh(wm, nslots=N)
    # each slot visited once per cycle -> ~cycles visits
    ok &= all(abs(fr['visit_corr'][r].size - cycles) <= 1 for r in range(N))
    # eff rate = fs / (N*dwell) since spacing between same-slot visits = N*dwell
    exp_eff = STREAM_SAMPLE_RATE_HZ / (N * dwell)
    ok &= all(abs(fr['eff_rate_hz'][r] - exp_eff) / exp_eff < 0.05 for r in range(N))
    # ZOH reconstruct should give finite (N,T) staircases distinct per slot
    zm = reconstruct_dual(wm, nslots=N)
    ok &= zm['corr'].shape == (N, Tm)
    means = [np.nanmean(zm['corr'][r]) for r in range(N)]
    ok &= means[1] > means[0]  # slot 1 baseline higher than slot 0
    print(f'[dry] 2-res fresh: visits/slot={[fr["visit_corr"][r].size for r in range(N)]}, '
          f'eff_rate={[round(fr["eff_rate_hz"][r], 1) for r in range(N)]} Hz '
          f'(expect {exp_eff:.1f})')

    print(f'[dry] {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Single-resonance frequency-tracking noise test')
    p.add_argument('--dry', action='store_true', help='run math self-test (no hardware)')
    p.add_argument('--rf-min', type=float, help='scan range min RF [Hz]')
    p.add_argument('--rf-max', type=float, help='scan range max RF [Hz]')
    p.add_argument('--points', type=int, default=1000)
    p.add_argument('--data-rate', type=float, default=1000.0, help='ODMR scan rate [Hz]')
    p.add_argument('--power', type=float, default=-10.0)
    p.add_argument('--open-duration', type=float, default=30.0)
    p.add_argument('--closed-duration', type=float, default=30.0)
    p.add_argument('--bandwidth', type=float, default=150.0)
    p.add_argument('--pi', action='store_true')
    p.add_argument('--sweep', action='store_true', help='also sweep loop settings')
    p.add_argument('--outdir', type=str, default=None)
    # --- Test 2 (2-resonance) ---
    p.add_argument('--test2', action='store_true',
                   help='run Test 2 (2-res vs 1-res); needs --rf-min/max AND --rf-min2/max2')
    p.add_argument('--test2-full', action='store_true',
                   help='run the full bracketed 2-res study (param sweep + diagnostics)')
    p.add_argument('--lowbw', action='store_true',
                   help='with --test2-full: use the focused low-bandwidth grid')
    p.add_argument('--rf-min2', type=float, help='second feature scan range min RF [Hz]')
    p.add_argument('--rf-max2', type=float, help='second feature scan range max RF [Hz]')
    p.add_argument('--dwell', type=float, default=1.0e-3, help='per-resonance dwell [s]')
    p.add_argument('--two-res-duration', type=float, default=18.0)
    p.add_argument('--demod-phase', type=float, default=None,
                   help='oscillator demod phase [deg] (default: keep persisted)')
    p.add_argument('--no-baselines', action='store_true',
                   help='skip the 1-res run_test1 baselines (scan+fit only)')
    args = p.parse_args(argv)

    if args.dry:
        return _dry_test()

    if args.test2 or args.test2_full:
        for req in ('rf_min', 'rf_max', 'rf_min2', 'rf_max2'):
            if getattr(args, req) is None:
                p.error('--test2/--test2-full need --rf-min/--rf-max and --rf-min2/--rf-max2')
        ranges = [(args.rf_min, args.rf_max), (args.rf_min2, args.rf_max2)]
        h = QudiHandles()
        try:
            if args.test2_full:
                run_test2_full(h, ranges, points=args.points, power=args.power,
                               data_rate=args.data_rate,
                               ref_closed_duration=args.closed_duration,
                               sweep_duration=args.two_res_duration,
                               grid=_2res_grid_lowbw() if args.lowbw else None,
                               outdir=args.outdir)
            else:
                run_test2(h, ranges, points=args.points, power=args.power,
                          data_rate=args.data_rate, baseline_duration=args.closed_duration,
                          two_res_duration=args.two_res_duration, bandwidth_hz=args.bandwidth,
                          dwell_time_s=args.dwell, demod_phase_deg=args.demod_phase,
                          run_baselines=not args.no_baselines, outdir=args.outdir)
        finally:
            h.close()
        return 0

    if args.rf_min is None or args.rf_max is None:
        p.error('--rf-min and --rf-max are required (or use --dry)')

    h = QudiHandles()
    try:
        summary = run_test1(h, args.rf_min, args.rf_max, points=args.points,
                            power=args.power, data_rate=args.data_rate,
                            open_duration=args.open_duration,
                            closed_duration=args.closed_duration,
                            bandwidth_hz=args.bandwidth, pi=args.pi, outdir=args.outdir)
        if args.sweep:
            sweep_settings(h, abs(summary['fit']['slope_lsb_per_hz']),
                           open_ref=summary.get('_open_ref'), outdir=summary['outdir'])
    finally:
        h.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
