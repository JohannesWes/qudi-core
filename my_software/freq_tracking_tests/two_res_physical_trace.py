"""Acquire and analyze a two-resonance ODMR physical trace.

The script drives the running Qudi instance, tracks the left (-1) and right (+1)
ODMR resonances simultaneously, and converts the two tracked frequency shifts
into differential magnetic-field and common-mode temperature channels.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, welch

from noise_test import (
    GYROMAGNETIC_RATIO_NT_PER_HZ,
    STREAM_SAMPLE_RATE_HZ,
    QudiHandles,
    _make_outdir,
    _plot_spectrum_fit,
    asd_noise_floor,
    record_traces_multi,
    scan_and_fit_feature,
    setup_single_resonance,
)


TEMP_COEFF_HZ_PER_K = -74.0e3


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _compute_asd_full(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmean(x)
    if np.any(~np.isfinite(x)):
        idx = np.arange(x.size)
        good = np.isfinite(x)
        x = np.interp(idx, idx[good], x[good]) if good.any() else np.zeros_like(x)
    nperseg = max(32, min(x.size, int(x.size)))
    f, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=0, window="hann", detrend="constant")
    return f, np.sqrt(pxx)


def _log_bin(f: np.ndarray, a: np.ndarray, bins_per_decade: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    f = np.asarray(f)
    a = np.asarray(a)
    good = (f > 0) & np.isfinite(f) & np.isfinite(a) & (a > 0)
    f = f[good]
    a = a[good]
    if f.size < 8:
        return f, a
    edges = np.logspace(np.log10(f.min()), np.log10(f.max()), int(np.ceil(
        bins_per_decade * np.log10(f.max() / f.min()))) + 1)
    fb, ab = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        if m.any():
            fb.append(np.exp(np.mean(np.log(f[m]))))
            ab.append(np.exp(np.mean(np.log(a[m]))))
    return np.asarray(fb), np.asarray(ab)


def _lowpass(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0 or cutoff_hz >= 0.45 * fs or x.size < 64:
        return np.asarray(x, dtype=np.float64)
    b, a = butter(3, cutoff_hz / (0.5 * fs), btype="low")
    return filtfilt(b, a, np.asarray(x, dtype=np.float64), method="gust")


def _thin(t: np.ndarray, *arrays: np.ndarray, max_points: int = 8000):
    step = max(1, int(np.ceil(t.size / max_points)))
    return (t[::step],) + tuple(np.asarray(a)[::step] for a in arrays)


def _floor_bands(f: np.ndarray, a: np.ndarray) -> Dict[str, float]:
    bands = {
        "0.005_0.05_Hz": (0.005, 0.05),
        "0.05_0.5_Hz": (0.05, 0.5),
        "0.5_5_Hz": (0.5, 5.0),
        "5_50_Hz": (5.0, 50.0),
    }
    return {name: asd_noise_floor(f, a, lo, hi) for name, (lo, hi) in bands.items()}


def _plot_time_series(outdir: str, t: np.ndarray, df_l: np.ndarray, df_r: np.ndarray,
                      b_nT: np.ndarray, temp_mK: np.ndarray, lp_hz: float) -> None:
    plt = _mpl()
    fs = 1.0 / np.median(np.diff(t))
    df_l_lp = _lowpass(df_l, fs, lp_hz)
    df_r_lp = _lowpass(df_r, fs, lp_hz)
    b_lp = _lowpass(b_nT, fs, lp_hz)
    temp_lp = _lowpass(temp_mK, fs, lp_hz)

    tt, dfl, dfr, dflp, dfrp = _thin(t, df_l / 1e3, df_r / 1e3, df_l_lp / 1e3, df_r_lp / 1e3)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(tt, dfl, color="C0", lw=0.35, alpha=0.35, label="left raw")
    ax.plot(tt, dfr, color="C1", lw=0.35, alpha=0.35, label="right raw")
    ax.plot(tt, dflp, color="C0", lw=1.4, label=f"left {lp_hz:g} Hz LP")
    ax.plot(tt, dfrp, color="C1", lw=1.4, label=f"right {lp_hz:g} Hz LP")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Resonance shift from initial median (kHz)")
    ax.set_title("Tracked resonance frequency shifts before B/T separation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "resonance_frequency_timeseries.png"), dpi=160)
    plt.close(fig)

    tt, bb, temp, blp, tlp = _thin(t, b_nT, temp_mK, b_lp, temp_lp)
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axs[0].plot(tt, bb, color="C2", lw=0.35, alpha=0.35, label="raw")
    axs[0].plot(tt, blp, color="C2", lw=1.4, label=f"{lp_hz:g} Hz LP")
    axs[0].set_ylabel("Magnetic-field change (nT)")
    axs[0].set_title("Differential channel: magnetic field")
    axs[0].grid(alpha=0.3)
    axs[0].legend()
    axs[1].plot(tt, temp, color="C3", lw=0.35, alpha=0.35, label="raw")
    axs[1].plot(tt, tlp, color="C3", lw=1.4, label=f"{lp_hz:g} Hz LP")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Temperature change (mK)")
    axs[1].set_title("Common-mode channel: temperature")
    axs[1].grid(alpha=0.3)
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "physical_channels_timeseries.png"), dpi=160)
    plt.close(fig)


def _plot_asds(outdir: str, fs: float, df_l: np.ndarray, df_r: np.ndarray,
               b_nT: np.ndarray, temp_mK: np.ndarray) -> Dict[str, Dict[str, float]]:
    plt = _mpl()
    channels = {
        "left_freq_Hz": df_l,
        "right_freq_Hz": df_r,
        "magnetic_field_nT": b_nT,
        "temperature_mK": temp_mK,
    }
    asds = {name: _compute_asd_full(x, fs) for name, x in channels.items()}

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, color, label in [
        ("left_freq_Hz", "C0", "left resonance"),
        ("right_freq_Hz", "C1", "right resonance"),
    ]:
        f, a = asds[name]
        fb, ab = _log_bin(f[1:], a[1:])
        ax.loglog(f[1:], a[1:], color=color, lw=0.35, alpha=0.25)
        ax.loglog(fb, ab, color=color, lw=1.5, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Frequency ASD (Hz/sqrtHz)")
    ax.set_title("Tracked resonance frequency ASDs before B/T separation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "resonance_frequency_asds.png"), dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for ax, name, color, ylabel, title in [
        (axs[0], "magnetic_field_nT", "C2", "B ASD (nT/sqrtHz)",
         "Differential magnetic-field ASD"),
        (axs[1], "temperature_mK", "C3", "Temperature ASD (mK/sqrtHz)",
         "Common-mode temperature ASD"),
    ]:
        f, a = asds[name]
        fb, ab = _log_bin(f[1:], a[1:])
        ax.loglog(f[1:], a[1:], color=color, lw=0.35, alpha=0.25)
        ax.loglog(fb, ab, color=color, lw=1.5)
        ax.axvspan(0.005, 0.5, color="0.9", label="sub-Hz band")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axs[1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "physical_channels_asds.png"), dpi=160)
    plt.close(fig)

    return {name: _floor_bands(f, a) for name, (f, a) in asds.items()}


def acquire_physical_trace(args: argparse.Namespace) -> Dict:
    outdir = _make_outdir(args.outdir)
    h = QudiHandles()
    try:
        odmr = h.odmr
        try:
            odmr.stop_multi_tracking()
        except Exception:
            pass

        ranges = [(args.rf_min, args.rf_max), (args.rf_min2, args.rf_max2)]
        zcs, slopes = [], []
        for i, (lo, hi) in enumerate(ranges):
            setup_single_resonance(h)
            sf = scan_and_fit_feature(h, lo, hi, points=args.points, power=args.power,
                                      data_rate=args.data_rate)
            zcs.append(float(sf["fit"]["f_zc_rf"]))
            slopes.append(abs(float(sf["fit"]["slope_lsb_per_hz"])))
            _plot_spectrum_fit(outdir, sf["rf"], sf["demod"], sf["fit"],
                               fname=f"spectrum_fit_res{i}.png")
            print(f"[fit] res{i}: zc={zcs[-1]/1e9:.6f} GHz, slope={slopes[-1]:.3g} LSB/Hz")

        odmr.set_scan_power(float(args.power))
        try:
            odmr.set_data_rate(float(args.data_rate))
        except Exception:
            pass
        odmr.set_dwell_time(float(args.dwell))
        odmr.set_settle_time(float(args.osc_settle))
        odmr._scan_settling_time = float(args.scan_settle)
        odmr.lock_bandwidth = float(args.bandwidth)
        odmr._resonance_freqs = [float(z) for z in zcs]
        odmr._resonance_slopes = [float(s) for s in slopes]
        odmr.set_max_correction_hz(float(args.max_correction))

        print("[track] configuring 2-res tracking")
        odmr.configure_multi_tracking()
        h.lock_hw.set_bandwidth(float(args.bandwidth), float(np.mean(slopes)), pi=False)
        print(f"[track] starting: duration={args.duration:.1f}s, BW={args.bandwidth:g}Hz, "
              f"dwell={args.dwell*1e3:g}ms, scan_settle={args.scan_settle*1e6:g}us, "
              f"osc_settle={args.osc_settle*1e6:g}us")
        odmr.start_multi_tracking()
        try:
            rec = record_traces_multi(h, args.duration, nslots=2,
                                      settle_drop_s=args.settle_drop, poll=args.poll)
        finally:
            odmr.stop_multi_tracking()

        fresh = rec["fresh"]
        eff = [float(x) for x in fresh["eff_rate_hz"]]
        if any(fresh["visit_corr"][i].size < 16 for i in range(2)):
            raise RuntimeError("Not enough per-visit samples in 2-res trace.")

        t0 = fresh["visit_idx"][0] / STREAM_SAMPLE_RATE_HZ
        t1 = fresh["visit_idx"][1] / STREAM_SAMPLE_RATE_HZ
        c0 = np.asarray(fresh["visit_corr"][0], dtype=np.float64)
        c1 = np.asarray(fresh["visit_corr"][1], dtype=np.float64)
        lo = max(float(t0.min()), float(t1.min()))
        hi = min(float(t0.max()), float(t1.max()))
        fs = float(np.nanmean(eff))
        t = np.arange(lo, hi, 1.0 / fs)
        left_hz = np.interp(t, t0, c0)
        right_hz = np.interp(t, t1, c1)
        t = t - t[0]

        base_n = max(8, min(t.size, int(args.baseline_s * fs)))
        left0 = float(np.median(left_hz[:base_n]))
        right0 = float(np.median(right_hz[:base_n]))
        df_left = left_hz - left0
        df_right = right_hz - right0

        b_nT = (df_right - df_left) / (2.0 * GYROMAGNETIC_RATIO_NT_PER_HZ)
        temp_K = (0.5 * (df_left + df_right)) / TEMP_COEFF_HZ_PER_K
        temp_mK = 1e3 * temp_K

        floors = _plot_asds(outdir, fs, df_left, df_right, b_nT, temp_mK)
        _plot_time_series(outdir, t, df_left, df_right, b_nT, temp_mK, args.lowpass)

        np.savez_compressed(
            os.path.join(outdir, "physical_trace_data.npz"),
            time_s=t,
            df_left_hz=df_left,
            df_right_hz=df_right,
            magnetic_field_nT=b_nT,
            temperature_mK=temp_mK,
            raw_words=np.asarray(rec["_words"], dtype=np.float64),
            zc_hz=np.asarray(zcs),
            slopes_lsb_per_hz=np.asarray(slopes),
            eff_rate_hz=np.asarray(eff),
        )
        summary = {
            "outdir": outdir,
            "duration_s_requested": args.duration,
            "duration_s_analyzed": float(t[-1] - t[0]) if t.size else 0.0,
            "sample_rate_hz": fs,
            "effective_visit_rate_hz": eff,
            "loss_frac": float(rec["loss_frac"]),
            "zcs_hz": zcs,
            "slopes_lsb_per_hz": slopes,
            "baseline_correction_hz": {"left": left0, "right": right0},
            "conversion": {
                "magnetic_field_nT": "(df_right - df_left)/(2*28.024)",
                "temperature_mK": "1000*((df_left + df_right)/2)/(-74000)",
            },
            "settings": {
                "power_dbm": args.power,
                "bandwidth_hz": args.bandwidth,
                "dwell_s": args.dwell,
                "scan_settle_s": args.scan_settle,
                "osc_settle_s": args.osc_settle,
                "settle_drop_s": args.settle_drop,
                "lowpass_for_time_plots_hz": args.lowpass,
            },
            "asd_floors": floors,
            "std": {
                "df_left_hz": float(np.std(df_left)),
                "df_right_hz": float(np.std(df_right)),
                "magnetic_field_nT": float(np.std(b_nT)),
                "temperature_mK": float(np.std(temp_mK)),
            },
        }
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"[done] analyzed {summary['duration_s_analyzed']:.1f}s at {fs:.1f} Hz, "
              f"loss={summary['loss_frac']*100:.3f}%")
        print(f"[out] {outdir}")
        return summary
    finally:
        h.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rf-min", type=float, default=2.732e9)
    p.add_argument("--rf-max", type=float, default=2.750e9)
    p.add_argument("--rf-min2", type=float, default=2.984e9)
    p.add_argument("--rf-max2", type=float, default=2.998e9)
    p.add_argument("--points", type=int, default=1000)
    p.add_argument("--power", type=float, default=-10.0)
    p.add_argument("--data-rate", type=float, default=1000.0)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--settle-drop", type=float, default=10.0)
    p.add_argument("--poll", type=float, default=0.05)
    p.add_argument("--bandwidth", type=float, default=10.0)
    p.add_argument("--dwell", type=float, default=1.0e-3)
    p.add_argument("--scan-settle", type=float, default=100e-6)
    p.add_argument("--osc-settle", type=float, default=200e-6)
    p.add_argument("--max-correction", type=float, default=1.0e6)
    p.add_argument("--baseline-s", type=float, default=10.0)
    p.add_argument("--lowpass", type=float, default=0.5)
    p.add_argument("--outdir", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    acquire_physical_trace(args)


if __name__ == "__main__":
    main()
