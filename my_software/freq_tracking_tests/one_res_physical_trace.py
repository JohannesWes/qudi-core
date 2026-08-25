"""Acquire an actual one-resonance ODMR tracking trace and compare to 2-res B.

This is intended as the direct comparison against two-resonance differential
magnetometry: track one ODMR resonance in closed loop, convert its measured
frequency correction into magnetic field, and plot the ASD next to a saved 2-res
physical trace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, welch

from noise_test import (
    GYROMAGNETIC_RATIO_NT_PER_HZ,
    STREAM_SAMPLE_RATE_HZ,
    LockSettings,
    QudiHandles,
    _make_outdir,
    _plot_spectrum_fit,
    asd_noise_floor,
    ensure_lock_polarity,
    park_cw,
    record_traces,
    scan_and_fit_feature,
    setup_single_resonance,
)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _asd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmean(x)
    if np.any(~np.isfinite(x)):
        idx = np.arange(x.size)
        good = np.isfinite(x)
        x = np.interp(idx, idx[good], x[good]) if good.any() else np.zeros_like(x)
    nperseg = max(32, min(x.size, x.size))
    f, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=0, window="hann", detrend="constant")
    return f, np.sqrt(pxx)


def _log_bin(f: np.ndarray, a: np.ndarray, bins_per_decade: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    f = np.asarray(f)
    a = np.asarray(a)
    good = (f > 0) & np.isfinite(a) & (a > 0)
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


def _thin(t: np.ndarray, y: np.ndarray, max_points: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
    step = max(1, int(np.ceil(t.size / max_points)))
    return t[::step], y[::step]


def _floor_bands(f: np.ndarray, a: np.ndarray) -> Dict[str, float]:
    bands = {
        "0.005_0.05_Hz": (0.005, 0.05),
        "0.05_0.5_Hz": (0.05, 0.5),
        "0.5_5_Hz": (0.5, 5.0),
        "5_50_Hz": (5.0, 50.0),
    }
    return {name: asd_noise_floor(f, a, lo, hi) for name, (lo, hi) in bands.items()}


def _plot_one_res(outdir: str, t: np.ndarray, b_nT: np.ndarray, f: np.ndarray,
                  a: np.ndarray, lowpass_hz: float, label: str) -> None:
    plt = _mpl()
    fs = 1.0 / np.median(np.diff(t))
    b_lp = _lowpass(b_nT, fs, lowpass_hz)
    tt, bb = _thin(t, b_nT)
    _, blp = _thin(t, b_lp)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(tt, bb, color="C0", lw=0.3, alpha=0.35, label="raw")
    ax.plot(tt, blp, color="C0", lw=1.4, label=f"{lowpass_hz:g} Hz LP")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnetic-field estimate (nT)")
    ax.set_title(f"Actual 1-res closed-loop magnetic-field trace ({label})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "one_res_B_timeseries.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    fb, ab = _log_bin(f[1:], a[1:])
    ax.loglog(f[1:], a[1:], color="C0", lw=0.35, alpha=0.25)
    ax.loglog(fb, ab, color="C0", lw=1.6, label="actual 1-res B")
    ax.axvspan(0.005, 0.5, color="0.9", label="sub-Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("B ASD (nT/sqrtHz)")
    ax.set_title(f"Actual 1-res magnetic-field ASD ({label})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "one_res_B_asd.png"), dpi=160)
    plt.close(fig)


def _plot_compare(outdir: str, one: Dict, two_res_dir: str, lowpass_hz: float) -> Dict:
    plt = _mpl()
    two_path = Path(two_res_dir)
    two = np.load(two_path / "physical_trace_data.npz")
    t2 = np.asarray(two["time_s"], dtype=np.float64)
    b2 = np.asarray(two["magnetic_field_nT"], dtype=np.float64)
    fs2 = 1.0 / np.median(np.diff(t2))
    f2, a2 = _asd(b2, fs2)

    t1 = one["time_s"]
    b1 = one["B_nT"]
    f1, a1 = one["asd"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for f, a, color, label in [
        (f1, a1, "C0", "actual 1-res B"),
        (f2, a2, "C2", "actual 2-res differential B"),
    ]:
        ax.loglog(f[1:], a[1:], color=color, lw=0.35, alpha=0.22)
        fb, ab = _log_bin(f[1:], a[1:])
        ax.loglog(fb, ab, color=color, lw=1.8, label=label)
    ax.axvspan(0.005, 0.5, color="0.9", label="sub-Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("B ASD (nT/sqrtHz)")
    ax.set_title("Actual 1-res measurement vs actual 2-res differential measurement")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "actual_1res_vs_2res_B_asd.png"), dpi=170)
    plt.close(fig)

    fs1 = 1.0 / np.median(np.diff(t1))
    b1_lp = _lowpass(b1, fs1, lowpass_hz)
    b2_lp = _lowpass(b2, fs2, lowpass_hz)
    tt1, yy1 = _thin(t1, b1_lp)
    tt2, yy2 = _thin(t2, b2_lp)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(tt1, yy1, color="C0", lw=1.3, label=f"actual 1-res B, {lowpass_hz:g} Hz LP")
    ax.plot(tt2, yy2, color="C2", lw=1.3, label=f"actual 2-res B, {lowpass_hz:g} Hz LP")
    ax.set_xlabel("Time within each acquisition (s)")
    ax.set_ylabel("B change (nT)")
    ax.set_title("Sequential acquisitions: actual 1-res B vs actual 2-res B")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "actual_1res_vs_2res_B_timeseries.png"), dpi=170)
    plt.close(fig)

    bands = {
        "actual_1res_B": _floor_bands(f1, a1),
        "actual_2res_B": _floor_bands(f2, a2),
    }
    bands["ratio_1res_over_2res"] = {
        k: bands["actual_1res_B"][k] / bands["actual_2res_B"][k]
        for k in bands["actual_1res_B"]
    }
    return bands


def acquire_one_res_trace(args: argparse.Namespace) -> Dict:
    outdir = _make_outdir(args.outdir)
    h = QudiHandles()
    try:
        setup_single_resonance(h)
        sf = scan_and_fit_feature(h, args.rf_min, args.rf_max, points=args.points,
                                  power=args.power, data_rate=args.data_rate)
        fit = sf["fit"]
        f_zc = float(fit["f_zc_rf"])
        slope = abs(float(fit["slope_lsb_per_hz"]))
        _plot_spectrum_fit(outdir, sf["rf"], sf["demod"], fit, fname="spectrum_fit.png")
        print(f"[fit] zc={f_zc/1e9:.6f} GHz, slope={slope:.3g} LSB/Hz")

        park_cw(h, f_zc, args.power)
        settings = LockSettings(bandwidth_hz=args.bandwidth, pi=False,
                                max_correction_hz=args.max_correction)
        pol = ensure_lock_polarity(h, slope, settings)
        print(f"[polarity] invert={pol['chosen']['invert']} "
              f"abs_err={pol['chosen']['abs_err']:.1f} LSB")
        rec = record_traces(h, args.duration, closed_loop=True,
                            slope_lsb_per_hz=slope, settings=settings, poll=args.poll)

        t = np.asarray(rec["times"], dtype=np.float64)
        corr_hz = np.asarray(rec["corr_hz"], dtype=np.float64)
        base_n = max(8, min(corr_hz.size, int(args.baseline_s * STREAM_SAMPLE_RATE_HZ)))
        corr0 = float(np.nanmedian(corr_hz[:base_n]))
        df_hz = corr_hz - corr0
        b_nT = float(args.sign) * df_hz / GYROMAGNETIC_RATIO_NT_PER_HZ
        f, a = _asd(b_nT, STREAM_SAMPLE_RATE_HZ)

        _plot_one_res(outdir, t, b_nT, f, a, args.lowpass, args.label)
        compare = {}
        if args.two_res_dir:
            compare = _plot_compare(outdir, {"time_s": t, "B_nT": b_nT, "asd": (f, a)},
                                    args.two_res_dir, args.lowpass)

        np.savez_compressed(
            os.path.join(outdir, "one_res_physical_trace_data.npz"),
            time_s=t,
            corr_hz=corr_hz,
            df_hz=df_hz,
            magnetic_field_nT=b_nT,
            rf=np.asarray(sf["rf"], dtype=np.float64),
            demod=np.asarray(sf["demod"], dtype=np.float64),
        )
        summary = {
            "outdir": outdir,
            "label": args.label,
            "duration_s": args.duration,
            "sample_rate_hz": STREAM_SAMPLE_RATE_HZ,
            "loss_frac": float(rec["loss_frac"]),
            "zc_hz": f_zc,
            "slope_lsb_per_hz": slope,
            "baseline_correction_hz": corr0,
            "field_sign": args.sign,
            "settings": {
                "power_dbm": args.power,
                "bandwidth_hz": args.bandwidth,
                "max_correction_hz": args.max_correction,
                "lowpass_for_time_plots_hz": args.lowpass,
            },
            "polarity": pol,
            "asd_floors": _floor_bands(f, a),
            "comparison_to_2res": compare,
            "std_magnetic_field_nT": float(np.nanstd(b_nT)),
        }
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=float)
        print(f"[done] loss={summary['loss_frac']*100:.3f}%")
        print(f"[out] {outdir}")
        return summary
    finally:
        h.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rf-min", type=float, default=2.732e9)
    p.add_argument("--rf-max", type=float, default=2.750e9)
    p.add_argument("--points", type=int, default=1000)
    p.add_argument("--power", type=float, default=-10.0)
    p.add_argument("--data-rate", type=float, default=1000.0)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--poll", type=float, default=0.05)
    p.add_argument("--bandwidth", type=float, default=10.0)
    p.add_argument("--max-correction", type=float, default=1.0e6)
    p.add_argument("--baseline-s", type=float, default=10.0)
    p.add_argument("--lowpass", type=float, default=0.5)
    p.add_argument("--sign", type=float, default=-1.0,
                   help="field conversion sign: left resonance -1, right resonance +1")
    p.add_argument("--label", type=str, default="left resonance")
    p.add_argument("--two-res-dir", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None)
    return p


def main() -> None:
    acquire_one_res_trace(build_argparser().parse_args())


if __name__ == "__main__":
    main()
