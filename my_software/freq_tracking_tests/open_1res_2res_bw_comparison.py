"""Compare open-loop, 1-res, and 2-res magnetic-field measurements.

This script acquires:

* one open-loop left-resonance trace, converted to B,
* several actual 1-res closed-loop left-resonance B traces,
* several actual 2-res closed-loop differential B traces,

then writes one ASD plot and one time-trace plot per selected 1-res/2-res
bandwidth pair.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, welch

from noise_test import (
    GYROMAGNETIC_RATIO_NT_PER_HZ,
    STREAM_SAMPLE_RATE_HZ,
    LockSettings,
    QudiHandles,
    _make_outdir,
    _plot_spectrum_fit,
    ensure_lock_polarity,
    park_cw,
    record_traces,
    record_traces_multi,
    scan_and_fit_feature,
    setup_single_resonance,
)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


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
    return np.asarray(t)[::step], np.asarray(y)[::step]


def _floor(f: np.ndarray, a: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f <= hi)
    return float(np.mean(a[m])) if m.any() else float("nan")


def _floors(f: np.ndarray, a: np.ndarray) -> Dict[str, float]:
    bands = {
        "0.005_0.05_Hz": (0.005, 0.05),
        "0.05_0.5_Hz": (0.05, 0.5),
        "0.5_5_Hz": (0.5, 5.0),
        "5_50_Hz": (5.0, 50.0),
    }
    return {name: _floor(f, a, lo, hi) for name, (lo, hi) in bands.items()}


def _field_from_left_corr(corr_hz: np.ndarray, baseline_s: float, fs: float) -> np.ndarray:
    corr_hz = np.asarray(corr_hz, dtype=np.float64)
    n0 = max(8, min(corr_hz.size, int(baseline_s * fs)))
    df = corr_hz - float(np.nanmedian(corr_hz[:n0]))
    return -df / GYROMAGNETIC_RATIO_NT_PER_HZ


def _field_from_open_left_error(err_lsb: np.ndarray, slope: float) -> np.ndarray:
    df = np.asarray(err_lsb, dtype=np.float64) / float(slope)
    df = df - np.nanmedian(df)
    return -df / GYROMAGNETIC_RATIO_NT_PER_HZ


def _interp_2res_field(rec: Dict, baseline_s: float) -> Tuple[np.ndarray, np.ndarray, float]:
    fresh = rec["fresh"]
    t0 = fresh["visit_idx"][0] / STREAM_SAMPLE_RATE_HZ
    t1 = fresh["visit_idx"][1] / STREAM_SAMPLE_RATE_HZ
    c0 = np.asarray(fresh["visit_corr"][0], dtype=np.float64)
    c1 = np.asarray(fresh["visit_corr"][1], dtype=np.float64)
    lo = max(float(t0.min()), float(t1.min()))
    hi = min(float(t0.max()), float(t1.max()))
    fs = float(np.nanmean(fresh["eff_rate_hz"]))
    t = np.arange(lo, hi, 1.0 / fs)
    left = np.interp(t, t0, c0)
    right = np.interp(t, t1, c1)
    n0 = max(8, min(t.size, int(baseline_s * fs)))
    left = left - float(np.nanmedian(left[:n0]))
    right = right - float(np.nanmedian(right[:n0]))
    b_nT = (right - left) / (2.0 * GYROMAGNETIC_RATIO_NT_PER_HZ)
    return t - t[0], b_nT, fs


def _plot_case(outdir: str, case: Dict, lowpass_hz: float, max_asd_hz: float) -> None:
    plt = _mpl()
    one_bw = case["one_bw_hz"]
    two_bw = case["two_bw_hz"]
    tag = f"1res{one_bw:g}_2res{two_bw:g}".replace(".", "p")

    fig, ax = plt.subplots(figsize=(9, 6))
    for key, color, label in [
        ("open", "0.35", "open-loop left B"),
        ("one", "C0", f"1-res B, BW {one_bw:g} Hz"),
        ("two", "C2", f"2-res diff B, BW {two_bw:g} Hz"),
    ]:
        f, a = case[key]["asd"]
        m = (f > 0) & (f <= max_asd_hz)
        ax.loglog(f[m], a[m], color=color, lw=0.3, alpha=0.22)
        fb, ab = _log_bin(f[m], a[m])
        ax.loglog(fb, ab, color=color, lw=1.7, label=label)
    ax.axvspan(0.005, 0.5, color="0.9", label="sub-Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("B ASD (nT/sqrtHz)")
    ax.set_title(f"Open-loop vs 1-res vs 2-res B ASD ({one_bw:g} Hz / {two_bw:g} Hz)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"compare_B_asd_{tag}.png"), dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for key, color, label in [
        ("open", "0.35", "open-loop left B"),
        ("one", "C0", f"1-res B, BW {one_bw:g} Hz"),
        ("two", "C2", f"2-res diff B, BW {two_bw:g} Hz"),
    ]:
        t = case[key]["time_s"]
        b = case[key]["B_nT"]
        fs = case[key]["fs_hz"]
        blp = _lowpass(b, fs, lowpass_hz)
        tt, yy = _thin(t, blp)
        ax.plot(tt, yy, color=color, lw=1.2, label=f"{label}, {lowpass_hz:g} Hz LP")
    ax.set_xlabel("Time within each acquisition (s)")
    ax.set_ylabel("B change (nT)")
    ax.set_title(f"Open-loop vs 1-res vs 2-res B time traces ({one_bw:g} Hz / {two_bw:g} Hz)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"compare_B_timeseries_{tag}.png"), dpi=170)
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict:
    outdir = _make_outdir(args.outdir)
    one_bws = _parse_csv_floats(args.one_bws)
    two_bws = _parse_csv_floats(args.two_bws)
    if len(one_bws) != len(two_bws):
        raise ValueError("--one-bws and --two-bws must have the same number of entries.")

    h = QudiHandles()
    try:
        odmr = h.odmr
        try:
            odmr.stop_multi_tracking()
        except Exception:
            pass

        # Fit both resonances once at the start.
        zcs, slopes = [], []
        for i, (lo, hi) in enumerate([(args.rf_min, args.rf_max), (args.rf_min2, args.rf_max2)]):
            setup_single_resonance(h)
            sf = scan_and_fit_feature(h, lo, hi, points=args.points, power=args.power,
                                      data_rate=args.data_rate)
            zcs.append(float(sf["fit"]["f_zc_rf"]))
            slopes.append(abs(float(sf["fit"]["slope_lsb_per_hz"])))
            _plot_spectrum_fit(outdir, sf["rf"], sf["demod"], sf["fit"],
                               fname=f"spectrum_fit_res{i}.png")
            print(f"[fit] res{i}: zc={zcs[-1]/1e9:.6f} GHz, slope={slopes[-1]:.3g} LSB/Hz")

        # Open-loop left resonance reference.
        setup_single_resonance(h)
        park_cw(h, zcs[0], args.power)
        print(f"[open] recording {args.duration:g}s")
        open_rec = record_traces(h, args.duration, closed_loop=False,
                                 slope_lsb_per_hz=slopes[0], poll=args.poll)
        open_B = _field_from_open_left_error(open_rec["err_lsb"], slopes[0])
        open_trace = {
            "time_s": np.asarray(open_rec["times"], dtype=np.float64),
            "B_nT": open_B,
            "fs_hz": STREAM_SAMPLE_RATE_HZ,
            "asd": _asd(open_B, STREAM_SAMPLE_RATE_HZ),
            "loss_frac": float(open_rec["loss_frac"]),
        }
        print(f"[open] loss={open_trace['loss_frac']*100:.3f}%")

        # Actual 1-res left traces.
        one_traces = {}
        setup_single_resonance(h)
        park_cw(h, zcs[0], args.power)
        for bw in one_bws:
            settings = LockSettings(bandwidth_hz=bw, pi=False,
                                    max_correction_hz=args.max_correction)
            pol = ensure_lock_polarity(h, slopes[0], settings)
            print(f"[1res] BW={bw:g}Hz invert={pol['chosen']['invert']} "
                  f"recording {args.duration:g}s")
            rec = record_traces(h, args.duration, closed_loop=True,
                                slope_lsb_per_hz=slopes[0], settings=settings,
                                poll=args.poll)
            b = _field_from_left_corr(rec["corr_hz"], args.baseline_s, STREAM_SAMPLE_RATE_HZ)
            one_traces[bw] = {
                "time_s": np.asarray(rec["times"], dtype=np.float64),
                "B_nT": b,
                "fs_hz": STREAM_SAMPLE_RATE_HZ,
                "asd": _asd(b, STREAM_SAMPLE_RATE_HZ),
                "loss_frac": float(rec["loss_frac"]),
                "polarity": pol,
            }
            print(f"[1res] BW={bw:g}Hz loss={rec['loss_frac']*100:.3f}%")

        # Actual 2-res differential traces.
        two_traces = {}
        odmr.set_scan_power(float(args.power))
        try:
            odmr.set_data_rate(float(args.data_rate))
        except Exception:
            pass
        odmr.set_dwell_time(float(args.dwell))
        odmr.set_settle_time(float(args.osc_settle))
        odmr._scan_settling_time = float(args.scan_settle)
        odmr._resonance_freqs = [float(z) for z in zcs]
        odmr._resonance_slopes = [float(s) for s in slopes]
        odmr.set_max_correction_hz(float(args.max_correction))

        for bw in two_bws:
            try:
                odmr.stop_multi_tracking()
            except Exception:
                pass
            try:
                h.mw.off()
                time.sleep(0.2)
            except Exception:
                pass
            odmr.lock_bandwidth = float(bw)
            print(f"[2res] BW={bw:g}Hz configure/start, recording {args.duration:g}s "
                  f"+ drop {args.settle_drop:g}s")
            odmr.configure_multi_tracking()
            h.lock_hw.set_bandwidth(float(bw), float(np.mean(slopes)), pi=False)
            odmr.start_multi_tracking()
            try:
                rec = record_traces_multi(h, args.duration, nslots=2,
                                          settle_drop_s=args.settle_drop,
                                          poll=args.poll)
            finally:
                odmr.stop_multi_tracking()
            t, b, fs = _interp_2res_field(rec, args.baseline_s)
            two_traces[bw] = {
                "time_s": t,
                "B_nT": b,
                "fs_hz": fs,
                "asd": _asd(b, fs),
                "loss_frac": float(rec["loss_frac"]),
            }
            print(f"[2res] BW={bw:g}Hz fs={fs:.1f}Hz loss={rec['loss_frac']*100:.3f}%")
            time.sleep(0.5)

        cases = []
        for one_bw, two_bw in zip(one_bws, two_bws):
            case = {
                "one_bw_hz": one_bw,
                "two_bw_hz": two_bw,
                "open": open_trace,
                "one": one_traces[one_bw],
                "two": two_traces[two_bw],
            }
            _plot_case(outdir, case, args.lowpass, args.max_asd_hz)
            cases.append(case)

        summary_cases = []
        for case in cases:
            row = {"one_bw_hz": case["one_bw_hz"], "two_bw_hz": case["two_bw_hz"]}
            for key in ("open", "one", "two"):
                f, a = case[key]["asd"]
                row[key] = {
                    "loss_frac": case[key]["loss_frac"],
                    "fs_hz": case[key]["fs_hz"],
                    "asd_floors": _floors(f, a),
                    "std_nT": float(np.nanstd(case[key]["B_nT"])),
                }
            row["ratios"] = {
                "open_over_two": {
                    band: row["open"]["asd_floors"][band] / row["two"]["asd_floors"][band]
                    for band in row["open"]["asd_floors"]
                },
                "one_over_two": {
                    band: row["one"]["asd_floors"][band] / row["two"]["asd_floors"][band]
                    for band in row["one"]["asd_floors"]
                },
            }
            summary_cases.append(row)

        # Store compact derived data for replotting.
        np.savez_compressed(
            os.path.join(outdir, "comparison_derived_traces.npz"),
            open_time_s=open_trace["time_s"],
            open_B_nT=open_trace["B_nT"],
            one_bws_hz=np.asarray(one_bws),
            two_bws_hz=np.asarray(two_bws),
            zcs_hz=np.asarray(zcs),
            slopes_lsb_per_hz=np.asarray(slopes),
            **{f"one_{bw:g}_time_s": one_traces[bw]["time_s"] for bw in one_bws},
            **{f"one_{bw:g}_B_nT": one_traces[bw]["B_nT"] for bw in one_bws},
            **{f"two_{bw:g}_time_s": two_traces[bw]["time_s"] for bw in two_bws},
            **{f"two_{bw:g}_B_nT": two_traces[bw]["B_nT"] for bw in two_bws},
        )

        summary = {
            "outdir": outdir,
            "duration_s": args.duration,
            "settings": {
                "power_dbm": args.power,
                "one_res_bandwidths_hz": one_bws,
                "two_res_bandwidths_hz": two_bws,
                "two_res_dwell_s": args.dwell,
                "two_res_scan_settle_s": args.scan_settle,
                "two_res_osc_settle_s": args.osc_settle,
                "two_res_settle_drop_s": args.settle_drop,
                "lowpass_for_time_plots_hz": args.lowpass,
            },
            "zcs_hz": zcs,
            "slopes_lsb_per_hz": slopes,
            "cases": summary_cases,
        }
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=float)
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
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--poll", type=float, default=0.05)
    p.add_argument("--one-bws", type=str, default="80,120,150")
    p.add_argument("--two-bws", type=str, default="5,10,20")
    p.add_argument("--dwell", type=float, default=1.0e-3)
    p.add_argument("--scan-settle", type=float, default=100e-6)
    p.add_argument("--osc-settle", type=float, default=200e-6)
    p.add_argument("--settle-drop", type=float, default=10.0)
    p.add_argument("--max-correction", type=float, default=1.0e6)
    p.add_argument("--baseline-s", type=float, default=10.0)
    p.add_argument("--lowpass", type=float, default=0.5)
    p.add_argument("--max-asd-hz", type=float, default=200.0)
    p.add_argument("--outdir", type=str, default=None)
    return p


def main() -> None:
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()
