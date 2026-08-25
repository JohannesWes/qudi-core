"""Plot saved 1-res/2-res frequency tracking diagnostics.

This script consumes the JSON/NPZ files written by ``noise_test.py`` during the
2026-06-30 hardware run and produces a compact visual summary of the remaining
2-res ASD issue. It does not connect to Qudi or hardware.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


FTW_PER_HZ = (2.0 ** 32) / 125e6
STREAM_SAMPLE_RATE_HZ = 125e6 / 4096
GYROMAGNETIC_RATIO_NT_PER_HZ = 28.024

BASE = Path(r"C:\Users\aj92uwef\qudi\Data\freq_tracking_tests")
ONE_RES_DIR = BASE / "analysis_1res_bw_left_20260630"
TWO_RES_DIR = BASE / "analysis_2res_targeted_20260630"
RAW_DIR = BASE / "analysis_2res_rawpoints_20260630"
SETTLE_DIR = BASE / "analysis_2res_settle_followup_20260630"
OUT_DIR = BASE / "analysis_plots_20260630"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def savefig(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT_DIR / f"{name}{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def ffill(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    valid = np.isfinite(a)
    if not valid.any():
        return a.copy()
    idx = np.where(valid, np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    out[: int(np.argmax(valid))] = np.nan
    return out


def parse_words(words: np.ndarray):
    n3 = (words.size // 3) * 3
    trip = np.asarray(words[:n3], dtype=np.float64).reshape(-1, 3)
    err, corr_ftw, step_raw = trip[:, 0], trip[:, 1], trip[:, 2]
    step = np.where(np.isfinite(ffill(step_raw)), np.rint(ffill(step_raw)), -1).astype(np.int64)
    return err, corr_ftw / FTW_PER_HZ, step


def correction_to_field(corr_hz: np.ndarray) -> np.ndarray:
    x = np.asarray(corr_hz, dtype=np.float64)
    return (x - np.nanmean(x)) / GYROMAGNETIC_RATIO_NT_PER_HZ


def compute_asd(samples_nT: np.ndarray, sample_rate: float):
    x = np.asarray(samples_nT, dtype=np.float64)
    if np.any(~np.isfinite(x)):
        idx = np.arange(x.size)
        good = np.isfinite(x)
        x = np.interp(idx, idx[good], x[good]) if good.any() else np.zeros_like(x)
    nperseg = max(16, min(int(sample_rate), x.size))
    f, pxx = welch(x, fs=sample_rate, nperseg=nperseg, noverlap=0, window="hann")
    return f, np.sqrt(pxx)


def fresh_visits(words: np.ndarray, nslots: int = 2):
    err, corr_hz, step = parse_words(words)
    result = {"visit_idx": [], "visit_corr": [], "visit_err": [], "eff_rate_hz": []}
    for r in range(nslots):
        live = step == r
        fresh = live & np.isfinite(err) & np.isfinite(corr_hz)
        edges = np.diff(np.r_[0, live.astype(np.int8), 0])
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        vix, vc, ve = [], [], []
        for s, e in zip(starts, ends):
            seg = fresh[s : e + 1]
            if seg.any():
                j = s + int(np.where(seg)[0][-1])
                vix.append(j)
                vc.append(corr_hz[j])
                ve.append(err[j])
        vix = np.asarray(vix, dtype=np.int64)
        result["visit_idx"].append(vix)
        result["visit_corr"].append(np.asarray(vc, dtype=np.float64))
        result["visit_err"].append(np.asarray(ve, dtype=np.float64))
        if vix.size >= 3:
            med = float(np.median(np.diff(vix.astype(np.float64))))
            result["eff_rate_hz"].append(STREAM_SAMPLE_RATE_HZ / med)
        else:
            result["eff_rate_hz"].append(float("nan"))
    return result


def run_lengths(step: np.ndarray):
    if step.size == 0:
        return []
    changes = np.flatnonzero(np.diff(step) != 0) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, step.size]
    return [(int(step[s]), int(e - s), int(s), int(e)) for s, e in zip(starts, ends)]


def timing_metrics(words: np.ndarray):
    _err, _corr, step = parse_words(words)
    runs = run_lengths(step)
    steady = runs[1:] if len(runs) > 1 else runs
    live_lens = [ln for state, ln, _s, _e in steady if state in (0, 1)]
    dead_lens = [ln for state, ln, _s, _e in steady if state < 0]
    visits = fresh_visits(words)
    spacing = []
    for vix in visits["visit_idx"]:
        if vix.size >= 3:
            spacing.append(float(np.median(np.diff(vix))))
    med_spacing = float(np.median(spacing)) if spacing else float("nan")
    med_live = float(np.median(live_lens)) if live_lens else float("nan")
    med_dead = float(np.median(dead_lens)) if dead_lens else float("nan")
    return {
        "step": step,
        "runs": runs,
        "med_live_ticks": med_live,
        "med_dead_ticks": med_dead,
        "med_visit_ticks": med_spacing,
        "live_fraction": med_live / med_spacing if med_spacing > 0 else float("nan"),
        "eff_rate_hz": STREAM_SAMPLE_RATE_HZ / med_spacing if med_spacing > 0 else float("nan"),
    }


def within_dwell_profile(words: np.ndarray, nslots: int = 2):
    err, _corr, step = parse_words(words)
    out = []
    for r in range(nslots):
        live = step == r
        edges = np.diff(np.r_[0, live.astype(np.int8), 0])
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        if starts.size == 0:
            out.append(np.array([]))
            continue
        lengths = ends - starts + 1
        med_len = float(np.median(lengths))
        steady = lengths <= max(1.5 * med_len, med_len + 2)
        starts = starts[steady]
        ends = ends[steady]
        if starts.size == 0:
            out.append(np.array([]))
            continue
        max_len = int(np.max(ends - starts + 1))
        acc = np.zeros(max_len)
        cnt = np.zeros(max_len)
        for s, e in zip(starts, ends):
            seg = err[s : e + 1]
            k = np.arange(seg.size)
            good = np.isfinite(seg)
            acc[k[good]] += seg[good]
            cnt[k[good]] += 1.0
        out.append(np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan))
    return out


def rows_for_axis(rows, axis):
    return sorted([r for r in rows if r["axis"] == axis], key=lambda r: r["bw"])


def plot_summary_sweeps():
    one = load_json(ONE_RES_DIR / "sweep_summary.json")
    two = load_json(TWO_RES_DIR / "targeted_summary.json")["rows"]
    settle = load_json(SETTLE_DIR / "summary.json")

    one_i = sorted([r for r in one if not r["settings"]["pi"]], key=lambda r: r["settings"]["bandwidth_hz"])
    one_pi = sorted([r for r in one if r["settings"]["pi"]], key=lambda r: r["settings"]["bandwidth_hz"])
    good = rows_for_axis(two, "bw_1ms_scan100_osc200")
    scan0 = rows_for_axis(two, "bw_1ms_scan0_osc200")
    fast = rows_for_axis(two, "bw_0p5ms_scan0_osc200")
    slow = rows_for_axis(two, "bw_2ms_scan0_osc200")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot([r["settings"]["bandwidth_hz"] for r in one_i], [r["track_floor_nT_sqrtHz"] for r in one_i],
            "o-", label="1-res I-only")
    ax.plot([r["bw"] for r in good], [r["floor_nT_sqrtHz"] for r in good],
            "o-", label="2-res 1 ms, scan100, osc200")
    ax.plot([r["bw"] for r in scan0], [r["floor_nT_sqrtHz"] for r in scan0],
            "s--", label="2-res 1 ms, scan0, osc200")
    ax.plot([r["bw"] for r in fast], [r["floor_nT_sqrtHz"] for r in fast],
            "^--", label="2-res 0.5 ms, scan0, osc200")
    ax.plot([r["bw"] for r in slow], [r["floor_nT_sqrtHz"] for r in slow],
            "v--", label="2-res 2 ms, scan0, osc200")
    ax.axvspan(70, 80, color="0.85", label="70-80 Hz target")
    ax.set_xlabel("Requested tracking bandwidth (Hz)")
    ax.set_ylabel("Low-frequency floor (nT/sqrtHz)")
    ax.set_title("Low-frequency ASD floor")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot([r["settings"]["bandwidth_hz"] for r in one_i], [r["servo_bump"] for r in one_i],
            "o-", label="1-res I-only")
    ax.plot([r["bw"] for r in good], [r["bump"] for r in good],
            "o-", label="2-res good timing")
    ax.plot([r["settings"]["bandwidth_hz"] for r in one_pi], [r["servo_bump"] for r in one_pi],
            "x:", label="1-res PI current tuning")
    ax.axvspan(70, 80, color="0.85")
    ax.set_xlabel("Requested tracking bandwidth (Hz)")
    ax.set_ylabel("Servo bump metric (x)")
    ax.set_title("Noise amplification / servo bump")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    for rows, label, marker in [
        (good, "scan100, osc200", "o"),
        (scan0, "scan0, osc200", "s"),
        (fast, "0.5 ms dwell, scan0", "^"),
        (slow, "2 ms dwell, scan0", "v"),
    ]:
        x = [r["bw"] / (r["eff_rate_hz"] / 2.0) for r in rows]
        y = [r["floor_nT_sqrtHz"] for r in rows]
        ax.plot(x, y, marker + "-", label=label)
    ax.axvspan(70 / 218.0, 80 / 218.0, color="0.85", label="70-80 Hz at 436 Hz visits")
    ax.set_xlabel("Requested BW / per-res Nyquist")
    ax.set_ylabel("Low-frequency floor (nT/sqrtHz)")
    ax.set_title("2-res is limited by visit-rate sampling")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    osc100 = rows_for_axis(settle, "scan100_osc100")
    osc50 = rows_for_axis(settle, "scan100_osc50")
    ax.plot([r["bw"] for r in good], [r["floor_nT_sqrtHz"] for r in good],
            "o-", label="osc settle 200 us")
    ax.plot([r["bw"] for r in osc100], [r["floor_nT_sqrtHz"] for r in osc100],
            "s--", label="osc settle 100 us")
    ax.plot([r["bw"] for r in osc50], [r["floor_nT_sqrtHz"] for r in osc50],
            "^--", label="osc settle 50 us")
    ax.set_xlabel("Requested tracking bandwidth (Hz)")
    ax.set_ylabel("Low-frequency floor (nT/sqrtHz)")
    ax.set_title("Shorter oscillator settling contaminates ASD")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("1-res vs 2-res closed-loop ASD metrics from saved hardware runs", fontsize=14)
    savefig(fig, "tracking_floor_bump_summary")


def plot_raw_asd_curves():
    files = [
        ("good_bw10_dwell1ms_scan100_osc200.npz", "BW10, 1 ms, scan100, osc200"),
        ("bad_bw10_dwell1ms_scan0_osc200.npz", "BW10, 1 ms, scan0, osc200"),
        ("fast_bw10_dwell0p5ms_scan0_osc200.npz", "BW10, 0.5 ms, scan0, osc200"),
        ("highbw_bw80_dwell1ms_scan100_osc200.npz", "BW80, 1 ms, scan100, osc200"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=False, sharey=True)
    for ax, (filename, label) in zip(axs.ravel(), files):
        data = np.load(RAW_DIR / filename)
        visits = fresh_visits(data["words"])
        for r in range(2):
            corr = visits["visit_corr"][r]
            eff = visits["eff_rate_hz"][r]
            f, a = compute_asd(correction_to_field(corr), eff)
            ax.loglog(f[1:], a[1:], lw=1.2, label=f"res{r}, visit {eff:.0f} Hz")
        nyq = np.nanmean(visits["eff_rate_hz"]) / 2.0
        ax.axvline(nyq, color="k", lw=0.8, alpha=0.5)
        ax.axvspan(5, 50, color="C2", alpha=0.08)
        ax.set_title(label)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Correction ASD (nT/sqrtHz)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Per-visit 2-res correction ASDs from raw FPGA captures", fontsize=14)
    savefig(fig, "raw_per_visit_asd_curves")


def plot_timing_and_profiles():
    files = [
        ("good_bw10_dwell1ms_scan100_osc200.npz", "good: 1 ms scan100 osc200 BW10"),
        ("bad_bw10_dwell1ms_scan0_osc200.npz", "scan0: 1 ms scan0 osc200 BW10"),
        ("fast_bw10_dwell0p5ms_scan0_osc200.npz", "fast: 0.5 ms scan0 osc200 BW10"),
        ("highbw_bw80_dwell1ms_scan100_osc200.npz", "high BW: 1 ms scan100 osc200 BW80"),
    ]

    fig, axs = plt.subplots(4, 2, figsize=(12, 10), constrained_layout=True)
    for row, (filename, label) in enumerate(files):
        words = np.load(RAW_DIR / filename)["words"]
        met = timing_metrics(words)
        step = met["step"]
        runs = met["runs"]
        start = runs[1][2] if len(runs) > 2 else 0
        stop = min(start + 220, step.size)
        t_us = (np.arange(start, stop) - start) / STREAM_SAMPLE_RATE_HZ * 1e6

        ax = axs[row, 0]
        ax.step(t_us, step[start:stop], where="post", lw=1.1)
        ax.set_ylim(-1.4, 1.4)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["dead", "res0", "res1"])
        ax.set_xlabel("Time in raw stream window (us)")
        ax.set_title(
            f"{label}\nmedian live {met['med_live_ticks']:.0f} ticks, dead {met['med_dead_ticks']:.0f}, "
            f"per-res live {100 * met['live_fraction']:.1f}%, visit {met['eff_rate_hz']:.0f} Hz"
        )
        ax.grid(True, alpha=0.25)

        ax = axs[row, 1]
        profiles = within_dwell_profile(words)
        for r, prof in enumerate(profiles):
            if prof.size:
                x_us = np.arange(prof.size) / STREAM_SAMPLE_RATE_HZ * 1e6
                prof_hz = prof / (34.369017079548705 if r == 0 else 31.792156059010964)
                ax.plot(x_us, prof_hz - np.nanmean(prof_hz), label=f"res{r}")
        ax.axhline(0, color="k", lw=0.7, alpha=0.4)
        ax.set_xlabel("Sample time within live dwell (us)")
        ax.set_ylabel("Mean demod error transient (Hz)")
        ax.set_title("Within-dwell mean error profile")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Raw marked-stream timing and dwell transients", fontsize=14)
    savefig(fig, "raw_timing_and_dwell_profiles")


def plot_visit_jumps():
    files = [
        ("good_bw10_dwell1ms_scan100_osc200.npz", "good BW10"),
        ("bad_bw10_dwell1ms_scan0_osc200.npz", "scan0 BW10"),
        ("fast_bw10_dwell0p5ms_scan0_osc200.npz", "0.5 ms BW10"),
        ("highbw_bw80_dwell1ms_scan100_osc200.npz", "good timing BW80"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, (filename, label) in zip(axs.ravel(), files):
        words = np.load(RAW_DIR / filename)["words"]
        visits = fresh_visits(words)
        for r in range(2):
            corr = visits["visit_corr"][r]
            eff = visits["eff_rate_hz"][r]
            t = np.arange(corr.size) / eff
            b = correction_to_field(corr)
            ax.plot(t[:1500], b[:1500], lw=0.8, label=f"res{r}")
        ax.set_title(label)
        ax.set_xlabel("Visit time (s)")
        ax.set_ylabel("Correction (nT, mean removed)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Per-visit correction traces: high BW and aggressive timing increase jumps", fontsize=14)
    savefig(fig, "raw_per_visit_correction_traces")


def main() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_summary_sweeps()
    plot_raw_asd_curves()
    plot_timing_and_profiles()
    plot_visit_jumps()
    print(f"Wrote plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
