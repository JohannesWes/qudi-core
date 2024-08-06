from pathlib import Path
import importlib.util
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fft as fft
from scipy.signal import find_peaks, welch, get_window, periodogram
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib

from my_software.tools.fitting import fit_hyperfine, evaluate_hyperfine

matplotlib.use("Qt5Agg")

GYROMAGNETIC_RATIO_NT_PER_HZ = 28.024  # Hz/nT


# ----------------------------------------- PROGRAM --------------------------------------------------- #

def magnetic_field_from_voltages(voltages, slope, scaling_factor=1):
    voltages = voltages - np.mean(voltages)

    # conversion factor for Volt -> nT
    volts_to_nT_multiplication_factor = 1 / slope * 1 / GYROMAGNETIC_RATIO_NT_PER_HZ  # 1/slope is Hz/V, 1/gyromagnetic ratio is nT/Hz

    # apply conversion factor to the data and scale voltages by factor of LIA output scaling
    # to account for the scaling of the analog LIA output -> that goes into the NIDAQ for the slope measurements
    samples_x_B_field = voltages * volts_to_nT_multiplication_factor * scaling_factor

    return samples_x_B_field


# ----------------------------------------------------------------------------------------------------------------------
# PLOT MAGNETIC FIELD NOISE TIME TRACES
def plot_magnetic_field_time_traces(samples_x_B_field, times, sample_rate, duration, save_fig=False, filename=None):
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(times[0:int(sample_rate * int(duration))], samples_x_B_field[0:int(sample_rate * int(duration))],
                linewidth=0.2, alpha=0.8, color="dimgray")
    axs[0].grid()
    axs[0].set_title(f"{int(duration)} s Time Trace of Magnetic Field Noise")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Magnetic Field [nT]")

    axs[1].plot(times[0:int(sample_rate)], samples_x_B_field[0:int(sample_rate)], linewidth=0.75,
                color="dimgray")  # , marker=".", markersize=0.5)
    axs[1].grid()
    axs[1].set_title("1 s Time Trace of Magnetic Field Noise")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Magnetic Field [nT]")
    fig.tight_layout()
    if save_fig:
        fig.savefig(filename + "_magnetic_field_time_traces.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------

def plot_voltage_time_traces(samples_x, times, sample_rate, duration, save_fig=False, filename=None):
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(times[0:int(sample_rate * int(duration))], samples_x[0:int(sample_rate * int(duration))],
                linewidth=0.2, alpha=0.8, color="navajowhite")
    axs[0].grid()
    axs[0].set_title(f"{int(duration)} s Time Trace of Voltage Noise")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Voltage [V]")

    axs[1].plot(times[0:int(sample_rate)], samples_x[0:int(sample_rate)], linewidth=0.75,
                color="navajowhite")  # , marker=".", markersize=0.5)
    axs[1].grid()
    axs[1].set_title("1 s Time Trace of Voltage Noise")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Voltage [V]")
    fig.tight_layout()
    if save_fig:
        fig.savefig(filename + "_voltage_field_time_traces.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
def plot_asds(samples_x_B_field, sample_rate, duration, f_ENBW, save_fig=False, save_data=True, filename=None):
    fig, ax = plt.subplots()

    welch_x_hanning = welch(samples_x_B_field, fs=sample_rate, nperseg=sample_rate, noverlap=0, window='hann')
    welch_x_boxcar = welch(samples_x_B_field, fs=sample_rate, nperseg=sample_rate, noverlap=0, window="boxcar")

    sensitivity_nT_root_Hz = np.mean(
        [np.std(samples_x_B_field[i * int(sample_rate):(i + 1) * int(sample_rate)]) for i in
         range(int(duration))]) / np.sqrt(2 * f_ENBW)

    asd_hanning, asd_boxcar = np.sqrt(welch_x_hanning[1]), np.sqrt(welch_x_boxcar[1])

    ax.plot(welch_x_hanning[0], asd_hanning, label="Hann window", alpha=0.7, linestyle="--", color="dimgray")
    ax.plot(welch_x_boxcar[0], asd_boxcar, label="Boxcar window", alpha=0.7, linestyle="-.")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f"Amplitude spectral density of the x-component of the magnetic field\n"
                 f"Sensitivity: {sensitivity_nT_root_Hz:.2f} nT/sqrt(Hz)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Sqrt of power spectral density [nT/sqrt(Hz)]")
    ax.set_ylim([1E-4, 1E3])
    ax.grid()
    # ax.set_ylim(bottom=np.min(asd_hanning, asd_boxcar) / 10, top=np.max(asd_hanning, asd_boxcar) * 10)
    ax.legend()
    fig.tight_layout()
    if save_fig and filename is not None:
        fig.savefig(filename + "_ASD.pdf")
    if save_data and filename is not None:
        asd_df = pd.DataFrame(
            data={"frequencies": welch_x_hanning[0], "asd_hanning": asd_hanning, "asd_boxcar": asd_boxcar})
        asd_df.to_csv(filename + "_ASD.csv", sep="\t")
    plt.close(fig)

    return {"frequencies": welch_x_hanning[0], "asd_hanning": asd_hanning, "asd_boxcar": asd_boxcar,
            "sensitivity": sensitivity_nT_root_Hz}
