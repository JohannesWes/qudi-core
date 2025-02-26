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
def plot_magnetic_field_time_traces(samples_x_B_field, times, sample_rate, duration, save_fig=False, filename_prefix=None):
    """
    Plot magnetic field noise for {duration} seconds.
    """
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
        fig.savefig(filename_prefix + "_magnetic_field_time_traces.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------

def plot_voltage_time_traces(samples_x, times, sample_rate, duration, save_fig=False, filename_prefix=None):
    """
    Plot voltage noise for {duration} seconds.
    """
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
        fig.savefig(filename_prefix + "_voltage_field_time_traces.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
def plot_asds(samples_x_B_field, sample_rate, duration, f_ENBW, save_fig=False, save_data=True, filename_prefix=None):
    fig, ax = plt.subplots()

    welch_x_hanning = welch(samples_x_B_field, fs=sample_rate, nperseg=sample_rate, noverlap=0, window='hann')
    welch_x_boxcar = welch(samples_x_B_field, fs=sample_rate, nperseg=sample_rate, noverlap=0, window="boxcar")
    frequencies = welch_x_hanning[0]

    # sensitivity via the standard deviation of the time series, not filtered
    sensitivity_std = np.mean(
        [np.std(samples_x_B_field[i * int(sample_rate):(i + 1) * int(sample_rate)]) for i in
         range(int(duration))]) / np.sqrt(2 * f_ENBW)

    asd_hanning, asd_boxcar = np.sqrt(welch_x_hanning[1]), np.sqrt(welch_x_boxcar[1])
    hanning_noise_floor, bandwidth = calculate_asd_noise_floor(frequencies , asd_hanning, 10, f_ENBW, filter_frequencies=[100, 150, 200, 250, 300, 350, 400, 450],
                                                               filter_intervals=[[48,52]])


    ax.plot(welch_x_hanning[0], asd_hanning, label="Hann window", alpha=0.7, linestyle="--", color="dimgray")
    ax.plot(welch_x_boxcar[0], asd_boxcar, label="Boxcar window", alpha=0.7, linestyle="-.")
    ax.axhline(hanning_noise_floor, color='r', linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f"Amplitude spectral density of the x-component of the magnetic field\n"
                 f"Sensitivity: {hanning_noise_floor:.2f} nT/sqrt(Hz)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Noise Amplitude Spectral Density [nT/"+r"$\sqrt{\mathrm{Hz}}$"+"]")
    ax.set_ylim([1E-4, 1E3])
    ax.grid()
    # ax.set_ylim(bottom=np.min(asd_hanning, asd_boxcar) / 10, top=np.max(asd_hanning, asd_boxcar) * 10)
    ax.legend()
    fig.tight_layout()
    if save_fig and filename_prefix is not None:
        fig.savefig(filename_prefix + "_ASD.pdf")
    if save_data and filename_prefix is not None:
        asd_df = pd.DataFrame(
            data={"frequencies": welch_x_hanning[0], "asd_hanning": asd_hanning, "asd_boxcar": asd_boxcar})
        asd_df.to_csv(filename_prefix + "_ASD.csv", sep="\t")
    plt.close(fig)

    return {"frequencies": welch_x_hanning[0], "asd_hanning": asd_hanning, "asd_boxcar": asd_boxcar,
            "sensitivity": hanning_noise_floor, "sensitivity_std": sensitivity_std}



def calculate_asd_noise_floor(freqs, asd_values, f1, f2, filter_frequencies=None, filter_intervals=None):
    """
    Calculate the noise floor of the Amplitude Spectral Density (ASD) within a specified frequency range.

    Args:
        freqs (array-like): Frequencies corresponding to the ASD values.
        asd_values (array-like): ASD values corresponding to `freqs`.
        f1 (float): Lower bound of the frequency range.
        f2 (float): Upper bound of the frequency range.
        filter_frequencies (list, optional): Specific frequencies to filter out from the ASD data.
        filter_intervals (list of tuples, optional): Frequency intervals to filter out from the ASD data.

    Returns:
        tuple: (mean_asd, equivalent_bandwidth)
            mean_asd (float): Mean ASD value within the specified frequency range.
            equivalent_bandwidth (float): Adjusted bandwidth after filtering.
    """
    # Convert inputs to numpy arrays
    freqs = np.asarray(freqs)
    asd_values = np.asarray(asd_values)

    # Initial bandwidth
    bandwidth = f2 - f1

    # Filter out specific frequencies if provided
    if filter_frequencies is not None and len(filter_frequencies) > 0:
        # Create mask to exclude filter_frequencies
        mask = ~np.isin(freqs, filter_frequencies)
        freqs = freqs[mask]
        asd_values = asd_values[mask]

        # Adjust bandwidth by the count of removed frequencies (as in the original code)
        bandwidth = bandwidth - len(filter_frequencies)

    # Filter out frequency intervals if provided
    if filter_intervals is not None:
        for (start_freq, end_freq) in filter_intervals:
            # Mask to keep only frequencies outside the given interval
            mask = (freqs < start_freq) | (freqs > end_freq)
            freqs = freqs[mask]
            asd_values = asd_values[mask]

            # Adjust bandwidth by the size of the removed interval
            bandwidth = bandwidth - (end_freq - start_freq)

    # Filter out frequencies outside the interval [f1, f2]
    within_range_mask = (freqs >= f1) & (freqs <= f2)
    freqs = freqs[within_range_mask]
    asd_values = asd_values[within_range_mask]

    # Compute mean ASD and handle the case where no frequencies remain
    if len(freqs) == 0:
        mean_asd = float('nan')
        equivalent_bandwidth = 0.0
    else:
        mean_asd = np.mean(asd_values)
        # The original logic returns the 'bandwidth' variable directly
        equivalent_bandwidth = bandwidth

    return mean_asd, equivalent_bandwidth



def allan_deviation(x, sampling_rate, m_values=None, m_mode='linear'):
    """
    Compute the Allan deviation of time-series data.

    Args:
        x (np.ndarray): 1D array of time-series data.
        sampling_rate (float): Sampling rate in Hz.
        m_values (array-like, optional): Array of block sizes (m). Each block size corresponds
                                         to an averaging time tau = m / sampling_rate.
                                         If None, the function will generate m-values based on m_mode.
        m_mode (str): If m_values is None, defines how to generate m-values.
                      'linear' -> use all integer m-values from 1 to N//2
                      'log2' -> use powers-of-two spaced m-values up to N//2

    Returns:
        tau_values (np.ndarray): Averaging times corresponding to each m (in seconds).
        allan_dev (np.ndarray): Allan deviation values for each tau.
    """
    x = np.asarray(x)
    N = len(x)
    dt = 1.0 / sampling_rate

    if m_values is None:
        max_m = N // 2
        if max_m < 1:
            raise ValueError("Time series too short to compute Allan deviation for any m > 1.")

        if m_mode == 'linear':
            # Use all integers from 1 to max_m
            m_values = np.arange(1, max_m + 1)
        elif m_mode == 'log2':
            # Use powers of 2 up to max_m
            m_values = 2 ** np.arange(int(np.floor(np.log2(max_m))) + 1)
        else:
            raise ValueError(f"Unknown m_mode '{m_mode}'. Choose 'linear' or 'log2'.")

    tau_values = m_values * dt
    allan_dev = np.zeros_like(m_values, dtype=float)

    # Compute Allan deviation for each m
    for i, m in enumerate(m_values):
        num_blocks = N // m
        if num_blocks < 2:
            allan_dev[i] = np.nan
            continue

        block_averages = np.array([np.mean(x[k * m:(k + 1) * m]) for k in range(num_blocks)])
        diff = np.diff(block_averages)
        allan_var = 0.5 * np.mean(diff ** 2)
        allan_dev[i] = np.sqrt(allan_var)

    return tau_values, allan_dev



def plot_allan_deviation(tau_values, allan_dev, save_fig=False, save_data=True, filename_prefix=None):
    fig, ax = plt.subplots()

    ax.plot(tau_values, allan_dev, label="Allan Deviation", alpha=0.7, linestyle="-", color="black")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Allan Deviation")
    ax.set_xlabel("Averaging Time [s]")
    ax.set_ylabel("Allan Deviation [nT]")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    if save_fig and filename_prefix is not None:
        fig.savefig(filename_prefix + "_Allan_Deviation.pdf")
    if save_data and filename_prefix is not None:
        allan_df = pd.DataFrame(data={"tau values [s]": tau_values, "allan deviaton [nT]": allan_dev})
        allan_df.to_csv(filename_prefix + "_Allan_Deviation.csv", sep="\t")
    plt.close(fig)



