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
import matplotlib

from my_software.tools.fitting import fit_hyperfine, evaluate_hyperfine

matplotlib.use("Qt5Agg")

# ----------------------------------------- FILENAMES AND PARAMETERS ------------------------------------------------- #

slope_index = 1
f_ENBW = 500  # [Hz]
filename_result = 'result_85mW_ohne_sinc.pkl'
filename_odmr = "20240326-1823-44_85mW_hyperfine_ohne_sinc_ODMR_signal.dat"

voltage_LIA_output_scaling = 100

# hyperfine peak finding parameters
feature_distance = 0.5e6  # [Hz]
min_feature_amp = 0.02
zero_crossing_fit_range = 0.1e6  # [Hz]

# ----------------------------------------- FILENAMES AND PARAMETERS --------------------------------------------------#


# ----------------------------------------- PROGRAM --------------------------------------------------- #
odmr_data = pd.read_csv(filename_odmr, comment='#', sep='\t',
                        header=None, names=['Frequency', 'Voltage'])
with open(filename_result, 'rb') as f:
    result = pickle.load(f)

# Find peaks and dips in the ODMR signal; specifically the maxima and minima of each peak
peaks_indices, dips_indices, zero_crossings_indices, slopes, intercepts = evaluate_hyperfine(
    np.array(odmr_data['Frequency']), np.array(odmr_data['Voltage']), feature_distance, min_feature_amp,
    zero_crossing_fit_range)

# --- Plot hyperfine spectrum and fitted lines --- #
fig, ax = plt.subplots()
ax.plot(odmr_data["Frequency"]/1e6, odmr_data["Voltage"], linewidth=3, color="dimgray")
ax.scatter(odmr_data['Frequency'].iloc[peaks_indices]/1e6, odmr_data['Voltage'].iloc[peaks_indices], color='red', label="Peaks", zorder=3)
ax.scatter(odmr_data['Frequency'].iloc[dips_indices]/1e6, odmr_data['Voltage'].iloc[dips_indices], color='cornflowerblue', label="Dips", zorder=3)

for i, slope in enumerate(slopes):
    lower_bound, upper_bound = zero_crossings_indices[i] - 30, zero_crossings_indices[i] + 30
    ax.plot(odmr_data['Frequency'][lower_bound:upper_bound]/1e6,
            slopes[i] * odmr_data['Frequency'][lower_bound:upper_bound] + intercepts[i], color='orange', linestyle='--')

ax.set_ylabel("Voltage [V]")
ax.set_xlabel("Frequency [MHz]")
ax.set_title("Fitting ZCSs")
#ax.grid()
ax.legend()
fig.tight_layout()
fig.show()

# conversion factor for Volt -> nT
gyromagnetic_ratio_Hz_per_nT = 28.024  # Hz/nT
volts_to_nT_multiplication_factor = 1 / slopes[
    slope_index] * 1 / gyromagnetic_ratio_Hz_per_nT  # 1/slope is Hz/V, 1/gyromagnetic ratio is nT/Hz

# --- Process time traces from the lock-in amplifier --- #
data_dict_x = result['/dev7279/demods/0/sample.x'][0]
data_dict_y = result['/dev7279/demods/0/sample.y'][0]


times = data_dict_x.time
duration = result['/duration'][0]  # [s]
sample_rate = int(len(times) / duration)  # [Hz]

# apply conversion factor to the data and scale voltages in data_dict_x.value[0] by factor voltage_LIA_output_scaling
# to account for the scaling of the analog LIA output -> that goes into the NIDAQ for the slope measurements
samples_x_B_field = data_dict_x.value[0] * volts_to_nT_multiplication_factor * voltage_LIA_output_scaling
samples_x_B_field_original = samples_x_B_field.copy()

# # remove the DC offset in each 1 second interval separately -> unsure if this is good practice
# for i in range(int(result['/duration'][0])):
#     samples_x_B_field[i*sample_rate:(i+1)*sample_rate] = samples_x_B_field[i*sample_rate:(i+1)*sample_rate] - np.mean(samples_x_B_field[i*sample_rate:(i+1)*sample_rate])

# samples_x_B_field = samples_x_B_field - np.mean(samples_x_B_field)
# samples_y_B_field = data_dict_y.value[0] * volts_to_nT_multiplication_factor
# samples_y_B_field = samples_y_B_field - np.mean(samples_y_B_field)


# ----------------------------------------------------------------------------------------------------------------------
# Plot Time-Traces

fig, axs = plt.subplots(2, 1)

axs[0].plot(times[0:int(sample_rate * int(duration))], samples_x_B_field_original[0:int(sample_rate * int(duration))],
            linewidth=0.2, alpha=0.8, color="dimgray")
axs[0].grid()
axs[0].set_title(f"{int(duration)} s Time Trace of Magnetic Field Noise")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Magnetic Field [nT]")

axs[1].plot(times[0:int(sample_rate)], samples_x_B_field[0:int(sample_rate)], linewidth=0.75, color="dimgray")
axs[1].grid()
axs[1].set_title("1 s Time Trace of Magnetic Field Noise")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Magnetic Field [nT]")
fig.tight_layout()
fig.show()

# ----------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()

welch_x_hanning = welch(samples_x_B_field, fs=sample_rate, nperseg=sample_rate, noverlap=0, window='hann')
welch_x_boxcar = welch(samples_x_B_field, fs=sample_rate, nperseg=sample_rate, noverlap=0, window="boxcar")

sensitivity_nT_root_Hz = np.mean(
    [np.std(samples_x_B_field[i * int(sample_rate):(i + 1) * int(sample_rate)]) for i in
     range(int(duration))]) / np.sqrt(2 * f_ENBW)

ax.plot(welch_x_hanning[0], np.sqrt(welch_x_hanning[1]), label="Hann window", alpha=0.7, linestyle="--", color="dimgray")
ax.plot(welch_x_boxcar[0], np.sqrt(welch_x_boxcar[1]), label="Boxcar window", alpha=0.7, linestyle="-.")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(f"Amplitude spectral density of the x-component of the magnetic field\n"
             f"Sensitivity: {sensitivity_nT_root_Hz:.2f} nT/Hz)")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Sqrt of power spectral density [nT/sqrt(Hz)]")
ax.grid()
ax.legend()
ax.set_ylim([1e-4, 1e2])
fig.tight_layout()
fig.show()

plt.show()
