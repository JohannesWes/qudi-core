import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from scipy.signal import savgol_filter

from lmfit.models import LinearModel, QuadraticModel

from my_software.tools.fitting import quadratic

matplotlib.use("Qt5Agg")


def fit_calibration_curve(filename="data/CALIBRATION_current_measurement_2024-04-12_162441_.csv", sep="\t",
                          id_odmr_dip=0, discard_below=2844.8e6):
    data = pd.read_csv(filename, sep=sep, header=0)

    current_label = "current[A]"  # Corrected to match the actual column name in provided CSV
    data.rename(columns={data.columns[0]: current_label}, inplace=True)

    columns = data.columns

    # OPX FM carrier frequency. We perform FM around 200 MHz. This is mixed with the LO frequency. We use the lower
    # sideband (f_LO - f_FM) -> Need to correct frequency by 200 MHz
    data[columns[1:]] += 200e6

    mask = data[columns[1 + id_odmr_dip]] > discard_below
    data = data[mask]

    frequency_column_data = data[columns[1 + id_odmr_dip]].dropna()

    # Smoothing with Savgol filter, handle short data
    frequencies_smoothed = None
    try:
        if len(frequency_column_data) >= 5:  # Minimum data points needed for savgol_filter with window_length=3 and polyorder=3
            window_length = min(len(frequency_column_data), 1000) # Ensure window_length is not larger than data length
            if window_length % 2 == 0: # window_length must be odd
                window_length -= 1
            if window_length < 3: # Minimum window_length is 3
                window_length = 3

            frequencies_smoothed = savgol_filter(frequency_column_data, window_length, 3)
        else:
            print(f"Warning: Too few data points ({len(frequency_column_data)}) for smoothing. Using raw data for fitting.")
            frequencies_smoothed = frequency_column_data.values # Use raw data if smoothing is not possible
    except Exception as e:
        print(f"Warning: Error during smoothing: {e}. Using raw data for fitting.")
        frequencies_smoothed = frequency_column_data.values # Fallback to raw data if smoothing fails


    # fit linear model to x = data[current_label, y = frequencies_smoothed
    linear_model = LinearModel()
    quadratic_model = QuadraticModel()

    # Use data without NaN for fitting
    valid_data_mask = ~data[current_label].isna() & ~data[columns[1 + id_odmr_dip]].isna()  # mask for valid data for both x and y
    valid_current_data = data[current_label][valid_data_mask]  # Apply mask to current data
    valid_frequency_data_for_fit = data[columns[1 + id_odmr_dip]][valid_data_mask] # Use original frequency data for fitting, smoothed or raw

    if frequencies_smoothed is not None and len(frequencies_smoothed) == len(valid_frequency_data_for_fit): # Use smoothed data if smoothing was successful and lengths match
        valid_frequencies_smoothed_data = frequencies_smoothed
    else: # Otherwise use raw valid frequency data
        valid_frequencies_smoothed_data = valid_frequency_data_for_fit.values
        print("Using raw valid frequency data for fitting because smoothing failed or data length mismatch.")


    pars_linear = linear_model.guess(valid_frequencies_smoothed_data, x=valid_current_data, intercept=2845.24e9,
                                     slope=0.5e9 / 0.025)  # use valid data for guess and fit
    pars_quadratic = quadratic_model.guess(valid_frequencies_smoothed_data, x=valid_current_data, a=-0.44e9, b=0.5e9 / 0.025,
                                        c=2.9e9)  # use valid data for guess and fit

    result_linear = linear_model.fit(valid_frequencies_smoothed_data, pars_linear, x=valid_current_data)  # use valid data for fit
    result_quadratic = quadratic_model.fit(valid_frequencies_smoothed_data, pars_quadratic, x=valid_current_data)  # use valid data for fit

    fig_calibration, ax_calibration = plt.subplots()

    ax_calibration.scatter(data[current_label], data[columns[1 + id_odmr_dip]] / 1e6, label="calibration data",
                           color="black", s=5,
                           alpha=0.1)

    if frequencies_smoothed is not None and len(valid_current_data) == len(frequencies_smoothed): # Only plot smoothed curve if smoothing was successful and lengths match
        ax_calibration.plot(valid_current_data, valid_frequencies_smoothed_data / 1e6, color="red", label="smoothed curve",
                             alpha=0.5)  # plot smoothed curve using valid current and smoothed frequency
    ax_calibration.plot(valid_current_data, result_linear.best_fit / 1e6, color="blue", label="linear fit",
                         alpha=0.5)  # plot linear fit using valid current and fit result
    ax_calibration.plot(valid_current_data, result_quadratic.best_fit / 1e6, color="green", label="quadratic fit",
                         alpha=0.5)  # plot quadratic fit using valid current and fit result
    ax_calibration.legend()

    ax_calibration.set_xlabel(current_label)
    ax_calibration.set_ylabel("Frequency [MHz]")
    ax_calibration.set_title("Calibration Curve")
    fig_calibration.show()
    draw()

    currents = data[current_label]

    # This part might need adjustment depending on if flipping is still needed when smoothing fails or is skipped.
    # if np.mean(frequencies_smoothed[-10:-1]) < np.mean(frequencies_smoothed[0:10]):
    #     frequencies_smoothed = np.flip(frequencies_smoothed)
    #     currents = np.flip(data[current_label])

    return result_quadratic.best_fit, data[current_label], result_quadratic.best_values["a"], result_quadratic.best_values["b"], \
           result_quadratic.best_values["c"]
    # return frequencies_smoothed, currents


def get_currents_from_relative_frequencies(frequencies, frequency_where_measured_current_is_zero):
    calibration_frequencies, calibration_currents, p_a, p_b, p_c = fit_calibration_curve()
    frequency_where_calibration_current_is_zero = quadratic(0, p_a, p_b, p_c)

    relative_frequencies_for_fitting = frequencies - frequency_where_measured_current_is_zero + frequency_where_calibration_current_is_zero

    y = relative_frequencies_for_fitting
    fitted_currents = -np.sqrt(-4 * p_a * p_c + 4 * p_a * y + p_b ** 2) / (2 * p_a) - p_b / (2 * p_a)

    # old
    # fitted_currents = np.interp(np.array(frequencies), calibration_frequencies, calibration_currents)

    return relative_frequencies_for_fitting, fitted_currents


def plot_change_in_odmr_freq_with_current(filename="NICESTUFF_current_measurement_2025-02-27_142024_.csv",
                                          sep="\t"):
    data = pd.read_csv(filename, sep=sep, header=0)

    current_label = "current[A]"  # corrected to match the actual column name in provided CSV
    data.rename(columns={data.columns[0]: current_label}, inplace=True)

    columns = data.columns

    # OPX FM carrier frequency. We perform FM around 200 MHz. This is mixed with the LO frequency. We use the lower
    # sideband (f_LO - f_FM) -> Need to correct frequency by 200 MHz
    data[columns[1:]] += 200e6

    # Check for NaN values and print a warning
    if data[columns[1:]].isnull().values.any():
        print("Warning: NaN values found in frequency data. These points will not be plotted.")

    num_nv_directions = (len(columns) - 1) // 1  # corrected because you only have one range in the example file.

    fig_frequencies, ax_frequencies = plt.subplots(1, num_nv_directions,
                                                    figsize=(7 * num_nv_directions, 5))  # Adjusted subplot to be a single row

    colors = {0: "red", 1: "blue", 2: "green", 3: "orange"}

    row_0 = ax_frequencies if num_nv_directions == 1 else ax_frequencies  # handle single subplot case

    for i_col, ax in enumerate([row_0] if num_nv_directions == 1 else row_0):  # iterate over a list of ax if num_nv_directions == 1
        ax.scatter(data[current_label], data[columns[1 + i_col]] / 1e6, s=5, alpha=0.4, color="black")
        ax.set_xlabel(current_label)
        ax.set_ylabel("Frequency [MHz]")
        ax.set_title("Dip " + str(i_col + 1) + "",
                     color=colors.get(i_col, "black"))  # use get to avoid key error if more ranges than colors
        ax.ticklabel_format(useOffset=False)
        ax.grid()

    fig_frequencies.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_change_in_odmr_freq_with_current("NICESTUFF_current_measurement_2025-03-27_162216_.csv")
    fit_calibration_curve("NICESTUFF_current_measurement_2025-03-27_162216_.csv")
    plt.show()
    get_currents_from_relative_frequencies(None, None)