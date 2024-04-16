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

    current_label = "Current [A]"
    data.rename(columns={data.columns[0]: current_label}, inplace=True)

    columns = data.columns

    # OPX FM carrier frequency. We perform FM around 200 MHz. This is mixed with the LO frequency. We use the lower
    # sideband (f_LO - f_FM) -> Need to correct frequency by 200 MHz
    data[columns[1:]] += 200e6

    mask = data[columns[1 + id_odmr_dip]] > discard_below
    data = data[mask]

    # fit linear function to x = data[current_label, y = data[columns[1+i]]
    frequencies_smoothed = savgol_filter(data[columns[1 + id_odmr_dip]], 1000, 3)

    # fit linear model to x = data[current_label, y = frequencies_smoothed
    linear_model = LinearModel()
    quadratic_model = QuadraticModel()

    pars_linear = linear_model.guess(frequencies_smoothed, x=data[current_label], intercept=2845.24e9, slope=0.5e9/0.025)
    pars_quadratic = quadratic_model.guess(frequencies_smoothed, x=data[current_label], a=-0.44e9, b=0.5e9/0.025, c=2.9e9)

    result_linear = linear_model.fit(frequencies_smoothed, pars_linear, x=data[current_label])
    result_quadratic = quadratic_model.fit(frequencies_smoothed, pars_quadratic, x=data[current_label])

    fig_calibration, ax_calibration = plt.subplots()

    ax_calibration.scatter(data[current_label], data[columns[1 + id_odmr_dip]] / 1e6, label="calibration data", color="black", s=5,
                alpha=0.1)

    ax_calibration.plot(data[current_label], frequencies_smoothed / 1e6, color="red", label="smoothed curve", alpha=0.5)
    ax_calibration.plot(data[current_label], result_linear.best_fit / 1e6, color="blue", label="linear fit", alpha=0.5)
    ax_calibration.plot(data[current_label], result_quadratic.best_fit / 1e6, color="green", label="quadratic fit", alpha=0.5)
    ax_calibration.legend()

    ax_calibration.set_xlabel(current_label)
    ax_calibration.set_ylabel("Frequency [MHz]")
    ax_calibration.set_title("Calibration Curve")
    fig_calibration.show()
    draw()


    currents = data[current_label]

    if np.mean(frequencies_smoothed[-10:-1]) < np.mean(frequencies_smoothed[0:10]):
        frequencies_smoothed = np.flip(frequencies_smoothed)
        currents = np.flip(data[current_label])

    return result_quadratic.best_fit, data[current_label], result_quadratic.best_values["a"], result_quadratic.best_values["b"], result_quadratic.best_values["c"]
    #return frequencies_smoothed, currents


def get_currents_from_relative_frequencies(frequencies, frequency_where_measured_current_is_zero):

    calibration_frequencies, calibration_currents, p_a, p_b, p_c = fit_calibration_curve()
    frequency_where_calibration_current_is_zero = quadratic(0, p_a, p_b, p_c)

    relative_frequencies_for_fitting = frequencies - frequency_where_measured_current_is_zero + frequency_where_calibration_current_is_zero

    y = relative_frequencies_for_fitting
    fitted_currents = np.sqrt(-4*p_a*p_c + 4*p_a*y + p_b**2)/(2 * p_a) - p_b / (2 * p_a)

    # old
    #fitted_currents = np.interp(np.array(frequencies), calibration_frequencies, calibration_currents)

    return relative_frequencies_for_fitting, fitted_currents


def plot_change_in_odmr_freq_with_current(filename="data/CALIBRATION_current_measurement_2024-04-12_162441_.csv",
                                          sep="\t"):
    data = pd.read_csv(filename, sep=sep, header=0)

    current_label = "Current [A]"
    data.rename(columns={data.columns[0]: current_label}, inplace=True)

    columns = data.columns

    # OPX FM carrier frequency. We perform FM around 200 MHz. This is mixed with the LO frequency. We use the lower
    # sideband (f_LO - f_FM) -> Need to correct frequency by 200 MHz
    data[columns[1:]] += 200e6

    num_nv_directions = (len(columns) - 1) // 2

    fig_frequencies, ax_frequencies = plt.subplots(2, num_nv_directions)

    colors = {0: "red", 1: "blue", 2: "green", 3: "orange"}

    row_0 = ax_frequencies[0]
    row_1 = ax_frequencies[1]

    if num_nv_directions == 1:
        row_0 = [row_0]
        row_1 = [row_1]

    for i_col, ax in enumerate(row_0):
        ax.scatter(data[current_label], data[columns[1 + i_col]] / 1e6, s=5, alpha=0.4, color="black")
        ax.set_xlabel(current_label)
        ax.set_ylabel("Frequency [MHz]")
        ax.set_title("Dip " + str(i_col + 1) + "", color=colors[i_col])
        ax.ticklabel_format(useOffset=False)
        ax.grid()

    for i_col, ax in enumerate(row_1):
        ax.scatter(data[current_label], data[columns[-1 - i_col]] / 1e6, s=5, alpha=0.4, color="black")
        ax.set_xlabel(current_label)
        ax.set_ylabel("Frequency [MHz]")
        #ax.set_title("NV direction " + str(i_col + 1) + " right side", color=colors[i_col])
        ax.ticklabel_format(useOffset=False)
        ax.grid()

    fig_frequencies.tight_layout()

    plt.show()


if __name__ == "__main__":
    #plot_change_in_odmr_freq_with_current()
    fit_calibration_curve()
    plt.show()
    get_currents_from_relative_frequencies(None, None)
