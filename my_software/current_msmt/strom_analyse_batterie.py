import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw
import pandas as pd

matplotlib.use("Qt5Agg")

from strom_analyse_kalibration import fit_calibration_curve
from my_software.tools.fitting import quadratic
#from my_software.tools.plotting.plotting


def load_frequency_data(filename=None, sep=","):
    df = pd.read_csv(filename, sep=sep, header=0)
    columns = df.columns
    time_column_label = columns[0]

    # convert time (str) in first column to datetime object
    df[time_column_label] = pd.to_datetime(df[time_column_label])
    relative_time_in_seconds = [(df[time_column_label][i] - df[time_column_label][0]).total_seconds() for i in range(len(df[time_column_label]))]

    # OPX FM carrier frequency. We perform FM around 200 MHz. This is mixed with the LO frequency. We use the lower
    # sideband (f_LO - f_FM) -> Need to correct frequency by 200 MHz
    df[columns[1:]] += 200e6

    return np.array(relative_time_in_seconds), np.array(df[columns[1:]])[:, 0]

# def current_from_frequency(frequencies, id_odmr_dip=0):


def get_currents_from_frequencies(frequencies):

    calibration_frequencies, calibration_currents, p_a, p_b, p_c = fit_calibration_curve()
    y = frequencies
    fitted_currents = +np.sqrt(-4*p_a*p_c + 4*p_a*y + p_b**2)/(2 * p_a) - p_b / (2 * p_a)
    #fitted_currents = np.interp(np.array(frequencies), calibration_frequencies, calibration_currents)

    return fitted_currents


def plot_frequencies_and_current_over_time(times, frequencies, currents, sep=","):

    # PLOTS CURRENTS OVER TIME
    fig_current, ax_current = plt.subplots()
    ax_current.scatter(times, currents*1000, s=5, alpha=0.4, color="red")
    ax_current.set_xlabel("Time [s]")
    ax_current.set_ylabel("Current [mA]")
    ax_current.set_title("Current over time")
    fig_current.tight_layout()
    draw()


    # PLOTS FREQUENCIES OVER TIME
    fig_frequencies, ax_frequencies = plt.subplots()

    ax_frequencies.scatter(times, frequencies/1e6, s=5, alpha=0.4, color="black")
    ax_frequencies.set_xlabel("Time [s]")
    ax_frequencies.set_ylabel("Frequency [MHz]")
    ax_frequencies.set_title("Frequency over time")
    fig_frequencies.tight_layout()
    draw()

    plt.show()



if __name__ == "__main__":
    times, frequencies = load_frequency_data("data/battery_current_measurement_2024-04-12_180114_.csv")

    interpolated_currents = get_currents_from_frequencies(frequencies)

    plot_frequencies_and_current_over_time(times, frequencies, interpolated_currents)

    print("Not working correctly yet. Either, the current ports were swapped compared to calibration measurement.")
