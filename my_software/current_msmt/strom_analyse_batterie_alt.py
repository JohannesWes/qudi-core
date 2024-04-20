import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Qt5Agg")

from strom_analyse_kalibration import fit_calibration_curve
from strom_analyse_kalibration import get_currents_from_relative_frequencies
from my_software.tools.fitting import quadratic
import my_software.tools.plotting.my_styles as ms
from my_software.tools.plotting.plotting import plot_data_in_subplots


# from my_software.tools.plotting.plotting


def load_frequency_data(filename=None, sep=","):
    df = pd.read_csv(filename, sep=sep, header=0)
    columns = df.columns
    time_column_label = columns[0]

    # convert time (str) in first column to datetime object
    df[time_column_label] = pd.to_datetime(df[time_column_label])
    relative_time_in_seconds = [(df[time_column_label][i] - df[time_column_label][0]).total_seconds() for i in
                                range(len(df[time_column_label]))]

    # OPX FM carrier frequency. We perform FM around 200 MHz. This is mixed with the LO frequency. We use the lower
    # sideband (f_LO - f_FM) -> Need to correct frequency by 200 MHz
    df[columns[1:]] += 200e6
    return np.array(relative_time_in_seconds), np.array(df[columns[1:]]).T
    # return np.array(relative_time_in_seconds), np.array(df[columns[1:]])[:, 0], np.array(df[columns[1:]])[:, 1]


# def current_from_frequency(frequencies, id_odmr_dip=0):


# def get_currents_from_frequencies(frequencies):
#
#     calibration_frequencies, calibration_currents, p_a, p_b, p_c = fit_calibration_curve()
#     y = frequencies
#     fitted_currents = +np.sqrt(-4*p_a*p_c + 4*p_a*y + p_b**2)/(2 * p_a) - p_b / (2 * p_a)
#     #fitted_currents = np.interp(np.array(frequencies), calibration_frequencies, calibration_currents)
#
#     return fitted_currents


def plot_frequencies_over_time(times, frequencies, title="Frequency over time", colors=None):
    # PLOTS FREQUENCIES OVER TIME
    if colors is None:
        colors = {0: "blue", 1: "red", 2: "green", 3: "orange"}

    if frequencies.ndim == 1:
        # plot the frequency time trace in one plot

        fig_frequencies, ax_frequencies = plt.subplots()

        ax_frequencies.scatter(times, frequencies / 1e6, s=5, alpha=0.4, color="black")
        ax_frequencies.set_xlabel("Time [s]")
        ax_frequencies.set_ylabel("Frequency [MHz]")
        ax_frequencies.set_title(title)
        fig_frequencies.tight_layout()
        plt.draw()
        fig_frequencies.show()


    elif frequencies.ndim == 2:
        # plot all frequency time traces in one big plot with subplots

        number_odmrs = frequencies.shape[0] // 2

        fig_frequencies, ax_frequencies = plt.subplots(2, number_odmrs)

        # such that we can iterate over the 2nd dimension of ax_frequencies, even if the figure has only one column
        if number_odmrs == 1:
            ax_frequencies = [[ax_frequencies[k]] for k in range(2)]
            ax_frequencies[1][0].set_title("Left Side of ODMR Spectrum")
            ax_frequencies[0][0].set_title("Right Side of ODMR Spectrum")

        # plotting
        for i in range(number_odmrs):
            ax_frequencies[1][i].scatter(times, frequencies[i] / 1e6, s=30, alpha=0.4, color=colors[i])
            ax_frequencies[1][i].set_xlabel("Time [s]")
            ax_frequencies[1][i].set_ylabel("Frequency [MHz]")

            ax_frequencies[0][i].scatter(times, frequencies[-1 - i] / 1e6, s=30, alpha=0.4, color=colors[i])
            ax_frequencies[0][i].set_xlabel("Time [s]")
            ax_frequencies[0][i].set_ylabel("Frequency [MHz]")

            fig_frequencies.suptitle(title)
            fig_frequencies.tight_layout()

            plt.draw()
            fig_frequencies.show()


def plot_currents_over_time(times, currents):
    # PLOTS CURRENTS OVER TIME
    fig_current, ax_current = plt.subplots()
    ax_current.scatter(times, currents * 1000, s=5, alpha=0.4, color="red")
    ax_current.set_xlabel("Time [s]")
    ax_current.set_ylabel("Current [mA]")
    ax_current.set_title("Current over time")
    fig_current.tight_layout()
    plt.draw()
    fig_current.show()


if __name__ == "__main__":
    # times, frequencies_left_of_zfs, frequencies_right_of_zfs = load_frequency_data("data/battery_current_measurement_2024-04-17_155811_.csv")

    # # alle NV Achsen aufgelöst in dieser Messung
    # times, frequencies = load_frequency_data("data/battery_current_measurement_2024-04-19_114448_.csv")

    # alle NV Achsen aufgelöst in dieser Messung NEU
    # times, frequencies = load_frequency_data("data/battery_current_measurement_2024-04-17_183542_.csv")

    # alle NV Achsen aufgelöst in dieser Messung NEUNEU
    # times, frequencies = load_frequency_data("data/battery_current_measurement_2024-04-17_193258_.csv")

    # nur zwei achsen aufgelöst in dieser Messung
    # times, frequencies = load_frequency_data("data/battery_current_measurement_2024-04-17_155811_.csv")
    # print(frequencies)

    # case where we only look at 1 NV direction -> 2 ODMR dips
    # if len(frequencies) == 2:
    #     frequencies_left_of_zfs = frequencies[0]
    #     frequencies_right_of_zfs = frequencies[1]
    #     freq_where_current_zero_lef_zfs = 2845.14e6
    #     freq_where_current_zero_right_zfs = 2845.14e6
    #
    #     fitting_frequencies_left_of_zfs, fitting_currents_left_of_zfs = get_currents_from_relative_frequencies(
    #         frequencies_left_of_zfs, freq_where_current_zero_lef_zfs)
    #
    #     plot_frequencies_over_time(times, frequencies)
    #
    #     # plot_frequencies_over_time(times, frequencies_left_of_zfs, title_left)
    #     plot_currents_over_time(times, fitting_currents_left_of_zfs)
    #
    #     plot_frequencies_over_time(times, frequencies_right_of_zfs - frequencies_left_of_zfs,
    #                                "Frequency difference over time")
    #
    # else:
    #     index_titles = [1, 2, 3, 4, 4, 3, 2, 1]
    #     plot_frequencies_over_time(times, frequencies)
    #
    #     index_titles = [1, 2, 3, 3, 2, 1]
        # alle NV Achsen aufgelöst in dieser Messung NEU
        # times, frequencies = load_frequency_data("data/battery_current_measurement_2024-04-17_183542_.csv")
        # plot_frequencies_over_time(times, frequencies)

        # times, frequencies = load_frequency_data("data/battery_current_measurement_2024-04-17_193258_.csv")
        # plot_frequencies_over_time(times, frequencies)

        # times, frequencies = load_frequency_data("data/battery_current_measurement_2024-04-17_204720_.csv")
        # plot_frequencies_over_time(times, frequencies)

        # times, frequencies = load_frequency_data("data/battery_current_measurement_2024-04-17_210258_.csv")
        # plot_frequencies_over_time(times, frequencies)

        # times, frequencies = load_frequency_data("data/battery_current_measurement_2024-04-15_121339_.csv")
        # plot_frequencies_over_time(times, frequencies[0])

    # filename =  "battery_current_measurement_2024-04-19_115144_.csv"
    # times, frequencies = load_frequency_data("data/" + filename)
    # plot_frequencies_over_time(times, frequencies, title=filename)
    #
    # filename = "battery_current_measurement_2024-04-19_120420_.csv"
    # times, frequencies = load_frequency_data("data/" + filename)
    # plot_frequencies_over_time(times, frequencies, title=filename)
    #
    # filename = "battery_current_measurement_2024-04-19_121240_.csv"
    # times, frequencies = load_frequency_data("data/" + filename)
    # plot_frequencies_over_time(times, frequencies, title=filename)
    #
    # filename = "battery_current_measurement_2024-04-19_121604_.csv"
    # times, frequencies = load_frequency_data("data/" + filename)
    # plot_frequencies_over_time(times, frequencies, title=filename)
    #
    # filename = "battery_current_measurement_2024-04-19_122413_.csv"
    # times, frequencies = load_frequency_data("data/" + filename)
    # plot_frequencies_over_time(times, frequencies, title=filename)
    #
    # filename = "battery_current_measurement_2024-04-19_122856_.csv"
    # times, frequencies = load_frequency_data("data/" + filename)
    # plot_frequencies_over_time(times, frequencies, title=filename)

    # back to back Messungen
    filenames = ["BACK_TO_BACK_battery_current_measurement_2024-04-19_130819_.csv",
                 "BACK_TO_BACK_battery_current_measurement_2024-04-19_130956_.csv",
                 "BACK_TO_BACK_battery_current_measurement_2024-04-19_131137_.csv",
                 "BACK_TO_BACK_battery_current_measurement_2024-04-19_131319_.csv",
                 "BACK_TO_BACK_battery_current_measurement_2024-04-19_131500_.csv",
                 "BACK_TO_BACK_battery_current_measurement_2024-04-19_131641_.csv",
                 "BACK_TO_BACK_battery_current_measurement_2024-04-19_131822_.csv",
                 "BACK_TO_BACK_battery_current_measurement_2024-04-19_132003_.csv",
                 "BACK_TO_BACK_battery_current_measurement_2024-04-19_132144_.csv",
                 "BACK_TO_BACK_battery_current_measurement_2024-04-19_132325_.csv"]

    upper_side_frequency_arrays = np.array([load_frequency_data("data/" + filename)[1][1] for filename in filenames])

    upper_side_time_arrays = np.array([load_frequency_data("data/" + filename)[0] for filename in filenames])

    plot_data_in_subplots(upper_side_time_arrays, upper_side_frequency_arrays/1e6, x_label="Time [s]", y_label="Frequency [MHz]", s=15, alpha=0.4)

    # zwischen den Messungen gewartet
    filenames = ["WAITING_IN_BETWEEN_battery_current_measurement_2024-04-19_133036_.csv",
                 "WAITING_IN_BETWEEN_battery_current_measurement_2024-04-19_133246_.csv",
                 "WAITING_IN_BETWEEN_battery_current_measurement_2024-04-19_133455_.csv",
                 "WAITING_IN_BETWEEN_battery_current_measurement_2024-04-19_133706_.csv",
                 "WAITING_IN_BETWEEN_battery_current_measurement_2024-04-19_133913_.csv",
                 "WAITING_IN_BETWEEN_battery_current_measurement_2024-04-19_134120_.csv",
                 "WAITING_IN_BETWEEN_battery_current_measurement_2024-04-19_134327_.csv",
    ]

    upper_side_frequency_arrays = np.array([load_frequency_data("data/" + filename)[1][1] for filename in filenames])

    upper_side_time_arrays = np.array([load_frequency_data("data/" + filename)[0] for filename in filenames])

    plot_data_in_subplots(upper_side_time_arrays, upper_side_frequency_arrays/1e6, x_label="Time [s]", y_label="Frequency [MHz]", s=15, alpha=0.4, color="red")

    plt.show()

    print("Not working correctly yet. Either, the current ports were swapped compared to calibration measurement.")
