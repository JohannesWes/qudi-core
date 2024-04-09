
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Qt5Agg")

def main(filename="current_measurement_2024-04-08_134356_.csv"):
    data = pd.read_csv

    data = pd.read_csv("current_measurement_2024-04-08_134356_.csv", sep="\t", header=0)
    columns = data.columns

    # OPX FM carrier frequency. We perform FM around 200 MHz. This is mixed with the LO frequency. We use the lower
    # sideband (f_LO - f_FM) -> Need to correct frequency by 200 MHz
    data[columns[1:]] += 200e6

    fig, ax = plt.subplots()

    num_nv_directions = 4

    fig_frequencies, ax_frequencies = plt.subplots(2, num_nv_directions)
    for i_row, row in enumerate(ax_frequencies):
        for i_col, ax in enumerate(row):
            ax.plot(data["current[A]"], data[columns[i_row*num_nv_directions + i_col + 1]])
            ax.set_title(columns[i_row*num_nv_directions + i_col + 1])
            ax.set_xlabel("Current [A]")
            ax.set_ylabel("Frequency [Hz]")
    fig_frequencies.tight_layout()

    for column in data.columns[1:]:
        ax.scatter(data["current[A]"], data[column], label=column)
    plt.show()





if __name__ == "__main__":
    main()