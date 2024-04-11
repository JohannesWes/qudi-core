
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

matplotlib.use("Qt5Agg")

def main(filename="current_measurement_2024-04-08_134356_.csv"):
    data = pd.read_csv

    data = pd.read_csv("current_measurement_2024-04-10_065800_.csv", sep="\t", header=0)
    columns = data.columns

    # OPX FM carrier frequency. We perform FM around 200 MHz. This is mixed with the LO frequency. We use the lower
    # sideband (f_LO - f_FM) -> Need to correct frequency by 200 MHz
    data[columns[1:]] += 200e6

    num_nv_directions = 4
    fig_frequencies, ax_frequencies = plt.subplots(2, num_nv_directions)

    colors = {0: "red", 1: "blue", 2: "green", 3: "orange"}

    row_0 = ax_frequencies[0]
    row_1 = ax_frequencies[1]
    for i_col, ax in enumerate(row_0):
        ax.scatter(data["current[A]"], data[columns[1+i_col]]/1e6, s=5, alpha=0.4, color="black")
        ax.set_xlabel("Current [A]")
        ax.set_ylabel("Frequency [MHz]")
        ax.set_title("NV direction " + str(i_col+1) + " left side", color=colors[i_col])
        ax.ticklabel_format(useOffset=False)
        ax.grid()
    for i_col, ax in enumerate(row_1):
        ax.scatter(data["current[A]"], data[columns[-1-i_col]]/1e6, s=5, alpha=0.4, color="black")
        ax.set_xlabel("Current [A]")
        ax.set_ylabel("Frequency [MHz]")
        ax.set_title("NV direction " + str(i_col+1) + " right side", color=colors[i_col])
        ax.ticklabel_format(useOffset=False)
        ax.grid()

    fig_frequencies.tight_layout()

    plt.show()
    #plt.show()

    # for column in data.columns[1:]:
    #     ax.scatter(data["current[A]"], data[column], label=column)


if __name__ == "__main__":
    main()