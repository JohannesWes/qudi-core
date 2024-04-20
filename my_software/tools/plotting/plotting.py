import matplotlib.pyplot as plt
import numpy as np


def plot_data_in_subplots(x, y, x_label=None, y_label=None, title=None, color="black", alpha=1, s=1):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    if x.ndim == 1:
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=30, alpha=0.4, color="black")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        fig.tight_layout()
        plt.draw()
        fig.show()

    else:

        number_plots = x.shape[0]
        number_rows = 2
        number_columns = number_plots // number_rows + number_plots % number_rows

        fig, ax = plt.subplots(number_rows, number_columns)

        if number_plots == 1:
            ax = [ax]

        for i in range(number_rows):
            for j in range(number_columns):
                if i * number_columns + j >= number_plots:
                    break
                ax[i][j].scatter(x[i * number_columns + j], y[i * number_columns + j], s=s, alpha=alpha, color=color)
                ax[i][j].set_xlabel(x_label)
                ax[i][j].set_ylabel(y_label)

        fig.suptitle(title)
        fig.tight_layout()
        plt.draw()
        fig.show()

    return fig, ax
