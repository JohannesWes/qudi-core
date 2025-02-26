import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pandas as pd


def parabola(x, x0, a, c):
    return a * (x - x0) ** 2 + c


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def evaluate_hyperfine(frequency_array, voltage_array, feature_distance_in_Hz=0.5e6, feature_prominence=0.02,
                       zero_crossings_fit_range=0.1e6):
    """
    This function finds the peaks and dips in the ODMR signal and fits straight lines around the zero crossings.
    It is ment for high-quality hyperfine ODMR spectra, where the peaks/dips can be identified by the maximum/minimum
    values instead of having to perform fits.

    Args:
        frequency_array (ndarray): Array of frequencies
        voltage_array (ndarray): Array of voltages
        feature_distance_in_Hz (float): The minimum distance between separate peaks/dips in Hz
        feature_prominence (float): The absolute of the minimum amplitude of a peak/dip
        zero_crossings_fit_range (float): The range in Hz around the zero crossings to fit straight lines

    Returns:
        peaks_indices (ndarray): Array indices of the peaks with respect to the frequency array
        dips_indices (ndarray): Array indices of the dips with respect to the frequency array
        zero_crossings_indices (ndarray): Array indices of the zero crossings with respect to the frequency array
        slopes (ndarray): Array of the slopes of the fitted lines
        intercepts (ndarray): Array of the intercepts of the fitted lines

    """
    # Check if input arrays are np.ndarray or pandas series
    if not isinstance(frequency_array, np.ndarray) and isinstance(frequency_array, pd.core.series.Series):
        frequency_array = np.array(frequency_array)
    if not isinstance(voltage_array, np.ndarray) and isinstance(voltage_array, pd.core.series.Series):
        voltage_array = np.array(voltage_array)

    frequency_spacing = (frequency_array[-1] - frequency_array[0]) / len(frequency_array)
    min_feature_distance_in_samples = feature_distance_in_Hz / frequency_spacing

    peaks_indices, _ = find_peaks(voltage_array, height=0.02, distance=min_feature_distance_in_samples,
                                  prominence=feature_prominence)
    dips_indices, _ = find_peaks(-voltage_array, height=0.02, distance=min_feature_distance_in_samples,
                                 prominence=feature_prominence)

    zero_crossings_indices = np.array(dips_indices + (peaks_indices - dips_indices) / 2).astype(int)

    # Initialize lists to store the slope and intercept of the fitted lines
    slopes = []
    intercepts = []

    # Fit straight lines around the zero crossings
    for zero_crossing in zero_crossings_indices:
        # Define the range for fitting
        fit_start = zero_crossing - int(zero_crossings_fit_range / frequency_spacing)
        fit_end = zero_crossing + int(zero_crossings_fit_range / frequency_spacing)

        # Get the frequencies and voltages around the zero crossing
        frequencies_around_zero = frequency_array[fit_start:fit_end]
        voltages_around_zero = voltage_array[fit_start:fit_end]

        # Fit a straight line to the data
        slope, intercept = np.polyfit(frequencies_around_zero, voltages_around_zero, 1)

        # Store the slope and intercept
        slopes.append(slope)
        intercepts.append(intercept)

    linewidths = np.sqrt(3) * (frequency_array[peaks_indices] - frequency_array[dips_indices])

    print(frequency_array[peaks_indices], frequency_array[dips_indices])

    return peaks_indices, dips_indices, zero_crossings_indices, slopes, intercepts, linewidths


def fit_hyperfine(frequency_array, voltage_array, feature_distance_in_Hz=0.5e6, feature_prominence=0.02,
                  min_feature_height=0.02, n_most_prominent_peaks=None, use_peak_finding_if_fitting_fails=True,
                  zero_crossings_fit_range=0.1e6,
                  feature_fit_range=0.2e6, plot_all=False, plot_result=False, save_result_plot=False, filename=None):
    """
    This function finds the peaks and dips in the ODMR signal and fits straight lines around the zero crossings.
    It is ment for high-quality hyperfine ODMR spectra, without much noise. Compared to evaluate_hyperfine, this function fits parabolas to the peaks
    and dips to more accurately determine their positions.

    Args:
        frequency_array (ndarray): Array of frequencies
        voltage_array (ndarray): Array of voltages
        feature_distance_in_Hz (float): The minimum distance between separate peaks/dips in Hz
        feature_prominence (float): The absolute of the minimum amplitude of a peak/dip
        min_feature_height (float): The minimum height of a peak/dip
        n_most_prominent_peaks: How many peaks should be found, only considers the n most prominent peaks
        zero_crossings_fit_range (float): The range in Hz around the zero crossings to fit straight lines
        feature_fit_range:
        plot_all:

    Returns:
        peaks_indices (ndarray): Array indices of the peaks with respect to the frequency array
        dips_indices (ndarray): Array indices of the dips with respect to the frequency array
        zero_crossings_indices (ndarray): Array indices of the zero crossings with respect to the frequency array
        slopes (ndarray): Array of the slopes of the fitted lines
        intercepts (ndarray): Array of the intercepts of the fitted lines


    """
    plot_fitting_initial = plot_all

    # Check if input arrays are np.ndarray or pandas series
    if not isinstance(frequency_array, np.ndarray) and isinstance(frequency_array, pd.core.series.Series):
        frequency_array = np.array(frequency_array)
    if not isinstance(voltage_array, np.ndarray) and isinstance(voltage_array, pd.core.series.Series):
        voltage_array = np.array(voltage_array)

    # -----------------------------------
    # DETERMINING ROUGH FEATURE POSITIONS
    # -----------------------------------
    frequency_spacing = (frequency_array[-1] - frequency_array[0]) / len(frequency_array)
    min_feature_distance_in_samples = feature_distance_in_Hz / frequency_spacing

    # smooth data before peak finding, s.t. strong outliers do not disturb the peak finding
    n_samples_smooth = int(0.25e6 / frequency_spacing)
    smoothed_voltage_array = smooth(voltage_array, n_samples_smooth)

    peaks_indices, peaks_properties = find_peaks(smoothed_voltage_array, height=min_feature_height,
                                                 distance=int(min_feature_distance_in_samples),
                                                 prominence=feature_prominence)
    dips_indices, dips_properties = find_peaks(-smoothed_voltage_array, height=min_feature_height,
                                               distance=int(min_feature_distance_in_samples),
                                               prominence=feature_prominence)

    # Get the indices of the n most prominent peaks
    if n_most_prominent_peaks is not None:
        peaks_indices = peaks_indices[np.argsort(peaks_properties["prominences"])[-n_most_prominent_peaks:]]



        dips_indices = dips_indices[np.argsort(dips_properties["prominences"])[-n_most_prominent_peaks:]]

        # Sort the peaks and dips by frequency
        peaks_indices = np.sort(peaks_indices)
        dips_indices = np.sort(dips_indices)

    # -------------------------
    # FITTING PARABOLAS TO DIPS
    # -------------------------

    fit_peak_positions, fit_peak_uncertainties = np.empty(n_most_prominent_peaks), np.empty(n_most_prominent_peaks)
    fit_dip_positions, fit_dip_uncertainties = np.empty(n_most_prominent_peaks), np.empty(n_most_prominent_peaks)

    fit_sample_range = int(feature_fit_range / frequency_spacing)
    for loop_index, peak_index, dip_index in zip(np.arange(n_most_prominent_peaks), peaks_indices, dips_indices):
        try:
            frequencies_around_peak = frequency_array[peak_index - fit_sample_range:peak_index + fit_sample_range]
            voltages_around_peak = voltage_array[peak_index - fit_sample_range:peak_index + fit_sample_range]
            smoothed_voltages_around_peak = smoothed_voltage_array[
                                            peak_index - fit_sample_range:peak_index + fit_sample_range]
            frequencies_around_dip = frequency_array[dip_index - fit_sample_range:dip_index + fit_sample_range]
            voltages_around_dip = voltage_array[dip_index - fit_sample_range:dip_index + fit_sample_range]
            smoothed_voltages_around_dip = smoothed_voltage_array[
                                           dip_index - fit_sample_range:dip_index + fit_sample_range]

            # estimate curvature of parabola for fitting
            estimated_curvature = np.sqrt(np.abs((voltages_around_peak[0] - voltages_around_peak[-1]) / (
                    frequencies_around_peak[0] - frequencies_around_peak[-1])))

            # fit parabola to peaks
            peak_params, peaks_cov = curve_fit(parabola, frequencies_around_peak, voltages_around_peak,  # noqa
                                               p0=[frequency_array[peak_index], -estimated_curvature,
                                                   voltage_array[peak_index]])

            # fit parabola to dips
            dip_params, dips_cov = curve_fit(parabola, frequencies_around_dip, voltages_around_dip,  # noqa
                                             p0=[frequency_array[dip_index], +estimated_curvature,
                                                 voltage_array[dip_index]])

            # Get the estimated positions of peaks and dips
            peak_position = peak_params[0]
            peak_uncertainty = np.sqrt(np.diag(peaks_cov))[0]
            dip_position = dip_params[0]
            dip_uncertainty = np.sqrt(np.diag(dips_cov))[0]

            if abs(peak_position - frequency_array[peak_index]) > 0.5e6:
                print(
                    "Warning: Peak position is more than 0.5 MHz away from most prominent data point at estimated peak.")
            if abs(dip_position - frequency_array[dip_index]) > 0.5e6:
                print(
                    "Warning: Dip position is more than 0.5 MHz away from most prominent data point at estimated dip.")

        except Exception as fitting_exception:
            print(str(fitting_exception))
            print("Fitting failed. Peaks/Dips were detected at" + str(frequency_array[peak_index]),
                  str(frequency_array[dip_index]))

            if use_peak_finding_if_fitting_fails:
                peak_position = frequency_array[peaks_indices[0]]
                dip_position = frequency_array[dips_indices[0]]
                peak_uncertainty, dip_uncertainty = np.nan, np.nan
            else:
                peak_position, dip_position = np.nan, np.nan
                peak_uncertainty, dip_uncertainty = np.nan, np.nan

            if plot_all:
                # plot data for which fitting failed
                fig_fail, ax_fail = plt.subplots()
                ax_fail.scatter(frequencies_around_peak, voltages_around_peak, color='blue', alpha=0.3)
                ax_fail.scatter(frequencies_around_dip, voltages_around_dip, color='blue', alpha=0.3, label='data')
                ax_fail.legend()
                fig_fail.show()
                # plt.show()

        try:
            if plot_all:
                fig_parabola, ax_parabola = plt.subplots()
                ax_parabola.scatter(frequencies_around_peak / 1e6, voltages_around_peak, color='blue', alpha=0.3)
                ax_parabola.scatter(frequencies_around_dip / 1e6, voltages_around_dip, color='blue', alpha=0.3,
                                    label='data')
                ax_parabola.plot(frequencies_around_dip / 1e6, smoothed_voltages_around_dip, zorder=3, color="yellow",
                                 linewidth=3)
                ax_parabola.plot(frequencies_around_peak / 1e6, smoothed_voltages_around_peak, zorder=3, color="yellow",
                                 label="smoothed data", linewidth=3)
                ax_parabola.plot(frequencies_around_peak / 1e6, parabola(frequencies_around_peak, *peak_params),
                                 color='red',
                                 zorder=2, linewidth=2)
                ax_parabola.plot(frequencies_around_dip / 1e6, parabola(frequencies_around_dip, *dip_params),
                                 color='red',
                                 label="parabolic fit", zorder=2, linewidth=2)
                ax_parabola.scatter(peak_position / 1e6, parabola(peak_position, *peak_params), color='green', zorder=3)
                ax_parabola.scatter(dip_position / 1e6, parabola(dip_position, *dip_params), color='green', zorder=3)
                ax_parabola.set_title("Fitting of Hyperfine Peaks and Dips")
                ax_parabola.set_xlabel("Frequency [MHz]")
                ax_parabola.set_ylabel("Voltage [V]")
                ax_parabola.legend()
                fig_parabola.show()
                # plt.show()
            else:
                pass

        except Exception as PlottingException:
            print("EXCEPTION: Plotting failed for peak at index", peak_index, "and dip at index", dip_index)
            print(str(PlottingException))

        fit_peak_positions[loop_index] = peak_position
        fit_dip_positions[loop_index] = dip_position
        fit_peak_uncertainties[loop_index] = peak_uncertainty
        fit_dip_uncertainties[loop_index] = dip_uncertainty


    # convert lists to numpy arrays
    fit_peak_positions, fit_peak_uncertainties = np.array(fit_peak_positions), np.array(fit_peak_uncertainties)
    fit_dip_positions, fit_dip_uncertainties = np.array(fit_dip_positions), np.array(fit_dip_uncertainties)

    try:
        linewidths = np.sqrt(3) * (frequency_array[peaks_indices] - frequency_array[dips_indices])
    except Exception as LineWidthException:
        print(str(LineWidthException))
        linewidths = np.nan

    # result plotting
    # Plot the smoothed curve with the peaks and dips for TESTING PURPOSES
    if plot_result or save_result_plot:
        fig_result, ax_result = plt.subplots()
        ax_result.plot(frequency_array/1e6, smoothed_voltage_array, label="smoothed", zorder=2, alpha=0.75)
        ax_result.plot(frequency_array/1e6, voltage_array, label="original", zorder=1, alpha=0.75, color="orange")
        ax_result.scatter(frequency_array[peaks_indices]/1e6, smoothed_voltage_array[peaks_indices], color='red',
                          label="Peaks",
                          zorder=3)
        ax_result.scatter(frequency_array[dips_indices]/1e6, smoothed_voltage_array[dips_indices], color='green',
                          label="Dips",
                          zorder=3)
        ax_result.set_xlabel("Frequency [MHz]")
        ax_result.set_ylabel("Voltage [V]")

    # -----------------------------------
    # DETERMINE ZERO CROSSINGS AND SLOPES
    # -----------------------------------
    zero_crossings_frequencies = np.array(fit_dip_positions / 2 + fit_peak_positions / 2)

    # Initialize lists to store the slope and intercept of the fitted lines
    slopes = np.empty(n_most_prominent_peaks)
    intercepts = np.empty(n_most_prominent_peaks)

    # Fit straight lines around the zero crossings
    for zcs_index, zero_crossing_f in enumerate(zero_crossings_frequencies):
        # Define the range for fitting
        f_fit_start, f_fit_end = zero_crossing_f - zero_crossings_fit_range, zero_crossing_f + zero_crossings_fit_range
        index_fit_start, index_fit_end = np.argmin(np.abs(frequency_array - f_fit_start)), np.argmin(
            np.abs(frequency_array - f_fit_end))

        # Get the frequencies and voltages around the zero crossing
        frequencies_around_zero = frequency_array[index_fit_start:index_fit_end]
        voltages_around_zero = voltage_array[index_fit_start:index_fit_end]

        # Fit a straight line to the data
        slope, intercept = np.polyfit(frequencies_around_zero, voltages_around_zero, 1)

        # Store the slope and intercept
        slopes[zcs_index] = slope
        intercepts[zcs_index] = intercept

        # plot fitted lines
        if plot_result or save_result_plot:
            ax_result.plot(frequencies_around_zero/1e6, slope * frequencies_around_zero + intercept, color="purple",
                           linestyle='-', linewidth=2, zorder=3)

    if plot_result or save_result_plot:
        ax_result.legend()
        # subtitle: linewidths of the three peaks rounded to .1 MHz
        ax_result.set_title('Linewidths [MHz]: ' + ', '.join([f'{round(linewidth/1e6, 1)}' for linewidth in linewidths]))

        fig_result.tight_layout()
        if save_result_plot and filename is not None:
            fig_result.savefig(filename + "_hyperfine_fitting.pdf")

    if plot_result:
        fig_result.show()
        plt.show()

    if plot_result or save_result_plot:
        plt.close(fig_result)



    return {
        "peak_positions [Hz]": fit_peak_positions,
        "peak_uncertainties [Hz]": fit_peak_uncertainties,
        "dip_positions [Hz]": fit_dip_positions,
        "dip_uncertainties [Hz]": fit_dip_uncertainties,
        "linewidths [Hz]": linewidths,
        "zero_crossing_frequencies [Hz]": zero_crossings_frequencies,
        "zero_crossing_slopes [V/Hz]": slopes
    }


def fit_hyperfine_OLD(frequency_array, voltage_array, feature_distance_in_Hz=0.5e6, min_feature_amplitude=0.02,
                      min_feature_height=0.02, zero_crossings_fit_range=0.1e6, feature_fit_range=0.2e6,
                      plot_fitting=False):
    """
    This function finds the peaks and dips in the ODMR signal and fits straight lines around the zero crossings.
    It is ment for high-quality hyperfine ODMR spectra, where the peaks/dips can be identified by the maximum/minimum
    values instead of having to perform fits. Compared to evaluate_hyperfine, this function fits parabolas to the peaks
    and dips to more accurately determine their positions.

    Args:
        frequency_array (ndarray): Array of frequencies
        voltage_array (ndarray): Array of voltages
        feature_distance_in_Hz (float): The minimum distance between separate peaks/dips in Hz
        min_feature_amplitude (float): The absolute of the minimum amplitude of a peak/dip
        min_feature_height:
        zero_crossings_fit_range (float): The range in Hz around the zero crossings to fit straight lines
        feature_fit_range:
        plot_fitting:

    Returns:
        peaks_indices (ndarray): Array indices of the peaks with respect to the frequency array
        dips_indices (ndarray): Array indices of the dips with respect to the frequency array
        zero_crossings_indices (ndarray): Array indices of the zero crossings with respect to the frequency array
        slopes (ndarray): Array of the slopes of the fitted lines
        intercepts (ndarray): Array of the intercepts of the fitted lines


    """
    plot_fitting_initial = plot_fitting

    # Check if input arrays are np.ndarray or pandas series
    if not isinstance(frequency_array, np.ndarray) and isinstance(frequency_array, pd.core.series.Series):
        frequency_array = np.array(frequency_array)
    if not isinstance(voltage_array, np.ndarray) and isinstance(voltage_array, pd.core.series.Series):
        voltage_array = np.array(voltage_array)

    # -----------------------------------
    # DETERMINING ROUGH FEATURE POSITIONS
    # -----------------------------------

    frequency_spacing = (frequency_array[-1] - frequency_array[0]) / len(frequency_array)
    min_feature_distance_in_samples = feature_distance_in_Hz / frequency_spacing

    # smooth data before peak finding, s.t. strong outliers do not disturb the peak finding
    smoothed_voltage_array = smooth(voltage_array, 20)

    peaks_indices, _ = find_peaks(smoothed_voltage_array, height=min_feature_height,
                                  distance=min_feature_distance_in_samples,
                                  prominence=min_feature_amplitude, width=3)
    dips_indices, _ = find_peaks(-smoothed_voltage_array, height=min_feature_height,
                                 distance=min_feature_distance_in_samples,
                                 prominence=min_feature_amplitude, width=3)

    # Plot the smoothed curve with the peaks and dips for TESTING PURPOSES
    if plot_fitting:
        fig, ax = plt.subplots()
        ax.plot(frequency_array, smoothed_voltage_array, label="smoothed", zorder=2, alpha=0.75)
        ax.plot(frequency_array, voltage_array, label="original", zorder=1, alpha=0.75, color="orange")
        ax.scatter(frequency_array[peaks_indices], smoothed_voltage_array[peaks_indices], color='red',
                   label="peaks smoothed",
                   zorder=3)
        ax.scatter(frequency_array[dips_indices], smoothed_voltage_array[dips_indices], color='green',
                   label="dips smoothed",
                   zorder=3)
        plt.legend()
        fig.suptitle('smoothed curve is used for peak finding\n Original curve is used for fitting')
        fig.tight_layout()
        fig.show()

    # -----------------------------------
    # DETERMINE ZERO CROSSINGS AND SLOPES
    # -----------------------------------
    # zero_crossings_indices = np.array(dips_indices + (peaks_indices - dips_indices) / 2).astype(int)
    #
    # # Initialize lists to store the slope and intercept of the fitted lines
    # slopes = []
    # intercepts = []
    #
    # # Fit straight lines around the zero crossings
    # for zero_crossing in zero_crossings_indices:
    #     # Define the range for fitting
    #     fit_start = zero_crossing - int(zero_crossings_fit_range / frequency_spacing)
    #     fit_end = zero_crossing + int(zero_crossings_fit_range / frequency_spacing)
    #
    #     # Get the frequencies and voltages around the zero crossing
    #     frequencies_around_zero = frequency_array[fit_start:fit_end]
    #     voltages_around_zero = voltage_array[fit_start:fit_end]
    #
    #     # Fit a straight line to the data
    #     slope, intercept = np.polyfit(frequencies_around_zero, voltages_around_zero, 1)
    #
    #     # Store the slope and intercept
    #     slopes.append(slope)
    #     intercepts.append(intercept)

    # -------------------------
    # FITTING PARABOLAS TO DIPS
    # -------------------------

    fit_peak_positions = []
    fit_peak_uncertainties = []
    fit_dip_positions = []
    fit_dip_uncertainties = []

    fit_sample_range = int(feature_fit_range / frequency_spacing)
    for peak_index, dip_index in zip(peaks_indices, dips_indices):

        try:
            frequencies_around_peak = frequency_array[peak_index - fit_sample_range:peak_index + fit_sample_range]
            voltages_around_peak = voltage_array[peak_index - fit_sample_range:peak_index + fit_sample_range]
            frequencies_around_dip = frequency_array[dip_index - fit_sample_range:dip_index + fit_sample_range]
            voltages_around_dip = voltage_array[dip_index - fit_sample_range:dip_index + fit_sample_range]

            # estimate curvature of parabola for fitting
            estimated_curvature = np.sqrt(np.abs((voltages_around_peak[0] - voltages_around_peak[-1]) / (
                    frequencies_around_peak[0] - frequencies_around_peak[-1])))

            # fit parabola to peaks
            peak_params, peaks_cov = curve_fit(parabola, frequencies_around_peak, voltages_around_peak,  # noqa
                                               p0=[frequency_array[peak_index], -estimated_curvature,
                                                   voltage_array[peak_index]])

            # fit parabola to dips
            dip_params, dips_cov = curve_fit(parabola, frequencies_around_dip, voltages_around_dip,  # noqa
                                             p0=[frequency_array[dip_index], +estimated_curvature,
                                                 voltage_array[dip_index]])

            # Get the estimated positions of peaks and dips
            peak_position = peak_params[0]
            peak_uncertainty = np.sqrt(np.diag(peaks_cov))[0]
            dip_position = dip_params[0]
            dip_uncertainty = np.sqrt(np.diag(dips_cov))[0]

            if abs(peak_position - frequency_array[peak_index]) > 0.5e6:
                print(
                    "Warning: Peak position is more than 0.5 MHz away from most prominent data point at estimated peak.")
            if abs(dip_position - frequency_array[dip_index]) > 0.5e6:
                print(
                    "Warning: Dip position is more than 0.5 MHz away from most prominent data point at estimated dip.")

        except Exception as fitting_exception:
            print(str(fitting_exception))
            print("Fitting failed. Peaks/Dips were detected at" + str(frequency_array[peak_index]),
                  str(frequency_array[dip_index]))
            peak_position, dip_position = np.nan, np.nan
            peak_uncertainty, dip_uncertainty = np.nan, np.nan

            # plot data for which fitting failed
            fig, ax = plt.subplots()
            ax.scatter(frequencies_around_peak, voltages_around_peak, color='blue', alpha=0.3)
            ax.scatter(frequencies_around_dip, voltages_around_dip, color='blue', alpha=0.3, label='data')
            ax.legend()
            fig.show()
            plt.show()

        try:
            if plot_fitting:
                fig, ax = plt.subplots()
                ax.scatter(frequencies_around_peak / 1e6, voltages_around_peak, color='blue', alpha=0.3)
                ax.scatter(frequencies_around_dip / 1e6, voltages_around_dip, color='blue', alpha=0.3, label='data')
                ax.plot(frequencies_around_dip / 1e6, smooth(voltages_around_dip, 20), zorder=3, color="yellow",
                        linewidth=3)
                ax.plot(frequencies_around_peak / 1e6, smooth(voltages_around_peak, 20), zorder=3, color="yellow",
                        label="smoothed data", linewidth=3)
                ax.plot(frequencies_around_peak / 1e6, parabola(frequencies_around_peak, *peak_params), color='red',
                        zorder=2, linewidth=2)
                ax.plot(frequencies_around_dip / 1e6, parabola(frequencies_around_dip, *dip_params), color='red',
                        label="parabolic fit", zorder=2, linewidth=2)
                ax.scatter(peak_position / 1e6, parabola(peak_position, *peak_params), color='green', zorder=3)
                ax.scatter(dip_position / 1e6, parabola(dip_position, *dip_params), color='green', zorder=3)
                ax.set_title("Fitting of Hyperfine Peaks and Dips")
                ax.set_xlabel("Frequency [MHz]")
                ax.set_ylabel("Voltage [V]")
                ax.legend()
                fig.show()
                plt.show()

        except Exception as PlottingException:
            print("EXCEPTIONfailed for peak at index", peak_index, "and dip at index", dip_index)
            print(str(PlottingException))

        fit_peak_positions.append(peak_position)
        fit_dip_positions.append(dip_position)
        fit_peak_uncertainties.append(peak_uncertainty)
        fit_dip_uncertainties.append(dip_uncertainty)

    return [
        [(fit_peak_positions[i], fit_peak_uncertainties[i]) for i in range(len(fit_dip_positions))],
        [(fit_dip_positions[i], fit_dip_uncertainties[i]) for i in range(len(fit_dip_positions))]
    ]
    # return peaks_indices, dips_indices, zero_crossings_indices, slopes, intercepts
