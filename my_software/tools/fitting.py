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


def evaluate_hyperfine(frequency_array, voltage_array, feature_distance_in_Hz=0.5e6, min_feature_amplitude=0.02,
                       zero_crossings_fit_range=0.1e6):
    """
    This function finds the peaks and dips in the ODMR signal and fits straight lines around the zero crossings.
    It is ment for high-quality hyperfine ODMR spectra, where the peaks/dips can be identified by the maximum/minimum
    values instead of having to perform fits.

    Args:
        frequency_array (ndarray): Array of frequencies
        voltage_array (ndarray): Array of voltages
        feature_distance_in_Hz (float): The minimum distance between separate peaks/dips in Hz
        min_feature_amplitude (float): The absolute of the minimum amplitude of a peak/dip
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

    Hz_per_sample = (frequency_array[-1] - frequency_array[0]) / len(frequency_array)
    min_feature_distance_in_samples = feature_distance_in_Hz / Hz_per_sample

    peaks_indices, _ = find_peaks(voltage_array, height=0.02, distance=min_feature_distance_in_samples,
                                  prominence=min_feature_amplitude)
    dips_indices, _ = find_peaks(-voltage_array, height=0.02, distance=min_feature_distance_in_samples,
                                 prominence=min_feature_amplitude)

    zero_crossings_indices = np.array(dips_indices + (peaks_indices - dips_indices) / 2).astype(int)

    # Initialize lists to store the slope and intercept of the fitted lines
    slopes = []
    intercepts = []

    # Fit straight lines around the zero crossings
    for zero_crossing in zero_crossings_indices:
        # Define the range for fitting
        fit_start = zero_crossing - int(zero_crossings_fit_range / Hz_per_sample)
        fit_end = zero_crossing + int(zero_crossings_fit_range / Hz_per_sample)

        # Get the frequencies and voltages around the zero crossing
        frequencies_around_zero = frequency_array[fit_start:fit_end]
        voltages_around_zero = voltage_array[fit_start:fit_end]

        # Fit a straight line to the data
        slope, intercept = np.polyfit(frequencies_around_zero, voltages_around_zero, 1)

        # Store the slope and intercept
        slopes.append(slope)
        intercepts.append(intercept)

    return peaks_indices, dips_indices, zero_crossings_indices, slopes, intercepts


def fit_hyperfine(frequency_array, voltage_array, feature_distance_in_Hz=0.5e6, min_feature_amplitude=0.02,
                  min_feature_height=0.02, zero_crossings_fit_range=0.1e6, feature_fit_range=0.2e6, plot_fitting=False):
    """
    This function finds the peaks and dips in the ODMR signal and fits straight lines around the zero crossings.
    It is ment for high-quality hyperfine ODMR spectra, where the peaks/dips can be identified by the maximum/minimum
    values instead of having to perform fits.

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
    plot_fitting_initial=plot_fitting

    # Check if input arrays are np.ndarray or pandas series
    if not isinstance(frequency_array, np.ndarray) and isinstance(frequency_array, pd.core.series.Series):
        frequency_array = np.array(frequency_array)
    if not isinstance(voltage_array, np.ndarray) and isinstance(voltage_array, pd.core.series.Series):
        voltage_array = np.array(voltage_array)

    # -----------------------------------
    # DETERMINING ROUGH FEATURE POSITIONS
    # -----------------------------------

    Hz_per_sample = (frequency_array[-1] - frequency_array[0]) / len(frequency_array)
    min_feature_distance_in_samples = feature_distance_in_Hz / Hz_per_sample

    # smooth data before peak finding, s.t. strong outliers do not disturb the peak finding
    smoothed_voltage_array = smooth(voltage_array, 20)

    peaks_indices, _ = find_peaks(smoothed_voltage_array, height=min_feature_height,
                                  distance=min_feature_distance_in_samples,
                                  prominence=min_feature_amplitude)
    dips_indices, _ = find_peaks(-smoothed_voltage_array, height=min_feature_height,
                                 distance=min_feature_distance_in_samples,
                                 prominence=min_feature_amplitude)

    # Plot the smoothed curve with the peaks and dips for TESTING PURPOSES
    if plot_fitting:
        fig, ax = plt.subplots()
        ax.plot(frequency_array, smoothed_voltage_array, label="smoothed", zorder=2, alpha=0.75)
        ax.plot(frequency_array, voltage_array, label="original", zorder=1, alpha=0.75)
        ax.scatter(frequency_array[peaks_indices], smoothed_voltage_array[peaks_indices], color='red', label="peaks smoothed",
                   zorder=3)
        ax.scatter(frequency_array[dips_indices], smoothed_voltage_array[dips_indices], color='green', label="dips smoothed",
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
    #     fit_start = zero_crossing - int(zero_crossings_fit_range / Hz_per_sample)
    #     fit_end = zero_crossing + int(zero_crossings_fit_range / Hz_per_sample)
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

    fit_sample_range = int(feature_fit_range / Hz_per_sample)
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
                print("Warning: Peak position is more than 0.5 MHz away from most prominent data point at estimated peak.")
            if abs(dip_position - frequency_array[dip_index]) > 0.5e6:
                print("Warning: Dip position is more than 0.5 MHz away from most prominent data point at estimated dip.")

        except Exception as fitting_exception:
            print(str(fitting_exception))
            print("Fitting failed. Peaks/Dips were detected at" + str(frequency_array[peak_index]), str(frequency_array[dip_index]))
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
                ax.scatter(frequencies_around_peak/1e6, voltages_around_peak, color='blue', alpha=0.3)
                ax.scatter(frequencies_around_dip/1e6, voltages_around_dip, color='blue', alpha=0.3, label='data')
                ax.plot(frequencies_around_dip/1e6, smooth(voltages_around_dip, 20), zorder=3, color="yellow", linewidth=3)
                ax.plot(frequencies_around_peak/1e6, smooth(voltages_around_peak, 20), zorder=3, color="yellow", label="smoothed data", linewidth=3)
                ax.plot(frequencies_around_peak/1e6, parabola(frequencies_around_peak, *peak_params), color='red', zorder=2, linewidth=2)
                ax.plot(frequencies_around_dip/1e6, parabola(frequencies_around_dip, *dip_params), color='red', label="parabolic fit", zorder=2, linewidth=2)
                ax.scatter(peak_position/1e6, parabola(peak_position, *peak_params), color='green', zorder=3)
                ax.scatter(dip_position/1e6, parabola(dip_position, *dip_params), color='green', zorder=3)
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
