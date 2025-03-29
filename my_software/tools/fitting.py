import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import find_peaks
import pandas as pd
import warnings  # Import warnings module
from typing import Union, Optional, Dict, Tuple, List  # For type hinting


# --- Helper Functions ---

def parabola(x: np.ndarray, x0: float, a: float, c: float) -> np.ndarray:
    """
    Parabola function for fitting.
    f(x) = a * (x - x0)^2 + c

    Args:
        x (np.ndarray): Input x values (e.g., frequencies).
        x0 (float): Position of the extremum (peak/dip center).
        a (float): Curvature parameter. Negative for peaks, positive for dips.
        c (float): Vertical offset (value at the extremum).

    Returns:
        np.ndarray: Calculated y values.
    """
    return a * (x - x0) ** 2 + c


def smooth(y: np.ndarray, box_pts: int) -> np.ndarray:
    """
    Smooth data using a moving average (boxcar filter).

    Args:
        y (np.ndarray): Input array to smooth.
        box_pts (int): Number of points in the moving average window.

    Returns:
        np.ndarray: Smoothed array with the same shape as y (using 'same' mode).
                    Note: 'same' mode introduces edge effects.
    """
    if box_pts <= 1:  # No smoothing needed if window is 1 or less
        return y
    if box_pts > len(y):  # Avoid window larger than data
        warnings.warn(f"Smoothing window ({box_pts}) is larger than data size ({len(y)}). No smoothing applied.",
                      UserWarning)
        return y

    box = np.ones(box_pts) / box_pts
    # Use 'reflect' mode for boundary handling to reduce edge artifacts compared to zero-padding in 'same'
    y_smooth = np.convolve(y, box, mode='same')
    # Correct edge effects by recalculating boundaries more carefully
    # (This is a simple approach; more sophisticated methods exist)
    half_box = box_pts // 2
    y_smooth[:half_box] = np.convolve(y[:box_pts - 1], box, mode='valid')[:half_box]
    y_smooth[-half_box:] = np.convolve(y[-(box_pts - 1):], box, mode='valid')[-half_box:]

    # Alternative (simpler, uses default edge handling of 'same'):
    # y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# --- Main Fitting Function ---

def fit_hyperfine(
        frequency_array:        Union[np.ndarray, pd.Series],
        voltage_array:          Union[np.ndarray, pd.Series],
        feature_distance_in_Hz: float = 0.5e6,
        feature_prominence:     float = 0.02,
        min_feature_height:     float = 0.02,
        n_most_prominent_peaks: Optional[int] = 3,  # Often expect 3 hyperfine lines
        use_peak_finding_if_fitting_fails: bool = True,
        zero_crossings_fit_range_hz: float = 0.1e6,
        feature_fit_range_hz:   float = 0.2e6,
        smooth_window_hz:       float = 0.25e6,
        plot_all:               bool = False,
        plot_result:            bool = False,
        save_result_plot:       bool = False,
        filename:               Optional[str] = None
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Finds and fits peaks/dips in ODMR hyperfine spectra.

    Finds features using scipy.signal.find_peaks on smoothed data, then refines
    their positions by fitting parabolas to the original data around each feature.
    Also calculates zero-crossing frequencies and the slope of the signal at
    these crossings by fitting linear functions.

    Args:
        frequency_array (Union[np.ndarray, pd.Series]): Array of frequencies (Hz).
        voltage_array (Union[np.ndarray, pd.Series]): Array of corresponding voltages (V).
        feature_distance_in_Hz (float): Minimum horizontal distance (Hz) between features.
        feature_prominence (float): Required prominence of features.
        min_feature_height (float): Minimum height (absolute value for dips) of features.
        n_most_prominent_peaks (Optional[int]): Target number of peak/dip pairs to find and fit.
                                                If None, find all peaks meeting criteria.
                                                Defaults to 3.
        use_peak_finding_if_fitting_fails (bool): If True, use the initial peak-finding
                                                  result if the parabola fit fails.
                                                  If False, return NaN for failed fits.
        zero_crossings_fit_range_hz (float): Frequency range (+/- Hz) around estimated
                                             zero crossings used for linear fitting.
        feature_fit_range_hz (float): Frequency range (+/- Hz) around found features
                                      used for parabolic fitting.
        smooth_window_hz (float): Frequency range (Hz) for the moving average window
                                  used *before* initial peak finding.
        plot_all (bool): If True, show plots for individual parabola fits.
        plot_result (bool): If True, show the final summary plot.
        save_result_plot (bool): If True and filename is provided, save the summary plot.
        filename (Optional[str]): Base filename for saving the plot (e.g., "scan_01").
                                  "_hyperfine_fitting.pdf" will be appended.

    Returns:
        Dict[str, Union[np.ndarray, float]]: A dictionary containing fitted parameters:
            - "peak_positions [Hz]": Fitted positions of peaks.
            - "peak_uncertainties [Hz]": Standard deviation uncertainty of peak positions from fit.
            - "dip_positions [Hz]": Fitted positions of dips.
            - "dip_uncertainties [Hz]": Standard deviation uncertainty of dip positions from fit.
            - "linewidths [Hz]": Estimated linewidths (sqrt(3) * |peak_pos - dip_pos|).
                                 Note: sqrt(3) assumes a Lorentzian derivative shape.
            - "zero_crossing_frequencies [Hz]": Midpoint between fitted peak/dip pairs.
            - "zero_crossing_slopes [V/Hz]": Slopes at the zero crossings from linear fits.
            - "n_features_found": Number of peak/dip pairs successfully processed.
    """

    # --- Input Validation and Preparation ---
    if not isinstance(frequency_array, (np.ndarray, pd.core.series.Series, list)):
        raise TypeError("frequency_array must be a numpy array, pandas Series, or list.")
    if not isinstance(voltage_array, (np.ndarray, pd.core.series.Series, list)):
        raise TypeError("voltage_array must be a numpy array, pandas Series, or list.")

    if isinstance(frequency_array, pd.core.series.Series):
        frequency_array = frequency_array.to_numpy()
    elif isinstance(frequency_array, list):
        frequency_array = np.array(frequency_array, dtype=np.float64)

    if isinstance(voltage_array, pd.core.series.Series):
        voltage_array = voltage_array.to_numpy()
    elif isinstance(voltage_array, list):
        voltage_array = np.array(voltage_array, dtype=np.float64)

    if frequency_array.shape != voltage_array.shape:
        raise ValueError("frequency_array and voltage_array must have the same shape after conversion.")
    if len(frequency_array) < 3: # Need at least a few points for fitting/smoothing
         raise ValueError("Input arrays must have at least 3 data points.")
    if frequency_array.ndim != 1:
         raise ValueError("Input arrays must be 1-dimensional.")
    if frequency_array[-1] <= frequency_array[0]:
         if not np.all(np.diff(frequency_array) > 0):
              raise ValueError("frequency_array must be monotonically increasing.")

    if n_most_prominent_peaks is not None and n_most_prominent_peaks <= 0:
        warnings.warn("n_most_prominent_peaks should be positive or None. Setting to None (find all).", UserWarning)
        n_most_prominent_peaks = None

    if feature_distance_in_Hz <= 0 or zero_crossings_fit_range_hz <= 0 or feature_fit_range_hz <= 0 or smooth_window_hz <= 0:
        raise ValueError(
            "Frequency ranges/distances (feature_distance_in_Hz, zero_crossings_fit_range_hz, feature_fit_range_hz, smooth_window_hz) must be positive.")

    n_points = len(frequency_array)
    frequency_spacing = (frequency_array[-1] - frequency_array[0]) / (n_points - 1)  # More robust calculation

    # Convert Hz parameters to sample indices
    min_feature_distance_in_samples = max(1,
                                          int(feature_distance_in_Hz / frequency_spacing))  # Ensure at least 1 sample
    n_samples_smooth = max(1, int(smooth_window_hz / frequency_spacing))
    fit_sample_half_range = max(1, int(feature_fit_range_hz / (2 * frequency_spacing)))  # Half-width for slicing
    zero_crossing_fit_half_range = max(1, int(zero_crossings_fit_range_hz / (2 * frequency_spacing)))

    # --- Initial Feature Finding (on Smoothed Data) ---
    smoothed_voltage_array = smooth(voltage_array, n_samples_smooth)

    try:
        peaks_indices, peaks_properties = find_peaks(
            smoothed_voltage_array,
            height=min_feature_height,
            distance=min_feature_distance_in_samples,
            prominence=feature_prominence
        )
        dips_indices, dips_properties = find_peaks(
            -smoothed_voltage_array,  # Invert for dips
            height=min_feature_height,  # Height here is positive depth
            distance=min_feature_distance_in_samples,
            prominence=feature_prominence
        )
    except Exception as e:
        print(f"Error during initial peak finding: {e}")
        # Return empty/NaN results if peak finding fails critically
        nan_array = np.array([np.nan])  # Placeholder for scalar return
        return {
            "peak_positions [Hz]": nan_array, "peak_uncertainties [Hz]": nan_array,
            "dip_positions [Hz]": nan_array, "dip_uncertainties [Hz]": nan_array,
            "linewidths [Hz]": nan_array, "zero_crossing_frequencies [Hz]": nan_array,
            "zero_crossing_slopes [V/Hz]": nan_array, "n_features_found": 0
        }

    # --- Filter by Prominence and Sort ---
    if n_most_prominent_peaks is not None:
        if len(peaks_indices) > n_most_prominent_peaks:
            prominent_peak_indices = np.argsort(peaks_properties["prominences"])[-n_most_prominent_peaks:]
            peaks_indices = peaks_indices[prominent_peak_indices]
        if len(dips_indices) > n_most_prominent_peaks:
            prominent_dip_indices = np.argsort(dips_properties["prominences"])[-n_most_prominent_peaks:]
            dips_indices = dips_indices[prominent_dip_indices]

    # Sort indices by frequency
    peaks_indices = np.sort(peaks_indices)
    dips_indices = np.sort(dips_indices)

    # Determine the number of features to actually process (limited by shortest list)
    # We expect pairs, so we take the minimum length.
    n_features_found = min(len(peaks_indices), len(dips_indices))

    if n_features_found == 0:
        print("No peak/dip pairs found matching the criteria.")
        nan_array = np.array([np.nan])  # Placeholder for scalar return
        return {
            "peak_positions [Hz]": nan_array, "peak_uncertainties [Hz]": nan_array,
            "dip_positions [Hz]": nan_array, "dip_uncertainties [Hz]": nan_array,
            "linewidths [Hz]": nan_array, "zero_crossing_frequencies [Hz]": nan_array,
            "zero_crossing_slopes [V/Hz]": nan_array, "n_features_found": 0
        }

    if n_most_prominent_peaks is not None and n_features_found < n_most_prominent_peaks:
        warnings.warn(
            f"Found only {n_features_found} peak/dip pairs, less than the requested {n_most_prominent_peaks}.",
            UserWarning)

    # --- Refine Feature Positions with Parabolic Fits (on Original Data) ---

    # Initialize result arrays with NaNs
    fit_peak_positions = np.full(n_features_found, np.nan)
    fit_peak_uncertainties = np.full(n_features_found, np.nan)
    fit_dip_positions = np.full(n_features_found, np.nan)
    fit_dip_uncertainties = np.full(n_features_found, np.nan)

    # Define fit warning threshold
    FIT_DEVIATION_WARN_HZ = 0.5e6

    for i in range(n_features_found):
        peak_idx = peaks_indices[i]
        dip_idx = dips_indices[i]

        # Define data slices for fitting, ensuring indices are within bounds
        peak_slice_start = max(0, peak_idx - fit_sample_half_range)
        peak_slice_end = min(n_points, peak_idx + fit_sample_half_range + 1)  # +1 for Python slicing
        dip_slice_start = max(0, dip_idx - fit_sample_half_range)
        dip_slice_end = min(n_points, dip_idx + fit_sample_half_range + 1)

        if peak_slice_end - peak_slice_start < 3 or dip_slice_end - dip_slice_start < 3:
            print(
                f"Warning: Not enough data points ({peak_slice_end - peak_slice_start} peak, {dip_slice_end - dip_slice_start} dip) around feature pair {i} for parabolic fitting. Skipping.")
            continue  # Skip this pair if insufficient data

        freq_around_peak = frequency_array[peak_slice_start:peak_slice_end]
        volt_around_peak = voltage_array[peak_slice_start:peak_slice_end]
        freq_around_dip = frequency_array[dip_slice_start:dip_slice_end]
        volt_around_dip = voltage_array[dip_slice_start:dip_slice_end]

        # Initial guesses for curve_fit
        # Use the initially found peak/dip frequency and voltage
        # Estimate curvature 'a' roughly from the edges of the fit window
        # Ensure curvature sign is correct (negative for peak, positive for dip)
        try:
            # Avoid division by zero if frequencies are identical
            delta_f_peak = freq_around_peak[-1] - freq_around_peak[0]
            delta_f_dip = freq_around_dip[-1] - freq_around_dip[0]

            # Rough curvature estimation - check for valid delta_f first
            # Use smoothed data for a potentially more stable curvature estimate? No, fit original data.
            # Use voltage at index, not smoothed voltage for initial guess 'c'
            if delta_f_peak > 1e-9:  # Check for non-zero frequency range
                peak_a_guess = -abs((volt_around_peak[-1] - volt_around_peak[0]) / delta_f_peak ** 2)
            else:
                peak_a_guess = -1.0  # Default guess if range is too small

            if delta_f_dip > 1e-9:
                dip_a_guess = abs((volt_around_dip[-1] - volt_around_dip[0]) / delta_f_dip ** 2)
            else:
                dip_a_guess = 1.0  # Default guess

            p0_peak = [frequency_array[peak_idx], peak_a_guess, voltage_array[peak_idx]]
            p0_dip = [frequency_array[dip_idx], dip_a_guess, voltage_array[dip_idx]]

            # Bounds: restrict x0 near initial guess, ensure correct sign for 'a'
            bounds_peak = (
                [freq_around_peak[0], -np.inf, -np.inf],  # Lower bounds (x0, a, c)
                [freq_around_peak[-1], 0, np.inf]  # Upper bounds (x0, a must be < 0, c)
            )
            bounds_dip = (
                [freq_around_dip[0], 0, -np.inf],  # Lower bounds (x0, a must be > 0, c)
                [freq_around_dip[-1], np.inf, np.inf]  # Upper bounds (x0, a, c)
            )

            # --- Fit Peak ---
            try:
                peak_params, peaks_cov = curve_fit(
                    parabola, freq_around_peak, volt_around_peak,
                    p0=p0_peak, bounds=bounds_peak, maxfev=5000  # Increase max iterations
                )
                peak_pos = peak_params[0]
                peak_unc = np.sqrt(np.diag(peaks_cov))[0] if np.all(np.isfinite(peaks_cov)) else np.nan

                if abs(peak_pos - frequency_array[peak_idx]) > FIT_DEVIATION_WARN_HZ:
                    warnings.warn(
                        f"Peak {i}: Fitted position ({peak_pos / 1e6:.3f} MHz) differs > {FIT_DEVIATION_WARN_HZ / 1e6} MHz from initial estimate ({frequency_array[peak_idx] / 1e6:.3f} MHz). Check fit quality.",
                        UserWarning)

                fit_peak_positions[i] = peak_pos
                fit_peak_uncertainties[i] = peak_unc

            except (RuntimeError, OptimizeWarning, ValueError) as fit_exc:
                print(
                    f"Warning: Parabolic fit failed for peak {i} near {frequency_array[peak_idx] / 1e6:.3f} MHz: {fit_exc}")
                if use_peak_finding_if_fitting_fails:
                    fit_peak_positions[i] = frequency_array[peak_idx]
                    # Uncertainty is unknown if fit failed
                    fit_peak_uncertainties[i] = np.nan
                # else: NaNs remain

            # --- Fit Dip ---
            try:
                dip_params, dips_cov = curve_fit(
                    parabola, freq_around_dip, volt_around_dip,
                    p0=p0_dip, bounds=bounds_dip, maxfev=5000
                )
                dip_pos = dip_params[0]
                dip_unc = np.sqrt(np.diag(dips_cov))[0] if np.all(np.isfinite(dips_cov)) else np.nan

                if abs(dip_pos - frequency_array[dip_idx]) > FIT_DEVIATION_WARN_HZ:
                    warnings.warn(
                        f"Dip {i}: Fitted position ({dip_pos / 1e6:.3f} MHz) differs > {FIT_DEVIATION_WARN_HZ / 1e6} MHz from initial estimate ({frequency_array[dip_idx] / 1e6:.3f} MHz). Check fit quality.",
                        UserWarning)

                fit_dip_positions[i] = dip_pos
                fit_dip_uncertainties[i] = dip_unc

            except (RuntimeError, OptimizeWarning, ValueError) as fit_exc:
                print(
                    f"Warning: Parabolic fit failed for dip {i} near {frequency_array[dip_idx] / 1e6:.3f} MHz: {fit_exc}")
                if use_peak_finding_if_fitting_fails:
                    fit_dip_positions[i] = frequency_array[dip_idx]
                    fit_dip_uncertainties[i] = np.nan
                # else: NaNs remain

            # --- Optional Plotting for Individual Fits ---
            if plot_all:
                # Plot only if both fits were attempted (even if they failed)
                fig_parabola, ax_parabola = plt.subplots()
                try:
                    # Plot raw data in fit range
                    ax_parabola.scatter(freq_around_peak / 1e6, volt_around_peak, color='blue', alpha=0.4, s=10,
                                        label='Peak Data')
                    ax_parabola.scatter(freq_around_dip / 1e6, volt_around_dip, color='green', alpha=0.4, s=10,
                                        label='Dip Data')

                    # Plot smoothed data in fit range for comparison
                    smooth_volt_around_peak = smooth(volt_around_peak,
                                                     max(1, n_samples_smooth // 4))  # Less smoothing maybe
                    smooth_volt_around_dip = smooth(volt_around_dip, max(1, n_samples_smooth // 4))
                    ax_parabola.plot(freq_around_peak / 1e6, smooth_volt_around_peak, color="cyan", lw=1.5,
                                     label='Smoothed Peak Data')
                    ax_parabola.plot(freq_around_dip / 1e6, smooth_volt_around_dip, color="lime", lw=1.5,
                                     label='Smoothed Dip Data')

                    # Plot fit results if successful
                    if not np.isnan(fit_peak_positions[i]):
                        ax_parabola.plot(freq_around_peak / 1e6, parabola(freq_around_peak, *peak_params),
                                         color='red', lw=2, label=f'Peak Fit {i}')
                        ax_parabola.axvline(fit_peak_positions[i] / 1e6, color='red', linestyle='--', lw=1)

                    if not np.isnan(fit_dip_positions[i]):
                        ax_parabola.plot(freq_around_dip / 1e6, parabola(freq_around_dip, *dip_params),
                                         color='purple', lw=2, label=f'Dip Fit {i}')
                        ax_parabola.axvline(fit_dip_positions[i] / 1e6, color='purple', linestyle='--', lw=1)

                    ax_parabola.set_title(f"Parabolic Fit - Feature Pair {i}")
                    ax_parabola.set_xlabel("Frequency [MHz]")
                    ax_parabola.set_ylabel("Voltage [V]")
                    ax_parabola.legend()
                    fig_parabola.tight_layout()
                    fig_parabola.show()
                except Exception as plot_exc:
                    print(f"EXCEPTION during individual plot for feature {i}: {plot_exc}")
                # No plt.show() here to avoid blocking execution for each plot
                # Consider managing figures better if n_features_found is large
                plt.close(fig_parabola)  # Close figure after showing

        except Exception as e:
            print(f"Error processing feature pair {i} (Peak Idx: {peak_idx}, Dip Idx: {dip_idx}): {e}")
            # Ensure NaNs remain for this pair if a general error occurred before fitting
            fit_peak_positions[i] = np.nan
            fit_peak_uncertainties[i] = np.nan
            fit_dip_positions[i] = np.nan
            fit_dip_uncertainties[i] = np.nan

    # --- Calculate Linewidths (using fitted positions) ---
    # Linewidth ~ sqrt(3) * |peak_pos - dip_pos| for Lorentzian derivative-like shape
    # Handle potential NaNs from failed fits
    linewidths = np.sqrt(3) * np.abs(fit_peak_positions - fit_dip_positions)

    # --- Determine Zero Crossings and Slopes (using fitted positions) ---
    # Estimate zero crossing as the midpoint between fitted peak and dip
    zero_crossings_frequencies = (fit_peak_positions + fit_dip_positions) / 2

    slopes = np.full(n_features_found, np.nan)
    intercepts = np.full(n_features_found, np.nan)  # Keep intercepts if needed later

    for i in range(n_features_found):
        zc_freq = zero_crossings_frequencies[i]

        # Skip if zero crossing is NaN (due to failed peak/dip fit)
        if np.isnan(zc_freq):
            continue

        # Define frequency range for linear fit around the zero crossing
        # Find closest indices, handling edges
        zc_idx_approx = np.argmin(np.abs(frequency_array - zc_freq))
        zc_slice_start = max(0, zc_idx_approx - zero_crossing_fit_half_range)
        zc_slice_end = min(n_points, zc_idx_approx + zero_crossing_fit_half_range + 1)

        if zc_slice_end - zc_slice_start < 2:  # Need at least 2 points for polyfit(1)
            print(
                f"Warning: Not enough data points ({zc_slice_end - zc_slice_start}) around zero crossing {i} for linear fit. Skipping.")
            continue  # Skip if insufficient data

        freq_around_zc = frequency_array[zc_slice_start:zc_slice_end]
        volt_around_zc = voltage_array[zc_slice_start:zc_slice_end]

        # Fit a straight line (degree 1 polynomial)
        try:
            # Check for degenerate data (e.g., all x or y values the same)
            if np.all(freq_around_zc == freq_around_zc[0]) or np.all(volt_around_zc == volt_around_zc[0]):
                raise ValueError("Degenerate data for linear fit (all X or Y values are the same).")

            slope, intercept = np.polyfit(freq_around_zc, volt_around_zc, 1)
            slopes[i] = slope
            intercepts[i] = intercept
        except (np.linalg.LinAlgError, ValueError) as lin_fit_exc:
            print(f"Warning: Linear fit failed for zero crossing {i} near {zc_freq / 1e6:.3f} MHz: {lin_fit_exc}")
            # slopes[i] remains NaN

    # --- Final Result Plotting ---
    fig_result, ax_result = None, None  # Initialize
    if plot_result or save_result_plot:
        fig_result, ax_result = plt.subplots(figsize=(10, 6))

        # Plot original and smoothed data
        ax_result.plot(frequency_array / 1e6, voltage_array, label="Original Data", color="darkgrey", alpha=0.6, lw=1)
        ax_result.plot(frequency_array / 1e6, smoothed_voltage_array, label="Smoothed Data", color="orange", alpha=0.8,
                       lw=1.5)

        # Plot initial found peaks/dips (useful for diagnosing find_peaks issues)
        ax_result.scatter(frequency_array[peaks_indices] / 1e6, smoothed_voltage_array[peaks_indices],
                          marker='^', color='red', s=50, label="Initial Peaks (smoothed)", zorder=4)
        ax_result.scatter(frequency_array[dips_indices] / 1e6, smoothed_voltage_array[dips_indices],
                          marker='v', color='blue', s=50, label="Initial Dips (smoothed)", zorder=4)

        # Plot fitted peak/dip positions
        ax_result.scatter(fit_peak_positions / 1e6,
                          parabola(fit_peak_positions, fit_peak_positions, -1, fit_peak_positions),  # Placeholder Y
                          marker='x', color='darkred', s=70, label="Fitted Peaks", zorder=5,
                          transform=ax_result.get_xaxis_transform())  # Hack to plot markers without needing Y data
        ax_result.scatter(fit_dip_positions / 1e6, parabola(fit_dip_positions, fit_dip_positions, 1, fit_dip_positions),
                          # Placeholder Y
                          marker='x', color='darkblue', s=70, label="Fitted Dips", zorder=5,
                          transform=ax_result.get_xaxis_transform())  # Hack to plot markers without needing Y data

        # Plot fitted lines at zero crossings
        for i in range(n_features_found):
            if not np.isnan(slopes[i]) and not np.isnan(zero_crossings_frequencies[i]):
                zc_freq = zero_crossings_frequencies[i]
                slope = slopes[i]
                intercept = intercepts[i]

                # Define plot range for the line (visual aid)
                plot_f_start = zc_freq - zero_crossings_fit_range_hz
                plot_f_end = zc_freq + zero_crossings_fit_range_hz
                line_freqs = np.linspace(plot_f_start, plot_f_end, 10)
                line_volts = slope * line_freqs + intercept

                ax_result.plot(line_freqs / 1e6, line_volts, color="purple",
                               linestyle='--', linewidth=1.5, zorder=3,
                               label="Zero Crossing Fit" if i == 0 else "")  # Label only once

        ax_result.set_xlabel("Frequency [MHz]")
        ax_result.set_ylabel("Voltage [V]")
        ax_result.legend()

        # Create title with linewidth info (handle NaNs)
        lw_str_parts = []
        for lw in linewidths:
            if np.isnan(lw):
                lw_str_parts.append("N/A")
            else:
                lw_str_parts.append(f'{lw / 1e6:.2f}')  # More precision
        title = 'ODMR Hyperfine Fit Results\nLinewidths [MHz]: ' + ', '.join(lw_str_parts)
        ax_result.set_title(title)

        fig_result.tight_layout()

        if save_result_plot and filename:
            try:
                save_path = f"{filename}_hyperfine_fitting.pdf"
                fig_result.savefig(save_path)
                print(f"Result plot saved to {save_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")

        if plot_result:
            plt.show()  # Blocks execution until plot window is closed

        # Close the figure window if it was created
        plt.close(fig_result)

    # --- Return Results ---
    return {
        "peak_positions [Hz]": fit_peak_positions,
        "peak_uncertainties [Hz]": fit_peak_uncertainties,
        "dip_positions [Hz]": fit_dip_positions,
        "dip_uncertainties [Hz]": fit_dip_uncertainties,
        "linewidths [Hz]": linewidths,
        "zero_crossing_frequencies [Hz]": zero_crossings_frequencies,
        "zero_crossing_slopes [V/Hz]": slopes,
        "n_features_found": n_features_found
    }