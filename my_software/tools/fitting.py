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
    """
    if box_pts <= 1:  # No smoothing needed if window is 1 or less
        return y
    if box_pts > len(y):  # Avoid window larger than data
        warnings.warn(f"Smoothing window ({box_pts}) is larger than data size ({len(y)}). No smoothing applied.",
                      UserWarning)
        return y

    box = np.ones(box_pts) / box_pts
    # Use 'same' mode which handles boundaries (e.g., zero-padding) and guarantees output size.
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def _pair_peaks_and_dips(
    peak_indices: np.ndarray,
    dip_indices: np.ndarray,
    frequency_array: np.ndarray,
    max_pair_distance_hz: float = 3e6
) -> List[Tuple[int, int]]:
    """
    Pair each peak with its nearest dip within max_pair_distance_hz.

    This is more robust than frequency-order pairing because it prevents
    spurious noise peaks from being incorrectly paired with legitimate dips.

    Args:
        peak_indices: Array of indices into frequency_array for peaks.
        dip_indices: Array of indices into frequency_array for dips.
        frequency_array: The frequency array (Hz).
        max_pair_distance_hz: Maximum allowed distance between peak and dip
                              to form a valid pair. Default 3 MHz.

    Returns:
        List of (peak_idx, dip_idx) tuples for valid pairs, sorted by
        the average frequency of each pair.
    """
    if len(peak_indices) == 0 or len(dip_indices) == 0:
        return []

    pairs = []
    used_dips = set()

    # Get frequencies for all peaks and dips
    peak_freqs = frequency_array[peak_indices]
    dip_freqs = frequency_array[dip_indices]

    # Process peaks in frequency order
    peak_order = np.argsort(peak_freqs)

    for sorted_idx in peak_order:
        peak_idx = peak_indices[sorted_idx]
        peak_f = frequency_array[peak_idx]
        best_dip_array_idx = None
        best_distance = float('inf')

        for j, dip_idx in enumerate(dip_indices):
            if j in used_dips:
                continue
            dip_f = frequency_array[dip_idx]
            distance = abs(peak_f - dip_f)
            if distance < best_distance and distance < max_pair_distance_hz:
                best_distance = distance
                best_dip_array_idx = j

        if best_dip_array_idx is not None:
            pairs.append((peak_idx, dip_indices[best_dip_array_idx]))
            used_dips.add(best_dip_array_idx)

    # Sort pairs by average frequency (center of each pair)
    pairs.sort(key=lambda p: (frequency_array[p[0]] + frequency_array[p[1]]) / 2)

    return pairs


def find_zero_crossings_direct(
    frequency_array: np.ndarray,
    voltage_array: np.ndarray,
    smooth_window_hz: float = 0.5e6,
    min_slope_threshold: float = 0.1,
    min_crossing_spacing_hz: float = 1.0e6,
    signal_envelope_threshold: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Find zero-crossings directly by locating where signal changes sign.

    This is more robust than peak-finding when ODMR lines are broadened,
    because zero-crossings are well-defined even when peaks/dips merge.

    Algorithm:
    1. Estimate baseline from edge regions
    2. Subtract baseline to center signal around zero
    3. Find signal envelope (where signal significantly deviates from baseline)
    4. Find sign changes within the signal envelope
    5. Interpolate to get precise crossing frequency
    6. Filter by minimum slope (reject noise crossings) and spacing

    Args:
        frequency_array: Array of frequencies (Hz).
        voltage_array: Array of corresponding voltages (V).
        smooth_window_hz: Smoothing window for noise reduction (Hz).
        min_slope_threshold: Minimum slope at crossing as fraction of max slope.
                             Rejects crossings in flat/noisy regions.
        min_crossing_spacing_hz: Minimum spacing between crossings (Hz).
                                 Helps reject noise-induced spurious crossings.
        signal_envelope_threshold: Fraction of signal amplitude to define
                                   "active" signal region.

    Returns:
        Tuple of:
            - zero_crossing_frequencies: Array of crossing frequencies (Hz)
            - slopes_at_crossings: Array of slopes (V/Hz) at each crossing
            - info: Dict with diagnostic info (envelope bounds, n_found, etc.)
    """
    n_points = len(frequency_array)
    freq_spacing = (frequency_array[-1] - frequency_array[0]) / (n_points - 1)

    # Smooth the data
    smooth_pts = max(3, int(smooth_window_hz / freq_spacing))
    smoothed = smooth(voltage_array, smooth_pts)

    # Estimate baseline from edge regions (first/last 10% of data)
    edge_frac = 0.1
    n_edge = max(5, int(n_points * edge_frac))
    baseline = np.median(np.concatenate([smoothed[:n_edge], smoothed[-n_edge:]]))

    # Subtract baseline to center around zero
    centered = smoothed - baseline

    # Find signal envelope: region where |signal| exceeds threshold
    signal_amplitude = np.max(np.abs(centered))
    envelope_threshold = signal_envelope_threshold * signal_amplitude
    in_envelope = np.abs(centered) > envelope_threshold

    # Dilate envelope to include nearby points (fill small gaps)
    dilate_pts = max(3, int(1e6 / freq_spacing))  # 1 MHz dilation
    dilated_envelope = np.copy(in_envelope)
    for i in range(n_points):
        if in_envelope[i]:
            start = max(0, i - dilate_pts)
            end = min(n_points, i + dilate_pts + 1)
            dilated_envelope[start:end] = True

    # Find envelope bounds
    envelope_indices = np.where(dilated_envelope)[0]
    if len(envelope_indices) == 0:
        # No significant signal found
        return np.array([]), np.array([]), {
            'envelope_start_hz': np.nan,
            'envelope_end_hz': np.nan,
            'n_crossings_raw': 0,
            'n_crossings_filtered': 0,
            'baseline': baseline,
            'signal_amplitude': signal_amplitude
        }

    envelope_start = envelope_indices[0]
    envelope_end = envelope_indices[-1]
    envelope_start_hz = frequency_array[envelope_start]
    envelope_end_hz = frequency_array[envelope_end]

    # Find sign changes in the centered, smoothed signal WITHIN the envelope
    sign_changes = []
    slopes = []

    for i in range(envelope_start, min(envelope_end, n_points - 1)):
        if centered[i] * centered[i + 1] < 0:  # Sign change
            # Linear interpolation to find precise crossing
            f0, f1 = frequency_array[i], frequency_array[i + 1]
            v0, v1 = centered[i], centered[i + 1]

            # Crossing frequency (linear interpolation)
            crossing_freq = f0 - v0 * (f1 - f0) / (v1 - v0)

            # Slope at crossing (V/Hz)
            slope = (v1 - v0) / (f1 - f0)

            sign_changes.append(crossing_freq)
            slopes.append(slope)

    n_crossings_raw = len(sign_changes)

    if n_crossings_raw == 0:
        return np.array([]), np.array([]), {
            'envelope_start_hz': envelope_start_hz,
            'envelope_end_hz': envelope_end_hz,
            'n_crossings_raw': 0,
            'n_crossings_filtered': 0,
            'baseline': baseline,
            'signal_amplitude': signal_amplitude
        }

    sign_changes = np.array(sign_changes)
    slopes = np.array(slopes)

    # Filter by minimum slope (reject crossings in flat/noisy regions)
    max_slope = np.max(np.abs(slopes))
    slope_threshold = min_slope_threshold * max_slope
    valid_slope = np.abs(slopes) >= slope_threshold

    # Filter by minimum spacing (reject closely-spaced noise crossings)
    # Keep crossings with highest slopes when multiple are close together
    valid_spacing = np.ones(len(sign_changes), dtype=bool)
    for i in range(len(sign_changes)):
        if not valid_slope[i]:
            continue
        for j in range(i + 1, len(sign_changes)):
            if not valid_slope[j]:
                continue
            if abs(sign_changes[j] - sign_changes[i]) < min_crossing_spacing_hz:
                # Keep the one with higher slope
                if abs(slopes[i]) >= abs(slopes[j]):
                    valid_spacing[j] = False
                else:
                    valid_spacing[i] = False
                    break

    valid = valid_slope & valid_spacing

    filtered_crossings = sign_changes[valid]
    filtered_slopes = slopes[valid]

    # Sort by frequency
    sort_idx = np.argsort(filtered_crossings)
    filtered_crossings = filtered_crossings[sort_idx]
    filtered_slopes = filtered_slopes[sort_idx]

    info = {
        'envelope_start_hz': envelope_start_hz,
        'envelope_end_hz': envelope_end_hz,
        'n_crossings_raw': n_crossings_raw,
        'n_crossings_filtered': len(filtered_crossings),
        'baseline': baseline,
        'signal_amplitude': signal_amplitude,
        'smoothed_data': smoothed,
        'centered_data': centered
    }

    return filtered_crossings, filtered_slopes, info


def fit_odmr_zero_crossing(
    frequency_array: Union[np.ndarray, pd.Series],
    voltage_array: Union[np.ndarray, pd.Series],
    n_expected_crossings: int = 5,
    expected_spacing_hz: float = 2.158e6,
    smooth_window_hz: float = 0.5e6,
    min_slope_threshold: float = 0.05,
    validate_spacing: bool = True,
    spacing_tolerance: float = 0.5
) -> Dict[str, Union[float, np.ndarray, str, int]]:
    """
    Fit ODMR spectrum by finding zero-crossings directly.

    This is more robust for broadened ODMR lines than peak-finding approaches.
    It finds where the signal crosses zero and validates the crossings based
    on expected hyperfine spacing.

    Args:
        frequency_array: Array of frequencies (Hz).
        voltage_array: Array of corresponding voltages (V).
        n_expected_crossings: Expected number of zero-crossings (default 5).
        expected_spacing_hz: Expected spacing between crossings (default 2.158 MHz).
        smooth_window_hz: Smoothing window (Hz).
        min_slope_threshold: Minimum slope as fraction of max (rejects noise).
        validate_spacing: If True, validate that crossing spacing is consistent
                          with expected_spacing_hz.
        spacing_tolerance: Allowed deviation from expected spacing (as fraction).

    Returns:
        Dictionary with fit results:
            - 'center_frequency': Mean of zero-crossing frequencies (Hz)
            - 'zero_crossing_frequencies [Hz]': Array of crossing positions
            - 'zero_crossing_slopes [V/Hz]': Array of slopes at crossings
            - 'n_features_found': Number of valid crossings found
            - 'fit_method': 'direct_zero_crossing'
            - 'envelope_start_hz', 'envelope_end_hz': Signal bounds
            - 'quality_grade': 'high', 'medium', or 'low' based on n_found vs expected
    """
    if isinstance(frequency_array, pd.Series):
        frequency_array = frequency_array.to_numpy()
    if isinstance(voltage_array, pd.Series):
        voltage_array = voltage_array.to_numpy()

    # Find zero-crossings
    crossings, slopes, info = find_zero_crossings_direct(
        frequency_array, voltage_array,
        smooth_window_hz=smooth_window_hz,
        min_slope_threshold=min_slope_threshold,
        min_crossing_spacing_hz=expected_spacing_hz * 0.5  # At least half the expected spacing
    )

    n_found = len(crossings)

    # Validate spacing if requested and we have multiple crossings
    if validate_spacing and n_found >= 2:
        # Check if spacing is consistent with expected
        spacings = np.diff(crossings)
        expected_spacings = expected_spacing_hz * np.ones(n_found - 1)

        # Allow some tolerance
        min_valid_spacing = expected_spacing_hz * (1 - spacing_tolerance)
        max_valid_spacing = expected_spacing_hz * (1 + spacing_tolerance)

        # Flag crossings with inconsistent spacing
        valid_spacings = (spacings >= min_valid_spacing) & (spacings <= max_valid_spacing)

        # If most spacings are invalid, the crossings may be unreliable
        if n_found > 2 and np.sum(valid_spacings) < (n_found - 1) * 0.5:
            # More than half of spacings are wrong - likely noise issues
            # Fall back to using the crossing closest to signal center
            signal_center = (info['envelope_start_hz'] + info['envelope_end_hz']) / 2
            center_distances = np.abs(crossings - signal_center)

            # Keep crossings near the center, reject outliers
            center_threshold = (info['envelope_end_hz'] - info['envelope_start_hz']) * 0.4
            valid_position = center_distances < center_threshold

            if np.any(valid_position):
                crossings = crossings[valid_position]
                slopes = slopes[valid_position]
                n_found = len(crossings)

    # Calculate center frequency
    if n_found > 0:
        center_frequency = np.mean(crossings)
    else:
        # Fallback: use center of signal envelope
        center_frequency = (info['envelope_start_hz'] + info['envelope_end_hz']) / 2

    # Determine quality grade
    if n_found >= n_expected_crossings:
        quality = 'high'
    elif n_found >= n_expected_crossings * 0.6:  # 60% or more
        quality = 'medium'
    else:
        quality = 'low'

    # Calculate linewidths if we have crossings
    # For hyperfine comb, linewidth relates to envelope width
    if n_found >= 2:
        # Approximate linewidth from total span
        total_span = crossings[-1] - crossings[0]
        estimated_linewidth = total_span / (n_found - 1) * 0.5  # Rough estimate
    else:
        estimated_linewidth = info['envelope_end_hz'] - info['envelope_start_hz']

    return {
        'center_frequency': center_frequency,
        'zero_crossing_frequencies [Hz]': crossings,
        'zero_crossing_slopes [V/Hz]': slopes,
        'n_features_found': n_found,
        'linewidths [Hz]': np.array([estimated_linewidth]),
        'fit_method': 'direct_zero_crossing',
        'quality_grade': quality,
        'envelope_start_hz': info['envelope_start_hz'],
        'envelope_end_hz': info['envelope_end_hz'],
        'signal_amplitude': info['signal_amplitude']
    }


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
        max_pair_distance_hz:   Optional[float] = None,  # Max distance for peak-dip pairing
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
        max_pair_distance_hz (Optional[float]): Maximum allowed distance (Hz) between
                                                 a peak and dip to form a valid pair.
                                                 If None, uses legacy frequency-order pairing.
                                                 Recommended: ~1.5x the expected peak-dip distance
                                                 (e.g., 3e6 for hyperfine comb signals).
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
            "Frequency ranges/distances (feature_distance_in_Hz, zero_crossings_fit_range, feature_fit_range, smooth_window_hz) must be positive.")

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

    # --- Pair peaks and dips ---
    if max_pair_distance_hz is not None:
        # Use proximity-based pairing (more robust to spurious features)
        paired_indices = _pair_peaks_and_dips(
            peaks_indices, dips_indices, frequency_array, max_pair_distance_hz
        )
        n_features_found = len(paired_indices)
    else:
        # Legacy behavior: pair by frequency order
        n_features_found = min(len(peaks_indices), len(dips_indices))
        paired_indices = [(peaks_indices[i], dips_indices[i]) for i in range(n_features_found)]

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

    for i, (peak_idx, dip_idx) in enumerate(paired_indices):

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

        # --- Plot fitted peak/dip positions ---
        # Filter out any NaN results from failed fits before interpolating
        valid_peak_indices = ~np.isnan(fit_peak_positions)
        valid_dip_indices = ~np.isnan(fit_dip_positions)

        if np.any(valid_peak_indices):
            fitted_peak_voltages = np.interp(
                fit_peak_positions[valid_peak_indices], # x-values for interpolation
                frequency_array,                       # original x-data
                voltage_array                          # original y-data
            )
            ax_result.scatter(
                fit_peak_positions[valid_peak_indices] / 1e6, # Fitted X (MHz)
                fitted_peak_voltages,                        # Interpolated Y (Voltage)
                marker='x', color='darkred', s=70, label="Fitted Peaks", zorder=5
            )

        if np.any(valid_dip_indices):
            # Interpolate original voltage data at the fitted dip frequencies
            fitted_dip_voltages = np.interp(
                fit_dip_positions[valid_dip_indices], # x-values for interpolation
                frequency_array,                     # original x-data
                voltage_array                        # original y-data
            )
            ax_result.scatter(
                fit_dip_positions[valid_dip_indices] / 1e6, # Fitted X (MHz)
                fitted_dip_voltages,                       # Interpolated Y (Voltage)
                marker='x', color='darkblue', s=70, label="Fitted Dips", zorder=5
            )

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


# --- Robust Fitting Functions ---

def _lorentzian_derivative(f: np.ndarray, f0: float, gamma: float, amp: float) -> np.ndarray:
    """
    Derivative of Lorentzian lineshape, typical for lock-in detected ODMR.

    The derivative of L(f) = A / (1 + ((f-f0)/gamma)^2) is:
    dL/df = -2 * A * (f-f0) / (gamma^2 * (1 + ((f-f0)/gamma)^2)^2)

    Simplified form used here: -2 * amp * x / (1 + x^2)^2 where x = (f-f0)/gamma

    Args:
        f: Frequency array (Hz).
        f0: Center frequency (Hz).
        gamma: Half-width at half-maximum (Hz).
        amp: Amplitude scaling factor.

    Returns:
        Lorentzian derivative values.
    """
    x = (f - f0) / gamma
    return -2 * amp * x / (1 + x**2)**2


def fit_hyperfine_comb_constrained(
    frequency_array: Union[np.ndarray, pd.Series],
    voltage_array: Union[np.ndarray, pd.Series],
    n_features: int = 5,
    hyperfine_spacing_hz: float = 2.158e6,
    initial_center_hz: Optional[float] = None,
    allow_amplitude_variation: bool = False
) -> Optional[Dict[str, Union[float, np.ndarray, str]]]:
    """
    Fit ODMR spectrum with hyperfine comb excitation using physics constraints.

    This is more robust than fit_hyperfine for degraded data because it:
    1. Uses the entire spectrum shape, not just individual peak positions.
    2. Enforces the known ~2.158 MHz hyperfine spacing.
    3. Directly fits a single center_frequency parameter (the goal).

    Model: Sum of n_features Lorentzian derivatives with fixed spacing.
    Free parameters: center_freq, linewidth, amplitude, baseline

    Args:
        frequency_array: Array of frequencies (Hz).
        voltage_array: Array of corresponding voltages (V).
        n_features: Number of hyperfine features (default 5 for comb excitation).
        hyperfine_spacing_hz: Fixed spacing between features (default 2.158 MHz for N14).
        initial_center_hz: Initial guess for center frequency. If None, uses
                           weighted center-of-mass estimate.
        allow_amplitude_variation: If True, fit individual amplitudes for each
                                   feature. If False, use single amplitude (simpler).

    Returns:
        Dictionary with fit results, or None if fit fails:
            - 'center_frequency': Fitted center frequency (Hz)
            - 'center_uncertainty': Uncertainty from covariance (Hz)
            - 'linewidth': Fitted linewidth (Hz)
            - 'amplitude': Fitted amplitude
            - 'baseline': Fitted baseline offset
            - 'residual_rms': RMS of fit residuals
            - 'fit_method': 'constrained_model'
    """
    # Input validation and conversion
    if isinstance(frequency_array, pd.Series):
        frequency_array = frequency_array.to_numpy()
    if isinstance(voltage_array, pd.Series):
        voltage_array = voltage_array.to_numpy()

    if len(frequency_array) < 10:
        warnings.warn("Too few data points for constrained fit.", UserWarning)
        return None

    # Define the model function
    def multi_lorentzian_derivative(f, center, linewidth, amplitude, baseline):
        """Sum of n_features Lorentzian derivatives with fixed spacing."""
        result = np.full_like(f, baseline, dtype=np.float64)
        # Features centered around 'center' with hyperfine_spacing
        offsets = np.arange(n_features) - (n_features - 1) / 2.0
        for offset in offsets:
            f0 = center + offset * hyperfine_spacing_hz
            result += _lorentzian_derivative(f, f0, linewidth, amplitude)
        return result

    # Initial guess for center frequency
    if initial_center_hz is None:
        # Use weighted center-of-mass as initial guess
        baseline_est = np.median(voltage_array)
        weights = np.abs(voltage_array - baseline_est)
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            initial_center_hz = np.sum(frequency_array * weights) / weights_sum
        else:
            initial_center_hz = np.mean(frequency_array)

    # Initial guesses for other parameters
    initial_linewidth = 1.0e6  # 1 MHz typical
    signal_range = np.max(voltage_array) - np.min(voltage_array)
    initial_amplitude = signal_range / (4 * n_features)  # Rough estimate
    initial_baseline = np.median(voltage_array)

    p0 = [initial_center_hz, initial_linewidth, initial_amplitude, initial_baseline]

    # Bounds: center within frequency range, linewidth 0.1-20 MHz, amplitude and baseline free
    freq_min, freq_max = frequency_array[0], frequency_array[-1]
    bounds = (
        [freq_min, 0.1e6, -np.inf, -np.inf],   # Lower bounds
        [freq_max, 20e6, np.inf, np.inf]       # Upper bounds
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                multi_lorentzian_derivative,
                frequency_array, voltage_array,
                p0=p0, bounds=bounds, maxfev=10000
            )

        center_freq = popt[0]
        linewidth = popt[1]
        amplitude = popt[2]
        baseline = popt[3]

        # Extract uncertainty from covariance
        if np.all(np.isfinite(pcov)):
            center_uncertainty = np.sqrt(pcov[0, 0])
        else:
            center_uncertainty = np.nan

        # Calculate residuals
        fitted = multi_lorentzian_derivative(frequency_array, *popt)
        residual_rms = np.sqrt(np.mean((voltage_array - fitted)**2))

        return {
            'center_frequency': center_freq,
            'center_uncertainty': center_uncertainty,
            'linewidth': linewidth,
            'amplitude': amplitude,
            'baseline': baseline,
            'residual_rms': residual_rms,
            'fit_method': 'constrained_model',
            'n_features': n_features
        }

    except (RuntimeError, ValueError, OptimizeWarning) as e:
        warnings.warn(f"Constrained fit failed: {e}", UserWarning)
        return None


def estimate_center_frequency_com(
    frequency_array: Union[np.ndarray, pd.Series],
    voltage_array: Union[np.ndarray, pd.Series]
) -> float:
    """
    Estimate center frequency using weighted center-of-mass.

    This is a simple, robust fallback that works even when individual
    features can't be resolved due to severe broadening or noise.

    The weighting uses the absolute deviation from baseline, so both
    peaks and dips contribute to the estimate.

    Args:
        frequency_array: Array of frequencies (Hz).
        voltage_array: Array of corresponding voltages (V).

    Returns:
        Estimated center frequency (Hz).
    """
    if isinstance(frequency_array, pd.Series):
        frequency_array = frequency_array.to_numpy()
    if isinstance(voltage_array, pd.Series):
        voltage_array = voltage_array.to_numpy()

    # Use median as baseline estimate (robust to outliers)
    baseline = np.median(voltage_array)

    # Weight by absolute deviation from baseline
    weights = np.abs(voltage_array - baseline)
    weights_sum = np.sum(weights)

    if weights_sum > 0:
        return np.sum(frequency_array * weights) / weights_sum
    else:
        # Fallback to simple center if all weights are zero
        return np.mean(frequency_array)


def estimate_spectrum_quality(
    frequency_array: Union[np.ndarray, pd.Series],
    voltage_array: Union[np.ndarray, pd.Series],
    n_expected_features: int = 5
) -> Tuple[str, str]:
    """
    Assess spectrum quality and recommend fitting strategy.

    Analyzes the spectrum to determine data quality based on:
    - Signal-to-noise ratio (SNR)
    - Number of distinguishable features

    Args:
        frequency_array: Array of frequencies (Hz).
        voltage_array: Array of corresponding voltages (V).
        n_expected_features: Expected number of peak/dip pairs.

    Returns:
        Tuple of (quality_grade, recommended_method):
            - quality_grade: 'high', 'medium', or 'low'
            - recommended_method: 'standard', 'constrained', or 'com'
    """
    if isinstance(frequency_array, pd.Series):
        frequency_array = frequency_array.to_numpy()
    if isinstance(voltage_array, pd.Series):
        voltage_array = voltage_array.to_numpy()

    n_points = len(voltage_array)

    # Estimate noise from outer regions (presumably baseline)
    edge_fraction = 0.1
    n_edge = max(5, int(n_points * edge_fraction))

    baseline_data = np.concatenate([voltage_array[:n_edge], voltage_array[-n_edge:]])
    noise_std = np.std(baseline_data)

    # Estimate signal amplitude
    signal_amplitude = np.max(voltage_array) - np.min(voltage_array)
    snr = signal_amplitude / (noise_std + 1e-12)

    # Count peaks above noise threshold
    smoothed = smooth(voltage_array, max(1, n_points // 50))
    height_threshold = 3 * noise_std
    distance_samples = max(1, n_points // 20)

    try:
        peaks, _ = find_peaks(smoothed, height=height_threshold, distance=distance_samples)
        dips, _ = find_peaks(-smoothed, height=height_threshold, distance=distance_samples)
        n_features_found = min(len(peaks), len(dips))
    except Exception:
        n_features_found = 0

    # Quality assessment
    if snr > 15 and n_features_found >= n_expected_features - 1:
        return 'high', 'standard'
    elif snr > 5:
        return 'medium', 'constrained'
    else:
        return 'low', 'com'


def fit_odmr_robust(
    frequency_array: Union[np.ndarray, pd.Series],
    voltage_array: Union[np.ndarray, pd.Series],
    n_expected_features: int = 5,
    hyperfine_spacing_hz: float = 2.158e6,
    max_pair_distance_hz: Optional[float] = None,
    force_method: Optional[str] = None,
    feature_prominence: float = 0.02,
    min_feature_height: float = 0.02,
    validate_results: bool = True
) -> Dict[str, Union[float, np.ndarray, str, int]]:
    """
    Robust ODMR fitting with automatic method selection and validation.

    This is a unified interface that tries multiple methods in order of preference
    and validates results to ensure physical consistency:

    Method priority:
    1. 'zero_crossing': Direct zero-crossing detection (most robust for broadened data)
    2. 'standard': fit_hyperfine with proximity-based pairing
    3. 'constrained': Multi-Lorentzian model with fixed spacing
    4. 'com': Center-of-mass fallback

    Args:
        frequency_array: Array of frequencies (Hz).
        voltage_array: Array of corresponding voltages (V).
        n_expected_features: Expected number of zero-crossings/features (default 5).
        hyperfine_spacing_hz: Expected spacing between features (default 2.158 MHz).
        max_pair_distance_hz: Maximum distance for peak-dip pairing. If None,
                              defaults to 0.75 * hyperfine_spacing_hz.
        force_method: Force a specific method ('zero_crossing', 'standard',
                      'constrained', 'com') instead of auto-selection.
        feature_prominence: Prominence threshold for peak finding (standard method).
        min_feature_height: Minimum height threshold for peak finding (standard method).
        validate_results: If True, validate that results are physically reasonable.

    Returns:
        Dictionary with fit results:
            - 'center_frequency': Fitted/estimated center frequency (Hz)
            - 'quality_grade': 'high', 'medium', or 'low'
            - 'method_used': Method that produced the final result
            - Additional keys depending on method used
    """
    # Input conversion
    if isinstance(frequency_array, pd.Series):
        frequency_array = frequency_array.to_numpy()
    if isinstance(voltage_array, pd.Series):
        voltage_array = voltage_array.to_numpy()

    # Default max_pair_distance if not provided
    if max_pair_distance_hz is None:
        max_pair_distance_hz = hyperfine_spacing_hz * 0.75

    result = {
        'quality_grade': 'unknown',
        'method_used': 'none'
    }

    # Helper function to validate results
    def is_valid_result(res, freq_array):
        """Check if result is physically reasonable."""
        if 'center_frequency' not in res or res['center_frequency'] is None:
            return False
        cf = res['center_frequency']
        if np.isnan(cf):
            return False
        # Center should be within the frequency range (with small margin)
        margin = (freq_array[-1] - freq_array[0]) * 0.1
        if cf < freq_array[0] - margin or cf > freq_array[-1] + margin:
            return False
        return True

    # Method 1: Direct zero-crossing detection (PREFERRED for robustness)
    if force_method is None or force_method == 'zero_crossing':
        try:
            zc_result = fit_odmr_zero_crossing(
                frequency_array, voltage_array,
                n_expected_crossings=n_expected_features,
                expected_spacing_hz=hyperfine_spacing_hz,
                smooth_window_hz=max(0.25e6, hyperfine_spacing_hz * 0.15),
                min_slope_threshold=0.05,
                validate_spacing=True
            )

            n_found = zc_result.get('n_features_found', 0)

            # Accept if we found a reasonable number of crossings
            if n_found >= 2 and is_valid_result(zc_result, frequency_array):
                result.update(zc_result)
                result['method_used'] = 'zero_crossing'

                # Set quality based on how many features found
                if n_found >= n_expected_features:
                    result['quality_grade'] = 'high'
                elif n_found >= n_expected_features * 0.6:
                    result['quality_grade'] = 'medium'
                else:
                    result['quality_grade'] = 'low'

                if force_method == 'zero_crossing':
                    return result
                # Continue to try other methods if not enough features found
                if n_found < n_expected_features * 0.6:
                    pass  # Try other methods
                else:
                    return result

        except Exception as e:
            warnings.warn(f"Zero-crossing method failed: {e}", UserWarning)

    # Method 2: Standard fit_hyperfine with proximity-based pairing
    if (force_method is None or force_method == 'standard') and result['method_used'] == 'none':
        try:
            fit_result = fit_hyperfine(
                frequency_array, voltage_array,
                n_most_prominent_peaks=n_expected_features,
                max_pair_distance_hz=max_pair_distance_hz,
                feature_prominence=feature_prominence,
                min_feature_height=min_feature_height
            )

            # Calculate center frequency from zero-crossings
            zc_freqs = fit_result.get('zero_crossing_frequencies [Hz]', np.array([np.nan]))
            if isinstance(zc_freqs, np.ndarray):
                valid_zc = zc_freqs[~np.isnan(zc_freqs)]
            else:
                valid_zc = []

            if len(valid_zc) >= 2:
                fit_result['center_frequency'] = np.mean(valid_zc)

                # Validate: check if crossings are within expected range
                if validate_results:
                    # All crossings should be relatively close together
                    span = valid_zc[-1] - valid_zc[0]
                    expected_span = (n_expected_features - 1) * hyperfine_spacing_hz
                    if span > expected_span * 3:
                        # Span too large - likely spurious crossings
                        warnings.warn(f"Standard fit span ({span/1e6:.1f} MHz) too large, rejecting", UserWarning)
                    elif is_valid_result(fit_result, frequency_array):
                        result.update(fit_result)
                        result['method_used'] = 'standard'
                        result['quality_grade'] = 'high' if len(valid_zc) >= n_expected_features else 'medium'
                        if force_method == 'standard':
                            return result

        except Exception as e:
            warnings.warn(f"Standard fitting failed: {e}", UserWarning)

    # Method 3: Constrained multi-Lorentzian model
    if (force_method is None or force_method == 'constrained') and result['method_used'] == 'none':
        try:
            fit_result = fit_hyperfine_comb_constrained(
                frequency_array, voltage_array,
                n_features=n_expected_features,
                hyperfine_spacing_hz=hyperfine_spacing_hz
            )
            if fit_result is not None and is_valid_result(fit_result, frequency_array):
                result.update(fit_result)
                result['method_used'] = 'constrained'
                result['quality_grade'] = 'medium'
                if force_method == 'constrained':
                    return result

        except Exception as e:
            warnings.warn(f"Constrained fitting failed: {e}", UserWarning)

    # Method 4: Center-of-mass fallback
    if force_method == 'com' or result['method_used'] == 'none':
        result['center_frequency'] = estimate_center_frequency_com(
            frequency_array, voltage_array
        )
        result['method_used'] = 'com'
        result['quality_grade'] = 'low'
        result['n_features_found'] = 0

    # Final validation
    if not is_valid_result(result, frequency_array):
        # Ultimate fallback: center of frequency range
        result['center_frequency'] = np.mean(frequency_array)
        result['method_used'] = 'range_center_fallback'
        result['quality_grade'] = 'low'

    return result