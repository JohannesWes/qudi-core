import os
import time
from datetime import datetime
import logging

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from my_software.tools.logging_config import setup_logging
setup_logging(level=logging.INFO)

from my_software.automation.qudi_remote_control import OdmrRemoteControl
from my_software.automation.power_supply_NGP_control import NGP_instance
from my_software.tools.fitting import fit_hyperfine


# --- Configuration ---
EXPERIMENT_TYPE = "current_calibration" # Subfolder name within the daily qudi directory
QUDI_DATA_ROOT = os.path.expanduser('~/qudi/Data') # Standard Qudi data location

# -------------------------------------------------------------------------------------------------------------------- #

def get_measurement_save_path():
    """
    Determines the correct save path based on Qudi's convention.
    Creates the necessary directories if they don't exist.
    Returns the path for the specific measurement run folder.
    Example: C:/Users/user/qudi/Data/YYYY/MM/YYYY-MM-DD/current_calibration/YYYYMMDD-HHMMSS_Calibration
    """
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    year_str = now.strftime('%Y')
    month_str = now.strftime('%m')
    timestamp_str = now.strftime('%Y%m%d-%H%M%S')

    # Construct the path according to Qudi's daily structure + experiment type
    daily_data_path = os.path.join(QUDI_DATA_ROOT, year_str, month_str, date_str)
    experiment_base_path = os.path.join(daily_data_path, EXPERIMENT_TYPE)

    # Create a unique folder for this specific measurement run
    measurement_folder_name = f"{timestamp_str}_Calibration"
    measurement_folder_path = os.path.join(experiment_base_path, measurement_folder_name)

    try:
        # Create all necessary directories
        os.makedirs(measurement_folder_path, exist_ok=True)
        logging.info(f"Created measurement directory: {measurement_folder_path}")
        return measurement_folder_path
    except OSError as e:
        logging.error(f"Failed to create directory {measurement_folder_path}: {e}")
        raise # Re-raise the exception to stop the script if dir creation fails

def get_qudi_relative_save_path(measurement_folder_path, filename_base):
    """
    Calculates the path relative to the Qudi daily data directory,
    which is needed for the odmr_remote.take_odmr_scan function.
    Example: current_calibration/YYYYMMDD-HHMMSS_Calibration/filename_base
    """
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    year_str = now.strftime('%Y')
    month_str = now.strftime('%m')
    daily_data_path = os.path.join(QUDI_DATA_ROOT, year_str, month_str, date_str)

    # Get the part of the path relative to the daily directory
    # Ensure the daily path exists or handle the potential error if it doesn't
    if not os.path.isdir(daily_data_path):
        logging.warning(f"Daily data path {daily_data_path} does not exist. Relative path calculation might be incorrect.")
        # Attempt to create it? Or just proceed? For now, proceed.
        try:
            os.makedirs(daily_data_path, exist_ok=True)
            logging.info(f"Created missing daily data path: {daily_data_path}")
        except OSError as e:
            logging.error(f"Failed to create daily data path {daily_data_path}: {e}")
            # Fallback: maybe return the filename_base directly or raise error?
            return filename_base.replace('\\', '/')


    try:
        relative_path = os.path.relpath(measurement_folder_path, daily_data_path)
    except ValueError as e:
         # This can happen on Windows if paths are on different drives
         logging.error(f"Could not determine relative path from {daily_data_path} to {measurement_folder_path}: {e}. Using absolute path structure relative to experiment type.")
         # Fallback: Construct path relative to EXPERIMENT_TYPE within the day folder structure assumed by get_measurement_save_path
         measurement_folder_name = os.path.basename(measurement_folder_path)
         relative_path = os.path.join(EXPERIMENT_TYPE, measurement_folder_name)


    # Combine relative path with the desired filename base
    return os.path.join(relative_path, filename_base).replace('\\', '/') # Use forward slashes for consistency


# -------------------------------------------------------------------------------------------------------------------- #

def current_measurement(measurement_folder_path,
                        current_min, current_max, current_points,
                        run_time_per_current_point=10, current_settling_time=5,
                        odmr_ranges=[[2.64e9, 2.65e9]], current_array=None, odmr_frequency_points=1000):

    """
    Performs ODMR measurements for a range of currents.

    Args:
        measurement_folder_path (str): Absolute path to the dedicated folder for this measurement run.
                                       Used for saving numpy arrays. Qudi data will be saved relative to this path.
        current_min (float): Minimum current value.
        current_max (float): Maximum current value.
        current_points (int): Number of current steps.
        run_time_per_current_point (float): Duration of each ODMR scan in seconds.
        current_settling_time (float): Time to wait after setting current before starting scan (in seconds).
        odmr_ranges (list of lists): List containing [freq_start, freq_stop] for each ODMR scan range.
        current_array (np.ndarray, optional): Predefined array of currents. If None, generated from min/max/points.
        odmr_frequency_points (int): Number of frequency points per ODMR scan.

    Returns:
        tuple: (odmr_frequencies_array, odmr_voltages_array) containing the raw measurement data.
    """
    odmr_remote = OdmrRemoteControl()

    # setting up the R&S NGP power supply
    ngp_active_channel = 1
    ngp_voltage = 60
    ngp = None
    try:
        ngp_address = "USB0::0x0AAD::0x0197::5601.4007k03-101169::INSTR"
        ngp = NGP_instance(usb_address=ngp_address)
        ngp.activate_channel(ngp_active_channel)
        ngp.output_on()
        logging.info(f"NGP Power supply initialized ({ngp_address}).")
        logging.info(f"Setting NGP channel {ngp_active_channel} voltage to {ngp_voltage} V.")
        logging.info(f"Turned NGP channel {ngp_active_channel} output ON.")

    except Exception as e:
        logging.error(f"Failed to initialize NGP Power Supply: {e}")
        odmr_remote.close_connection() # Close ODMR connection if NGP fails
        if ngp: # Try to close if object was partially created
            try:
                ngp.close()
            except Exception:
                logging.warning("Failed to close NGP connection after initialization error.")
        raise

    # define currents to be applied
    if current_array is None: # Check correctly for None
        current_array = np.linspace(current_min, current_max, current_points)
        logging.info(f"Generated current array from {current_min:.4f}A to {current_max:.4f}A with {current_points} points.")
    else:
        current_points = len(current_array)
        logging.info(f"Using provided current array with {current_points} points.")


    # data arrays for storing the measured ODMR and frequency data
    odmr_voltages_array = np.zeros(shape=(current_points, len(odmr_ranges), odmr_frequency_points))
    # Frequency array is likely the same for all currents within one range, so shape is (num_ranges, num_points)
    odmr_frequencies_array = np.zeros(shape=(len(odmr_ranges), odmr_frequency_points))

    # -------------------------------------------------------------------------------------------- #
    # MEASUREMENT
    # -------------------------------------------------------------------------------------------- #
    measurement_start_time = time.time()
    logging.info("Starting current sweep measurement...")

    try:
        for current_index, current in enumerate(current_array):
            logging.info(f"Setting current to {current:.4f} A (Point {current_index + 1}/{current_points})")

            # --- NGP Control ---
            try:
                ngp.set_voltage(ngp_active_channel, ngp_voltage)
                ngp.set_current(ngp_active_channel, current)
            except Exception as e:
                logging.error(f"Failed to set NGP current to {current:.4f}A: {e}. Skipping this point.")
                odmr_voltages_array[current_index, :, :].fill(np.nan)
                continue

            logging.debug(f"Waiting {current_settling_time}s for current to settle...")
            time.sleep(current_settling_time)

            # --- ODMR Scans for this current ---
            for odmr_range_index, odmr_range in enumerate(odmr_ranges):
                freq_start, freq_stop = odmr_range
                logging.info(f"  Starting ODMR Scan {odmr_range_index + 1}/{len(odmr_ranges)}: Range [{freq_start/1e9:.4f}, {freq_stop/1e9:.4f}] GHz")

                filename_base = f"current_{current:.4f}A_range_{odmr_range_index}"

                # Get the path relative to the daily Qudi folder for saving via Qudi
                qudi_relative_save_path = get_qudi_relative_save_path(measurement_folder_path, filename_base)
                logging.debug(f"  Qudi save path (relative): {qudi_relative_save_path}")

                # ODMR scan data are stored via Qudi in its data structure
                try:
                    frequencies, odmr_voltages = odmr_remote.take_odmr_scan(
                        filename=qudi_relative_save_path, # Pass relative path
                        min_run_time=run_time_per_current_point,
                        frequency_start=freq_start,
                        frequency_stop=freq_stop,
                        frequency_points=odmr_frequency_points,
                        save_data=True
                    )

                    # Store the returned data, check for NaN/incorrect shape before assignment
                    if isinstance(frequencies, np.ndarray) and frequencies.shape == (odmr_frequency_points,) and \
                       isinstance(odmr_voltages, np.ndarray) and odmr_voltages.shape == (odmr_frequency_points,):

                        if odmr_frequencies_array[odmr_range_index].sum() == 0: # Store frequency axis only once per range if not already stored
                             odmr_frequencies_array[odmr_range_index] = frequencies
                        odmr_voltages_array[current_index, odmr_range_index] = odmr_voltages
                        logging.info(f"  ODMR Scan {odmr_range_index + 1} completed.")
                    else:
                        logging.error(f"  ODMR scan for current {current:.4f}A, range {odmr_range_index} returned unexpected data format/shape. Filling with NaNs.")
                        odmr_voltages_array[current_index, odmr_range_index].fill(np.nan)
                        if odmr_frequencies_array[odmr_range_index].sum() == 0:
                            odmr_frequencies_array[odmr_range_index].fill(np.nan)


                except Exception as e:
                     logging.error(f"  Error during ODMR scan for current {current:.4f}A, range {odmr_range_index}: {e}")
                     odmr_voltages_array[current_index, odmr_range_index].fill(np.nan)
                     if odmr_frequencies_array[odmr_range_index].sum() == 0:
                         odmr_frequencies_array[odmr_range_index].fill(np.nan)
                     continue


    finally: # Ensure NGP and ODMR connection are closed
        if ngp:
            try:
                ngp.output_off()
                logging.info(f"Turning NGP channel {ngp_active_channel} output OFF.")
                ngp.close()
                logging.info('NGP connection closed.')
            except Exception as e:
                logging.warning(f"Could not properly close NGP connection: {e}")
        try:
            odmr_remote.close_connection()
            logging.info('ODMR remote connection closed.')
        except Exception as e:
            logging.warning(f"Could not close ODMR remote connection: {e}")


    # Additionally save raw data aggregated into numpy arrays in the measurement folder
    try:
        freq_save_path = os.path.join(measurement_folder_path, "raw_odmr_frequencies.npy")
        volt_save_path = os.path.join(measurement_folder_path, "raw_odmr_voltages.npy")
        np.save(freq_save_path, odmr_frequencies_array)
        np.save(volt_save_path, odmr_voltages_array)
        logging.info(f"Raw aggregated data saved to {measurement_folder_path}")
    except Exception as e:
        logging.error(f"Error saving raw numpy data: {e}")

    measurement_duration = time.time() - measurement_start_time
    logging.info(f"Current sweep measurement finished. Total duration: {measurement_duration:.2f} seconds.")

    return odmr_frequencies_array, odmr_voltages_array


def fit_odmr_series(odmr_ranges, odmr_frequencies_array, odmr_voltages_array,
                     feature_prominence=0.05,
                     min_feature_height=0.025,
                     feature_fit_range_hz=0.3e6,
                     n_most_prominent_peaks=5):
    """
    Fits hyperfine structure to a series of ODMR measurements.
    (Code from the original script, with minor logging additions and error handling)
    """
    logging.info("Starting ODMR series fitting...")
    # --- Input Validation ---
    if not isinstance(odmr_frequencies_array, np.ndarray) or odmr_frequencies_array.ndim != 2:
        logging.error("odmr_frequencies_array must be a 2D NumPy array (num_ranges, num_freq_points)")
        raise ValueError("odmr_frequencies_array must be a 2D NumPy array")
    if not isinstance(odmr_voltages_array, np.ndarray) or odmr_voltages_array.ndim != 3:
        logging.error("odmr_voltages_array must be a 3D NumPy array (num_series, num_ranges, num_voltage_points)")
        raise ValueError("odmr_voltages_array must be a 3D NumPy array")

    num_series = odmr_voltages_array.shape[0]
    num_ranges = odmr_voltages_array.shape[1]
    num_voltage_points = odmr_voltages_array.shape[2]

    if len(odmr_ranges) != num_ranges:
         msg = (f"Length of odmr_ranges ({len(odmr_ranges)}) does not match "
                f"odmr_voltages_array.shape[1] ({num_ranges})")
         logging.error(msg)
         raise ValueError(msg)
    if odmr_frequencies_array.shape[0] != num_ranges:
        msg = (f"odmr_frequencies_array.shape[0] ({odmr_frequencies_array.shape[0]}) does not match "
               f"odmr_voltages_array.shape[1] ({num_ranges})")
        logging.error(msg)
        raise ValueError(msg)
    if odmr_frequencies_array.shape[1] != num_voltage_points:
         logging.warning(f"odmr_frequencies_array.shape[1] ({odmr_frequencies_array.shape[1]}) does not match "
                         f"odmr_voltages_array.shape[2] ({num_voltage_points}). Check data alignment.")


    # --- Initialization ---
    avg_odmr_pos = np.full(shape=(num_series, num_ranges), fill_value=np.nan)
    uncertainty_avg_odmr_pos = np.full(shape=(num_series, num_ranges), fill_value=np.nan)

    # --- Main Loop ---
    for series_idx in range(num_series):
        for range_idx, odmr_range_descriptor in enumerate(odmr_ranges): # Using enumerate for description
            # Select data for the current series index and range index
            current_frequencies = odmr_frequencies_array[range_idx]
            current_voltages = odmr_voltages_array[series_idx, range_idx]

            # Check if data is all NaN (e.g., due to previous error)
            if np.all(np.isnan(current_voltages)) or np.all(np.isnan(current_frequencies)):
                logging.warning(f"Skipping fit for series {series_idx}, range {range_idx}: Input data contains NaNs.")
                continue # Skip to next iteration, results remain NaN

            try:
                logging.debug(f"Fitting series {series_idx}, range {range_idx}...")
                # Ensure fit_hyperfine can handle the data type (e.g., float64)
                fit_results = fit_hyperfine(current_frequencies.astype(np.float64),
                                            current_voltages.astype(np.float64),
                                            feature_prominence=feature_prominence,
                                            min_feature_height=min_feature_height,
                                            feature_fit_range_hz=feature_fit_range_hz,
                                            n_most_prominent_peaks=n_most_prominent_peaks)

                # --- Process Fit Results ---
                # (Using the robust processing from the original thought process)
                try:
                    # Ensure keys exist before accessing, provide default empty list or list of NaNs
                    peak_positions_raw = fit_results.get("peak_positions [Hz]", [])
                    dip_positions_raw = fit_results.get("dip_positions [Hz]", [])
                    peak_uncertainties_raw = fit_results.get("peak_uncertainties [Hz]", [])
                    dip_uncertainties_raw = fit_results.get("dip_uncertainties [Hz]", [])

                    # Convert None to NaN and ensure numpy array format
                    peak_positions = np.array([p if p is not None else np.nan for p in peak_positions_raw], dtype=float)
                    dip_positions = np.array([d if d is not None else np.nan for d in dip_positions_raw], dtype=float)
                    peak_uncertainties = np.array([p if p is not None else np.nan for p in peak_uncertainties_raw], dtype=float)
                    dip_uncertainties = np.array([d if d is not None else np.nan for d in dip_uncertainties_raw], dtype=float)

                    # Handle cases where fit might return single values instead of lists
                    peak_positions = np.atleast_1d(peak_positions)
                    dip_positions = np.atleast_1d(dip_positions)
                    peak_uncertainties = np.atleast_1d(peak_uncertainties)
                    dip_uncertainties = np.atleast_1d(dip_uncertainties)


                except Exception as e:
                    logging.error(f"Fit Result Access/Conversion Error for series {series_idx}, range {range_idx}: {e}")
                    # Default to NaN arrays if extraction fails
                    peak_positions = np.array([np.nan])
                    dip_positions = np.array([np.nan])
                    peak_uncertainties = np.array([np.nan])
                    dip_uncertainties = np.array([np.nan])


                # --- Calculate Average Position and Uncertainty ---
                all_positions = np.concatenate([peak_positions, dip_positions])
                all_uncertainties_sq = np.concatenate([peak_uncertainties**2, dip_uncertainties**2])

                # Filter out NaNs before calculation
                valid_mask = ~np.isnan(all_positions) & ~np.isnan(all_uncertainties_sq) # Also check uncertainty isn't NaN
                valid_positions = all_positions[valid_mask]
                valid_uncertainties_sq = all_uncertainties_sq[valid_mask]
                num_valid_features = len(valid_positions)


                try:
                    if num_valid_features > 0:
                        # Calculate weighted average if uncertainties are reliable, otherwise use simple mean
                        # Simple mean:
                        avg_odmr_pos[series_idx, range_idx] = np.mean(valid_positions)
                        # Uncertainty of the mean (std dev / sqrt(N)) or propagate errors?
                        # Propagating errors: sqrt(sum(sigma_i^2)) / N
                        sum_var = np.sum(valid_uncertainties_sq)
                        uncertainty_avg_odmr_pos[series_idx, range_idx] = np.sqrt(sum_var) / num_valid_features

                        # # Weighted mean approach (if uncertainties are reliable weights)
                        # weights = 1.0 / valid_uncertainties_sq
                        # avg_odmr_pos[series_idx, range_idx] = np.average(valid_positions, weights=weights)
                        # uncertainty_avg_odmr_pos[series_idx, range_idx] = np.sqrt(1.0 / np.sum(weights))


                        logging.debug(f"  Fit results (series {series_idx}, range {range_idx}): Avg Pos={avg_odmr_pos[series_idx, range_idx]:.2f} Hz, Uncertainty={uncertainty_avg_odmr_pos[series_idx, range_idx]:.2f} Hz ({num_valid_features} features)")
                    else:
                        logging.warning(f"No valid features found for series {series_idx}, range {range_idx}. Setting avg pos to NaN.")
                        # Results remain NaN (set during initialization)

                except FloatingPointError as fe: # Catch potential division by zero if weights are bad
                    logging.error(f"Floating point error calculating avg/uncertainty for series {series_idx}, range {range_idx}: {fe}")
                    avg_odmr_pos[series_idx, range_idx] = np.NAN
                    uncertainty_avg_odmr_pos[series_idx, range_idx] = np.NAN
                except Exception as e:
                     logging.error(f"Error calculating avg/uncertainty for series {series_idx}, range {range_idx}: {e}")
                     avg_odmr_pos[series_idx, range_idx] = np.NAN
                     uncertainty_avg_odmr_pos[series_idx, range_idx] = np.NAN

            except Exception as e:
                logging.exception(f"Hyperfine Fit Error occurred for series {series_idx}, range {range_idx}:")
                avg_odmr_pos[series_idx, range_idx] = np.NAN
                uncertainty_avg_odmr_pos[series_idx, range_idx] = np.NAN

    logging.info("ODMR series fitting finished.")
    # Return results matching the input structure (voltages/frequencies might contain NaNs now)
    return avg_odmr_pos, uncertainty_avg_odmr_pos


# --- UPDATED PLOTTING FUNCTION WITH LINEAR FIT ---
def plot_calibration_curve(current_array, avg_positions, uncertainties, odmr_ranges, save_filepath):
    """
    Plots the average ODMR peak positions vs. current with error bars, fits a linear line to each dataset,
    and saves the plot.

    Args:
        current_array (np.ndarray): 1D array of current values.
        avg_positions (np.ndarray): 2D array (current_points, num_ranges) of average peak positions [Hz].
        uncertainties (np.ndarray): 2D array (current_points, num_ranges) of uncertainties [Hz].
        odmr_ranges (list): List of [start, stop] frequencies for labeling.
        save_filepath (str): Full path including filename (e.g., '/path/to/calibration_curve.pdf') to save the plot.
    """
    logging.info(f"Generating calibration plot with linear fit...")
    try:
        num_ranges = avg_positions.shape[1]
        num_points = len(current_array)

        if avg_positions.shape[0] != num_points or uncertainties.shape[0] != num_points:
            logging.error("Mismatch between length of current_array and results arrays for plotting.")
            raise ValueError("Data array length mismatch for plotting.")
        if avg_positions.shape[1] != len(odmr_ranges) or uncertainties.shape[1] != len(odmr_ranges):
            logging.error("Mismatch between number of ODMR ranges and results arrays for plotting.")
            raise ValueError("ODMR range mismatch for plotting.")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Define markers or colors if needed for multiple ranges
        markers = ['o', 's', '^', 'd', 'v', '<', '>']
        colors = plt.cm.viridis(np.linspace(0, 0.9, num_ranges))  # Use a colormap

        for i in range(num_ranges):
            # Extract data for this range, convert Hz to GHz for better readability on axis
            y_data_ghz = avg_positions[:, i] / 1e9
            y_error_ghz = uncertainties[:, i] / 1e9

            # Create label for legend
            range_label = f"Range {i}: {odmr_ranges[i][0] / 1e9:.3f}-{odmr_ranges[i][1] / 1e9:.3f} GHz"

            # Plot with error bars
            ax.errorbar(
                current_array,
                y_data_ghz,
                yerr=y_error_ghz,
                label=range_label,
                fmt=markers[i % len(markers)],  # Format: marker only (no line)
                capsize=4,  # Add caps to error bars
                markersize=6,
                color=colors[i], alpha=0.7
            )

            # Perform linear fit (weighted by uncertainties if they're significant)
            if np.all(y_error_ghz > 0):
                # Use weighted fit if we have valid uncertainties
                slope, intercept, r_value, p_value, std_err = stats.linregress(current_array, y_data_ghz)
            else:
                # Use standard fit if uncertainties are zeros or invalid
                slope, intercept, r_value, p_value, std_err = stats.linregress(current_array, y_data_ghz)

            # Create smooth x values for plotting the fit line
            x_fit = np.linspace(min(current_array), max(current_array), 100)
            y_fit = slope * x_fit + intercept

            # Plot the fit line
            ax.plot(x_fit, y_fit, '-', color=colors[i], linewidth=1.5, alpha=0.7,
                    label=f"Fit {i}: {slope:.3e} GHz/A")

            # Add fit equation text to the plot
            fit_text = f"Range {i}: y = {slope:.3e} × x + {intercept:.3e}, R² = {r_value ** 2:.4f}"
            logging.info(fit_text)

        ax.set_xlabel("Applied Current [A]")
        ax.set_ylabel("Average ODMR Peak Frequency [GHz]")
        ax.set_title("Current Calibration: ODMR Frequency vs. Current")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Add legend with better layout
        ax.legend(loc='best', fontsize=9)

        # Save the figure
        plt.savefig(save_filepath, format='pdf', bbox_inches='tight')
        logging.info(f"Calibration plot with linear fit saved successfully to: {save_filepath}")

        # Return the fit parameters for potential later use
        fit_results = []
        for i in range(num_ranges):
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                current_array, avg_positions[:, i] / 1e9)
            fit_results.append({
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'std_err': std_err
            })

        plt.close(fig)  # Close the figure to free memory
        return fit_results

    except Exception as e:
        logging.error(f"Failed to generate or save calibration plot with linear fit: {e}")
        # Ensure plot is closed even if saving fails mid-way
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None


def aufnahme_calibration(current_max=0.1, current_points=20, odmr_ranges=[[2.64e9, 2.65e9]], current_settling_time=5):
    """
    Main function to run the current calibration measurement and analysis.
    """
    # --- Measurement Parameters ---
    current_min = 0.001
    run_time_per_odmr_scan = 10
    odmr_frequency_points = 1001 # Number of points in each ODMR scan


    # --- Setup ---
    logging.info("Starting Current Calibration Measurement...")
    start_time = datetime.now()
    logging.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get the specific folder for this measurement run
    measurement_folder_path = get_measurement_save_path()

    # Generate the current array
    current_array = np.linspace(current_min, current_max, current_points)

    # --- MEASUREMENT ---
    # Pass the full path for saving numpy arrays, Qudi saves relatively
    odmr_frequencies_array, odmr_voltages_array = current_measurement(
                                                        measurement_folder_path=measurement_folder_path,
                                                        current_min=current_min,
                                                        current_max=current_max,
                                                        current_points=current_points, # Will be recalculated if current_array is passed
                                                        odmr_ranges=odmr_ranges,
                                                        run_time_per_current_point=run_time_per_odmr_scan,
                                                        current_settling_time=current_settling_time,
                                                        current_array=current_array,
                                                        odmr_frequency_points=odmr_frequency_points)


    # --- ANALYSIS ---
    # Fit the measured data
    avg_odmr_positions, uncertainty_avg_odmr_positions = fit_odmr_series(
        odmr_ranges, odmr_frequencies_array, odmr_voltages_array,
        feature_prominence=0.01,
        min_feature_height=0.005,
        feature_fit_range_hz=0.2e6,
        n_most_prominent_peaks=5
        )


    # --- DATA SAVING (Processed Results) ---
    logging.info("Saving processed calibration data...")

    # Create a dictionary for the pandas DataFrame
    combined_data_dict = {"current[A]": current_array}
    for i in range(len(odmr_ranges)):
        pos_col_name = f"avg_peak_pos_range{i}[Hz]"
        unc_col_name = f"uncertainty_avg_peak_pos_range{i}[Hz]"

        # Ensure the slicing is correct (all series points for range i)
        # Check if the results arrays have the expected dimensions before slicing
        if avg_odmr_positions.shape == (current_points, len(odmr_ranges)) and \
           uncertainty_avg_odmr_positions.shape == (current_points, len(odmr_ranges)):
            combined_data_dict[pos_col_name] = avg_odmr_positions[:, i]
            combined_data_dict[unc_col_name] = uncertainty_avg_odmr_positions[:, i]
        else:
             logging.error(f"Results array shape mismatch. Expected ({current_points}, {len(odmr_ranges)}), "
                           f"Got avg: {avg_odmr_positions.shape}, unc: {uncertainty_avg_odmr_positions.shape}. "
                           f"Cannot populate DataFrame for range {i}.")
             # Fill with NaNs or handle error as appropriate
             combined_data_dict[pos_col_name] = [np.nan] * current_points
             combined_data_dict[unc_col_name] = [np.nan] * current_points


    try:
        data_combined = pd.DataFrame(combined_data_dict)

        # Define a consistent filename for the processed data CSV
        csv_filename = "processed_odmr_vs_current_peaks.csv"
        csv_file_path = os.path.join(measurement_folder_path, csv_filename)

        data_combined.to_csv(csv_file_path, index=False, sep="\t", float_format='%.6e') # Use scientific notation for precision
        logging.info(f"Processed data saved successfully to: {csv_file_path}")

    except Exception as e:
        logging.error(f"Failed to create or save processed data DataFrame/CSV: {e}")


    # --- PLOTTING ---
    # Define the full path for the plot PDF file
    plot_filename = "calibration_curve.pdf"
    plot_file_path = os.path.join(measurement_folder_path, plot_filename)

    # Check if results are valid before plotting
    if isinstance(avg_odmr_positions, np.ndarray) and isinstance(uncertainty_avg_odmr_positions, np.ndarray) \
       and avg_odmr_positions.shape == (current_points, len(odmr_ranges)):
        plot_calibration_curve(
            current_array=current_array,
            avg_positions=avg_odmr_positions,
            uncertainties=uncertainty_avg_odmr_positions,
            odmr_ranges=odmr_ranges,
            save_filepath=plot_file_path
        )
    else:
         logging.warning("Skipping plotting due to invalid or missing analysis results.")


    end_time = datetime.now()
    logging.info(f"Calibration finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total duration: {end_time - start_time}")
    logging.info(f"All data saved within folder: {measurement_folder_path}")


if __name__ == "__main__":
    odmr_ranges = [[2.804e9, 2.832e9]]

    aufnahme_calibration(current_max=0.5, odmr_ranges=odmr_ranges, current_points=20, )

    print('--- Script Execution Done ---')