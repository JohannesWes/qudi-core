# --- START OF FILE live_current_measurement.py ---

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque # Import deque for efficient data handling

from my_software.tools.logging_config import setup_logging
setup_logging(level=logging.INFO)

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # For formatting axes if needed

from my_software.automation.qudi_remote_control import OdmrRemoteControl
from my_software.tools.fitting import fit_hyperfine

try:
    import pyrpl

    PYRPL_AVAILABLE = True
except ImportError:
    logging.warning("Pyrpl library not found. PWM output feature will be disabled.")
    PYRPL_AVAILABLE = False
    pyrpl = None  # Define pyrpl as None if import fails

# --- Configuration ---

CALIBRATION_FILE_PATH = os.path.expanduser('~\\qudi\\Data\\2025\\05\\2025-05-14\\current_calibration\\20250514-105333_Calibration\\processed_odmr_vs_current_peaks.csv') # <--- CHANGE THIS

# Which ODMR range index from the calibration file to use for live tracking?
CALIBRATION_RANGE_INDEX = 0

# ODMR settings for the live measurement (should match the chosen calibration range reasonably)
LIVE_ODMR_RANGE = [2.804e9, 2.83e9] # [frequency_start (Hz), frequency_stop (Hz)]
LIVE_ODMR_POINTS = 1001
LIVE_ODMR_RUN_TIME = 1 # Seconds per scan (adjust for desired sensitivity/speed)
# LIVE_MEASUREMENT_DURATION_S = 60

# Fitting parameters (adjust based on signal quality)
LIVE_FIT_PROMINENCE = 0.01
LIVE_FIT_MIN_HEIGHT = 0.005
LIVE_FIT_RANGE_HZ = 0.2e6
LIVE_FIT_N_PEAKS = 5 # Expected number of features in the range

# --- Live Plot Configuration ---
ENABLE_LIVE_PLOT = True
PLOT_TIME_WINDOW = 60 # Seconds - How much history to show on the x-axis
Y_AXIS_PADDING_FACTOR = 0.1 # 10% padding above max and below min on y-axis

# --- PWM Output Configuration ---
ENABLE_PWM_OUTPUT = False  # Set to True to enable sending data via PWM
RED_PITAYA_HOSTNAME = "10.203.129.28" # <--- CHANGE THIS to your Red Pitaya IP/hostname
PWM_CHANNEL = 2 # Which PWM channel to use (0-3) - Corresponds to expansion P0-P3 if direct output is enabled
PWM_MODE = "normal" # "normal" or "dithered" - Set the desired mode for the channel

# How often to update the current measurement (in seconds)
# This includes ODMR scan time, fitting time, calculation time, and PWM communication time.
UPDATE_INTERVAL = 5 # Seconds - Adjust as needed, ensure it's longer than the total processing time

# --- Data Saving Configuration for Live Trace ---
SAVE_LIVE_TRACE_DATA = True  # Set to True to save the time trace data
# Define QUDI_DATA_ROOT if not already present, or ensure it's correctly defined
if 'QUDI_DATA_ROOT' not in globals(): # Check if already defined (e.g. by other imports)
    QUDI_DATA_ROOT = os.path.expanduser('~/qudi/Data') # Standard Qudi data location
LIVE_TRACE_EXPERIMENT_TYPE = "live_current_traces" # Subfolder for saved traces within the daily Qudi directory

# --- Helper Functions ---

def load_calibration_data(filepath: str, range_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Loads current and frequency data from the calibration CSV file."""
    logging.info(f"Loading calibration data from: {filepath}")
    try:
        df = pd.read_csv(filepath, sep='\t')
        current_col = 'current[A]'
        # Corrected column name format based on calibration script
        freq_col = f'avg_peak_pos_range{range_index}[Hz]'

        if current_col not in df.columns:
            raise ValueError(f"Current column '{current_col}' not found in calibration file.")
        if freq_col not in df.columns:
            raise ValueError(f"Frequency column '{freq_col}' for range {range_index} not found.")

        currents = df[current_col].to_numpy()
        frequencies = df[freq_col].to_numpy()

        # Remove NaN values resulting from failed fits during calibration
        valid_mask = ~np.isnan(currents) & ~np.isnan(frequencies)
        if np.sum(valid_mask) < 2:
            raise ValueError("Not enough valid (non-NaN) data points found in calibration file.")

        logging.info(f"Successfully loaded {np.sum(valid_mask)} valid calibration points for range {range_index}.")
        return currents[valid_mask], frequencies[valid_mask]

    except FileNotFoundError:
        logging.error(f"Calibration file not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading calibration data: {e}")
        raise

def calculate_calibration_slope(currents: np.ndarray, frequencies: np.ndarray) -> float:
    """Calculates the slope (df/dI) from calibration data using linear regression."""
    if len(currents) < 2:
        raise ValueError("Need at least two data points to calculate slope.")

    # Fit a line: frequency = slope * current + intercept
    slope, intercept = np.polyfit(currents, frequencies, 1)

    if not np.isfinite(slope):
         raise ValueError("Linear fit resulted in a non-finite slope (NaN or +/-inf). Check calibration data.")

    logging.info(f"Calculated calibration slope (df/dI): {slope:.4e} Hz/A")
    return slope # Return only the slope (df/dI)

def get_representative_frequency(fit_results: dict) -> float:
    """
    Extracts a single representative frequency from the fit_hyperfine results.
    Uses the average of valid zero-crossing frequencies.
    Returns np.nan if no valid features are found or calculation fails.
    """
    try:
        zc_freqs = fit_results.get("zero_crossing_frequencies [Hz]", np.array([]))
        valid_zc_freqs = zc_freqs[~np.isnan(zc_freqs)]

        if len(valid_zc_freqs) > 0:
            avg_freq = np.mean(valid_zc_freqs)
            logging.debug(f"Found {len(valid_zc_freqs)} valid zero crossings. Average: {avg_freq:.2f} Hz")
            return avg_freq
        else:
            logging.warning("No valid zero-crossings found in fit results.")
            return np.nan

    except Exception as e:
        logging.error(f"Error extracting representative frequency from fit results: {e}")
        return np.nan

def measure_single_odmr_frequency(odmr_remote: OdmrRemoteControl,
                                  odmr_range: list,
                                  odmr_points: int,
                                  odmr_run_time: float,
                                  fit_prominence: float,
                                  fit_min_height: float,
                                  fit_range_hz: float,
                                  fit_n_peaks: int) -> float:
    """Performs a single ODMR scan, fits it, and returns the representative frequency."""
    try:
        logging.debug(f"Starting ODMR scan: Range={odmr_range}, Points={odmr_points}, Time={odmr_run_time}s")
        # We don't need to save intermediate scans for live view
        frequencies, voltages = odmr_remote.take_odmr_scan(
            filename="live_scan_temp", # Qudi might require a filename, even if not saving
            min_run_time=odmr_run_time,
            frequency_start=odmr_range[0],
            frequency_stop=odmr_range[1],
            frequency_points=odmr_points,
            save_data=False # Don't save individual scans here
        )

        if frequencies is None or voltages is None or len(frequencies) != odmr_points or len(voltages) != odmr_points:
             logging.error("ODMR scan returned invalid data.")
             return np.nan

        # Fit the obtained spectrum
        logging.debug("Fitting ODMR spectrum...")
        fit_results = fit_hyperfine(
            frequencies, voltages,
            feature_prominence=fit_prominence,
            min_feature_height=fit_min_height,
            feature_fit_range_hz=fit_range_hz,
            n_most_prominent_peaks=fit_n_peaks,
            plot_result=False, # Don't show intermediate fit plots
            plot_all=False,
            save_result_plot=False,
            filename="C:\\Users\\aj92uwef\\qudi\\Data\\2025\\05\\2025-05-14\\odmr_logic\\bullshit" # This filename seems like a placeholder, ensure it's okay
        )

        # Extract the representative frequency
        representative_freq = get_representative_frequency(fit_results)
        return representative_freq

    except Exception as e:
        logging.error(f"Error during single ODMR measurement or fit: {e}")
        return np.nan

# --- Main Live Measurement Function ---

def live_current_measurement():
    """Runs the continuous current measurement loop."""
    logging.info("--- Starting Live Current Measurement ---")

    # --- Plot Initialization (if enabled) ---
    fig, ax, line = None, None, None
    time_data = None
    current_data = None
    plotting_active = False  # Use a local flag to track if plotting is *actually* active
    start_time = time.time()  # Reference time for relative x-axis (initialize even if plot fails)

    if ENABLE_LIVE_PLOT:
        try:
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots(figsize=(10, 5))
            line, = ax.plot([], [], 'bo-', markersize=4)  # Blue dots connected by lines
            ax.set_xlabel("Time (Relative Seconds)")
            ax.set_ylabel("Measured Current Change (mA)")
            ax.set_title("Live Current Measurement")
            ax.grid(True)
            time_data = deque()
            current_data = deque()
            logging.info(f"Live plot initialized. Window size: {PLOT_TIME_WINDOW}s")
            ax.set_xlim(0, PLOT_TIME_WINDOW)
            ax.set_ylim(-1, 1)  # Initial guess for y-limits
            fig.show()  # Show the plot window non-blockingly
            plotting_active = True  # Set local flag to True on success
        except Exception as e:
            logging.error(f"Failed to initialize live plot: {e}")
            # plotting_active remains False

    # --- Data Storage for Saving Entire Trace ---
    all_abs_timestamps_for_save = []
    all_relative_times_for_save = []  # Relative to actual live measurement start
    all_current_values_for_save = []
    all_frequencies_for_save = []

    # --- Measurement Initialization ---
    odmr_remote = None
    rp = None  # Pyrpl Red Pitaya object
    pyrpl_instance = None  # Pyrpl main instance

    try:
        # 1. Load Calibration
        cal_currents, cal_frequencies = load_calibration_data(CALIBRATION_FILE_PATH, CALIBRATION_RANGE_INDEX)

        # 2. Calculate Slope
        df_dI = calculate_calibration_slope(cal_currents, cal_frequencies)  # Slope in Hz/A

        # 3. Initialize ODMR Control
        odmr_remote = OdmrRemoteControl()
        logging.info("ODMR Remote Control initialized.")

        # 4. Initialize Pyrpl and Red Pitaya (if enabled)
        if ENABLE_PWM_OUTPUT:
            if not PYRPL_AVAILABLE:
                logging.error("Cannot enable PWM output: Pyrpl library is not installed or failed to import.")
            else:
                try:
                    logging.info(f"Initializing Pyrpl for Red Pitaya at {RED_PITAYA_HOSTNAME}...")
                    pyrpl_instance = pyrpl.Pyrpl(config="", hostname=RED_PITAYA_HOSTNAME, reloadfpga=False,
                                                 reloadserver=False)
                    rp = pyrpl_instance.redpitaya
                    logging.info("Pyrpl Red Pitaya connection successful.")
                    logging.info(f"Configuring PWM Channel {PWM_CHANNEL} for direct output...")
                    rp.hk.enable_pwm_direct_output(True)
                    logging.info(f"Setting PWM Channel {PWM_CHANNEL} mode to '{PWM_MODE}'...")
                    rp.ams.set_pwm_mode(PWM_CHANNEL, PWM_MODE)
                    logging.info(f"Setting initial PWM state (Error Mode) on channel {PWM_CHANNEL}")
                    rp.ams.set_current_pwm(PWM_CHANNEL, 0, error_state=True)
                except Exception as e:
                    logging.error(f"Failed to initialize Pyrpl or configure PWM: {e}")
                    rp = None

    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        logging.error("Cannot start live measurement. Exiting.")
        if odmr_remote:
            try:
                odmr_remote.close_connection()
            except:
                pass
        if fig:
            try:
                plt.close(fig)
            except:
                pass
        return

    # --- Zero Measurement ---
    logging.info("Performing initial 'zero' measurement to establish reference frequency...")
    logging.info("Ensure the current is at the desired reference state (e.g., 0 A).")
    time.sleep(2)  # Give user time to read

    f_zero = np.nan
    attempts = 3
    for i in range(attempts):
        f_zero = measure_single_odmr_frequency(
            odmr_remote, LIVE_ODMR_RANGE, LIVE_ODMR_POINTS, LIVE_ODMR_RUN_TIME,
            LIVE_FIT_PROMINENCE, LIVE_FIT_MIN_HEIGHT, LIVE_FIT_RANGE_HZ, LIVE_FIT_N_PEAKS
        )
        if not np.isnan(f_zero):
            logging.info(f"Reference frequency (f_zero) established: {f_zero / 1e6:.6f} MHz")
            break
        else:
            logging.warning(f"Attempt {i + 1}/{attempts} failed to get reference frequency. Retrying...")
            time.sleep(1)
    else:  # Loop finished without break
        logging.error("Failed to establish reference frequency after multiple attempts. Exiting.")
        if odmr_remote:
            try:
                odmr_remote.close_connection()
            except:
                pass
        if rp:
            try:
                logging.info(f"Resetting PWM state (Error Mode) on channel {PWM_CHANNEL} before exiting.")
                rp.ams.set_current_pwm(PWM_CHANNEL, 0, error_state=True)
            except Exception as pwm_e:
                logging.warning(f"Could not reset PWM state on exit: {pwm_e}")
        if fig:
            try:
                plt.close(fig)
            except:
                pass
        return

    # --- Live Measurement Loop ---
    live_measurement_actual_start_time = time.time()  # Start time for duration control and saved relative time

    logging.info("Starting continuous measurement loop. Press Ctrl+C to stop.")
    # --- REMOVED DURATION-BASED LOGGING ---
    # - if LIVE_MEASUREMENT_DURATION_S and LIVE_MEASUREMENT_DURATION_S > 0: # This variable is removed
    # -     logging.info(f"Measurement will run for approximately {LIVE_MEASUREMENT_DURATION_S} seconds.")

    try:
        while True:  # This loop will now run indefinitely until Ctrl+C is pressed
            loop_iteration_start_time = time.time()
            current_datetime_obj = datetime.now()

            # --- REMOVED DURATION-BASED STOPPING LOGIC ---
            # # Check for duration limit
            # if LIVE_MEASUREMENT_DURATION_S and LIVE_MEASUREMENT_DURATION_S > 0: # This variable is removed
            #     elapsed_measurement_time = time.time() - live_measurement_actual_start_time
            #     if elapsed_measurement_time >= LIVE_MEASUREMENT_DURATION_S:
            #         logging.info(f"Live measurement duration ({LIVE_MEASUREMENT_DURATION_S}s) reached. Stopping.")
            #         break # This break is removed; loop now only stops via Ctrl+C or error

            # Measure current frequency
            f_current = measure_single_odmr_frequency(
                odmr_remote, LIVE_ODMR_RANGE, LIVE_ODMR_POINTS, LIVE_ODMR_RUN_TIME,
                LIVE_FIT_PROMINENCE, LIVE_FIT_MIN_HEIGHT, LIVE_FIT_RANGE_HZ, LIVE_FIT_N_PEAKS
            )

            delta_I_mA = np.nan
            measurement_valid = not np.isnan(f_current)

            if measurement_valid:
                delta_f = f_current - f_zero
                delta_I = delta_f / df_dI
                delta_I_mA = delta_I * 1000

                timestamp_str = current_datetime_obj.strftime('%H:%M:%S')
                logging.info(f"[{timestamp_str}] f_curr: {f_current / 1e6:,.6f} MHz | "
                             f"Delta_f: {delta_f / 1e3:,.3f} kHz | "
                             f"Delta_I: {delta_I_mA:,.3f} mA")
            else:
                logging.warning("Failed to get valid frequency measurement in this cycle.")

            # --- Store data for saving ---
            time_relative_to_live_start = loop_iteration_start_time - live_measurement_actual_start_time
            all_abs_timestamps_for_save.append(
                current_datetime_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            all_relative_times_for_save.append(time_relative_to_live_start)
            all_current_values_for_save.append(delta_I_mA)
            all_frequencies_for_save.append(f_current)

            # --- Send PWM Data (if enabled and Red Pitaya is connected) ---
            if ENABLE_PWM_OUTPUT and rp:
                try:
                    error_state = not measurement_valid
                    current_to_send = delta_I_mA if measurement_valid else 0
                    rp.ams.set_current_pwm(PWM_CHANNEL, current_to_send, error_state=error_state)
                    if error_state:
                        logging.info(f"Sent PWM Channel {PWM_CHANNEL} to Error State (30 Hz)")
                    else:
                        pwm_freq = 10 if abs(current_to_send) <= 250 else 20
                        logging.debug(f"Sent PWM Channel {PWM_CHANNEL}: {current_to_send:.2f} mA (Freq: {pwm_freq} Hz)")
                except Exception as pwm_e:
                    logging.warning(f"Failed to send PWM data to Red Pitaya: {pwm_e}")

            # --- Update Live Plot (if active) ---
            if plotting_active and fig and ax and line:
                current_time_rel_for_plot = loop_iteration_start_time - start_time
                time_data.append(current_time_rel_for_plot)
                current_data.append(delta_I_mA)

                while time_data and time_data[0] < current_time_rel_for_plot - PLOT_TIME_WINDOW:
                    time_data.popleft()
                    current_data.popleft()

                if time_data:
                    line.set_data(list(time_data), list(current_data))
                    x_max_plot = current_time_rel_for_plot if current_time_rel_for_plot > PLOT_TIME_WINDOW else PLOT_TIME_WINDOW
                    x_min_plot = x_max_plot - PLOT_TIME_WINDOW
                    ax.set_xlim(x_min_plot, x_max_plot)

                    valid_current_plot_data = [val for val in current_data if not np.isnan(val)]
                    if valid_current_plot_data:
                        min_curr_plot = min(valid_current_plot_data)
                        max_curr_plot = max(valid_current_plot_data)
                    elif current_data:
                        min_curr_plot, max_curr_plot = -1, 1
                    else:
                        min_curr_plot, max_curr_plot = -1, 1

                    if min_curr_plot == max_curr_plot:
                        y_range_plot = max(abs(min_curr_plot * Y_AXIS_PADDING_FACTOR * 2), 0.1)
                        min_curr_plot -= y_range_plot / 2
                        max_curr_plot += y_range_plot / 2
                    else:
                        y_range_plot = max_curr_plot - min_curr_plot
                        padding_plot = y_range_plot * Y_AXIS_PADDING_FACTOR
                        min_curr_plot -= padding_plot
                        max_curr_plot += padding_plot
                    ax.set_ylim(min_curr_plot, max_curr_plot)

                    try:
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                    except Exception as plot_e:
                        logging.warning(f"Could not update plot: {plot_e}")

            # The following block for waiting until UPDATE_INTERVAL is currently commented out in your original script.
            # If you want to control the update rate, you would uncomment this.
            # Otherwise, the loop runs as fast as possible.
            # elapsed_this_loop = time.time() - loop_iteration_start_time
            # sleep_time = UPDATE_INTERVAL - elapsed_this_loop
            # if sleep_time > 0:
            #     time.sleep(sleep_time)
            # else:
            #     logging.warning(f"Measurement cycle took longer ({elapsed_this_loop:.2f}s) than update interval ({UPDATE_INTERVAL}s).")
            #     time.sleep(0.01)


    except KeyboardInterrupt:
        logging.info("Ctrl+C received. Stopping live measurement.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        # This block will execute when the loop is exited,
        # either by KeyboardInterrupt or an error.
        # Data saving logic is here and will use all data collected.
        logging.info("Cleaning up resources...")
        if odmr_remote:
            try:
                odmr_remote.close_connection()
                logging.info("ODMR remote connection closed.")
            except Exception as e:
                logging.warning(f"Could not close ODMR remote connection: {e}")

        if ENABLE_PWM_OUTPUT and rp:
            try:
                logging.info(f"Setting final PWM state (Error Mode) on channel {PWM_CHANNEL}")
                rp.ams.set_current_pwm(PWM_CHANNEL, 0, error_state=True)
            except Exception as e:
                logging.warning(f"Could not reset PWM state during cleanup: {e}")

        if plotting_active and fig:
            try:
                plt.ioff()
                plt.close(fig)
                logging.info("Live plot window closed.")
            except Exception as e:
                logging.warning(f"Error closing live plot window: {e}")

        # --- Save Live Trace Data and Plot ---
        # This part is crucial: it saves the data collected in the lists.
        if SAVE_LIVE_TRACE_DATA and all_relative_times_for_save:
            logging.info("Saving live current trace data and plot...")
            save_folder = get_live_trace_save_folder()  # This function is defined below
            if save_folder:
                folder_timestamp = os.path.basename(save_folder).split('_')[0]

                try:
                    trace_df = pd.DataFrame({
                        'absolute_timestamp_ms': all_abs_timestamps_for_save,
                        'time_relative_s': all_relative_times_for_save,
                        'f_current_Hz': all_frequencies_for_save,
                        'delta_I_mA': all_current_values_for_save
                    })
                    csv_filename = f"{folder_timestamp}_live_current_trace.csv"
                    full_csv_save_path = os.path.join(save_folder, csv_filename)
                    trace_df.to_csv(full_csv_save_path, index=False, sep='\t', float_format='%.6e')
                    logging.info(f"Live trace data saved to: {full_csv_save_path}")
                except Exception as e_save_csv:
                    logging.error(f"Failed to save live trace CSV data: {e_save_csv}")

                try:
                    final_fig, final_ax = plt.subplots(figsize=(12, 6))
                    final_ax.plot(all_relative_times_for_save, all_current_values_for_save, 'b.-', markersize=5,
                                  linewidth=1)
                    final_ax.set_xlabel("Time (Relative Seconds from Measurement Start)")
                    final_ax.set_ylabel("Measured Current Change (mA)")
                    final_ax.set_title(f"Full Live Current Time Trace ({folder_timestamp})")
                    final_ax.grid(True)
                    valid_currents_for_plot = [val for val in all_current_values_for_save if not np.isnan(val)]
                    if valid_currents_for_plot:
                        min_val = min(valid_currents_for_plot)
                        max_val = max(valid_currents_for_plot)
                        data_range = max_val - min_val
                        if data_range == 0:
                            padding = max(abs(min_val * 0.1), 0.1)
                        else:
                            padding = data_range * Y_AXIS_PADDING_FACTOR
                        final_ax.set_ylim(min_val - padding, max_val + padding)
                    plot_filename = f"{folder_timestamp}_live_current_trace_plot.pdf"
                    full_plot_save_path = os.path.join(save_folder, plot_filename)
                    final_fig.savefig(full_plot_save_path, bbox_inches='tight')
                    logging.info(f"Full time trace plot saved to: {full_plot_save_path}")
                    plt.close(final_fig)
                except Exception as e_save_plot:
                    logging.error(f"Failed to save full time trace plot: {e_save_plot}")
                    if 'final_fig' in locals() and plt.fignum_exists(final_fig.number):
                        plt.close(final_fig)
            else:
                logging.error("Could not get or create save folder for live trace data and plot. Data/Plot not saved.")
        elif SAVE_LIVE_TRACE_DATA:
            logging.info("No live trace data was collected to save or plot.")

    logging.info("--- Live Current Measurement Finished ---")


def get_live_trace_save_folder():
    """
    Determines the save path for the live current trace data based on Qudi's convention.
    Creates the necessary directories if they don't exist.
    Returns the path to the specific measurement run folder.
    Example: ~/qudi/Data/YYYY/MM/YYYY-MM-DD/live_current_traces/YYYYMMDD-HHMMSS_LiveTrace
    """
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    year_str = now.strftime('%Y')
    month_str = now.strftime('%m')
    timestamp_str = now.strftime('%Y%m%d-%H%M%S')

    daily_data_path = os.path.join(QUDI_DATA_ROOT, year_str, month_str, date_str)
    experiment_base_path = os.path.join(daily_data_path, LIVE_TRACE_EXPERIMENT_TYPE)
    trace_folder_name = f"{timestamp_str}_LiveTrace"
    trace_folder_path = os.path.join(experiment_base_path, trace_folder_name)

    try:
        os.makedirs(trace_folder_path, exist_ok=True)
        logging.info(f"Created live trace data directory: {trace_folder_path}")
        return trace_folder_path
    except OSError as e:
        logging.error(f"Failed to create directory {trace_folder_path}: {e}")
        return None

if __name__ == "__main__":
    if "YYYY/MM/YYYY-MM-DD" in CALIBRATION_FILE_PATH or "YYYYMMDD-HHMMSS" in CALIBRATION_FILE_PATH:
         print("\n" + "*"*60)
         print("ERROR: Please update the 'CALIBRATION_FILE_PATH' variable")
         print("       in the live_current_measurement.py script!")
         print("*"*60 + "\n")
    elif ENABLE_PWM_OUTPUT and ("your_ip" in RED_PITAYA_HOSTNAME or RED_PITAYA_HOSTNAME == ""):
         print("\n" + "*"*60)
         print("ERROR: Please update the 'RED_PITAYA_HOSTNAME' variable")
         print("       in the live_current_measurement.py script!")
         print("*"*60 + "\n")
    else:
         live_current_measurement()

    print('--- Script Execution Done ---')
# --- END OF FILE live_current_measurement.py ---