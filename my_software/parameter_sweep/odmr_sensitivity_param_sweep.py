import numpy as np
import time
import matplotlib
import pandas as pd
import os
import json
import winsound
import traceback
import logging
from datetime import datetime
from itertools import product

matplotlib.use("Qt5Agg")

from qudi_remote_control import OdmrRemoteControl
from my_software.tools.fitting import fit_hyperfine
from my_software.sensitivity_msmt.sensitivity_measurement import LockInAmp, SensitivityMeasurement
from my_software.sensitivity_msmt.auswertung.sensitivity_auswertung_modular import (
    magnetic_field_from_voltages,
    plot_magnetic_field_time_traces,
    plot_asds,
)


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def create_measurement_folder(base_foldername):
    """Creates a timestamped folder for the measurement."""
    timenow = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folder_path = os.path.join(base_foldername, timenow)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def parameter_sweep(odmr_power_dbm_array=np.array([-20]),
                    fm_modulation_frequency_array=np.array([16e3]),
                    fm_deviation_khz_array=np.array([300.0]),
                    odmr_range=(2.86e9, 2.88e9), odmr_frequency_points=1001,
                    single_odmr_run_time=60,
                    min_fit_amplitude=0.01, min_feature_height=0.003, n_most_prominent_peaks=3,
                    which_zc=2, n_time_traces=32, data_rate=1000,
                    loop_order=None, include_off_resonant_sensitivity=False):
    """
    Performs a parameter sweep for ODMR sensitivity using a Red Pitaya-based microwave source.

    This script iterates over microwave power, FM modulation frequency, and FM deviation,
    performing an ODMR scan and sensitivity measurement for each parameter combination.

    Args:
        odmr_power_dbm_array (array-like): Array of ODMR power levels in dBm.
        fm_modulation_frequency_array (array-like): Array of FM modulation frequencies in Hz.
        fm_deviation_khz_array (array-like): Array of FM frequency deviations in kHz.
        odmr_range (tuple): Frequency range for ODMR scan (start_hz, stop_hz).
        odmr_frequency_points (int): Number of frequency points for ODMR scan.
        single_odmr_run_time (int): Runtime for a single ODMR scan in seconds.
        min_fit_amplitude (float): Minimum amplitude for feature detection in fit.
        min_feature_height (float): Minimum feature height for fit.
        n_most_prominent_peaks (int): Number of prominent peaks to consider in fit.
        which_zc (int): Which zero-crossing to use from the fit result (0-indexed).
        n_time_traces (int): Number of time traces for sensitivity measurement.
        data_rate (int): Data rate for data acquisition in Hz.
        loop_order (list): Specifies the iteration order. Keys: ["power", "f_mod", "f_dev"].
        include_off_resonant_sensitivity (bool): If True, measure off-resonant sensitivity.

    Returns:
        str: Name of the folder where data is saved.
    """
    if loop_order is None:
        loop_order = ["power", "f_dev", "f_mod"]

    param_dict = {
        "power": odmr_power_dbm_array,
        "f_mod": fm_modulation_frequency_array,
        "f_dev": fm_deviation_khz_array
    }

    for param in loop_order:
        if param not in param_dict:
            raise ValueError(f"Parameter '{param}' in loop_order not found in param_dict keys.")

    folder_name = create_measurement_folder("measurements/RedPitaya_Parameter_Sweep")
    logging.info(f"Data will be saved in: {folder_name}")

    odmr_remote = OdmrRemoteControl()
    lia = LockInAmp()

    num_combinations = len(list(product(*param_dict.values())))
    results_list = []
    min_sensitivity_so_far = np.inf
    best_params_so_far = {}

    # Create the list of parameter combinations to iterate over
    # This makes tracking previous parameters easier
    product_params = list(product(*[param_dict[p] for p in loop_order]))

    for idx, combination in enumerate(product_params):
        param_values = dict(zip(loop_order, combination))
        power_val = param_values["power"]
        f_mod_val = param_values["f_mod"]
        f_dev_val = param_values["f_dev"]

        current_sensitivity = np.nan

        try:
            print(f"\n--- Starting Measurement {idx + 1}/{num_combinations} ---")
            print(f"Parameters: Power={power_val:.2f}dBm, f_mod={f_mod_val / 1e3:.1f}kHz, f_dev={f_dev_val:.1f}kHz")

            # --- Configure Microwave Source ---
            # This is the main change: instead of setting up an OPX, we configure
            # the Qudi microwave module (which controls the Red Pitaya) remotely.
            # We assume a 3-tone signal ('triple' mode) as this is typical.
            odmr_remote.configure_mw_source(
                power_dbm=power_val,
                f_mod_hz=f_mod_val,
                f_dev_khz=f_dev_val,
                multi_freq_mode='triple'
            )

            # --- Thermal Stabilization ---
            # Wait if power was changed, as this can cause thermal drifts.
            if idx > 0:
                previous_params = dict(zip(loop_order, product_params[idx - 1]))
                if previous_params["power"] != power_val:
                    sleep_timer = 180
                    print(f"Microwave power changed ({previous_params['power']:.2f} -> {power_val:.2f} dBm). "
                          f"Waiting {sleep_timer}s for thermal stabilization...")
                    time.sleep(sleep_timer)

            filename_prefix = os.path.join(folder_name,
                                           f"P_{power_val:.2f}dBm_fmod_{f_mod_val / 1e3:.1f}k_fdev_{f_dev_val:.1f}k_timenow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            # --- 1) Take ODMR Scan ---
            print("Taking ODMR Scan...")
            frequencies, odmr_voltages = odmr_remote.take_single_odmr_scan(
                filename=filename_prefix + "_ODMR",
                run_time=single_odmr_run_time,
                frequency_start=odmr_range[0],
                frequency_stop=odmr_range[1],
                frequency_points=odmr_frequency_points,
                data_rate=data_rate,
                save_data=True
            )
            # Normalize by subtracting the mean
            odmr_voltages = odmr_voltages - np.mean(odmr_voltages)

            # --- 2) Fit ODMR Data ---
            print("Fitting ODMR data...")
            fit_result = fit_hyperfine(frequencies, odmr_voltages, feature_prominence=min_fit_amplitude,
                                       n_most_prominent_peaks=n_most_prominent_peaks, plot_result=False,
                                       save_result_plot=True,
                                       min_feature_height=min_feature_height, filename=filename_prefix)

            if fit_result is None or which_zc >= len(fit_result.get("zero_crossing_frequencies [Hz]", [])):
                print(f"Fit failed or did not find {which_zc + 1} zero crossings. Skipping point.")
                result_row = {"power_dbm": power_val, "f_mod_hz": f_mod_val, "f_dev_khz": f_dev_val,
                              "sensitivity_nT_rtHz": np.nan}
                results_list.append(result_row)
                continue

            # --- 3) Set CW Frequency for Sensitivity Measurement ---
            cw_frequency = fit_result["zero_crossing_frequencies [Hz]"][which_zc]
            # Note: The power for CW mode is set on the ODMR logic module, not directly on the MW source here.
            # This is because the sensitivity measurement is a CW experiment.
            odmr_remote.set_cw_parameters(frequency=float(cw_frequency), power=float(power_val))
            odmr_remote.toggle_cw_output(True)

            # --- 4) Run Sensitivity Measurement ---
            print(f"Running on-resonant sensitivity measurement at {cw_frequency / 1e9:.4f} GHz...")
            sensitivity = SensitivityMeasurement(lia)
            sensitivity_result = sensitivity.run_measurement(
                filename_prefix=filename_prefix + "_sensitivity_ON-resonant",
                n_time_traces=n_time_traces,
                plot_data=False, save_raw_data=False, save_metadata=True, timeout=5000)

            # --- 5) Optional: Off-Resonant Measurement ---
            if include_off_resonant_sensitivity:
                off_resonant_cw_frequency = cw_frequency + 30e6  # 20 MHz offset
                print(f"Running off-resonant sensitivity measurement at {off_resonant_cw_frequency / 1e9:.4f} GHz...")
                odmr_remote.toggle_cw_output(False) # turn off CW, to set new parameters
                odmr_remote.set_cw_parameters(frequency=float(off_resonant_cw_frequency), power=float(power_val))
                odmr_remote.toggle_cw_output(True)
                sensitivity_result_off = sensitivity.run_measurement(
                    filename_prefix=filename_prefix + "_sensitivity_OFF-resonant",
                    n_time_traces=n_time_traces,
                    plot_data=False, save_raw_data=False, save_metadata=True, timeout=5000)

            odmr_remote.toggle_cw_output(False)

            # --- 6) Analyze Sensitivity Data ---
            slope = fit_result["zero_crossing_slopes [V/Hz]"][which_zc]

            # On-resonant analysis
            meta = sensitivity_result["metadata"]
            b_noise_trace = magnetic_field_from_voltages(
                sensitivity_result["x_value [V]"], slope,
                scaling_factor=meta["aux_0_scaling"])

            plot_magnetic_field_time_traces(b_noise_trace, sensitivity_result["times [s]"],
                                            meta["sampling_rate [Hz]"], meta["duration [s]"], save_fig=True,
                                            filename_prefix=filename_prefix + "_ON-resonant")


            asd_result = plot_asds(b_noise_trace, meta["sampling_rate [Hz]"], meta["duration [s]"],
                                   meta["filter_3db_freq [Hz]"],
                                   filename_prefix=filename_prefix + "_ON-resonant", save_data=True, save_fig=True)
            current_sensitivity = asd_result["sensitivity"]

            # Off-resonant analysis
            sens_off, sens_std_off = np.nan, np.nan
            if include_off_resonant_sensitivity:
                meta_off = sensitivity_result_off["metadata"]
                b_noise_trace_off = magnetic_field_from_voltages(
                    sensitivity_result_off["x_value [V]"], slope,
                    scaling_factor=meta_off["aux_0_scaling"])

                plot_magnetic_field_time_traces(b_noise_trace_off, sensitivity_result_off["times [s]"],
                                                meta_off["sampling_rate [Hz]"], meta_off["duration [s]"], save_fig=True,
                                                filename_prefix=filename_prefix + "_OFF-resonant")

                asd_result_off = plot_asds(b_noise_trace_off, meta_off["sampling_rate [Hz]"], meta_off["duration [s]"],
                                           meta_off["filter_3db_freq [Hz]"],
                                           filename_prefix=filename_prefix + "_OFF-resonant", save_data=True,
                                           save_fig=True)
                sens_off = asd_result_off["sensitivity"]
                sens_std_off = asd_result_off["sensitivity_std"]

            # --- 7) Store and Report Results ---
            result_row = {
                "power_dbm": power_val,
                "f_mod_hz": f_mod_val,
                "f_dev_khz": f_dev_val,
                "linewidth_hz": fit_result["linewidths [Hz]"][which_zc],
                "zc_slope_V_per_Hz": slope,
                "sensitivity_nT_rtHz": current_sensitivity,
                "sensitivity_std": asd_result["sensitivity_std"],
                "sensitivity_off_resonant_nT_rtHz": sens_off,
                "sensitivity_std_off_resonant": sens_std_off,
            }
            results_list.append(result_row)

            print("-" * 20 + " Sensitivity Result " + "-" * 20)
            print(f"  Current Sensitivity: {current_sensitivity:.3f} nT/sqrt(Hz)")
            if include_off_resonant_sensitivity:
                print(f"  Off-Resonant Sensitivity: {sens_off:.3f} nT/sqrt(Hz)")

            if not np.isnan(current_sensitivity) and current_sensitivity < min_sensitivity_so_far:
                min_sensitivity_so_far = current_sensitivity
                best_params_so_far = param_values
                print("  >>> New Minimum Sensitivity Found! <<<")

            if best_params_so_far:
                print(f"  Best Sensitivity So Far: {min_sensitivity_so_far:.3f} nT/sqrt(Hz) "
                      f"(@ P={best_params_so_far['power']:.2f}dBm, "
                      f"f_mod={best_params_so_far['f_mod'] / 1e3:.1f}kHz, "
                      f"f_dev={best_params_so_far['f_dev']:.1f}kHz)")
            print("-" * (40 + len(" Sensitivity Result ")))

        except Exception as e:
            print(f"!!!!!!!! ERROR during measurement {idx + 1} !!!!!!!!")
            print(f"Parameters: P={power_val:.2f}dBm, f_mod={f_mod_val / 1e3:.1f}kHz, f_dev={f_dev_val:.1f}kHz")
            traceback.print_exc()
            result_row = {"power_dbm": power_val, "f_mod_hz": f_mod_val, "f_dev_khz": f_dev_val,
                          "sensitivity_nT_rtHz": np.nan}
            results_list.append(result_row)
            print(f"!!!!!!!! Skipping rest of measurement {idx + 1} due to error. !!!!!!!!")

    # --- Sweep Finished: Save All Results ---
    print("\n--- Sweep Finished ---")
    print("Saving summary results...")
    result_df = pd.DataFrame(results_list)
    result_df.to_csv(os.path.join(folder_name, "parameter_sweep_summary.csv"), sep="\t", index=False, na_rep='NaN')

    if best_params_so_far:
        print("\n--- Best Result Found During Sweep ---")
        print(f"  Minimum Sensitivity: {min_sensitivity_so_far:.3f} nT/sqrt(Hz)")
        print(f"  Parameters: Power={best_params_so_far['power']:.2f}dBm, "
              f"f_mod={best_params_so_far['f_mod'] / 1e3:.1f}kHz, "
              f"f_dev={best_params_so_far['f_dev']:.1f}kHz")
    else:
        print("\nNo successful sensitivity measurements were completed.")

    return folder_name


if __name__ == "__main__":
    setup_logging()
    start_time = time.time()

    # --- Define Sweep Parameters ---
    # These arrays define the parameter space to explore.

    # Total power of the microwave signal in dBm - linear scale in actual power
    min_dbm, max_dbm, num_points = -25, -18, 15
    odmr_power_dbm_array = 10 * np.log10(np.linspace(10 ** (min_dbm / 10), 10 ** (max_dbm / 10), num_points))
#     odmr_power_dbm_array = np.array([-20, -12.5, -20, -12.5, -20, -12.5, -20, -12.5, -20, -12.5, -12.5, -12.5, -12.5, -12.5, -12.5, -20, -20, -20, -20, -20, -20])
# #     odmr_power_dbm_array = np.concatenate([
# #     np.full(5, -12.5),
# #     np.full(5, -20),
# #     np.full(5, -12.5),
# #     np.full(5, -20)
# # ])



    # FM modulation frequency in Hz.
    fm_modulation_frequency_array = np.linspace(16e3, 25e3, 1)

    # FM frequency deviation in kHz.
    fm_deviation_khz_array = np.linspace(500, 600, 1)

    # Define the order of the loops. The outer loop changes least frequently.
    # Good for thermal stability if 'power' is the outermost loop.
    custom_loop_order = ["power", "f_mod", "f_dev"]

    # --- Define Measurement Settings ---
    # These settings are constant for the entire sweep.

    # ODMR scan settings
    odmr_range = (2.748e9, 2.766e9)  # Hz
    single_odmr_runtime = 10  # seconds
    odmr_frequency_points = 1001

    # Data acquisition settings
    data_rate = 1000  # Hz
    n_time_traces = 16

    # Fitting settings
    which_zc = 2  # Use the 3rd zero-crossing (m_s=0 -> m_s=-1 transition)
    n_most_prominent_peaks = 5
    min_fit_amplitude = 0.005
    min_feature_height = 0.003

    include_off_resonant_sensitivity = True
    laser_power_mW = 1750 * (0.11/0.5)  # For metadata

    # --- Run the Sweep ---
    folder_name = parameter_sweep(
        odmr_power_dbm_array=odmr_power_dbm_array,
        fm_modulation_frequency_array=fm_modulation_frequency_array,
        fm_deviation_khz_array=fm_deviation_khz_array,
        odmr_range=odmr_range,
        single_odmr_run_time=single_odmr_runtime,
        odmr_frequency_points=odmr_frequency_points,
        data_rate=data_rate,
        n_time_traces=n_time_traces,
        min_fit_amplitude=min_fit_amplitude,
        min_feature_height=min_feature_height,
        n_most_prominent_peaks=n_most_prominent_peaks,
        which_zc=which_zc,
        loop_order=custom_loop_order,
        include_off_resonant_sensitivity=include_off_resonant_sensitivity
    )

    # --- Save Metadata ---
    total_runtime_min = (time.time() - start_time) / 60
    print(f"Total measurement time: {total_runtime_min:.2f} minutes ({total_runtime_min / 60:.2f} hours).")

    metadata = {
        "laser_power [mW]": laser_power_mW,
        "n_time_traces": n_time_traces,
        "which_zc": which_zc,
        "single_odmr_runtime [s]": single_odmr_runtime,
        "total_runtime [min]": round(total_runtime_min, 1),
        "loop_order": custom_loop_order
    }
    with open(os.path.join(folder_name, "sweep_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # --- Notify User ---
    winsound.Beep(frequency=500, duration=1000)
    print("Parameter sweep complete!")