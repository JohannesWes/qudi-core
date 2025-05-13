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

from my_software.automation.qudi_remote_control import OdmrRemoteControl
from my_software.tools.fitting import fit_hyperfine
from my_software.opx_tools.frequency_modulation.FM_working_oscillator_RF_mixer_hyperfine import FM_setup

from my_software.sensitivity_msmt.sensitivity_measurement import LockInAmp, SensitivityMeasurement
from my_software.sensitivity_msmt.auswertung.sensitivity_auswertung_modular import *

def folder(foldername):
    timenow = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folder_path = os.path.join(foldername)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def load_calibration_data():
    calibration_df = pd.read_csv("..\opx_tools\IQ_calibration\calibration_2024-08-02-10-33-57.csv", sep="\t",
                                 index_col=0)
    return calibration_df


def parameter_sweep(OPX_LO_voltage_array=np.array([0.1]), OPX_IF_voltage_array=np.array([0.1]),
                    f_mod_array=np.array([6.2e3]), f_dev_array=np.array([600e3]),
                    odmr_range=[2.64e9, 2.65e9], odmr_frequency_points=1000,
                    single_odmr_run_time=60, min_fit_amplitude=0.01, min_feature_height=0.003, n_most_prominent_peaks=3,
                    which_zc=2,
                    n_time_traces=32, data_rate=1000,
                    loop_order=None, include_off_resonant_sensitivity=False):
    """
    Perform a parameter sweep with flexible iteration order over given parameter arrays.

    Args:
        OPX_LO_voltage_array: array-like of OPX LO voltages
        OPX_IF_voltage_array: array-like of OPX IF voltages
        f_mod_array: array-like of modulation frequencies
        f_dev_array: array-like of frequency deviations
        odmr_range: frequency range for ODMR
        odmr_frequency_points: number of frequency points for ODMR scan
        single_odmr_run_time: runtime for a single ODMR scan
        min_fit_amplitude: minimum amplitude for feature detection in fit
        min_feature_height: minimum feature height for fit
        n_most_prominent_peaks: number of prominent peaks to consider in fit
        which_zc: which zero-crossing to use from the fit result
        n_time_traces: number of time traces for sensitivity measurement
        data_rate: data rate (Hz)
        loop_order: list specifying the order in which parameters are iterated.
                    Possible keys: ["OPX_LO_voltage", "OPX_IF_voltage", "f_mod", "f_dev"].
                    Example: ["OPX_IF_voltage", "f_dev", "f_mod", "OPX_LO_voltage"]
        include_off_resonant_sensitivity: boolean to indicate if off-resonant sensitivity should be measured.

    Returns:
        folder_name: name of the folder where data is saved.
    """
    # Default loop order if none is provided
    if loop_order is None:
        loop_order = ["OPX_LO_voltage", "OPX_IF_voltage", "f_mod", "f_dev"]

    # Store parameters in a dict for flexible iteration
    param_dict = {
        "OPX_LO_voltage": OPX_LO_voltage_array,
        "OPX_IF_voltage": OPX_IF_voltage_array,
        "f_mod": f_mod_array,
        "f_dev": f_dev_array
    }

    # Verify loop_order correctness
    for param in loop_order:
        if param not in param_dict:
            raise ValueError(f"Parameter '{param}' in loop_order not found in param_dict keys.")

    # Create folder
    timestamp = datetime.now()
    folder_name = "Tims_Laser_Parameter_Sweep" + timestamp.strftime('%Y%m%d-%H%M-%S') + "/"
    folder(folder_name)

    # Connect to ODMR remote module and LIA
    odmr_remote = OdmrRemoteControl()
    LIA = LockInAmp()

    # Load IQ calibration data
    calibration_data = load_calibration_data()
    cal_voltages = np.array(calibration_data["voltage [V]"])
    g_cal, phi_cal = np.array(calibration_data["g"]), np.array(calibration_data["phi"])
    I_cal, Q_cal = np.array(calibration_data["I"]), np.array(calibration_data["Q"])

    # Prepare arrays to store results
    num_parameter_combinations = (len(OPX_LO_voltage_array) * len(OPX_IF_voltage_array) *
                                  len(f_mod_array) * len(f_dev_array))
    linewidths = np.zeros(num_parameter_combinations)
    sensitivities = np.zeros(num_parameter_combinations)
    sensitivities_std = np.zeros(num_parameter_combinations)
    peak_positions = np.zeros(num_parameter_combinations)
    dip_positions = np.zeros(num_parameter_combinations)
    peak_uncertainties = np.zeros(num_parameter_combinations)
    dip_uncertainties = np.zeros(num_parameter_combinations)
    zero_crossing_frequencies = np.zeros(num_parameter_combinations)
    zero_crossing_slopes = np.zeros(num_parameter_combinations)

    # Store parameter values for each combination
    OPX_LO_voltage_loop_array = np.zeros(num_parameter_combinations)
    OPX_IF_voltage_loop_array = np.zeros(num_parameter_combinations)
    f_mod_loop_array = np.zeros(num_parameter_combinations)
    f_dev_loop_array = np.zeros(num_parameter_combinations)

    if include_off_resonant_sensitivity:
        sensitivities_off_resonant = np.zeros(num_parameter_combinations)
        sensitivities_std_off_resonant = np.zeros(num_parameter_combinations)

    # --- Added for tracking minimum sensitivity ---
    min_sensitivity_so_far = np.inf
    best_params_so_far = {}

    # product_params will be a tuple like (OPX_LO_val, OPX_IF_val, f_mod_val, f_dev_val) depending on loop_order
    product_params = list(product(*[param_dict[p] for p in loop_order]))

    for idx, combination in enumerate(product_params):
        # combination is in the order specified by loop_order
        # We need to extract each parameter accordingly:
        param_values = dict(zip(loop_order, combination))
        if idx > 0: # Get previous params for comparison, handle first iteration
             previous_param_values = dict(zip(loop_order, product_params[idx - 1]))
        else:
             previous_param_values = {} # No previous params for the first iteration

        OPX_LO_val =  OPX_LO_voltage_loop_array[idx] = param_values["OPX_LO_voltage"]
        OPX_IF_val = OPX_IF_voltage_loop_array[idx] = param_values["OPX_IF_voltage"]
        f_mod_val = f_mod_loop_array[idx] = param_values["f_mod"]
        f_dev_val = f_dev_loop_array[idx] = param_values["f_dev"]

        current_sensitivity = np.nan # Initialize sensitivity for this loop to NaN

        try:
            print(f"\n--- Starting Measurement {idx + 1}/{num_parameter_combinations} ---")
            print(f"Parameters: LO={OPX_LO_val:.3f}V, IF={OPX_IF_val:.3f}V, f_mod={f_mod_val/1e3:.1f}kHz, f_dev={f_dev_val/1e3:.1f}kHz")


            # Interpolate calibration data for this OPX_LO_val
            g = np.interp(OPX_LO_val, cal_voltages, g_cal)
            phi = np.interp(OPX_LO_val, cal_voltages, phi_cal)
            I_offset = np.interp(OPX_LO_val, cal_voltages, I_cal)
            Q_offset = np.interp(OPX_LO_val, cal_voltages, Q_cal)

            filename_prefix = (folder_name +
                               f"OPX_LO_{OPX_LO_val:.3f}_OPX_IF_{OPX_IF_val:.3f}_f_mod_{f_mod_val/1e3:.1f}k_f_dev_{f_dev_val/1e3:.1f}k")

            # 1) Set up frequency modulation
            fm = FM_setup(OPX_LO_voltage=OPX_LO_val, OPX_IF_voltage=OPX_IF_val,
                          f_mod=f_mod_val, f_dev=f_dev_val, g_cor=g, phi_cor=phi,
                          I_offset=I_offset, Q_offset=Q_offset)
            fm.execute_FM()

            # if OPX_LO_val or OPX_IF_val was changed, wait 60 s - as changing MW power leads to heating
            # Check if previous_param_values is not empty before accessing keys
            if previous_param_values:
                lo_changed = previous_param_values.get("OPX_LO_voltage") != OPX_LO_val
                if_changed = previous_param_values.get("OPX_IF_voltage") != OPX_IF_val
                if lo_changed or if_changed:
                    sleep_timer = 180
                    print(f"MW Power changed, waiting {sleep_timer} s for thermal stabilization...")
                    if lo_changed: print(f"  OPX_LO_V: {previous_param_values.get('OPX_LO_voltage'):.3f} -> {OPX_LO_val:.3f}")
                    if if_changed: print(f"  OPX_IF_V: {previous_param_values.get('OPX_IF_voltage'):.3f} -> {OPX_IF_val:.3f}")
                    time.sleep(sleep_timer)
            elif idx > 0: # Should not happen if logic above is correct, but added as safety
                 print("Warning: Could not compare previous MW parameters.")


            # 2) Hyperfine ODMR
            print("Taking ODMR Scan...")
            frequencies, odmr_voltages = odmr_remote.take_odmr_scan(filename_prefix + "_ODMR", single_odmr_run_time,
                                                                    odmr_range[0], odmr_range[1],
                                                                    odmr_frequency_points, save_data=True,
                                                                    data_rate=data_rate)
            odmr_voltages = odmr_voltages - np.mean(odmr_voltages)
            odmr_df = pd.DataFrame(data={"frequencies [Hz]": frequencies, "voltages [V]": odmr_voltages})
            odmr_df.to_csv(filename_prefix + "_ODMR.csv", sep="\t")

            # 3) Fit hyperfine
            print("Fitting ODMR data...")
            fit_result = fit_hyperfine(frequencies, odmr_voltages, feature_prominence=min_fit_amplitude,
                                       n_most_prominent_peaks=n_most_prominent_peaks, plot_result=False,
                                       save_result_plot=True,
                                       min_feature_height=min_feature_height, filename=filename_prefix)

            # Check if fit was successful and results exist for the chosen ZC
            if fit_result is None or which_zc >= len(fit_result.get("zero_crossing_frequencies [Hz]", [])):
                 print(f"Fit failed or did not find {which_zc+1} zero crossings. Skipping sensitivity measurement for this point.")
                 # Store NaN or default values for this point
                 linewidths[idx] = np.nan
                 peak_positions[idx] = np.nan
                 dip_positions[idx] = np.nan
                 peak_uncertainties[idx] = np.nan
                 dip_uncertainties[idx] = np.nan
                 zero_crossing_frequencies[idx] = np.nan
                 zero_crossing_slopes[idx] = np.nan
                 sensitivities[idx] = np.nan
                 sensitivities_std[idx] = np.nan
                 if include_off_resonant_sensitivity:
                     sensitivities_off_resonant[idx] = np.nan
                     sensitivities_std_off_resonant[idx] = np.nan
                 continue # Skip to the next iteration


            # 4) Set CW frequency
            cw_frequency = fit_result["zero_crossing_frequencies [Hz]"][which_zc]
            odmr_remote.set_cw_parameters(frequency=float(cw_frequency), power=13)
            odmr_remote.toggle_cw_output(True)

            # 5) Sensitivity measurement
            print("Running sensitivity measurement...")
            sensitivity = SensitivityMeasurement(LIA)
            sensitivity_result = sensitivity.run_measurement(filename_prefix=filename_prefix + "_cw_time_trace",
                                                             n_time_traces=n_time_traces,
                                                             plot_data=False,
                                                             save_raw_data=False,
                                                             save_metadata=True,
                                                             timeout=5000)

            odmr_remote.toggle_cw_output(False)

            # 5.5) Take off-resonant sensitivity measurement
            if include_off_resonant_sensitivity:
                off_resonant_cw_frequency = cw_frequency + 15e6
                odmr_remote.set_cw_parameters(frequency=float(off_resonant_cw_frequency), power=13)
                odmr_remote.toggle_cw_output(True)

                sensitivity_result_off_resonant = sensitivity.run_measurement(filename_prefix=filename_prefix + "_cw_time_trace_off_resonant",
                                                                             n_time_traces=n_time_traces,
                                                                             plot_data=False,
                                                                             save_raw_data=False,
                                                                             save_metadata=True,
                                                                             timeout=5000)

            # 6) Reset CW
            odmr_remote.toggle_cw_output(False)

            # 7) Extract fit results
            linewidths[idx] = fit_result["linewidths [Hz]"][which_zc]
            peak_positions[idx] = fit_result["peak_positions [Hz]"][which_zc]
            dip_positions[idx] = fit_result["dip_positions [Hz]"][which_zc]
            peak_uncertainties[idx] = fit_result["peak_uncertainties [Hz]"][which_zc]
            dip_uncertainties[idx] = fit_result["dip_uncertainties [Hz]"][which_zc]
            zero_crossing_frequencies[idx] = fit_result["zero_crossing_frequencies [Hz]"][which_zc]
            zero_crossing_slopes[idx] = fit_result["zero_crossing_slopes [V/Hz]"][which_zc]


            # 8) Sensitivity analysis
            times = sensitivity_result["times [s]"]
            demod_x_values = sensitivity_result["x_value [V]"]
            scaling = sensitivity_result["metadata"]["aux_0_scaling"]
            sampling_rate = sensitivity_result["metadata"]["sampling_rate [Hz]"]
            duration = sensitivity_result["metadata"]["duration [s]"]
            f_3db = sensitivity_result["metadata"]["filter_3db_freq [Hz]"]

            B_noise_time_trace = magnetic_field_from_voltages(demod_x_values,
                                                              zero_crossing_slopes[idx],
                                                              scaling_factor=scaling)
            plot_magnetic_field_time_traces(B_noise_time_trace, times, sampling_rate, duration, save_fig=True,
                                            filename_prefix=filename_prefix)
            asd_result = plot_asds(B_noise_time_trace, sampling_rate, duration, f_3db, filename_prefix=filename_prefix,
                                   save_data=True, save_fig=True)

            # tau_values, adev_values = allan_deviation(B_noise_time_trace, sampling_rate)
            # plot_allan_deviation(tau_values, adev_values, save_fig=True, filename_prefix=filename_prefix)

            sensitivities[idx] = current_sensitivity = asd_result["sensitivity"] # Store and assign to current_sensitivity
            sensitivities_std[idx] = asd_result["sensitivity_std"]


            # --- Added for reporting sensitivity ---
            print("-" * 20 + " Sensitivity Result " + "-" * 20)
            print(f"  Current Sensitivity: {current_sensitivity:.3f} nT/sqrt(Hz)")

            # Update and report minimum sensitivity
            if not np.isnan(current_sensitivity) and current_sensitivity < min_sensitivity_so_far:
                min_sensitivity_so_far = current_sensitivity
                best_params_so_far = {
                    "OPX_LO_voltage": OPX_LO_val,
                    "OPX_IF_voltage": OPX_IF_val,
                    "f_mod": f_mod_val,
                    "f_dev": f_dev_val
                }
                print(f"  >>> New Minimum Sensitivity Found! <<<")

            if best_params_so_far: # Check if we have found a minimum yet
                 print(f"  Best Sensitivity So Far: {min_sensitivity_so_far:.3f} nT/sqrt(Hz)")
                 print(f"    Achieved with: LO={best_params_so_far['OPX_LO_voltage']:.3f}V, IF={best_params_so_far['OPX_IF_voltage']:.3f}V, "
                       f"f_mod={best_params_so_far['f_mod']/1e3:.1f}kHz, f_dev={best_params_so_far['f_dev']/1e3:.1f}kHz")
            else:
                 print("  Minimum sensitivity not yet determined.")
            print("-" * 58) # Match length
            # --- End of added code ---


            if include_off_resonant_sensitivity:
                print("Analyzing off-resonant sensitivity...")
                times_off = sensitivity_result_off_resonant["times [s]"]
                demod_x_values_off = sensitivity_result_off_resonant["x_value [V]"]
                scaling_off = sensitivity_result_off_resonant["metadata"]["aux_0_scaling"]
                sampling_rate_off = sensitivity_result_off_resonant["metadata"]["sampling_rate [Hz]"]
                duration_off = sensitivity_result_off_resonant["metadata"]["duration [s]"]
                f_3db_off = sensitivity_result_off_resonant["metadata"]["filter_3db_freq [Hz]"]

                # NOTE: Using the *on-resonant* slope here intentionally to compare noise levels under same conversion factor
                B_noise_time_trace_off = magnetic_field_from_voltages(demod_x_values_off,
                                                                      zero_crossing_slopes[idx],
                                                                      scaling_factor=scaling_off)
                plot_magnetic_field_time_traces(B_noise_time_trace_off, times_off, sampling_rate_off, duration_off, save_fig=True,
                                                filename_prefix=filename_prefix + "_off_resonant")

                asd_result_off = plot_asds(B_noise_time_trace_off, sampling_rate_off, duration_off, f_3db_off, filename_prefix=filename_prefix + "_off_resonant",
                                           save_data=True, save_fig=True)

                # tau_values, adev_values = allan_deviation(B_noise_time_trace, sampling_rate)
                # plot_allan_deviation(tau_values, adev_values, save_fig=True, filename_prefix=filename_prefix + "_off_resonant")

                sensitivities_off_resonant[idx] = asd_result_off["sensitivity"]
                sensitivities_std_off_resonant[idx] = asd_result_off["sensitivity_std"]
                print(f"  Off-Resonant Sensitivity: {sensitivities_off_resonant[idx]:.3f} nT/sqrt(Hz)")


        except Exception as e:
            print(f"!!!!!!!! ERROR during measurement {idx + 1} !!!!!!!!")
            print(f"Parameters: LO={OPX_LO_val:.3f}V, IF={OPX_IF_val:.3f}V, f_mod={f_mod_val/1e3:.1f}kHz, f_dev={f_dev_val/1e3:.1f}kHz")
            traceback.print_exc()
            print(f"Error message: {str(e)}")
            # Store NaN or default values for this point if error occurred
            linewidths[idx] = np.nan
            peak_positions[idx] = np.nan
            dip_positions[idx] = np.nan
            peak_uncertainties[idx] = np.nan
            dip_uncertainties[idx] = np.nan
            zero_crossing_frequencies[idx] = np.nan
            zero_crossing_slopes[idx] = np.nan
            sensitivities[idx] = np.nan # Ensure sensitivity is NaN if error occurred
            sensitivities_std[idx] = np.nan
            if include_off_resonant_sensitivity:
                sensitivities_off_resonant[idx] = np.nan
                sensitivities_std_off_resonant[idx] = np.nan
            print(f"!!!!!!!! Skipping rest of measurement {idx + 1} due to error. !!!!!!!!")


    # Save fit results to CSV
    print("\n--- Sweep Finished ---")
    print("Saving results...")
    # Build a DataFrame from results
    result_data = {
        "OPX_LO_voltage [V]":           OPX_LO_voltage_loop_array,
        "OPX_IF_voltage [V]":           OPX_IF_voltage_loop_array,
        "f_mod [Hz]":                   f_mod_loop_array,
        "f_dev [Hz]":                   f_dev_loop_array,
        "linewidths [Hz]":              linewidths,
        "peak_positions [Hz]":          peak_positions,
        "dip_positions [Hz]":           dip_positions,
        "peak_uncertainties [Hz]":      peak_uncertainties,
        "dip_uncertainties [Hz]":       dip_uncertainties,
        "zero_crossing_frequencies [Hz]": zero_crossing_frequencies,
        "zero_crossing_slopes [V/Hz]":  zero_crossing_slopes,
        "sensitivities [nT/root(Hz)]":  sensitivities,
        "sensitivities_std [nT/root(Hz)]": sensitivities_std
    }


    if include_off_resonant_sensitivity:
        result_data["sensitivities_off_resonant [nT/root(Hz)]"] = sensitivities_off_resonant
        result_data["sensitivities_std_off_resonant [nT/root(Hz)]"] = sensitivities_std_off_resonant

    result_df = pd.DataFrame(result_data)
    # Save with NaN representation specified
    result_df.to_csv(folder_name + "fit_results.csv", sep="\t", index=False, na_rep='NaN')

    # Report the final best result
    if best_params_so_far:
        print("\n--- Best Result Found During Sweep ---")
        print(f"  Minimum Sensitivity: {min_sensitivity_so_far:.3f} nT/sqrt(Hz)")
        print(f"  Parameters: LO={best_params_so_far['OPX_LO_voltage']:.3f}V, IF={best_params_so_far['OPX_IF_voltage']:.3f}V, "
              f"f_mod={best_params_so_far['f_mod']/1e3:.1f}kHz, f_dev={best_params_so_far['f_dev']/1e3:.1f}kHz")
    else:
        print("\nNo successful sensitivity measurements were completed.")


    return folder_name


if __name__ == "__main__":
    # Example usage with a specified loop order
    # Here we specify a loop order:
    # First iterate over OPX_IF_voltage, then f_dev, then f_mod, then OPX_LO_voltage
    custom_loop_order = ["OPX_IF_voltage", "f_dev", "f_mod", "OPX_LO_voltage"]

    for odmr_range in [[2.804e9, 2.828e9]]:
        start_time = time.time()

        OPX_LO_voltage_array = np.linspace(0.4, 0.4, 1)
        OPX_IF_voltage_array = np.linspace(0.2, 0.275, 1)
        f_mod_array = np.linspace(15e3, 22e3, 1)
        f_dev_array = np.linspace(400e3, 600e3, 1)

        # OPX_LO_voltage_array = np.linspace(0.4, 0.4, 1)ö
        # OPX_IF_voltage_array = np.linspace(0.05, 0.2, 1)
        # f_mod_array = np.linspace(5e3, 10e3, 1)
        # f_dev_array = np.linspace(400e3, 700e3, 1)

        include_off_resonant_sensitivity = True

        n_time_traces = 16
        single_odmr_runtime = 10
        data_rate = 1000  # Hz
        odmr_frequency_points = 1000
        which_zc = 2
        n_most_prominent_peaks = 5
        min_fit_amplitude = 0.005
        laser_power = 0

        folder_name = parameter_sweep(OPX_LO_voltage_array= OPX_LO_voltage_array,
                                      OPX_IF_voltage_array= OPX_IF_voltage_array,
                                      f_mod_array=          f_mod_array,
                                      f_dev_array=          f_dev_array,
                                      odmr_range=           odmr_range,
                                      single_odmr_run_time= single_odmr_runtime,
                                      min_fit_amplitude=    min_fit_amplitude,
                                      which_zc=             which_zc,
                                      n_most_prominent_peaks=n_most_prominent_peaks,
                                      n_time_traces=        n_time_traces,
                                      data_rate=            data_rate,
                                      odmr_frequency_points=odmr_frequency_points,
                                      loop_order=           custom_loop_order,
                                      include_off_resonant_sensitivity=include_off_resonant_sensitivity)

        print(f"Measurement took {(time.time() - start_time) / 60} minutes, or {(time.time() - start_time) / 3600} hours.")

        metadata = {"laser_power [mW]":       laser_power,
                    "n_time_traces":        n_time_traces,
                    "which_zc":             which_zc,
                    "single_odmr_runtime [s]":  single_odmr_runtime,
                    "Measurement time [min]": round((time.time() - start_time)/60, 1)}

        with open(folder_name + "sweep_metadata" + ".json", "w") as f:
            json.dump(metadata, f)

        duration = 500  # ms
        freq = 440  # Hz
        winsound.Beep(freq, duration)
