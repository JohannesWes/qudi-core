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
                    loop_order=None):
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
    peak_positions = np.zeros(num_parameter_combinations)
    dip_positions = np.zeros(num_parameter_combinations)
    peak_uncertainties = np.zeros(num_parameter_combinations)
    dip_uncertainties = np.zeros(num_parameter_combinations)
    zero_crossing_frequencies = np.zeros(num_parameter_combinations)
    zero_crossing_slopes = np.zeros(num_parameter_combinations)

    # Use itertools.product to iterate in the specified order
    from itertools import product

    # product_params will be a tuple like (OPX_LO_val, OPX_IF_val, f_mod_val, f_dev_val) depending on loop_order
    product_params = list(product(*[param_dict[p] for p in loop_order]))

    for idx, combination in enumerate(product_params):
        # combination is in the order specified by loop_order
        # We need to extract each parameter accordingly:
        param_values = dict(zip(loop_order, combination))
        previous_param_values = dict(zip(loop_order, product_params[idx - 1]))
        OPX_LO_val = param_values["OPX_LO_voltage"]
        OPX_IF_val = param_values["OPX_IF_voltage"]
        f_mod_val = param_values["f_mod"]
        f_dev_val = param_values["f_dev"]

        try:
            print(f"Starting Measurement {idx + 1}/{num_parameter_combinations}")

            # Interpolate calibration data for this OPX_LO_val
            g = np.interp(OPX_LO_val, cal_voltages, g_cal)
            phi = np.interp(OPX_LO_val, cal_voltages, phi_cal)
            I_offset = np.interp(OPX_LO_val, cal_voltages, I_cal)
            Q_offset = np.interp(OPX_LO_val, cal_voltages, Q_cal)

            filename_prefix = (folder_name +
                               f"OPX_LO_{OPX_LO_val}_OPX_IF_{OPX_IF_val}_f_mod_{f_mod_val/1e3:.2f}_f_dev_{f_dev_val/1e3:.2f}")

            # 1) Set up frequency modulation
            fm = FM_setup(OPX_LO_voltage=OPX_LO_val, OPX_IF_voltage=OPX_IF_val,
                          f_mod=f_mod_val, f_dev=f_dev_val, g_cor=g, phi_cor=phi,
                          I_offset=I_offset, Q_offset=Q_offset)
            fm.execute_FM()
            # if OPX_LO_val or OPX_IF_val was changed, wait 60 s - as changing MW power leads to heating
            if previous_param_values["OPX_LO_voltage"] != OPX_LO_val or previous_param_values["OPX_IF_voltage"] != OPX_IF_val:
                time.sleep(60)


            # 2) Hyperfine ODMR
            frequencies, odmr_voltages = odmr_remote.take_odmr_scan(filename_prefix + "_ODMR", single_odmr_run_time,
                                                                    odmr_range[0], odmr_range[1],
                                                                    odmr_frequency_points, save_data=True,
                                                                    data_rate=data_rate)
            odmr_voltages = odmr_voltages - np.mean(odmr_voltages)
            odmr_df = pd.DataFrame(data={"frequencies [Hz]": frequencies, "voltages [V]": odmr_voltages})
            odmr_df.to_csv(filename_prefix + "_ODMR.csv", sep="\t")

            # 3) Fit hyperfine
            fit_result = fit_hyperfine(frequencies, odmr_voltages, min_feature_amplitude=min_fit_amplitude,
                                       n_most_prominent_peaks=n_most_prominent_peaks, plot_result=False,
                                       save_result_plot=True,
                                       min_feature_height=min_feature_height, filename=filename_prefix)

            # 4) Set CW frequency
            cw_frequency = fit_result["zero_crossing_frequencies [Hz]"][which_zc]
            odmr_remote.set_cw_parameters(frequency=float(cw_frequency), power=13)
            odmr_remote.toggle_cw_output(True)

            # 5) Sensitivity measurement
            sensitivity = SensitivityMeasurement(LIA)
            sensitivity_result = sensitivity.run_measurement(filename_prefix=filename_prefix + "_cw_time_trace",
                                                             n_time_traces=n_time_traces,
                                                             plot_data=False,
                                                             save_raw_data=False,
                                                             save_metadata=True,
                                                             timeout=60)

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

            tau_values, adev_values = allan_deviation(B_noise_time_trace, sampling_rate)
            plot_allan_deviation(tau_values, adev_values, save_fig=True, filename_prefix=filename_prefix)

            sensitivities[idx] = asd_result["sensitivity"]

        except Exception as e:
            traceback.print_exc()
            print(f"Error in measurement: {str(e)}")

    # Save fit results to CSV
    # Build a DataFrame from results
    result_data = {
        "OPX_LO_voltage [V]": [],
        "OPX_IF_voltage [V]": [],
        "f_mod [Hz]": [],
        "f_dev [Hz]": [],
        "linewidths [Hz]": linewidths,
        "peak_positions [Hz]": peak_positions,
        "dip_positions [Hz]": dip_positions,
        "peak_uncertainties [Hz]": peak_uncertainties,
        "dip_uncertainties [Hz]": dip_uncertainties,
        "zero_crossing_frequencies [Hz]": zero_crossing_frequencies,
        "zero_crossing_slopes [V/Hz]": zero_crossing_slopes,
        "sensitivities [nT/root(Hz)]": sensitivities
    }

    # Fill in parameter columns in the order they were generated
    # We know the order of generation from product_params
    # product_params were generated with `loop_order`
    # Let's reconstruct them into a DataFrame
    for param_name in loop_order:
        param_values_list = [combination[loop_order.index(param_name)] for combination in product_params]
        result_data[f"{param_name} [unit]"] = param_values_list

    # Remove old parameter columns that were pre-named
    # We'll rely on the newly created columns from the actual param values
    for old_param in ["OPX_LO_voltage [V]", "OPX_IF_voltage [V]", "f_mod [Hz]", "f_dev [Hz]"]:
        if old_param in result_data:
            del result_data[old_param]

    # The arrays have been added in loop_order with generic "[unit]" placeholders above
    # Let's rename them properly:
    rename_map = {
        "OPX_LO_voltage [unit]": "OPX_LO_voltage [V]",
        "OPX_IF_voltage [unit]": "OPX_IF_voltage [V]",
        "f_mod [unit]": "f_mod [Hz]",
        "f_dev [unit]": "f_dev [Hz]"
    }
    result_df = pd.DataFrame(result_data).rename(columns=rename_map)

    result_df.to_csv(folder_name + "fit_results.csv", sep="\t", index=False)

    return folder_name


if __name__ == "__main__":
    # Example usage with a specified loop order
    # Here we specify a loop order:
    # First iterate over OPX_IF_voltage, then f_dev, then f_mod, then OPX_LO_voltage
    custom_loop_order = ["OPX_IF_voltage", "f_dev", "f_mod", "OPX_LO_voltage"]

    for odmr_range in [[2.784e9, 2.803e9]]:
        start_time = time.time()

        OPX_LO_voltage_array = np.linspace(0.4, 0.4, 1)
        OPX_IF_voltage_array = np.linspace(0.02, 0.33, 5)
        f_mod_array = np.array([6.3e3])
        f_dev_array = np.linspace(400e3, 405e3, 2)

        n_time_traces = 16
        single_odmr_runtime = 30
        data_rate = 1000  # Hz
        odmr_frequency_points = 1000
        which_zc = 2
        n_most_prominent_peaks = 5
        min_fit_amplitude = 0.001
        laser_power = 600

        folder_name = parameter_sweep(OPX_LO_voltage_array=OPX_LO_voltage_array,
                                      OPX_IF_voltage_array=OPX_IF_voltage_array,
                                      f_mod_array=f_mod_array,
                                      f_dev_array=f_dev_array,
                                      odmr_range=odmr_range,
                                      single_odmr_run_time=single_odmr_runtime,
                                      min_fit_amplitude=min_fit_amplitude,
                                      which_zc=which_zc,
                                      n_most_prominent_peaks=n_most_prominent_peaks,
                                      n_time_traces=n_time_traces,
                                      data_rate=data_rate,
                                      odmr_frequency_points=odmr_frequency_points,
                                      loop_order=custom_loop_order)

        metadata = {"laser_power_mW": laser_power, "n_time_traces": n_time_traces,
                    "which_zc": which_zc, "single_odmr_runtime": single_odmr_runtime}
        with open(folder_name + "sweep_metadata" + ".json", "w") as f:
            json.dump(metadata, f)

        duration = 1000  # ms
        freq = 440  # Hz
        winsound.Beep(freq, duration)

        print(f"Measurement took {(time.time() - start_time) / 60} minutes, or {(time.time() - start_time) / 3600} hours.")
