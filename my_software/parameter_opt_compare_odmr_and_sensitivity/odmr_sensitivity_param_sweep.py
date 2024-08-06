import numpy as np
import importlib.util
import sys
from datetime import datetime
import time
import matplotlib
import pandas as pd
import os
import json
import winsound
import traceback

matplotlib.use("Qt5Agg")

from my_software.automation.qudi_remote_control import OdmrRemoteControl
from my_software.tools.fitting import fit_hyperfine
from my_software.opx_tools.frequency_modulation.FM_working_oscillator_RF_mixer_hyperfine import FM_setup

from my_software.sensitivity_msmt.sensitivity_measurement import lock_in_amp
from my_software.sensitivity_msmt.auswertung.sensitivity_auswertung_modular import *


def folder(foldername):
    timenow = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folder_path = os.path.join(foldername)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        # print('Created:', folder_path)


# -------------------------------------------------------------------------------------------------------------------- #
def load_calibration_data():
    calibration_df = pd.read_csv("..\opx_tools\IQ_calibration\calibration_2024-08-02-10-33-57.csv", sep="\t",
                                 index_col=0)
    return calibration_df


def parameter_sweep(OPX_LO_voltage_array=np.array([0.1]), OPX_IF_voltage_array=np.array([0.1]),
                    f_mod_array=np.array([6.2e3]), f_dev_array=np.array([600e3]),
                    odmr_range=[2.64e9, 2.65e9], odmr_frequency_points=1000,
                    single_odmr_run_time=60, min_fit_amplitude=0.01, min_feature_height=0.003, n_most_prominent_peaks=3,
                    which_zc=2,
                    n_time_traces=32, data_rate=1000):
    # folder_name for current measurement
    timestamp = datetime.now()
    folder_name = "Tims_Laser_Parameter_Sweep" + timestamp.strftime('%Y%m%d-%H%M-%S') + "/"
    folder(folder_name)

    # connect to ODMR remote module
    odmr_remote = OdmrRemoteControl()
    LIA = lock_in_amp()

    # Create a meshgrid from the parameters
    OPX_LO_voltage_mesh, OPX_IF_voltage_mesh, f_mod_mesh, f_dev_mesh = np.meshgrid(OPX_LO_voltage_array,
                                                                                   OPX_IF_voltage_array, f_mod_array,
                                                                                   f_dev_array)

    # Flatten the meshgrid arrays
    OPX_LO_voltage_flat = OPX_LO_voltage_mesh.flatten()
    OPX_IF_voltage_flat = OPX_IF_voltage_mesh.flatten()
    f_mod_flat = f_mod_mesh.flatten()
    f_dev_flat = f_dev_mesh.flatten()
    num_parameter_combinations = len(OPX_LO_voltage_flat)

    # load IQ calibration data
    calibration_data = load_calibration_data()
    cal_voltages = np.array(calibration_data["voltage [V]"])
    g_cal, phi_cal = np.array(calibration_data["g"]), np.array(calibration_data["phi"])
    I_cal, Q_cal = np.array(calibration_data["I"]), np.array(calibration_data["Q"])

    # interpolate calibration data to get g, phi, I, Q for the now chosen voltages
    g, phi = np.interp(OPX_LO_voltage_flat, cal_voltages, g_cal), np.interp(OPX_LO_voltage_flat, cal_voltages, phi_cal)
    I, Q = np.interp(OPX_LO_voltage_flat, cal_voltages, I_cal), np.interp(OPX_LO_voltage_flat, cal_voltages, Q_cal)

    linewidths, sensitivities = np.zeros(num_parameter_combinations), np.zeros(num_parameter_combinations)
    peak_positions, dip_positions = np.zeros(num_parameter_combinations), np.zeros(num_parameter_combinations)
    peak_uncertainties, dip_uncertainties = np.zeros(num_parameter_combinations), np.zeros(num_parameter_combinations)
    zero_crossing_frequencies, zero_crossing_slopes = np.zeros(num_parameter_combinations), np.zeros(
        num_parameter_combinations)

    try:
        for i in range(num_parameter_combinations):
            # 1) OPX set parameters; adjust g,p,I,q parameters for the set voltages; start frequency modulation
            # 2) take a hyperfine ODMR for that voltage in known odmr range
            # 3) fit the hyperfine ODMR -> robust enough?
            # 4) set CW frequency to the chosen zero-crossing
            # 5) collect time-trace for sensitivity measurement

            filename_pre = folder_name + "OPX_LO_" + str(OPX_LO_voltage_flat[i]) + "_OPX_IF_" + str(
                OPX_IF_voltage_flat[i]) + "_f_mod_" + "{:.2f}".format(
                f_mod_flat[i] / 1e3) + "_f_dev_" + "{:.2f}".format(f_dev_flat[i] / 1e3)

            # 1) OPX set voltage; adjust g,p,I,q parameters for that voltage; start frequency modulation
            fm = FM_setup(OPX_LO_voltage=OPX_LO_voltage_flat[i], OPX_IF_voltage=OPX_IF_voltage_flat[i],
                          f_mod=f_mod_flat[i],
                          f_dev=f_dev_flat[i], g_cor=g[i], phi_cor=phi[i], I_offset=I[i], Q_offset=Q[i])
            fm.execute_FM()
            time.sleep(0.5)

            # 2) take a hyperfine ODMR for that voltage in known odmr range
            frequencies, odmr_voltages = odmr_remote.take_odmr_scan(filename_pre + "_ODMR", single_odmr_run_time,
                                                                    odmr_range[0],
                                                                    odmr_range[1],
                                                                    odmr_frequency_points, save_data=True,
                                                                    data_rate=data_rate)
            odmr_voltages = odmr_voltages - np.mean(odmr_voltages)

            # create pandas dataframe for the ODMR data and save to csv
            odmr_df = pd.DataFrame(data={"frequencies [Hz]": frequencies, "voltages [V]": odmr_voltages})
            odmr_df.to_csv(filename_pre + "_ODMR.csv", sep="\t")

            # 3) fit the hyperfine ODMR -> robust enough?
            fit_result = fit_hyperfine(frequencies, odmr_voltages, min_feature_amplitude=min_fit_amplitude,
                                       n_most_prominent_peaks=n_most_prominent_peaks, plot_result=False,
                                       save_result_plot=True,
                                       min_feature_height=min_feature_height, filename=filename_pre)

            # 4) set CW frequency to the chosen zero-crossing
            cw_frequency = fit_result["zero_crossing_frequencies [Hz]"][which_zc]
            # rpyc has problems with numpy float serialization
            odmr_remote.set_cw_parameters(frequency=float(cw_frequency), power=13)
            odmr_remote.toggle_cw_output(True)

            # 5) collect time-trace for sensitivity measurement
            sensitivity_result = LIA.sensitivity_measurement(filename_pre + "_cw_time_trace",
                                                             n_time_traces=n_time_traces,
                                                             save_raw_data=True, save_metadata=True)

            # 6) Reset everything
            odmr_remote.toggle_cw_output(False)

            # save fit results to arrays
            linewidths[i] = fit_result["linewidths [Hz]"][which_zc]
            peak_positions[i] = fit_result["peak_positions [Hz]"][which_zc]
            dip_positions[i] = fit_result["dip_positions [Hz]"][which_zc]
            peak_uncertainties[i] = fit_result["peak_uncertainties [Hz]"][which_zc]
            dip_uncertainties[i] = fit_result["dip_uncertainties [Hz]"][which_zc]
            zero_crossing_frequencies[i] = fit_result["zero_crossing_frequencies [Hz]"][
                which_zc]
            zero_crossing_slopes[i] = fit_result["zero_crossing_slopes [V/Hz]"][which_zc]

            # 7) sensitivity analysis
            times = sensitivity_result["times"]
            demod_x_values = sensitivity_result["x_value"]
            scaling = sensitivity_result["metadata"]["aux_0_scaling"]
            sampling_rate = sensitivity_result["metadata"]["sampling_rate"]
            duration = sensitivity_result["metadata"]["duration"]
            f_3db = sensitivity_result["metadata"]["filter_3db_freq [Hz]"]

            B_noise_time_trace = magnetic_field_from_voltages(demod_x_values,
                                                              zero_crossing_slopes[i],
                                                              scaling_factor=scaling)
            plot_magnetic_field_time_traces(B_noise_time_trace, times, sampling_rate, duration, save_fig=True,
                                            filename=filename_pre)
            asd_result = plot_asds(B_noise_time_trace, sampling_rate, duration, f_3db, filename=filename_pre,
                                   save_data=True, save_fig=True)
            sensitivities[i] = asd_result["sensitivity"]

    except Exception as e:
        traceback.print_exc()
        print(f"Error in measurement: {str(e)}")

    # create dataframe for fit results and save to csv
    fit_results_df = pd.DataFrame(
        data={"OPX_LO_voltage [V]": OPX_LO_voltage_flat, "OPX_IF_voltage [V]": OPX_IF_voltage_flat,
              "f_mod [Hz]": f_mod_flat,
              "f_dev [Hz]": f_dev_flat, "linewidths [Hz]": linewidths, "peak_positions [Hz]": peak_positions,
              "dip_positions [Hz]": dip_positions, "peak_uncertainties [Hz]": peak_uncertainties,
              "dip_uncertainties [Hz]": dip_uncertainties, "zero_crossing_frequencies [Hz]": zero_crossing_frequencies,
              "zero_crossing_slopes [V/Hz]": zero_crossing_slopes, "sensitivities": sensitivities})
    fit_results_df.to_csv(folder_name + "fit_results.csv", sep="\t")

    return folder_name


if __name__ == "__main__":
    OPX_LO_voltage_array = np.linspace(0.5, 0.5, 1)
    OPX_IF_voltage_array = np.linspace(0.1, 0.2, 1)
    f_mod_array = np.array([6.3e3])
    f_dev_array = np.array([620e3])

    n_time_traces = 32
    single_odmr_runtime = 30
    data_rate = 200  # Hz

    which_zc = 0
    n_most_prominent_peaks = 5
    min_fit_amplitude = 0.005

    laser_power = 500

    # has to be list instead of np.array
    odmr_range = [2.81e9, 2.826e9]

    folder_name = parameter_sweep(OPX_LO_voltage_array=OPX_LO_voltage_array, OPX_IF_voltage_array=OPX_IF_voltage_array,
                                  f_mod_array=f_mod_array, f_dev_array=f_dev_array, odmr_range=odmr_range,
                                  single_odmr_run_time=single_odmr_runtime, min_fit_amplitude=min_fit_amplitude,
                                  which_zc=which_zc,
                                  n_most_prominent_peaks=n_most_prominent_peaks,
                                  n_time_traces=n_time_traces, data_rate=data_rate)

    metadata = {"laser_power_mW": laser_power, "n_time_traces": n_time_traces,
                "which_zc": which_zc, "single_odmr_runtime": single_odmr_runtime}
    # save metadata to a json file
    with open(folder_name + "sweep_metadata" + ".json", "w") as f:
        json.dump(metadata, f)

    # play sound when measurement is finished
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
