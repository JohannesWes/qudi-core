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

matplotlib.use("Qt5Agg")

from my_software.automation.qudi_remote_control import OdmrRemoteControl
from my_software.tools.fitting import fit_hyperfine
from my_software.opx_tools.frequency_modulation.FM_working_oscillator import FM_setup

from my_software.sensitivity_msmt.sensitivity_measurement import lock_in_amp
from my_software.sensitivity_msmt.auswertung.sensitivity_auswertung_modular import *


def folder(foldername):
    timenow = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folderpath = os.path.join(foldername)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        # print('Created:', folderpath)


# -------------------------------------------------------------------------------------------------------------------- #
def load_calibration_data():
    calibration_df = pd.read_csv("..\opx_tools\IQ_calibration\calib_01-08-24\calibration_data_2024-08-01_renamed.csv", sep="\t", index_col=0)
    return calibration_df


def voltage_sweep(voltages, f_mod=4.5e3, f_dev=700e3, odmr_range=[2.64e9, 2.65e9], odmr_frequency_points=1000,
                  single_odmr_run_time=60, min_fit_amplitude=0.01, min_feature_height=0.003, which_zc=2,
                  n_time_traces=32, data_rate=1000):
    # folder_name for current measurement
    timestamp = datetime.now()
    folder_name = "Tims_Laser_Voltage_Sweep" + timestamp.strftime('%Y%m%d-%H%M-%S') + "/"
    folder(folder_name)

    # connect to ODMR remote module
    odmr_remote = OdmrRemoteControl()
    LIA = lock_in_amp()

    # load IQ calibration data
    calibration_data = load_calibration_data()
    cal_voltages = np.array(calibration_data["voltage [V]"])
    g_cal = np.array(calibration_data["g"])
    phi_cal = np.array(calibration_data["phi"])
    I_cal = np.array(calibration_data["I"])
    Q_cal = np.array(calibration_data["Q"])

    # interpolate calibration data to get g, phi, I, Q for the now chosen voltages
    g = np.interp(voltages, cal_voltages, g_cal)
    phi = np.interp(voltages, cal_voltages, phi_cal)
    I = np.interp(voltages, cal_voltages, I_cal)
    Q = np.interp(voltages, cal_voltages, Q_cal)

    linewidths = np.zeros(len(voltages))
    peak_positions = np.zeros(len(voltages))
    dip_positions = np.zeros(len(voltages))
    peak_uncertainties = np.zeros(len(voltages))
    dip_uncertainties = np.zeros(len(voltages))
    zero_crossing_frequencies = np.zeros(len(voltages))
    zero_crossing_slopes = np.zeros(len(voltages))
    sensitivities = np.zeros(len(voltages))

    # -------------------------------------------------------------------------------------------- #
    # MEASUREMENT
    # -------------------------------------------------------------------------------------------- #
    filename_pre = ""

    try:
        for v_index, voltage in enumerate(voltages):
            # 1) OPX set voltage; adjust g,p,I,q parameters for that voltage; start frequency modulation
            # 2) take a hyperfine ODMR for that voltage in known odmr range
            # 3) fit the hyperfine ODMR -> robust enough?
            # 4) set CW frequency to the chosen zero-crossing
            # 5) collect time-trace for sensitivity measurement

            filename_pre = folder_name + "voltage_" + str(voltage) + "V"

            print(f"Starting measurement for voltage {voltage} V")

            # 1) OPX set voltage; adjust g,p,I,q parameters for that voltage; start frequency modulation
            fm = FM_setup(f_mod=f_mod, f_dev=f_dev, voltage_opx=voltage, g_cor=g[v_index], phi_cor=phi[v_index],
                          I_offset=I[v_index], Q_offset=Q[v_index])
            fm.execute_FM()

            time.sleep(0.5)

            # 2) take a hyperfine ODMR for that voltage in known odmr range
            frequencies, odmr_voltages = odmr_remote.take_odmr_scan(filename_pre + "_ODMR", single_odmr_run_time,
                                                                    odmr_range[0],
                                                                    odmr_range[1],
                                                                    odmr_frequency_points, save_data=True, data_rate=data_rate)
            odmr_voltages = odmr_voltages - np.mean(odmr_voltages)

            # create pandas dataframe for the ODMR data and save to csv
            odmr_df = pd.DataFrame(data={"frequencies [Hz]": frequencies, "voltages [V]": odmr_voltages})
            odmr_df.to_csv(filename_pre + "_ODMR.csv", sep="\t")

            # 3) fit the hyperfine ODMR -> robust enough?
            fit_result = fit_hyperfine(frequencies, odmr_voltages, min_feature_amplitude=min_fit_amplitude,
                                       n_most_prominent_peaks=3, plot_result=False, save_result_plot=True,
                                       min_feature_height=min_feature_height, filename=filename_pre)

            # 4) set CW frequency to the chosen zero-crossing
            cw_frequency = fit_result["zero_crossing_frequencies [Hz]"][which_zc]
            # rpyc has problems with numpy float serialization
            odmr_remote.set_cw_parameters(frequency=float(cw_frequency), power=13)
            odmr_remote.toggle_cw_output(True)

            # 5) collect time-trace for sensitivity measurement
            sensitivity_result = LIA.sensitivity_measurement(filename_pre + "_cw_time_trace", n_time_traces=n_time_traces,
                                                             save_raw_data=True)

            # 6) Reset everything
            odmr_remote.toggle_cw_output(False)

            linewidth = fit_result["linewidths [Hz]"][which_zc]
            linewidths[v_index] = linewidth
            peak_positions[v_index] = fit_result["peak_positions [Hz]"][which_zc]
            dip_positions[v_index] = fit_result["dip_positions [Hz]"][which_zc]
            peak_uncertainties[v_index] = fit_result["peak_uncertainties [Hz]"][which_zc]
            dip_uncertainties[v_index] = fit_result["dip_uncertainties [Hz]"][which_zc]
            zero_crossing_frequencies[v_index] = fit_result["zero_crossing_frequencies [Hz]"][which_zc]
            zero_crossing_slopes[v_index] = fit_result["zero_crossing_slopes [V/Hz]"][which_zc]

            # 7) sensitivity analysis
            times = sensitivity_result["times"]
            demod_x_values = sensitivity_result["x_value"]
            scaling = sensitivity_result["metadata"]["aux_0_scaling"]
            sampling_rate = sensitivity_result["metadata"]["sampling_rate"]
            duration = sensitivity_result["metadata"]["duration"]
            f_3db = sensitivity_result["metadata"]["filter_3db_freq [Hz]"]

            B_noise_time_trace = magnetic_field_from_voltages(demod_x_values, zero_crossing_slopes[v_index],
                                                              scaling_factor=scaling)
            plot_magnetic_field_time_traces(B_noise_time_trace, times, sampling_rate, duration, save_fig=True, filename=filename_pre)
            asd_result = plot_asds(B_noise_time_trace, sampling_rate, duration, f_3db, filename=filename_pre,
                                   save_data=True, save_fig=True)
            sensitivities[v_index] = asd_result["sensitivity"]

    except Exception as e:
        print(f"Error in measurement: {str(e)}")

    # create dataframe for fit results and save to csv
    fit_results_df = pd.DataFrame(
        data={"voltage [V]": voltages, "linewidths [Hz]": linewidths, "peak_positions [Hz]": peak_positions,
              "dip_positions [Hz]": dip_positions, "peak_uncertainties [Hz]": peak_uncertainties,
              "dip_uncertainties [Hz]": dip_uncertainties, "zero_crossing_frequencies [Hz]": zero_crossing_frequencies,
              "zero_crossing_slopes [V/Hz]": zero_crossing_slopes, "sensitivities [nT/root(Hz)]": sensitivities})
    fit_results_df.to_csv(folder_name + "fit_results.csv", sep="\t")

    return folder_name

if __name__ == "__main__":
    voltage_array = np.linspace(0.01, 0.08, 50)


    f_mod = 6.3e3
    f_dev = 620e3
    n_time_traces = 32
    which_zc = 0
    single_odmr_runtime = 30

    laser_power = 500


    odmr_range = [2.81e9, 2.826e9]

    folder_name = voltage_sweep(voltage_array, f_mod=f_mod, f_dev=f_dev, odmr_range=odmr_range, single_odmr_run_time=single_odmr_runtime,
                  n_time_traces=n_time_traces, min_fit_amplitude=0.005, which_zc=which_zc, data_rate=200)

    metadata = {"laser_power_mW": laser_power, "f_mod_Hz": f_mod, "f_dev_Hz": f_dev, "n_time_traces": n_time_traces, "which_zc": which_zc, "single_odmr_runtime": single_odmr_runtime}
    # save metadata to ajson file
    with open(folder_name + "sweep_metadata" + ".json", "w") as f:
        json.dump(metadata, f)

    # play sound when measurement is finished
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)