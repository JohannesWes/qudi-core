import numpy as np
import importlib.util
import sys
from datetime import datetime
import time
import matplotlib
import pandas as pd
import os

matplotlib.use("Qt5Agg")

from my_software.automation.qudi_remote_control import OdmrRemoteControl
from my_software.tools.fitting import fit_hyperfine


def folder():
    timenow = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folderpath = os.path.join("../test/", "folder", str(timenow))
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        print('Created:', folderpath)


# -------------------------------------------------------------------------------------------------------------------- #

def current_measurement(num_current_measurements, run_time_per_odmr_scan=5,
                        odmr_ranges=[[2.63e9, 2.66e9]], odmr_frequency_points=1000):
    # folder_name for current measurement
    timestamp = datetime.now()
    folder_name = "Current_Measurement_" + timestamp.strftime('%Y%m%d-%H%M-%S') + "/"


    odmr_remote = OdmrRemoteControl()

    # data arrays for storing the measured ODMR and frequency data
    odmr_voltages_array = np.zeros(shape=(len(odmr_ranges), num_current_measurements, odmr_frequency_points))
    odmr_frequencies_array = np.zeros(shape=(len(odmr_ranges), odmr_frequency_points))
    times = np.array([datetime.now() for i in range(num_current_measurements)], dtype='datetime64')

    # -------------------------------------------------------------------------------------------- #
    # MEASUREMENT
    # -------------------------------------------------------------------------------------------- #

    for measurement_index in range(num_current_measurements):

        for odmr_range_index, odmr_range in enumerate(odmr_ranges):
            # ODMR scan data are stored in the qudi->Data folder
            filename = folder_name + "measurement_" + str(measurement_index) + "A_" + "range_" + str(odmr_range) + "Hz"
            frequencies, odmr_voltages = odmr_remote.take_odmr_scan(filename, run_time_per_odmr_scan, odmr_range[0],
                                                                    odmr_range[1],
                                                                    odmr_frequency_points, save_data=True)

            odmr_frequencies_array[odmr_range_index] = frequencies
            odmr_voltages_array[odmr_range_index][measurement_index] = odmr_voltages
            times[measurement_index] = datetime.now()

    # additionally save raw data in numpy arrays
    try:
        np.save(folder_name + "save_odmr_times_array", odmr_frequencies_array)
        np.save(folder_name + "save_odmr_frequencies_array", odmr_frequencies_array)
        np.save(folder_name + "save_odmr_voltages_array", odmr_voltages_array)
    except:
        pass

    # MEASUREMENT FINISHED
    return times, odmr_frequencies_array, odmr_voltages_array


def fit_avg_position_lock_in_hyperfine(odmr_ranges, num_current_measurements, odmr_frequencies_array, odmr_voltages_array, min_feature_amplitude=0.02,
                                       min_feature_height=0.005,
                                       feature_fit_range=0.5e6, testing_flag=False):

    # data array for storing the "middle position/average position" of the 3 hyperfine dips for each odmr_range
    avg_odmr_pos = np.zeros(shape=(len(odmr_ranges), num_current_measurements))
    uncertainty_avg_odmr_pos = np.zeros(shape=(len(odmr_ranges), num_current_measurements))

    for measurement_index in range(num_current_measurements):
        for odmr_range_index, odmr_range in enumerate(odmr_ranges):
            peak_data, dip_data = fit_hyperfine(odmr_frequencies_array[odmr_range_index],
                                                odmr_voltages_array[odmr_range_index][measurement_index],
                                                min_feature_amplitude=min_feature_amplitude,
                                                min_feature_height=min_feature_height,
                                                feature_fit_range=feature_fit_range, plot_fitting=testing_flag)

            peak_positions, dip_positions = np.array(peak_data)[:, 0], np.array(dip_data)[:, 0]
            peak_uncertainties, dip_uncertainties = np.array(peak_data)[:, 1], np.array(dip_data)[:, 1]

            num_features = len(peak_positions) + len(dip_positions)
            try:
                avg_odmr_pos[odmr_range_index][measurement_index] = (np.mean(peak_positions) + np.mean(
                    dip_positions)) / 2
                uncertainty_avg_odmr_pos[odmr_range_index][measurement_index] = np.sqrt(
                    1 / num_features ** 2 * (np.sum(peak_uncertainties ** 2) + np.sum(dip_uncertainties ** 2)))
            except:
                pass

    return avg_odmr_pos, uncertainty_avg_odmr_pos, odmr_frequencies_array, odmr_voltages_array


if __name__ == '__main__':
    num_current_measurements = 60
    run_time_per_odmr_scan = 3

    # for some reason, the odmr ranges must be passed as floats and NOT as numpy floats. This probably is some problem
    # connected to rpyc in some way

    # this is the measurement range used in the calibration measurement
    odmr_ranges = [[2.638e9, 2.652e9]]
    odmr_frequency_points = 1000

    print("Measurement starting at " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    times, odmr_frequencies_array, odmr_voltages_array = current_measurement(num_current_measurements, run_time_per_odmr_scan=run_time_per_odmr_scan,
                                                                      odmr_ranges=odmr_ranges, odmr_frequency_points=odmr_frequency_points)

    avg_odmr_positions, uncertainty_avg_odmr_positions, odmr_frequencies_array, odmr_voltages_array = fit_avg_position_lock_in_hyperfine(
        odmr_ranges, num_current_measurements, odmr_frequencies_array, odmr_voltages_array, testing_flag=False)

    print("Measurement finished at " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    columns = {f"odmr_peak_pos_{i}[Hz]": avg_odmr_positions[i] for i in range(len(avg_odmr_positions))}
    data = pd.DataFrame({"times": times} | columns)
    data.to_csv("battery_current_measurement_" + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + "_.csv", index=False)

    columns_uncertainty = {f"odmr_peak_pos_uncertainty_{i}[Hz]": uncertainty_avg_odmr_positions[i] for i in
                           range(len(uncertainty_avg_odmr_positions))}
    data_uncertainty = pd.DataFrame({"times": times} | columns_uncertainty)
    data_uncertainty.to_csv(
        "battery_current_measurement_uncertainty" + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + "_.csv", index=False)

    print('Measurement finished')
