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
            filename = folder_name + "measurement_" + str(measurement_index) + "ODMRrange_" + str(odmr_range) + "Hz"
            frequencies, odmr_voltages = odmr_remote.take_odmr_scan(filename, run_time_per_odmr_scan, odmr_range[0],
                                                                    odmr_range[1],
                                                                    odmr_frequency_points, save_data=True)

            odmr_frequencies_array[odmr_range_index] = frequencies
            odmr_voltages_array[odmr_range_index][measurement_index] = odmr_voltages
        times[measurement_index] = datetime.now()

    # additionally save raw data in numpy arrays
    try:
        np.save(folder_name + "save_odmr_times_array", times)
        np.save(folder_name + "save_odmr_frequencies_array", odmr_frequencies_array)
        np.save(folder_name + "save_odmr_voltages_array", odmr_voltages_array)
    except:
        pass

    # MEASUREMENT FINISHED
    return times, odmr_frequencies_array, odmr_voltages_array


def fit_avg_position_lock_in_hyperfine(odmr_ranges, num_current_measurements, odmr_frequencies_array,
                                       odmr_voltages_array, min_feature_amplitude=0.02, fit_intervals=None,
                                       min_feature_height=0.005,
                                       feature_fit_range=0.5e6, testing_flag=False):
    # data array for storing the "middle position/average position" of the 3 hyperfine dips for each odmr_range
    avg_odmr_pos = np.zeros(shape=(len(odmr_ranges), num_current_measurements))
    uncertainty_avg_odmr_pos = np.zeros(shape=(len(odmr_ranges), num_current_measurements))

    if not fit_intervals == None:
        for odmr_range_index, odmr_range in enumerate(odmr_ranges):
            no_fit_indices = np.where(np.logical_or(odmr_frequencies_array[odmr_range_index] < fit_intervals[odmr_range_index][0],
                                                  odmr_frequencies_array[odmr_range_index] > fit_intervals[odmr_range_index][1]))

            # set voltages outside of fit interval to 0, such that no fit is done there
            odmr_voltages_array[odmr_range_index][no_fit_indices] = 0

    for measurement_index in range(num_current_measurements):
        for odmr_range_index, odmr_range in enumerate(odmr_ranges):
            peak_data, dip_data = fit_hyperfine(odmr_frequencies_array[odmr_range_index],
                                                odmr_voltages_array[odmr_range_index][measurement_index],
                                                min_feature_amplitude=min_feature_amplitude,
                                                min_feature_height=min_feature_height,
                                                feature_fit_range=feature_fit_range, plot_fitting=testing_flag)

            peak_positions, dip_positions = np.array(peak_data)[:, 0], np.array(dip_data)[:, 0]
            peak_uncertainties, dip_uncertainties = np.array(peak_data)[:, 1], np.array(dip_data)[:, 1]

            # don't use last peak and first dip for average position, fitting not so good there
            peak_positions, peak_uncertainties = peak_positions[:-1], peak_uncertainties[:-1]
            dip_positions, dip_uncertainties = dip_positions[1:], dip_uncertainties[1:]

            if len(peak_positions) != 2 or len(dip_positions) != 2:
                print(
                    f"Error in fitting hyperfine structure. Len(peak_positions): {len(peak_positions)}, Len(dip_positions): {len(dip_positions)}")
                print(odmr_range)
            num_features = len(peak_positions) + len(dip_positions)
            try:
                avg_odmr_pos[odmr_range_index][measurement_index] = (np.mean(peak_positions) + np.mean(
                    dip_positions)) / 2
                uncertainty_avg_odmr_pos[odmr_range_index][measurement_index] = np.sqrt(
                    1 / num_features ** 2 * (np.sum(peak_uncertainties ** 2) + np.sum(dip_uncertainties ** 2)))
            except:
                print("Error in calculating average position")
                pass

    return avg_odmr_pos, uncertainty_avg_odmr_pos, odmr_frequencies_array, odmr_voltages_array


if __name__ == '__main__':

    # PARAMETERS
    num_current_measurements = 2000
    pre_name = "REAL_MSMT"

    testing_flag = False

    just_fitting_flag = True

    run_time_per_odmr_scan = 2 # [s]

    min_feature_amplitude = 0.25  # [V]
    min_feature_amplitude = 0.05  # [V]

    min_feature_height = 0.2  # [V]
    min_feature_height = 0.025  # [V]

    feature_fit_range = 0.3e6  # [Hz]

    # for some reason, the odmr ranges must be passed as floats and NOT as numpy floats. This probably is some problem
    # connected to rpyc in some way

    # this is the measurement range used in the calibration measurement
    # odmr_ranges = [[2.640000e9, 2.6511e9]]
    odmr_ranges = [[2.640000e9, 2.6511e9], [2.721e9, 2.731e9]]
    odmr_ranges = [[2.51e9, 2.54e9], [2.54e9, 2.56e9], [2.63e9, 2.66e9], [2.66e9, 2.68e9], [2.69e9, 2.715e9],
                   [2.715e9, 2.74e9], [2.79e9, 2.815e9], [2.815e9, 2.835e9]]

    odmr_ranges = [[2.64e9, 2.655e9],
                   [2.718e9, 2.734e9]]

    odmr_frequency_points = 500

    fit_ranges = np.array([[2.6438e9, 2.6528e9], [2.72e9, 2.7295e9]])

    print("Measurement starting at " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    if not just_fitting_flag:
        times, odmr_frequencies_array, odmr_voltages_array = current_measurement(num_current_measurements,
                                                                                 run_time_per_odmr_scan=run_time_per_odmr_scan,
                                                                                 odmr_ranges=odmr_ranges,
                                                                                 odmr_frequency_points=odmr_frequency_points)

    else:
        times, odmr_frequencies_array, odmr_voltages_array = np.load("save_odmr_times_array.npy"), np.load(
            "save_odmr_frequencies_array.npy"), np.load("save_odmr_voltages_array.npy")

    avg_odmr_positions, uncertainty_avg_odmr_positions, odmr_frequencies_array, odmr_voltages_array = fit_avg_position_lock_in_hyperfine(
        odmr_ranges, num_current_measurements, odmr_frequencies_array, odmr_voltages_array, testing_flag=testing_flag,
        min_feature_amplitude=min_feature_amplitude, min_feature_height=min_feature_height,
        feature_fit_range=feature_fit_range, fit_intervals=fit_ranges)

    print("Measurement finished at " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    columns = {f"odmr_peak_pos_{i}[Hz]": avg_odmr_positions[i] for i in range(len(avg_odmr_positions))}
    data = pd.DataFrame({"times": times} | columns)
    data.to_csv(pre_name + "battery_current_measurement_" + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + "_.csv", index=False)

    columns_uncertainty = {f"odmr_peak_pos_uncertainty_{i}[Hz]": uncertainty_avg_odmr_positions[i] for i in
                           range(len(uncertainty_avg_odmr_positions))}
    data_uncertainty = pd.DataFrame({"times": times} | columns_uncertainty)
    data_uncertainty.to_csv(
        "battery_current_measurement_uncertainty" + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + "_.csv",
        index=False)

    print('Measurement finished')




    # MESSUNG ZUR ANALYSE DER MAGNETISIERUNG

    # for funny_index in range(10):
    #     # PARAMETERS
    #     num_current_measurements = 30
    #     testing_flag = False
    #
    #     just_fitting_flag = False
    #
    #     run_time_per_odmr_scan = 1  # [s]
    #
    #     min_feature_amplitude = 0.25  # [V]
    #     min_feature_amplitude = 0.05  # [V]
    #
    #     min_feature_height = 0.2  # [V]
    #     min_feature_height = 0.025  # [V]
    #
    #     feature_fit_range = 0.3e6  # [Hz]
    #
    #     # for some reason, the odmr ranges must be passed as floats and NOT as numpy floats. This probably is some problem
    #     # connected to rpyc in some way
    #
    #     # this is the measurement range used in the calibration measurement
    #     # odmr_ranges = [[2.640000e9, 2.6511e9]]
    #     odmr_ranges = [[2.640000e9, 2.6511e9], [2.721e9, 2.731e9]]
    #     odmr_ranges = [[2.51e9, 2.54e9], [2.54e9, 2.56e9], [2.63e9, 2.66e9], [2.66e9, 2.68e9], [2.69e9, 2.715e9],
    #                    [2.715e9, 2.74e9], [2.79e9, 2.815e9], [2.815e9, 2.835e9]]
    #     odmr_ranges = [[2.64e9, 2.655e9],
    #                    [2.718e9, 2.734e9]]
    #
    #     odmr_frequency_points = 500
    #
    #     print("Measurement starting at " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    #
    #     if not just_fitting_flag:
    #         times, odmr_frequencies_array, odmr_voltages_array = current_measurement(num_current_measurements,
    #                                                                                  run_time_per_odmr_scan=run_time_per_odmr_scan,
    #                                                                                  odmr_ranges=odmr_ranges,
    #                                                                                  odmr_frequency_points=odmr_frequency_points)
    #
    #     else:
    #         times, odmr_frequencies_array, odmr_voltages_array = np.load("save_odmr_times_array.npy"), np.load(
    #             "save_odmr_frequencies_array.npy"), np.load("save_odmr_voltages_array.npy")
    #
    #     avg_odmr_positions, uncertainty_avg_odmr_positions, odmr_frequencies_array, odmr_voltages_array = fit_avg_position_lock_in_hyperfine(
    #         odmr_ranges, num_current_measurements, odmr_frequencies_array, odmr_voltages_array, testing_flag=testing_flag,
    #         min_feature_amplitude=min_feature_amplitude, min_feature_height=min_feature_height,
    #         feature_fit_range=feature_fit_range)
    #
    #     print("Measurement finished at " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    #
    #     columns = {f"odmr_peak_pos_{i}[Hz]": avg_odmr_positions[i] for i in range(len(avg_odmr_positions))}
    #     data = pd.DataFrame({"times": times} | columns)
    #     data.to_csv("BACK_TO_BACK_battery_current_measurement_" + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + "_.csv", index=False)
    #
    #     columns_uncertainty = {f"odmr_peak_pos_uncertainty_{i}[Hz]": uncertainty_avg_odmr_positions[i] for i in
    #                            range(len(uncertainty_avg_odmr_positions))}
    #     data_uncertainty = pd.DataFrame({"times": times} | columns_uncertainty)
    #     data_uncertainty.to_csv(
    #         "battery_current_measurement_uncertainty" + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + "_.csv",
    #         index=False)
    #
    #     print('Measurement finished')
    #
    # time.sleep(300)
    #
    # for funny_index in range(10):
    #     time.sleep(30)
    #     # PARAMETERS
    #     num_current_measurements = 30
    #     testing_flag = False
    #
    #     just_fitting_flag = False
    #
    #     run_time_per_odmr_scan = 1  # [s]
    #
    #     min_feature_amplitude = 0.25  # [V]
    #     min_feature_amplitude = 0.05  # [V]
    #
    #     min_feature_height = 0.2  # [V]
    #     min_feature_height = 0.025  # [V]
    #
    #     feature_fit_range = 0.3e6  # [Hz]
    #
    #     # for some reason, the odmr ranges must be passed as floats and NOT as numpy floats. This probably is some problem
    #     # connected to rpyc in some way
    #
    #     # this is the measurement range used in the calibration measurement
    #     # odmr_ranges = [[2.640000e9, 2.6511e9]]
    #     odmr_ranges = [[2.640000e9, 2.6511e9], [2.721e9, 2.731e9]]
    #     odmr_ranges = [[2.51e9, 2.54e9], [2.54e9, 2.56e9], [2.63e9, 2.66e9], [2.66e9, 2.68e9], [2.69e9, 2.715e9],
    #                    [2.715e9, 2.74e9], [2.79e9, 2.815e9], [2.815e9, 2.835e9]]
    #     odmr_ranges = [[2.64e9, 2.655e9],
    #                    [2.718e9, 2.734e9]]
    #
    #     odmr_frequency_points = 500
    #
    #     print("Measurement starting at " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    #
    #     if not just_fitting_flag:
    #         times, odmr_frequencies_array, odmr_voltages_array = current_measurement(num_current_measurements,
    #                                                                                  run_time_per_odmr_scan=run_time_per_odmr_scan,
    #                                                                                  odmr_ranges=odmr_ranges,
    #                                                                                  odmr_frequency_points=odmr_frequency_points)
    #
    #     else:
    #         times, odmr_frequencies_array, odmr_voltages_array = np.load("save_odmr_times_array.npy"), np.load(
    #             "save_odmr_frequencies_array.npy"), np.load("save_odmr_voltages_array.npy")
    #
    #     avg_odmr_positions, uncertainty_avg_odmr_positions, odmr_frequencies_array, odmr_voltages_array = fit_avg_position_lock_in_hyperfine(
    #         odmr_ranges, num_current_measurements, odmr_frequencies_array, odmr_voltages_array,
    #         testing_flag=testing_flag,
    #         min_feature_amplitude=min_feature_amplitude, min_feature_height=min_feature_height,
    #         feature_fit_range=feature_fit_range)
    #
    #     print("Measurement finished at " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    #
    #     columns = {f"odmr_peak_pos_{i}[Hz]": avg_odmr_positions[i] for i in range(len(avg_odmr_positions))}
    #     data = pd.DataFrame({"times": times} | columns)
    #     data.to_csv("WAITING_IN_BETWEEN_battery_current_measurement_" + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + "_.csv",
    #                 index=False)
    #
    #     columns_uncertainty = {f"odmr_peak_pos_uncertainty_{i}[Hz]": uncertainty_avg_odmr_positions[i] for i in
    #                            range(len(uncertainty_avg_odmr_positions))}
    #     data_uncertainty = pd.DataFrame({"times": times} | columns_uncertainty)
    #     data_uncertainty.to_csv(
    #         "battery_current_measurement_uncertainty" + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + "_.csv",
    #         index=False)
    #
    #     print('Measurement finished')
    #
    #
    #
    # # PARAMETERS
    # num_current_measurements = 500
    # testing_flag = False
    #
    # just_fitting_flag = False
    #
    # run_time_per_odmr_scan = 1  # [s]
    #
    # min_feature_amplitude = 0.25  # [V]
    # min_feature_amplitude = 0.05  # [V]
    #
    # min_feature_height = 0.2  # [V]
    # min_feature_height = 0.025  # [V]
    #
    # feature_fit_range = 0.3e6  # [Hz]
    #
    # # for some reason, the odmr ranges must be passed as floats and NOT as numpy floats. This probably is some problem
    # # connected to rpyc in some way
    #
    # # this is the measurement range used in the calibration measurement
    # # odmr_ranges = [[2.640000e9, 2.6511e9]]
    # odmr_ranges = [[2.640000e9, 2.6511e9], [2.721e9, 2.731e9]]
    # odmr_ranges = [[2.51e9, 2.54e9], [2.54e9, 2.56e9], [2.63e9, 2.66e9], [2.66e9, 2.68e9], [2.69e9, 2.715e9],
    #                [2.715e9, 2.74e9], [2.79e9, 2.815e9], [2.815e9, 2.835e9]]
    # odmr_ranges = [[2.64e9, 2.655e9],
    #                [2.718e9, 2.734e9]]
    #
    # odmr_frequency_points = 500
    #
    # print("Measurement starting at " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    #
    # if not just_fitting_flag:
    #     times, odmr_frequencies_array, odmr_voltages_array = current_measurement(num_current_measurements,
    #                                                                              run_time_per_odmr_scan=run_time_per_odmr_scan,
    #                                                                              odmr_ranges=odmr_ranges,
    #                                                                              odmr_frequency_points=odmr_frequency_points)
    #
    # else:
    #     times, odmr_frequencies_array, odmr_voltages_array = np.load("save_odmr_times_array.npy"), np.load(
    #         "save_odmr_frequencies_array.npy"), np.load("save_odmr_voltages_array.npy")
    #
    # avg_odmr_positions, uncertainty_avg_odmr_positions, odmr_frequencies_array, odmr_voltages_array = fit_avg_position_lock_in_hyperfine(
    #     odmr_ranges, num_current_measurements, odmr_frequencies_array, odmr_voltages_array, testing_flag=testing_flag,
    #     min_feature_amplitude=min_feature_amplitude, min_feature_height=min_feature_height,
    #     feature_fit_range=feature_fit_range)
    #
    # print("Measurement finished at " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    #
    # columns = {f"odmr_peak_pos_{i}[Hz]": avg_odmr_positions[i] for i in range(len(avg_odmr_positions))}
    # data = pd.DataFrame({"times": times} | columns)
    # data.to_csv("LONG_TIME_battery_current_measurement_" + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + "_.csv", index=False)
    #
    # columns_uncertainty = {f"odmr_peak_pos_uncertainty_{i}[Hz]": uncertainty_avg_odmr_positions[i] for i in
    #                        range(len(uncertainty_avg_odmr_positions))}
    # data_uncertainty = pd.DataFrame({"times": times} | columns_uncertainty)
    # data_uncertainty.to_csv(
    #     "battery_current_measurement_uncertainty" + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + "_.csv",
    #     index=False)

    print('Measurement finished')