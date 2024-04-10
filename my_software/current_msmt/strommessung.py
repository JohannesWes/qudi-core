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
from my_software.automation.NGP_control import NGP_instance

# import fitting tools from my tools module
path_to_mod = "C:\\Users\\aj92uwef\\PycharmProjects\\johannes_tools\\data_analysis\\fitting.py"
spec = importlib.util.spec_from_file_location("fitting", path_to_mod)
johannes_tools = importlib.util.module_from_spec(spec)
sys.modules["fitting"] = johannes_tools
spec.loader.exec_module(johannes_tools)
from fitting import fit_hyperfine  # noqa


def folder():
    timenow = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folderpath = os.path.join("../test/", "folder", str(timenow))
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        print('Created:', folderpath)


# -------------------------------------------------------------------------------------------------------------------- #

def current_measurement(current_min, current_max, current_points, run_time_per_current_point=5,
                        odmr_ranges=[[2.6368e9, 2.651e9]], odmr_frequency_points=1000):
    # folder_name for current measurement
    timestamp = datetime.now()
    folder_name = "Current_Measurement_" + timestamp.strftime('%Y%m%d-%H%M-%S') + "/"

    # setting up the R&S NGP power supply
    ngp_active_channel = 4
    ngp = NGP_instance()
    ngp.activate_channel(ngp_active_channel)
    ngp.output_on()

    odmr_remote = OdmrRemoteControl()

    # define currents to be applied
    current_array = np.linspace(current_min, current_max, current_points)

    # data arrays for storing the measured ODMR and frequency data
    odmr_voltages_array = np.zeros(shape=(len(odmr_ranges), current_points, odmr_frequency_points))
    odmr_frequencies_array = np.zeros(shape=(len(odmr_ranges), odmr_frequency_points))

    # -------------------------------------------------------------------------------------------- #
    # MEASUREMENT
    # -------------------------------------------------------------------------------------------- #

    for current_index, current in enumerate(current_array):
        ngp.set_current(ngp_active_channel, current)
        time.sleep(0.1)

        for odmr_range_index, odmr_range in enumerate(odmr_ranges):
            # ODMR scan data are stored in the qudi->Data folder
            filename = folder_name + "current_" + str(current) + "A_" + "NVaxis_" + str(odmr_range_index)
            frequencies, odmr_voltages = odmr_remote.take_odmr_scan(filename, run_time_per_current_point, odmr_range[0],
                                                                    odmr_range[1],
                                                                    odmr_frequency_points, save_data=True)

            odmr_frequencies_array[odmr_range_index] = frequencies
            odmr_voltages_array[odmr_range_index][current_index] = odmr_voltages

    # additionally save raw data in numpy arrays
    try:
        np.save("save_odmr_frequencies_array", odmr_frequencies_array)
        np.save("save_odmr_voltages_array", odmr_voltages_array)
    except:
        pass

    # MEASUREMENT FINISHED
    ngp.close()
    print('NGP closed')

    return odmr_frequencies_array, odmr_voltages_array


def fit(odmr_ranges, current_array, odmr_frequencies_array, odmr_voltages_array, min_feature_amplitude=0.01,
        min_feature_height=0.005,
        feature_fit_range=0.3e6, testing_flag=True):
    # data array for storing the "middle position/average position" of the 3 hyperfine dips for each odmr_range
    avg_odmr_pos = np.zeros(shape=(len(odmr_ranges), current_points))
    uncertainty_avg_odmr_pos = np.zeros(shape=(len(odmr_ranges), current_points))

    for current_index, current in enumerate(current_array):
        for odmr_range_index, odmr_range in enumerate(odmr_ranges):
            peak_data, dip_data = fit_hyperfine(odmr_frequencies_array[odmr_range_index],
                                                odmr_voltages_array[odmr_range_index][current_index],
                                                min_feature_amplitude=min_feature_amplitude,
                                                min_feature_height=min_feature_height,
                                                feature_fit_range=feature_fit_range, testing=testing_flag)

            peak_positions, dip_positions = np.array(peak_data)[:, 0], np.array(dip_data)[:, 0]
            peak_uncertainties, dip_uncertainties = np.array(peak_data)[:, 1], np.array(dip_data)[:, 1]

            num_features = len(peak_positions) + len(dip_positions)
            try:
                avg_odmr_pos[odmr_range_index][current_index] = (np.mean(peak_positions) + np.mean(
                    dip_positions)) / 2
                uncertainty_avg_odmr_pos[odmr_range_index][current_index] = np.sqrt(
                    1 / num_features ** 2 * (np.sum(peak_uncertainties ** 2) + np.sum(dip_uncertainties ** 2)))
            except:
                pass

    return current_array, avg_odmr_pos, uncertainty_avg_odmr_pos, odmr_frequencies_array, odmr_voltages_array


if __name__ == '__main__':
    current_min, current_max, current_points = 0.001, 0.3, 1
    current_array = np.linspace(current_min, current_max, current_points)
    run_time_per_odmr_scan = 2

    # for some reason, the odmr ranges must be passed as floats and NOT as numpy floats. This probably is some problem
    # connected to rpyc in some way
    odmr_ranges = [[2.51e9, 2.54e9], [2.54e9, 2.56e9], [2.63e9, 2.66e9], [2.66e9, 2.68e9], [2.69e9, 2.715e9],
                   [2.715e9, 2.74e9], [2.79e9, 2.815e9], [2.815e9, 2.835e9]]
    # odmr_ranges = [[2.51e9, 2.54e9], [2.54e9, 2.56e9]]  # for testing

    print("Measurement starting at " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    odmr_frequencies_array, odmr_voltages_array = current_measurement(current_min, current_max, current_points,
                                                                      odmr_ranges=odmr_ranges,
                                                                      run_time_per_current_point=run_time_per_odmr_scan)

    current_array, avg_odmr_positions, uncertainty_avg_odmr_positions, odmr_frequencies_array, odmr_voltages_array = fit(
        odmr_ranges, current_array, odmr_frequencies_array, odmr_voltages_array)

    print("Measurement finished at " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    columns = {f"odmr_peak_pos_{i}[Hz]": avg_odmr_positions[i] for i in range(len(avg_odmr_positions))}
    data = pd.DataFrame({"current[A]": current_array} | columns)
    data.to_csv("current_measurement_" + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + "_.csv", index=False,
                sep="\t")

    columns_uncertainty = {f"odmr_peak_pos_uncertainty_{i}[Hz]": uncertainty_avg_odmr_positions[i] for i in
                           range(len(uncertainty_avg_odmr_positions))}
    data_uncertainty = pd.DataFrame({"current[A]": current_array} | columns_uncertainty)
    data_uncertainty.to_csv(
        "current_measurement_uncertainty" + str(datetime.now().strftime('%Y-%m-%d_%H%M%S')) + "_.csv", index=False,
        sep="\t")

    print('Measurement finished')
