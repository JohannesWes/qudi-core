import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
from zhinst.toolkit import Session
#from my_software.automation.qudi_remote_control import OdmrRemoteControl

matplotlib.use("Qt5Agg")

# Constants
SERVER_HOST = '192.168.113.190'
DEVICE_ID = "DEV7279"


class lock_in_amp():
    def __init__(self):
        # Create a session, connect to the device and the DAQ module
        self.session = Session(SERVER_HOST)
        self.device = self.session.connect_device(DEVICE_ID)
        self.daq_module = self.session.modules.daq

    def demod_data_acquisition(self, total_duration, n_samples):
        """
        Perform demodulated data acquisition using a DAQ module.

        Args:
            total_duration (float): Total duration of the data acquisition in seconds.
            n_samples (int): Number of samples to acquire.

        Returns:
            dict: A dictionary containing the following information:
                - result_object: The result object returned by the DAQ module.
                - times: The time values of the acquired data.
                - x_value: The demodulated x-value.
                - y_value: The demodulated y-value.
                - filter_order: The order of the demodulator filter.
                - filter_3db_freq: The 3dB cutoff frequency of the demodulator filter.
                - filter_sinc: The sinc filter setting of the demodulator (0/1)
                - aux_0_scaling: The scaling factor of the auxiliary output 0.
        """


        self.daq_module.device(self.device)
        self.daq_module.type(0)  # Continuous acquisition
        self.daq_module.grid.mode(2)  # Linear interpolation
        self.daq_module.count(1)  # Number of trigger events to acquire in single-shot mode
        self.daq_module.duration(total_duration)
        self.daq_module.grid.cols(n_samples)  # Number of columns/samples in the returned data grid

        # subscribe to the demodulator sample nodes
        demod_sample_nodes = [
            self.device.demods[0].sample.x,
            self.device.demods[0].sample.y,
        ]
        for node in demod_sample_nodes:
            self.daq_module.subscribe(node)

        # Start the DAQ module and wait until the data acquisition finished
        self.daq_module.execute()
        while not self.daq_module.raw_module.finished():
            time.sleep(1)

        result = self.daq_module.read(raw=False, clk_rate=self.device.clockbase())

        # note that the demod data is NOT scaled by the aux_0_scaling factor. However the lock-in ODMR that we measure
        # is scaled by that factor
        times = result[demod_sample_nodes[0]][0].time
        x_value = result[demod_sample_nodes[0]][0].value[0]
        y_value = result[demod_sample_nodes[1]][0].value[0]

        filter_order = self.device.demods[0].order()
        filter_time_constant = self.device.demods[0].timeconstant()
        filter_3db_freq = np.sqrt(2**(1/filter_order) - 1)/(2*np.pi*filter_time_constant)
        filter_sinc = self.device.demods[0].sinc()
        aux_0_scaling = self.device.auxouts[0].scale()

        return {"result_object": result, "times [s]": times, "x_value [V]": x_value, "y_value [V]": y_value, "filter_order": filter_order,
                "filter_3db_freq [Hz]": filter_3db_freq, "filter_sinc": filter_sinc, "aux_0_scaling": aux_0_scaling}


    # def save_results(filename, result):
    #     """Save results to a pickle file."""
    #     with open(filename, 'wb') as f:
    #         pickle.dump(result, f)

    def plot_demod_time_traces(self, times, x, y):
        """Plot the results using matplotlib."""
        _, axis = plt.subplots(1, 1)

        for data in (x, y):
            axis.plot(
                times,
                data,
                label="x"
            )

        axis.grid(True)
        axis.legend()
        axis.set_title("Data acquired through the DAQ module")
        axis.set_xlabel("Time(s)")
        axis.set_ylabel("Signal(V)")
        plt.show()

    def sensitivity_measurement(self, file_name_save, n_time_traces=32, plot_demod=False,save_data=False):
        """Perform a sensitivity measurement."""

        total_duration = 1 * n_time_traces  # [s]
        sampling_rate = 20e3  # [Hz]
        n_samples = int(sampling_rate * n_time_traces)  # Number of points

        acq_data = self.demod_data_acquisition(total_duration=total_duration, n_samples=n_samples)
        times, x_value, y_value = acq_data["times [s]"], acq_data["x_value [V]"], acq_data["y_value [V]"]

        metadata = {"filter_order": acq_data["filter_order"], "filter_3db_freq [Hz]": acq_data["filter_3db_freq [Hz]"],
                    "filter_sinc": acq_data["filter_sinc"], "aux_0_scaling": acq_data["aux_0_scaling"], "duration": total_duration,
                    "sampling_rate":  sampling_rate}

        if save_data:
            pd_data = pd.DataFrame({"times [s]": times, "x_value [V]": x_value, "y_value [V]": y_value})
            pd_data.to_csv(file_name_save + ".csv", sep="\t")

            # store metadata in a json file
            with open(file_name_save + "_metadata.json", 'w') as f:
                json.dump(metadata, f)
            


        if plot_demod:
            self.plot_demod_time_traces(times, x_value, y_value)

        return {"acq_data": acq_data, "times": times, "x_value": x_value, "y_value": y_value, "metadata": metadata}


if __name__ == "__main__":
    LIA = lock_in_amp()
    LIA.sensitivity_measurement("test", n_time_traces=1, save_data=True, plot_demod=False)