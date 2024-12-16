import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from zhinst.toolkit import Session
import logging

matplotlib.use("Qt5Agg")

logger = logging.getLogger(__name__)

# Constants (can be parameterized or moved to a config file)
DEFAULT_SERVER_HOST = '192.168.113.190'
DEFAULT_DEVICE_ID = "DEV7279"
DEFAULT_SAMPLING_RATE = 20e3  # Hz
DEFAULT_TIMEOUT = 60  # seconds, maximum wait for acquisition to finish
DEFAULT_FILTER_PLOT = False

class LockInAmp:
    """
    Class for controlling and acquiring demodulated data from a Zurich Instruments
    Lock-In Amplifier using the zhinst-toolkit.

    Attributes:
        server_host (str): The IP address or hostname of the Zurich Instruments server.
        device_id (str): The device identifier (e.g., "DEV7279").
        session (Session): The toolkit session associated with the device.
        device (ToolkitDevice): The connected device object.
        daq_module (DAQModule): The DAQ module used for data acquisition.
    """

    def __init__(self, server_host=DEFAULT_SERVER_HOST, device_id=DEFAULT_DEVICE_ID):
        self.server_host = server_host
        self.device_id = device_id
        self.session = None
        self.device = None
        self.daq_module = None
        self._connect()

    def _connect(self):
        """
        Establishes a session and connects to the specified device.
        """
        try:
            self.session = Session(self.server_host)
            self.device = self.session.connect_device(self.device_id)
            self.daq_module = self.session.modules.daq
            logger.info(f"Connected to device {self.device_id} at {self.server_host}.")
        except Exception as e:
            logger.exception(f"Failed to connect to device {self.device_id} at {self.server_host}: {e}")
            raise


    def _configure_daq_module(self, total_duration, n_samples):
        """
        Configures the DAQ module for continuous acquisition with given parameters.

        Args:
            total_duration (float): Total duration of the data acquisition in seconds.
            n_samples (int): Number of samples to acquire.
        """
        self.daq_module.device(self.device)
        self.daq_module.type(0)  # Continuous acquisition
        self.daq_module.grid.mode(2)  # Linear interpolation
        self.daq_module.count(1)  # Acquire one segment in single-shot mode
        self.daq_module.duration(total_duration)
        self.daq_module.grid.cols(n_samples)  # Number of columns/samples in the returned data grid

    def _wait_for_acquisition(self, timeout=DEFAULT_TIMEOUT):
        """
        Wait until the data acquisition is finished or a timeout occurs.

        Args:
            timeout (int): Maximum number of seconds to wait for acquisition.

        Raises:
            TimeoutError: If acquisition does not finish within the given timeout.
        """
        start_time = time.time()
        while not self.daq_module.raw_module.finished():
            if (time.time() - start_time) > timeout:
                raise TimeoutError("Data acquisition did not finish within the timeout period.")
            time.sleep(0.5)

    def demod_data_acquisition(self, total_duration, n_samples, timeout=DEFAULT_TIMEOUT):
        """
        Perform demodulated data acquisition using the DAQ module.

        Args:
            total_duration (float): Total duration of the data acquisition in seconds.
            n_samples (int): Number of samples to acquire.
            timeout (int): Maximum wait time in seconds for acquisition to finish.

        Returns:
            dict: A dictionary containing the acquired data and metadata.
        """
        # Basic input validation
        if total_duration <= 0:
            raise ValueError("Total duration must be positive.")
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive.")

        self._configure_daq_module(total_duration, n_samples)

        # Subscribe to demodulator sample nodes
        demod_sample_nodes = [
            self.device.demods[0].sample.x,
            self.device.demods[0].sample.y
        ]

        for node in demod_sample_nodes:
            self.daq_module.subscribe(node)

        # Start the acquisition
        self.daq_module.execute()
        logger.info("Data acquisition started.")

        # Wait until finished or timeout
        self._wait_for_acquisition(timeout=timeout)

        # Read out the result
        result = self.daq_module.read(raw=False, clk_rate=self.device.clockbase())
        logger.info("Data acquisition finished and results read.")

        # Extract time, x, and y data
        times = result[demod_sample_nodes[0]][0].time
        x_value = result[demod_sample_nodes[0]][0].value[0]
        y_value = result[demod_sample_nodes[1]][0].value[0]

        # Retrieve filter and scaling metadata
        filter_order = self.device.demods[0].order()
        filter_time_constant = self.device.demods[0].timeconstant()
        filter_3db_freq = np.sqrt(2**(1/filter_order) - 1)/(2*np.pi*filter_time_constant)
        filter_sinc = self.device.demods[0].sinc()
        aux_0_scaling = self.device.auxouts[0].scale()

        # Unsubscribe from nodes to clean up
        for node in demod_sample_nodes:
            self.daq_module.unsubscribe(node)

        return {
            "result_object": result,
            "times [s]": times,
            "x_value [V]": x_value,
            "y_value [V]": y_value,
            "filter_order": filter_order,
            "filter_3db_freq [Hz]": filter_3db_freq,
            "filter_sinc": filter_sinc,
            "aux_0_scaling": aux_0_scaling
        }

    def plot_demod_time_traces(self, times, x_data, y_data, title="Data acquired through the DAQ module"):
        """
        Plot the demodulated time traces.

        Args:
            times (array-like): Time values in seconds.
            x_data (array-like): Demodulated x-values in Volts.
            y_data (array-like): Demodulated y-values in Volts.
            title (str): Title of the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(times, x_data, label="X-Channel [V]")
        ax.plot(times, y_data, label="Y-Channel [V]", alpha=0.7)
        ax.grid(True)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Signal [V]")
        plt.tight_layout()
        plt.show()

    def save_raw_data(self, filename_prefix, times, x_data, y_data, sep="\t"):
        """
        Save raw time traces to a CSV file.

        Args:
            filename_prefix (str): The base filename (without extension).
            times (array-like): Time values.
            x_data (array-like): X demodulated data.
            y_data (array-like): Y demodulated data.
            sep (str): Field delimiter for CSV.
        """
        try:
            df = pd.DataFrame({
                "times [s]": times,
                "x_value [V]": x_data,
                "y_value [V]": y_data
            })
            csv_filename = filename_prefix + ".csv"
            df.to_csv(csv_filename, sep=sep, index=False)
            logger.info(f"Raw data saved to {csv_filename}.")
        except Exception as e:
            logger.exception(f"Error saving raw data to {filename_prefix}: {e}")

    def save_metadata(self, filename_prefix, metadata):
        """
        Save metadata to a JSON file.

        Args:
            filename_prefix (str): The base filename (without extension).
            metadata (dict): Dictionary containing metadata.
        """
        try:
            json_filename = filename_prefix + "_metadata.json"
            with open(json_filename, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata saved to {json_filename}.")
        except Exception as e:
            logger.exception(f"Error saving metadata to {filename_prefix}: {e}")

    def sensitivity_measurement(self,
                                filename_prefix,
                                n_time_traces=32,
                                sampling_rate=DEFAULT_SAMPLING_RATE,
                                plot_data=False,
                                save_raw_data=False,
                                save_metadata=True,
                                timeout=DEFAULT_TIMEOUT):
        """
        Perform a sensitivity measurement by acquiring demodulated data over several time traces.

        Args:
            filename_prefix (str): Base filename for saving data and metadata.
            n_time_traces (int): Number of time traces to acquire.
            sampling_rate (float): Sampling rate in Hz.
            plot_data (bool): Whether to plot the acquired data.
            save_raw_data (bool): Whether to save the raw data to a CSV file.
            save_metadata (bool): Whether to save the metadata to a JSON file.
            timeout (int): Max wait time in seconds for acquisition to finish.

        Returns:
            dict: Dictionary containing acquisition data and metadata.
        """
        if n_time_traces <= 0:
            raise ValueError("Number of time traces must be positive.")
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive.")

        total_duration = n_time_traces  # e.g. 1 second per time trace
        n_samples = int(sampling_rate * n_time_traces)

        logger.info("Starting sensitivity measurement.")
        # Acquire data
        acq_data = self.demod_data_acquisition(total_duration=total_duration,
                                               n_samples=n_samples,
                                               timeout=timeout)

        times = acq_data["times [s]"]
        x_value = acq_data["x_value [V]"]
        y_value = acq_data["y_value [V]"]

        # Prepare metadata
        metadata = {
            "filter_order": acq_data["filter_order"],
            "filter_3db_freq [Hz]": acq_data["filter_3db_freq [Hz]"],
            "filter_sinc": acq_data["filter_sinc"],
            "aux_0_scaling": acq_data["aux_0_scaling"],
            "duration [s]": total_duration,
            "sampling_rate [Hz]": sampling_rate,
            "n_time_traces": n_time_traces
        }

        # Save data if requested
        if save_raw_data:
            self.save_raw_data(filename_prefix, times, x_value, y_value)

        if save_metadata:
            self.save_metadata(filename_prefix, metadata)

        # Plot data if requested
        if plot_data:
            self.plot_demod_time_traces(times, x_value, y_value, title="Sensitivity Measurement")

        logger.info("Sensitivity measurement completed.")
        return {
            "acq_data": acq_data,
            "times [s]": times,
            "x_value [V]": x_value,
            "y_value [V]": y_value,
            "metadata": metadata
        }


if __name__ == "__main__":
    # Example usage (assuming logging is configured in the main script):
    LIA = LockInAmp()
    LIA.sensitivity_measurement("test_measurement",
                                    n_time_traces=1,
                                    save_raw_data=True,
                                    plot_data=True)

