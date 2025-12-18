# FILE: qudi_remote_control.py

# Implementing remote control of Qudi modules using RPyC
# Qudi and the gui windows for the modules (e.g. ODMR, TimeSeries) must be opened already.

import time
import numpy as np
import logging
import functools
import rpyc


def logger():
    return logging.getLogger(__name__)


def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger().info(f"Calling {func.__name__}...")
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            logger().debug(f"Calling {func.__name__} with args {signature}.")
            result = func(*args, **kwargs)
            logger().info(f"Execution of {func.__name__} resolved.")
            return result
        except Exception as e:
            logger().exception(f"Exception raised in {func.__name__}. exception: {str(e)}")
            raise e

    return wrapper


class QudiRemoteControl:
    """Base class for remote control of Qudi modules."""

    def __init__(self, host='localhost', port=12345, conn_config=None):
        if conn_config is None:
            conn_config = {'allow_all_attrs': True, 'allow_pickle': True}
        self.host = host
        self.port = port
        self.conn_config = conn_config
        self.connection = rpyc.connect(self.host, self.port, config=self.conn_config)
        self.modules = self.connection.root.exposed_get_available_module_names()

    def close_connection(self):
        self.connection.close()

class TimeSeriesRemoteControl(QudiRemoteControl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._module_name = "time_series_reader_logic"
        if self._module_name not in self.modules:
            logging.error(f"Module {self._module_name} not available.")

    @log
    def start_timeseries(self):
        try:
            tm = self.connection.root.exposed_get_module_instance(self._module_name)
            tm.start_reading()
        except:
            logging.error(f"Couldn't get module instance {self._module_name}.")
            raise

    @log
    def stop_timeseries(self):
        try:
            tm = self.connection.root.exposed_get_module_instance(self._module_name)
            tm.stop_reading()
        except:
            logging.error(f"Couldn't get module instance {self._module_name}.")
            raise


class OdmrRemoteControl(QudiRemoteControl):
    """
    Remote control for the OdmrLogic module.
    This class is specifically adapted to control an ODMR setup where the microwave
    source is a complex device (like MicrowaveRedPitayaWindfreak) that handles
    multi-frequency generation and frequency modulation (FM).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._module_name = "odmr_logic"
        if self._module_name not in self.modules:
            logging.error(f"Module {self._module_name} not available.")
            raise NameError(f"Module '{self._module_name}' not found in available Qudi modules.")

        try:
            self._odmr_module = self.connection.root.exposed_get_module_instance(self._module_name)
            # Get a proxy to the microwave module connected to the ODMR logic
            self._microwave_proxy = self._odmr_module._microwave()
        except Exception as e:
            logging.error(f"Couldn't get module instance for '{self._module_name}' or its microwave connector.",
                          exc_info=True)
            raise

    @log
    def configure_mw_source(self, power_dbm, f_mod_hz, f_dev_khz,
                            multi_freq_mode='triple', multi_freq_amplitudes=None):
        """
        Configures the microwave source for an ODMR scan with specified FM parameters.
        This is a high-level function that sets both the power and FM settings.

        Args:
            power_dbm (float): Total RF power in dBm.
            f_mod_hz (float): FM modulation frequency in Hz.
            f_dev_khz (float): FM deviation in kHz.
            multi_freq_mode (str): 'single', 'dual', or 'triple'.
            multi_freq_amplitudes (list): Relative amplitudes for multi-frequency mode.
        """
        # 1. Set the scan power on the ODMR logic module.
        #    This will be passed to the microwave module when a scan is configured.
        self._odmr_module.set_scan_power(power_dbm)

        # 2. Configure the multi-frequency mode of the microwave source.
        self._microwave_proxy.set_multi_frequency_mode(multi_freq_mode, multi_freq_amplitudes)

        # 3. Configure the FM parameters on the microwave source.
        self._microwave_proxy.set_fm_parameters(
            enable=True,
            deviation_khz=f_dev_khz,
            modulation_frequency=f_mod_hz
        )
        logging.info(f"Microwave source configured: Power={power_dbm} dBm, f_mod={f_mod_hz} Hz, f_dev={f_dev_khz} kHz")

    @log
    def take_single_odmr_scan(self, filename, run_time, frequency_start, frequency_stop,
                              frequency_points, data_rate=1000, save_data=True,
                              save_thumbnails=True, use_timestamp=True):
        """
        Configures and executes a single ODMR scan, waits for it to finish,
        and optionally saves the data.

        Args:
            filename (str): Base filename for saving data.
            run_time (float): Desired measurement time in seconds.
            frequency_start (float): Start frequency in Hz.
            frequency_stop (float): Stop frequency in Hz.
            frequency_points (int): Number of frequency points.
            data_rate (int): Data rate for the data_scanner in Hz.
            save_data (bool): If True, saves the data.
            save_thumbnails (bool): If True, saves plot thumbnails.
            use_timestamp (bool): If True, appends a timestamp to the filename.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (frequency_array, signal_array)
        """
        # Configure ODMR logic parameters
        self._odmr_module.set_runtime(run_time)
        self._odmr_module.set_frequency_range(frequency_start, frequency_stop, frequency_points, 0)
        self._odmr_module.set_data_rate(data_rate)

        # Start the scan
        self._odmr_module.start_odmr_scan()

        # Wait for the scan to complete
        # A simple wait loop checking the module state
        while self._odmr_module.module_state() == 'locked':
            time.sleep(run_time / 10 if run_time > 1 else 0.1)

        # Optionally save the data
        if save_data:
            self._odmr_module._save_thumbnails = save_thumbnails
            self._odmr_module._use_timestamp = use_timestamp
            self._odmr_module.save_odmr_data(filename)

        # Return ODMR data (element 0 is frequency array, element 1 is signal array)
        # Note: This assumes a single data channel is of interest.
        # This might need adjustment if multiple channels are processed.
        joined_data = self._odmr_module._join_signal_data()
        return np.array(joined_data.T[0]), np.array(joined_data.T[1])

    @log
    def set_cw_parameters(self, frequency=2.7e9, power=13):
        """Sets the CW parameters on the ODMR logic module."""
        try:
            self._odmr_module.set_cw_parameters(float(frequency), float(power))
        except:
            logging.error(f"Couldn't set CW parameters.")
            raise

    @log
    def toggle_cw_output(self, enable):
        try:
            self._odmr_module.toggle_cw_output(enable)
        except Exception as e:
            logging.error(f"Couldn't toggle CW output: {str(e)}")
            raise


    @log
    def get_mw_info(self):
        """Retrieves multi-frequency and FM information from the microwave source."""
        return self._microwave_proxy.get_multi_frequency_info()