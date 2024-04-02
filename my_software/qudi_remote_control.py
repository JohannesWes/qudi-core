import jupyter_client as jc
import ast
import numpy as np
import logging
import json
import time
import rpyc
import functools


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


# def run_kernel_command(cmd):
#     print('Running Kernel Command')
#     conn_file = jc.find_connection_file()
#     client = jc.BlockingKernelClient(connection_file=conn_file)
#     client.load_connection_file()
#     client.start_channels()
#     client.execute(cmd)


class QudiRemoteControl():
    def __init__(self, host='localhost', port=12345, conn_config={'allow_all_attrs': True}):
        self.host = host
        self.port = port
        self.conn_config = conn_config
        self.connection = rpyc.connect(self.host, self.port, config=self.conn_config)
        self.modules = self.connection.root.exposed_get_available_module_names()


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

    @log
    def stop_timeseries(self):
        try:
            tm = self.connection.root.exposed_get_module_instance(self._module_name)
            tm.stop_reading()
        except:
            logging.error(f"Couldn't get module instance {self._module_name}.")

    def close_connection(self):
        self.connection.close()


class OdmrRemoteControl(QudiRemoteControl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._module_name = "odmr_logic"
        if self._module_name not in self.modules:
            logging.error(f"Module {self._module_name} not available.")

        try:
            self._odmr_module = self.connection.root.exposed_get_module_instance(self._module_name)
        except:
            logging.error(f"Couldn't get module instance {self._module_name}.")

        # self._scan_power = None
        # self._run_time = None
        # self._frequency_start = None
        # self._frequency_stop = None
        # self._frequency_points = None

    @log
    def start_odmr_scan(self):
        try:
            self._odmr_module.start_odmr_scan()
        except:
            logging.error(f"Couldn't start ODMR scan.")

    @log
    def set_odmr_parameters(self, scan_power=13, run_time=10, frequency_start=2.815e9, frequency_stop=2.835e9,
                            frequency_points=1000, data_rate=1000):
        try:
            self._odmr_module.set_scan_power(scan_power)
            self._odmr_module.set_runtime(run_time)
            self._odmr_module.set_frequency_range(frequency_start, frequency_stop, frequency_points, 0)
            self._odmr_module.set_data_rate(data_rate)

        except:
            logging.error(f"Couldn't set ODMR parameters.")

    @log
    def save_odmr_scan(self, filename, use_timestamp=False, save_thumbnails=False):
        try:
            self._odmr_module._save_thumbnails = save_thumbnails
            self._odmr_module._use_timestamp = use_timestamp
            self._odmr_module.save_odmr_scan(filename)
        except:
            logging.error(f"Couldn't save ODMR scan.")

    @log
    def take_odmr_scan(self, frequency_start, frequency_stop, frequency_points):
        self.set_odmr_parameters(frequency_start=frequency_start, frequency_stop=frequency_stop, frequency_points=frequency_points)
        self.start_odmr_scan()


    def close_connection(self):
        self.connection.close()


if __name__ == '__main__':
    odmr_remote = OdmrRemoteControl()
    odmr_remote.set_odmr_parameters()

    folder_name = "folder"
    file_name = folder_name + "/" + "filename"

    #odmr_remote.start_odmr_scan()

