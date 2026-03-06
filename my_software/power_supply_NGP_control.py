from RsNgx import *
import numpy as np
from scipy.constants import mu_0
import pyvisa as visa
import logging
import functools
import time

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


# -------------------------------------------------------------------------------------------------------------------- #


# from RsInstrument import *

class NGP_instance():
    def __init__(self, usb_address=None):

        try:
            self.driver = RsNgx(usb_address, reset=True)
        except:
            logging.error(f"Could not connect to the NGP.")
            raise

        self.driver.utilities.visa_timeout = 3000

        print(f'Connected to {self.driver.utilities.query("*IDN?")}')
        print(f'RsNgx package version: {self.driver.utilities.driver_version}')
        print(f'Visa Manufacturer: {self.driver.utilities.visa_manufacturer}')
        print(f'Instrument full name: {self.driver.utilities.full_instrument_model_name}')

        self.output_off()
        for chn in range(1, 4):
            self.deactivate_channel(chn)

        self.driver.output.general.set_state(False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def set_voltage(self, chn, vlt):
        try:
            self.driver.instrument.select.set(chn)
            self.driver.source.voltage.level.immediate.set_amplitude(vlt)
        except:
            logging.error(f"Could not set voltage to {vlt} on channel {chn}")
            raise


    def set_current(self, chn, curr):
        try:
            self.driver.instrument.select.set(chn)
            self.driver.source.current.level.immediate.set_amplitude(curr)
        except:
            logging.error(f"Could not set current to {curr} on channel {chn}")
            raise

    def get_voltage(self, chn):
        try:
            self.driver.instrument.select.set(chn)
            return self.driver.source.voltage.level.immediate.get_amplitude()
        except:
            logging.error(f"Could not get voltage on channel {chn}")
            raise

    def get_current(self, chn):
        try:
            self.driver.instrument.select.set(chn)
            return self.driver.source.current.level.immediate.get_amplitude()
        except:
            logging.error(f"Could not get current on channel {chn}")
            raise

    def activate_channel(self, chn):
        try:
            self.driver.instrument.select.set(chn)
            self.driver.output.set_select(True)
        except:
            logging.error(f"Could not activate channel {chn}")
            raise

    def deactivate_channel(self, chn):
        try:
            self.driver.instrument.select.set(chn)
            self.driver.output.set_select(False)
        except:
            logging.error(f"Could not deactivate channel {chn}")
            raise

    def deactivate_all_channels(self):
        try:
            for chn in range(1, 4+1):
                self.deactivate_channel(chn)
        except:
            logging.error(f"Could not deactivate all channels.")
            raise

    def output_on(self):
        try:
            self.driver.output.general.set_state(True)
        except:
            logging.error(f"Could not turn on the output.")
            raise

    def output_off(self):
        try:
            self.driver.output.general.set_state(False)
        except:
            logging.error(f"Could not turn off the output.")
            raise

    def read_channel_state(self, chn):
        try:
            self.driver.instrument.select.set(chn)
            return self.driver.read()
        except:
            logging.error(f"Could not read channel state.")
            raise

    def read_mode(self, chn):
        mode = self.driver.status.questionable.instrument.isummary.condition.get(chn)
        if mode == 1:
            return 'CONSTANT CURRENT'
        elif mode == 2:
            return 'CONSTANT VOLTAGE'
        else:
            return 'UNKNOWN MODE'

    def report_metrics(self, chn):
        data = self.read_channel_state(chn)
        res = data.Voltage / data.Current if data.Current > 0 else 0
        print(
            f'Channel: {chn}\t|\tAxis: {self._chn_to_axis(chn)}\t|\tVoltage: {data.Voltage}\t|\tCurrent: {data.Current}\t|\tResistance: {res:05.2f}\t|\tMode: {self.read_mode(chn)}')

    def report_all_metrics(self):
        for chn in range(1, 4):
            self.report_metrics(chn)

    def close(self):
        self.deactivate_all_channels()
        self.output_off()
        self.driver.close()

    def _chn_to_axis(self, chn):
        if chn == 1:
            return 'X'
        elif chn == 2:
            return 'Y'
        elif chn == 3:
            return 'Z'
        else:
            return 'NO CHANNEL DEFINED'