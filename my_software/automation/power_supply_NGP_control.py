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
    def __init__(self):
        self.small_dist = 0.054
        self.small_N = 400
        self.small_R = 0.022
        self.medium_dist = 0.078
        self.medium_N = 450
        self.medium_R = 0.059
        self.large_dist = 0.09
        self.large_N = 600
        self.large_R = 0.1045

        try:
            self.driver = RsNgx('ASRL5::INSTR', reset=True)
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

    # def _chn_to_axis(self, chn):
    #     if chn == 1:
    #         return 'X'
    #     elif chn == 2:
    #         return 'Y'
    #     elif chn == 3:
    #         return 'Z'
    #     else:
    #         return 'NO CHANNEL DEFINED'
    #
    # def unit_vec(self, dir, mag):
    #     B_dir = mag * dir / np.linalg.norm(dir)
    #     return np.array([np.dot(B_dir, np.array([1, 0, 0])), np.dot(B_dir, np.array([0, 1, 0])),
    #                      np.dot(B_dir, np.array([0, 0, 1]))])
    #
    # def field_to_current(self, direction, magnitude):
    #     """
    #     Calculate required current per coil for the provided magnetic field vector (in Gauss).
    #     """
    #     vec = self.unit_vec(direction, magnitude)
    #     I_small = ((self.small_R ** 2 + (self.small_dist / 2) ** 2) ** (3 / 2) * vec[0] * 1e-4) / (
    #             mu_0 * self.small_R ** 2 * self.small_N)
    #     I_large = ((self.large_R ** 2 + (self.large_dist / 2) ** 2) ** (3 / 2) * vec[1] * 1e-4) / (
    #             mu_0 * self.large_R ** 2 * self.large_N)
    #     I_medium = ((self.medium_R ** 2 + (self.medium_dist / 2) ** 2) ** (3 / 2) * vec[2] * 1e-4) / (
    #             mu_0 * self.medium_R ** 2 * self.medium_N)
    #     return I_small, I_large, I_medium
    #
    # def magnetic_field_vector(self, direction, magnitude):
    #     I_small, I_large, I_medium = self.field_to_current(direction, magnitude)
    #     self.set_channel(1, 6, I_small) if I_small > 0 else self.disable_channel(1)
    #     self.set_channel(2, 16, I_large) if  I_large> 0 else self.disable_channel(2)
    #     self.set_channel(3, 30, I_medium) if I_medium > 0 else self.disable_channel(3)
    #     return (self.read_channel_state(1), self.read_channel_state(2), self.read_channel_state(3))

