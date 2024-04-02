from RsNgx import *
import numpy as np
from scipy.constants import mu_0
import pyvisa as visa


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
        self.driver = RsNgx('ASRL5::INSTR', reset=True)
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
        self.driver.instrument.select.set(chn)
        self.driver.source.voltage.level.immediate.set_amplitude(vlt)

    def set_current(self, chn, curr):
        self.driver.instrument.select.set(chn)
        self.driver.source.current.level.immediate.set_amplitude(curr)

    def activate_channel(self, chn):
        self.driver.instrument.select.set(chn)
        self.driver.output.set_select(True)

    def deactivate_channel(self, chn):
        self.driver.instrument.select.set(chn)
        self.driver.output.set_select(False)

    def deactivate_all_channels(self):
        for chn in range(1, 4):
            self.deactivate_channel(chn)

    def output_on(self):
        self.driver.output.general.set_state(True)

    def output_off(self):
        self.driver.output.general.set_state(False)

    def read_channel_state(self, chn):
        self.driver.instrument.select.set(chn)
        return self.driver.read()

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

