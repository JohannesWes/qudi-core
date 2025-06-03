# auto_mixer_tools_visa.py
from abc import ABC, abstractmethod
import numpy as np
import pyvisa as visa
from RsInstrument import *
from time import sleep
import pyrpl
import logging
from logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)


class VisaRS_RedPitaya(ABC):
    def __init__(self, pyrpl_instance, address='TCPIP0::192.168.88.12::INSTR'):
        super().__init__()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing VISA instrument at {address}")

        self.resource = address
        self.sa = RsInstrument(self.resource, True, True, "SelectVisa='rs'")
        sleep(1)

        visa_manufacturer = self.sa.visa_manufacturer
        self.logger.info(f'VISA Manufacturer: {visa_manufacturer}')

        self.sa.visa_timeout = 5000
        self.sa.opc_timeout = 5000
        self.sa.instrument_status_checking = True
        self.sa.clear_status()

        idnResponse = self.sa.query_str('*IDN?')
        self.logger.info(f'Connected to instrument: {idnResponse}')

        self.logger.debug("Resetting instrument...")
        self.sa.write_str_with_opc('*RST')
        self.sa.write_str_with_opc('SYST:DISP:UPD ON')
        self.sa.write_str_with_opc('CALC:MATH1 "FFTmag(Ch1)"')
        self.sa.write_str_with_opc('CALC:MATH1:STATE ON')
        self.sa.write_str_with_opc('TIM:SCAL 5E-9')
        self.sa.write_str_with_opc('CHAN1:COUP DC')

        self.pyrpl = pyrpl_instance
        self.fgen3 = self.pyrpl.rp.fgen3
        self.method = None

        self.logger.info("VISA instrument initialized successfully")

    def _start_redpitaya_signal(self):
        """Initialize and start the Red Pitaya signal generation."""
        self.logger.debug("Starting Red Pitaya signal generation...")

        self.fgen3.gen_enable = True
        self.fgen3.output_zero = False

        self.fgen3.frequency0 = 21.158e6
        self.fgen3.amplitude_a0 = 0.0
        self.fgen3.amplitude_b0 = 0.0
        self.fgen3.phase_offset_a0 = 0.0
        self.fgen3.phase_offset_b0 = 90.0
        self.fgen3.enable0 = True

        self.fgen3.enable1 = False
        self.fgen3.enable2 = False

        self.fgen3.overall_dc_offset_a = 0.0
        self.fgen3.overall_dc_offset_b = 0.0

        self.logger.debug("Red Pitaya signal generation started")

    def IQ_imbalance_correction(self, g, phi):
        """Apply IQ imbalance correction to Red Pitaya."""
        self.logger.debug(f"Applying IQ imbalance correction: g={g:.5f}, phi={phi:.5f}")


        # todo: vielleicht direkt aus gewünschter IF-amplitude und g die neuen amplituden berechnen, statt aus aktuellen amplitudes?
        current_amp_i = self.fgen3.amplitude_a0
        current_amp_q = self.fgen3.amplitude_b0

        nominal_amp = (current_amp_i + current_amp_q) / 2.0

        amp_i = nominal_amp * (1 + g)
        amp_q = nominal_amp * (1 - g)

        phase_i = 0.0
        phase_q = 90.0 + np.degrees(phi)

        self.logger.debug(f"Calculated corrections: amp_i={amp_i:.5f}, amp_q={amp_q:.5f}, "
                          f"phase_i={phase_i:.2f}, phase_q={phase_q:.2f}")

        return amp_i, amp_q, phase_i, phase_q

    def get_leakage(self, i0, q0):
        """Set DC offsets and measure LO leakage."""
        self.logger.debug(f"Measuring LO leakage with DC offsets: I={i0:.5f}, Q={q0:.5f}")
        self.fgen3.overall_dc_offset_a = i0
        self.fgen3.overall_dc_offset_b = q0
        amp_ = self.get_amp()
        self.logger.debug(f"LO leakage power: {amp_} dBm")
        return amp_

    def get_image(self, g, p):
        """Apply IQ correction and measure image rejection."""
        self.logger.debug(f"Measuring image rejection with g={g:.5f}, p={p:.5f}")
        amp_i, amp_q, phase_i, phase_q = self.IQ_imbalance_correction(g, p)

        self.fgen3.amplitude_a0 = amp_i
        self.fgen3.amplitude_b0 = amp_q
        self.fgen3.phase_offset_a0 = phase_i
        self.fgen3.phase_offset_b0 = phase_q

        amp_ = self.get_amp()
        self.logger.debug(f"Image rejection power: {amp_} dBm")
        return amp_

    def set_if_amplitude(self, amplitude):
        """Set the IF signal amplitude (0 to 1)."""
        self.logger.debug(f"Setting IF amplitude to {amplitude:.3f}")
        self.fgen3.amplitude_a0 = amplitude
        self.fgen3.amplitude_b0 = amplitude

    def __del__(self):
        self.logger.debug("Cleaning up VISA instrument...")
        if hasattr(self, 'fgen3') and self.fgen3:
            try:
                self.fgen3.gen_enable = False
                self.logger.debug("Red Pitaya output disabled")
            except:
                pass

        if hasattr(self, 'sa') and self.sa:
            try:
                # Check if the instrument session is still valid
                if hasattr(self.sa, '_session') and self.sa._session:
                    self.sa.clear_status()
                    self.sa.close()
                    self.logger.debug("VISA connection closed")
            except Exception as e:
                # Silently ignore errors in destructor
                self.logger.debug(f"Error in destructor: {e}")

    @abstractmethod
    def get_amp(self):
        pass



class RhodeSchwarzRTO6_RedPitaya(VisaRS_RedPitaya):
    """Rhode & Schwarz RTO6 oscilloscope with Red Pitaya signal source."""

    def get_amp(self):
        self.get_single_trigger()
        if self.method == 1:
            sig = self.get_measurement_data()
        elif self.method == 2:
            sig = self.query_marker(1)
        else:
            sig = float("NaN")
        return sig

    def set_automatic_video_bandwidth(self, state: int):
        self.logger.debug(f"set_automatic_video_bandwidth({state}) - Not implemented for RTO6")

    def set_automatic_bandwidth(self, state: int):
        self.logger.debug(f"set_automatic_bandwidth({state}) - Not implemented for RTO6")

    def set_bandwidth(self, bw: int):
        self.logger.debug(f"Setting FFT bandwidth to {bw} Hz")
        self.sa.write_str_with_opc(f'CALC:MATH1:FFT:BAND {int(bw)}')

    def set_sweep_points(self, n_points: int):
        self.logger.debug(f"set_sweep_points({n_points}) - Not implemented for RTO6")

    def set_center_freq(self, freq: int):
        self.logger.debug(f"Setting center frequency to {freq / 1e9:.3f} GHz")
        self.sa.write_str_with_opc(f"CALC:MATH1:FFT:CFR {int(freq)}")

    def set_span(self, span: int):
        self.logger.debug(f"Setting span to {span / 1e6:.1f} MHz")
        self.sa.write_str_with_opc(f"CALC:MATH1:FFT:SPAN {int(span)}")

    def set_cont_off(self):
        self.logger.debug("Stopping continuous acquisition")
        return self.sa.write_str_with_opc("STOP")

    def set_cont_on(self):
        self.logger.debug("Starting continuous acquisition")
        return self.sa.write_str_with_opc("RUN")

    def get_single_trigger(self):
        self.logger.debug("Triggering single acquisition")
        return self.sa.write_str_with_opc('SING')

    def active_marker(self, marker: int):
        self.logger.debug(f"Activating marker {marker}")
        self.sa.write_str_with_opc(f"CURS{int(marker)}:STAT ON")
        self.sa.write_str_with_opc(f"CURS{int(marker)}:SOUR M1")
        self.sa.write_str_with_opc(f"CURS{int(marker)}:TRAC ON")

    def set_marker_freq(self, marker: int, freq: int):
        self.logger.debug(f"Setting marker {marker} to {freq / 1e9:.3f} GHz")
        self.get_single_trigger()
        self.sa.write(f"CURS{int(marker)}:X1P {int(freq)}")

    def query_marker(self, marker: int):
        value = float(self.sa.query_float(f"CURS{int(marker)}:Y1P?"))
        self.logger.debug(f"Marker {marker} value: {value} dBm")
        return value

    def get_full_trace(self):
        self.logger.debug("Getting full FFT trace...")
        amp = self.sa.query_bin_or_ascii_float_list('FORM REAL,32;CALC:MATH1:DATA?')
        btrace = self.sa.query_bin_or_ascii_float_list('CALC:MATH1:DATA:HEAD?')
        f_vec = np.linspace(btrace[0], btrace[1], int(btrace[2]))
        self.logger.debug(f"Retrieved {len(f_vec)} frequency points")
        return f_vec, amp

    def enable_measurement(self):
        self.logger.warning('enable_measurement() - Not implemented for RTO6')

    def disables_measurement(self):
        self.logger.warning('disables_measurement() - Not implemented for RTO6')

    def sets_measurement_integration_bw(self, ibw: int):
        self.logger.warning(f'sets_measurement_integration_bw({ibw}) - Not implemented for RTO6')

    def disables_measurement_averaging(self):
        self.logger.debug('disables_measurement_averaging() - Not implemented for RTO6')

    def get_measurement_data(self):
        self.logger.warning('get_measurement_data() - Not implemented for RTO6')
        return None