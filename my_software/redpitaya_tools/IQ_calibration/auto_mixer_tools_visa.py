# This file contains classes of spectrum analyzers using the VISA interface with Red Pitaya
# They should have almost uniform commands, making adaptions to new models/brands quite easy

from abc import ABC, abstractmethod
import numpy as np
import pyvisa as visa
from RsInstrument import *
from time import sleep
import pyrpl


class VisaRS_RedPitaya(ABC):
    def __init__(self, pyrpl_instance, address='TCPIP0::192.168.88.12::INSTR'):
        # Gets an existing Red Pitaya pyrpl instance
        super().__init__()
        self.resource = address  # VISA resource string for the oscilloscope
        self.sa = RsInstrument(self.resource, True, True, "SelectVisa='rs'")
        sleep(1)  # Eventually add some waiting time when reset is performed during initialization
        print('\n', f'VISA Manufacturer: {self.sa.visa_manufacturer}' + '\n')  # Confirm VISA package
        self.sa.visa_timeout = 5000  # Timeout for VISA Read Operations
        self.sa.opc_timeout = 5000  # Timeout for opc-synchronised operations
        self.sa.instrument_status_checking = True  # Error check after each command
        self.sa.clear_status()  # Clear status register
        idnResponse = self.sa.query_str('*IDN?')  # Perform an Identification Query
        print('Hello, I am ' + idnResponse + '\n')
        self.sa.write_str_with_opc('*RST')  # PRESET
        self.sa.write_str_with_opc('SYST:DISP:UPD ON')  # Time base set to
        self.sa.write_str_with_opc('CALC:MATH1 "FFTmag(Ch1)"')  # Enable FFT
        self.sa.write_str_with_opc('CALC:MATH1:STATE ON')  # FFT on
        self.sa.write_str_with_opc('TIM:SCAL 5E-9')  # Time base set to
        self.sa.write_str_with_opc('CHAN1:COUP DC')  # Set coupling to DC

        self.pyrpl = pyrpl_instance
        self.fgen3 = self.pyrpl.rp.fgen3
        self.method = None

        # Don't start the signal here - it will be configured by the calibrator

    def _start_redpitaya_signal(self):
        """Initialize and start the Red Pitaya signal generation."""
        # Enable the fgen3 module
        self.fgen3.gen_enable = True
        self.fgen3.output_zero = False

        # Set initial configuration for component 0
        # We'll use component 0 for our IF signal
        self.fgen3.frequency0 = 21.158e6  # 21.158 MHz IF frequency
        self.fgen3.amplitude_a0 = 0.5  # Default amplitude for I (DAC A)
        self.fgen3.amplitude_b0 = 0.5  # Default amplitude for Q (DAC B)
        self.fgen3.phase_offset_a0 = 0.0  # Initial phase for I
        self.fgen3.phase_offset_b0 = 90.0  # 90 degree phase shift for Q
        self.fgen3.enable0 = True

        # Disable other components
        self.fgen3.enable1 = False
        self.fgen3.enable2 = False

        # Set initial DC offsets
        self.fgen3.overall_dc_offset_a = 0.0
        self.fgen3.overall_dc_offset_b = 0.0

    def IQ_imbalance_correction(self, g, phi):
        """Apply IQ imbalance correction to Red Pitaya."""
        # For Red Pitaya, we implement IQ correction by adjusting amplitudes and phase
        # g is the gain imbalance, phi is the phase imbalance

        # Get the current amplitude setting (before correction)
        current_amp_i = self.fgen3.amplitude_a0
        current_amp_q = self.fgen3.amplitude_b0

        # Use the average of current I and Q amplitudes as the nominal
        # This preserves the IF power level we're calibrating for
        nominal_amp = (current_amp_i + current_amp_q) / 2.0

        # Apply gain correction
        amp_i = nominal_amp * (1 + g)
        amp_q = nominal_amp * (1 - g)

        # Apply phase correction
        phase_i = 0.0
        phase_q = 90.0 + np.degrees(phi)

        return amp_i, amp_q, phase_i, phase_q

    def get_leakage(self, i0, q0):
        """Set DC offsets and measure LO leakage."""
        self.fgen3.overall_dc_offset_a = i0
        self.fgen3.overall_dc_offset_b = q0
        amp_ = self.get_amp()
        return amp_

    def get_image(self, g, p):
        """Apply IQ correction and measure image rejection."""
        amp_i, amp_q, phase_i, phase_q = self.IQ_imbalance_correction(g, p)

        # Apply corrections to Red Pitaya
        self.fgen3.amplitude_a0 = amp_i
        self.fgen3.amplitude_b0 = amp_q
        self.fgen3.phase_offset_a0 = phase_i
        self.fgen3.phase_offset_b0 = phase_q

        amp_ = self.get_amp()
        return amp_

    def set_if_amplitude(self, amplitude):
        """Set the IF signal amplitude (0 to 1)."""
        # Scale both I and Q components proportionally
        self.fgen3.amplitude_a0 = amplitude
        self.fgen3.amplitude_b0 = amplitude

    def __del__(self):
        # Turn off Red Pitaya output
        if hasattr(self, 'fgen3'):
            self.fgen3.gen_enable = False

        # Clean up oscilloscope
        if hasattr(self, 'sa'):
            self.sa.clear_status()
            self.sa.close()

    @abstractmethod
    def get_amp(self):
        pass

    @abstractmethod
    def set_automatic_video_bandwidth(self, state: int):
        pass

    @abstractmethod
    def set_automatic_bandwidth(self, state: int):
        pass

    @abstractmethod
    def set_bandwidth(self, bw: int):
        pass

    @abstractmethod
    def set_sweep_points(self, n_points: int):
        pass

    @abstractmethod
    def set_center_freq(self, freq: int):
        pass

    @abstractmethod
    def set_span(self, span: int):
        pass

    @abstractmethod
    def set_cont_off(self):
        pass

    @abstractmethod
    def set_cont_on(self):
        pass

    @abstractmethod
    def get_single_trigger(self):
        pass

    @abstractmethod
    def active_marker(self, marker: int):
        pass

    @abstractmethod
    def set_marker_freq(self, marker: int, freq: int):
        pass

    @abstractmethod
    def query_marker(self, marker: int):
        pass

    @abstractmethod
    def get_full_trace(self):
        pass

    @abstractmethod
    def enable_measurement(self):
        pass

    @abstractmethod
    def disables_measurement(self):
        pass

    @abstractmethod
    def sets_measurement_integration_bw(self, ibw: int):
        pass

    @abstractmethod
    def disables_measurement_averaging(self):
        pass

    @abstractmethod
    def get_measurement_data(self):
        pass


class RhodeSchwarzRTO6_RedPitaya(VisaRS_RedPitaya):
    """Rhode & Schwarz RTO6 oscilloscope with Red Pitaya signal source."""

    def get_amp(self):
        self.get_single_trigger()
        if self.method == 1:  # Channel power
            sig = self.get_measurement_data()
        elif self.method == 2:  # Marker
            sig = self.query_marker(1)
        else:
            sig = float("NaN")
        return sig

    def set_automatic_video_bandwidth(self, state: int):
        pass

    def set_automatic_bandwidth(self, state: int):
        pass

    def set_bandwidth(self, bw: int):
        self.sa.write_str_with_opc(f'CALC:MATH1:FFT:BAND {int(bw)}')

    def set_sweep_points(self, n_points: int):
        pass

    def set_center_freq(self, freq: int):
        self.sa.write_str_with_opc(f"CALC:MATH1:FFT:CFR {int(freq)}")

    def set_span(self, span: int):
        self.sa.write_str_with_opc(f"CALC:MATH1:FFT:SPAN {int(span)}")

    def set_cont_off(self):
        return self.sa.write_str_with_opc("STOP")

    def set_cont_on(self):
        return self.sa.write_str_with_opc("RUN")

    def get_single_trigger(self):
        return self.sa.write_str_with_opc('SING')

    def active_marker(self, marker: int):
        self.sa.write_str_with_opc(f"CURS{int(marker)}:STAT ON")
        self.sa.write_str_with_opc(f"CURS{int(marker)}:SOUR M1")
        self.sa.write_str_with_opc(f"CURS{int(marker)}:TRAC ON")

    def set_marker_freq(self, marker: int, freq: int):
        self.get_single_trigger()
        self.sa.write(f"CURS{int(marker)}:X1P {int(freq)}")

    def query_marker(self, marker: int):
        return float(self.sa.query_float(f"CURS{int(marker)}:Y1P?"))

    def get_full_trace(self):
        amp = self.sa.query_bin_or_ascii_float_list('FORM REAL,32;CALC:MATH1:DATA?')
        btrace = self.sa.query_bin_or_ascii_float_list('CALC:MATH1:DATA:HEAD?')
        f_vec = np.linspace(btrace[0], btrace[1], int(btrace[2]))
        return f_vec, amp

    def enable_measurement(self):
        print('NOT ENABLING ANYTHING')

    def disables_measurement(self):
        print('NOT DISABLING ANYTHING')

    def sets_measurement_integration_bw(self, ibw: int):
        print('NOT SETTING MEASUREMENT INTEGRATION BW')

    def disables_measurement_averaging(self):
        pass

    def get_measurement_data(self):
        print('NOT GETTING MEASUREMENT DATA')
        return None