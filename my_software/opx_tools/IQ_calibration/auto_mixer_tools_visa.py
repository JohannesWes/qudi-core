# This file contains classes of spectrum analyzers using the VISA interface to communicate with the computers.
# They should have almost uniform commands, making adaptions to new models/brands quite easy

from qm.qua import *
from abc import ABC, abstractmethod
import numpy as np
import pyvisa as visa
from RsInstrument import *
from time import sleep



class VisaRS(ABC):
    def __init__(self, qm, address='TCPIP0::192.168.88.12::INSTR'):
        # Gets an existing qm, assumes there is an element called "qubit" with an operation named "test_pulse" which
        # plays a constant pulse
        super().__init__()
        self.resource = address  # VISA resource string for the device
        self.sa = RsInstrument(self.resource, True, True, "SelectVisa='rs'")
        sleep(1) # Eventually add some waiting time when reset is performed during initialization
        print('\n',f'VISA Manufacturer: {self.sa.visa_manufacturer}'+'\n') # Confirm VISA package to be chosen
        self.sa.visa_timeout = 5000 # Timeout for VISA Read Operations
        self.sa.opc_timeout = 5000 # Timeout for opc-synchronised operations
        self.sa.instrument_status_checking = True # Error check after each command, can be True or False
        self.sa.clear_status() # Clear status register
        idnResponse = self.sa.query_str('*IDN?')  # Perform an Identification Query
        print('Hello, I am ' + idnResponse + '\n')
        self.sa.write_str_with_opc('*RST') # PRESET
        self.sa.write_str_with_opc('SYST:DISP:UPD ON') # Time base set to
        self.sa.write_str_with_opc('CALC:MATH1 "FFTmag(Ch1)"') # Enable FFT
        self.sa.write_str_with_opc('CALC:MATH1:STATE ON') # FFT on
        self.sa.write_str_with_opc('TIM:SCAL 5E-9') # Time base set to
        self.sa.write_str_with_opc('CHAN1:COUP DC') # Set vertical offset to 1.76 V

        with program() as mixer_cal:
            with infinite_loop_():
                play("test_pulse", "qubit")

        self.qm = qm
        self.job = qm.execute(mixer_cal)
        self.method = None

    def IQ_imbalance_correction(self, g, phi):
        c = np.cos(phi)
        s = np.sin(phi)
        N = 1 / ((1 - g**2) * (2 * c**2 - 1))
        return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]

    def get_leakage(self, i0, q0):
        self.qm.set_output_dc_offset_by_element("qubit", "I", i0)
        self.qm.set_output_dc_offset_by_element("qubit", "Q", q0)
        amp_ = self.get_amp()
        return amp_

    def get_image(self, g, p):
        self.job.set_element_correction("qubit", self.IQ_imbalance_correction(g, p))
        # self.qm.set_mixer_correction("mixer_qubit", qubit_IF, qubit_LO, self.IQ_imbalance_correction(g ,p))
        amp_ = self.get_amp()
        return amp_

    def __del__(self):
        self.sa.clear_status() # Clear status register
        self.sa.close()

    @abstractmethod
    def get_amp(self):
        pass

    @abstractmethod
    def set_automatic_video_bandwidth(self, state: int):
        # State should be 1 or 0
        pass

    @abstractmethod
    def set_automatic_bandwidth(self, state: int):
        # State should be 1 or 0
        pass

    @abstractmethod
    def set_bandwidth(self, bw: int):
        # Sets the bandwidth
        pass

    @abstractmethod
    def set_sweep_points(self, n_points: int):
        # Sets the number of points for a sweep
        pass

    @abstractmethod
    def set_center_freq(self, freq: int):
        # Sets the central frequency
        pass

    @abstractmethod
    def set_span(self, span: int):
        # Sets the span
        pass

    @abstractmethod
    def set_cont_off(self):
        # Sets continuous mode off
        pass

    @abstractmethod
    def set_cont_on(self):
        # Sets continuous mode on
        pass

    @abstractmethod
    def get_single_trigger(self):
        # Performs a single sweep
        pass

    @abstractmethod
    def active_marker(self, marker: int):
        # Active the given marker
        pass

    @abstractmethod
    def set_marker_freq(self, marker: int, freq: int):
        # Sets the marker's frequency
        pass

    @abstractmethod
    def query_marker(self, marker: int):
        # Query the marker
        pass

    @abstractmethod
    def get_full_trace(self):
        # Returns the full trace
        pass

    @abstractmethod
    def enable_measurement(self):
        # Sets the measurement to channel power
        pass

    @abstractmethod
    def disables_measurement(self):
        # Sets the measurement to none
        pass

    @abstractmethod
    def sets_measurement_integration_bw(self, ibw: int):
        # Sets the measurement integration bandwidth
        pass

    @abstractmethod
    def disables_measurement_averaging(self):
        # Disables averaging in the measurement
        pass

    @abstractmethod
    def get_measurement_data(self):
        # Returns the result of the measurement
        pass

class RhodeSchwarzRTO6(VisaRS):
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
        # State should be 1 or 0
        #self.sa.write(f"SENS:BAND:VID:AUTO {int(state)}")
        pass

    def set_automatic_bandwidth(self, state: int):
        # State should be 1 or 0. Resolution (or measurement) bandwidth
        #self.sa.write(f"SENS:BAND:AUTO {int(state)}")
        pass

    def set_bandwidth(self, bw: int):
        # Sets the resolution (or measurement) bandwidth, 1 Hz to 3 MHz, default unit is Hz
        # Example SENS:BAND 100000
        self.sa.write_str_with_opc(f'CALC:MATH1:FFT:BAND {int(bw)}')  # FFT Bandwidth

    def set_sweep_points(self, n_points: int):
        # Sets the number of points for a sweep, allowed range 101 to 2501, default is 201
        #self.sa.write(f"SENS:SWE:POIN {int(n_points)}")
        pass

    def set_center_freq(self, freq: int):
        # Sets the central frequency, default unit is Hz
        self.sa.write_str_with_opc(f"CALC:MATH1:FFT:CFR {int(freq)}")

    def set_span(self, span: int):
        # Sets the span, default unit is Hz
        self.sa.write_str_with_opc(f"CALC:MATH1:FFT:SPAN {int(span)}")

    def set_cont_off(self):
        # This command selects the sweep mode (but does not start the measurement!)
        # OFF or 0 is a single sweep mode
        # *OPC? is to make sure there is no overlapping execution
        return self.sa.write_str_with_opc("STOP")

    def set_cont_on(self):
        # This command selects the sweep mode (but does not start the measurement!)
        # ON or 1 is a continuous sweep mode
        # *OPC? is to make sure there is no overlapping execution
        return self.sa.write_str_with_opc("RUN")

    def get_single_trigger(self):
        # Initiates a new measurement sequence (starts the sweep)
        return self.sa.write_str_with_opc('SING') # Single Acquisition

    def active_marker(self, marker: int):
        # Activate the given marker
        self.sa.write_str_with_opc(f"CURS{int(marker)}:STAT ON")
        self.sa.write_str_with_opc(f"CURS{int(marker)}:SOUR M1")
        self.sa.write_str_with_opc(f"CURS{int(marker)}:TRAC ON")

    def set_marker_freq(self, marker: int, freq: int):
        # Sets the marker's frequency. Default unit is Hz
        self.get_single_trigger()
        self.sa.write(f"CURS{int(marker)}:X1P {int(freq)}")

    def query_marker(self, marker: int):
        # Query the amplitude (default unit is dBm) of the marker
        return float(self.sa.query_float(f"CURS{int(marker)}:Y1P?"))

    def get_full_trace(self):
        # Returns the full trace. Implicit assumption that this is trace1 (there could be 1-4)
        amp = self.sa.query_bin_or_ascii_float_list('FORM REAL,32;CALC:MATH1:DATA?')
        btrace = self.sa.query_bin_or_ascii_float_list('CALC:MATH1:DATA:HEAD?')
        f_vec = np.linspace(btrace[0], btrace[1], int(btrace[2]))
        return f_vec, amp

    def enable_measurement(self):
        # Sets the measurement to channel power
        print('NOT ENABLING ANYTHING')
        #self.sa.write(
        #    "CALC:MARK:FUNC:POW:SEL CPOW; CALC:MARK:FUNC:LEV:ONCE; CALC:MARK:FUNC:CPOW:UNIT DBM; CALC:MARK:FUNC:POW:RES:PHZ ON"
        #)

    def disables_measurement(self):
        # Sets the channel power measurement to none
        #self.sa.write("CALC:MARK:FUNC:POW OFF")
        print('NOT DISABLING ANYTHING')

    def sets_measurement_integration_bw(self, ibw: int):
        # Sets the measurement integration bandwidth for channel power measurements
        print('NOT SETTING MEASUREMENT INTEGRATION BW')
        #self.sa.write(f"CALC:MARK:FUNC:CPOW:BAND {int(ibw)}")

    def disables_measurement_averaging(self):
        # disables averaging in the measurement
        pass

    def get_measurement_data(self):
        # Returns the result of the measurement
        print('NOT GETTING MEASUREMENT DATA')
        #return self.sa.query(f"CALC:MARK:FUNC:POW:RES? CPOW")
        return None
