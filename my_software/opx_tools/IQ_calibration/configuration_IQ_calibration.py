import numpy as np
from dataclasses import dataclass

simulate = False

##############
# Parameters #
##############
# ----------------------------------------------------------------------------------------------------------------------

class parameters:
    def __init__(self):
        self.f_mod = 5.0e4  # modulation frequency
        self.f_dev = 200.00e3  # deviation of the modulation
        self.f_base = 200.00e6  # frequency around that we modulate: base frequency chosen as middle of bandwidth (400 MHz) of OPX
        self.ensemble_lo = 2.6e9
        #self.voltage_opx = 0.4  # peak (not peak-to-peak) output voltage of the opx

        # corrections for I & Q voltages; values from IQ-calibration script
        self.I_offset, self.Q_offset = -0.00914, -0.01080

        # corrections for g and phi; values from IQ-calibration script
        self.g_cor, self.phi_cor = 0.13180, 0.17266

        self.port_I, self.port_Q = 3, 4
        self.port_reference = 5

        self.optimization_repititions = 3  # How often LO-leakage and IQ-imbalances are repeated
        self.mw_usb_com_port = "COM3"  # The USB port for the MW, opened using visa.
        self.opx_ip_address = '192.168.88.10'
        self.osci_address = "TCPIP0::192.168.88.12::inst0::INSTR"  # The TCIP adress for the oscilloscope, opened using visa.
        # ----------------------------------------------------------------------------------------------------------------------

        # duration of one FM period (in ns); also serves as the pulse length; pulse length must be integer multiple of 4
        self.FM_period_duration = int(1 / self.f_mod / 1e-9) - int(1 / self.f_mod / 1e-9) % 4
        print(f"FM_period_duration [ns]: {self.FM_period_duration}")

        # Frequencies
        self.qubit_LO = self.ensemble_lo
        self.qubit_IF = self.f_base

        # MW parameters
        self.mw_len_NV = self.FM_period_duration  # in units of ns



        # Important Parameters:
        self.bDoSweeps = True  # If True, performs a large sweep before and after the optimization.
        self.method = 2  # If set to 1, checks power using a channel power measurement. If set to 2, checks power using a marker.

        # Parameters for oscilloscope spectrum easurement:
        self.measBW = 400e6  # Measurement bandwidth
        self.measNumPoints = 101

        # Parameters for oscilloscope spectrum sweep:
        self.sweepBW = 1e6
        self.fullNumPoints = 1201
        self.fullSpan = int(abs(self.qubit_IF * 4.1))  # Larger than 4 such that we'll see spurs
        self.startFreq = self.qubit_LO - self.fullSpan / 2
        self.stopFreq = self.qubit_LO + self.fullSpan / 2
        self.freq_vec = np.linspace(float(self.startFreq), float(self.stopFreq), int(self.fullNumPoints))

        self.microwave_max_power = 13

        self.xatol = 1e-4  # 1e-4 change in DC offset or gain/phase
        self.fatol = 3  # dB change tolerance
        self.maxiter = 50  # 50 iterations should be more than enough, but can be changed.




    def create_config(self, voltage_NV):
        config = {

            'version': 1,
            'controllers': {
                'con1': {
                    'type': 'opx1',
                    'analog_outputs': {
                        # LO Leakage for Antenna for 0.07 Vp OPX Config
                        self.port_I: {"offset": self.I_offset, 'delay': 73},
                        # 75},  # NV I for OPX+ 1: {'offset': 0.0, 'delay': 0}
                        self.port_Q: {"offset": self.Q_offset, 'delay': 73},  # 75},  # NV Q
                    },
                }
            },

            'elements': {
                'qubit': {
                    'mixInputs': {
                        'I': ('con1', self.port_I),
                        'Q': ('con1', self.port_Q),
                        'lo_frequency': self.qubit_LO,
                        'mixer': 'mixer_qubit'
                    },
                    'intermediate_frequency': self.qubit_IF,
                    'operations': {
                        'test_pulse': 'NV_cw_pulse',
                    },
                },
            },

            'pulses': {
                'NV_cw_pulse': {
                    'operation': 'control',
                    'length': self.mw_len_NV,
                    'waveforms': {
                        "I": "NV_const_wf",
                        "Q": "zero_wf"
                    },
                },
            },

            'waveforms': {
                'NV_const_wf': {'type': 'constant', 'sample': voltage_NV},
                'zero_wf': {'type': 'constant', 'sample': 0.0},
            },

            'mixers': {
                'mixer_qubit': [
                    # IQ-calibration
                    {'intermediate_frequency': self.qubit_IF,
                     'lo_frequency': self.qubit_LO,
                     'correction': IQ_imbalance(self.g_cor, self.phi_cor)},  # (Amplitude, Phase) Imbalances!
                ],
            },
        }
        return config

def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    n = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(n * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]



