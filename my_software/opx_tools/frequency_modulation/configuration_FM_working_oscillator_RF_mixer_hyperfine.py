import numpy as np
import math

def round_to_multiple(x, base=4):
    return base * round(x / base)

def lcm(x, y):
    """"Least common multiple of x and y."""
    return abs(x * y) // math.gcd(x, y)

def round_to_common_multiple(n, a, b):
    common_multiple = lcm(a, b)
    return common_multiple * round(n / common_multiple)


##############
# Parameters #
##############
# ----------------------------------------------------------------------------------------------------------------------

class parameters:

    def __init__(self):
        # number of time-segments for the FM
        self.n_chirp_segments = 100

        self.f_mod = 4.5e3  # modulation frequency. Note, that the effective modulation frequency is smaller than f_mod,
        # due to the integer multiple of 4 ns constraint on the pulse length

        self.f_dev = 700e3  # deviation of the modulation
        self.f_base = 200.00e6  # frequency around that we modulate: base frequency chosen as middle of bandwidth (400 MHz) of OPX
        self.ensemble_lo = 2.5e9
        self.OPX_LO_voltage = 0.5  # peak (not peak-to-peak) output voltage of the opx

        # corrections for I & Q voltages; values from IQ-calibration script
        self.I_offset, self.Q_offset = -0.00431, -0.00452

        # corrections for g and phi; values from IQ-calibration script
        self.g_cor, self.phi_cor = -0.0043692122361752225, -0.004434408861159571

        self.port_I, self.port_Q = 3, 4
        self.port_reference = 5


        self.f_IF_hyperfine = 2.158e6
        self.port_IF_hyperfine = 6
        self.OPX_IF_voltage = 0.05

        self.opx_ip_address = '10.203.129.12'

        # duration of one FM period (in ns); also serves as the pulse length; pulse length must be integer multiple of 4
        # note, that the effective modulation frequency is smaller than f_mod, due to the above constraint on the pulse length
        #FM_period_duration = round_to_multiple(1 / f_mod / 1e-9, base=4)

        # print(f"\nFM_period_duration [ns], also serves as pulse length: {self.FM_period_duration()}")
        # print(f"f_mod [kHz] {1/(self.FM_period_duration()/1e9)/1e3} \n")

    def FM_period_duration(self):
        FM_period_duration = round_to_common_multiple(1 / self.f_mod / 1e-9, 4, 4 * self.n_chirp_segments)
        return FM_period_duration

    def create_config(self):
        config = {
            "version": 1,
            "controllers": {
                "con1": {
                    "type": "opx1",
                    "analog_outputs": {
                        # the voltage offsets correct for the LO Leakage; They are the outputs of the IQ-calibration script
                        self.port_I: {"offset": self.I_offset},
                        self.port_Q: {"offset": self.Q_offset},

                        self.port_reference: {"offset": +0.0},

                        self.port_IF_hyperfine: {"offset": +0.0},
                    },
                }
            },
            "elements": {

                "ensemble": {
                    "mixInputs": {
                        "I": ("con1", self.port_I),
                        "Q": ("con1", self.port_Q),

                        "lo_frequency": self.ensemble_lo,
                        "mixer": "mixer_ensemble"
                    },
                    "intermediate_frequency": self.f_base,
                    'sticky': {
                        'analog': True,
                        'duration': 200
                    },
                    "operations": {
                        "const": "constPulse",
                        "zero": "zeroPulse",
                    },
                },

                "LIA": {
                    "singleInput": {"port": ("con1", self.port_reference)},
                    # frequency is updated in the program to sub-Hz resolution
                    "intermediate_frequency": 1 / (self.FM_period_duration() / 1e9),
                    "operations": {
                        "const_single": "constPulse_single",
                        "zero_single": "zeroPulse_single",
                    },
                },

                "IF_hyperfine": {
                    "singleInput": {"port": ("con1", self.port_IF_hyperfine)},
                    # frequency is updated in the program to sub-Hz resolution
                    "intermediate_frequency": self.f_IF_hyperfine,
                    "operations": {
                        "const_single_IF_hyperfine": "constPulse_IF_hyperfine",
                        "zero_single_IF_hyperfine": "zeroPulse_IF_hyperfine",
                    },
                },

            },

            "pulses": {
                "constPulse": {
                    "operation": "control",
                    "length": self.FM_period_duration(),  # in ns; multiple of 4 ns (clock cycle)
                    "waveforms": {
                        "I": "const_wf",
                        "Q": "zero_wf"
                    },
                },
                "zeroPulse": {
                    "operation": "control",
                    "length": self.FM_period_duration(),  # in ns; multiple of 4 ns (clock cycle)
                    "waveforms": {
                        "I": "zero_wf",
                        "Q": "zero_wf"
                    },
                },
                "constPulse_single": {
                    "operation": "control",
                    "length": self.FM_period_duration(),  # in ns; multiple of 4 ns (clock cycle)
                    "waveforms": {"single": "const_wf"},
                },
                "zeroPulse_single": {
                    "operation": "control",
                    "length": self.FM_period_duration(),  # in ns; multiple of 4 ns (clock cycle)
                    "waveforms": {"single": "zero_wf"},
                },

                "constPulse_IF_hyperfine": {
                    "operation": "control",
                    "length": 100*self.FM_period_duration(),  # in ns; multiple of 4 ns (clock cycle)
                    "waveforms": {"single": "const_wf_IF_hyperfine"},
                },
                "zeroPulse_IF_hyperfine": {
                    "operation": "control",
                    "length": self.FM_period_duration(),  # in ns; multiple of 4 ns (clock cycle)
                    "waveforms": {"single": "zero_wf"},
                },

            },

            "waveforms": {
                "const_wf": {"type": "constant", "sample": self.OPX_LO_voltage},
                "const_wf_IF_hyperfine": {"type": "constant", "sample": self.OPX_IF_voltage},
                "zero_wf": {"type": "constant", "sample": 0.0},
            },

            "mixers": {
                "mixer_ensemble": [
                    {
                        "intermediate_frequency": self.f_base,
                        "lo_frequency": self.ensemble_lo,
                        "correction": IQ_imbalance(self.g_cor , self.phi_cor),
                    },
                ],
            }
        }

        return config




# IQ_calibration function
# -----------------------------------------------------------------------------------------

def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    n = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(n * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


# CONFIGURATION
# ----------------------------------------------------------------------------------------------------------------------


