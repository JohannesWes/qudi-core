import numpy as np
from dataclasses import dataclass


##############
# Parameters #
##############
# ----------------------------------------------------------------------------------------------------------------------

class parameters:
    def __init__(self):
        self.f_mod = 5.0e4  # modulation frequency
        self.f_dev = 200.00e3  # deviation of the modulation
        self.f_base = 21.158e6  # IF frequency for Red Pitaya (changed from 20 MHz)
        self.ensemble_lo = 2.87e9

        # Red Pitaya specific parameters
        self.redpitaya_hostname = '10.203.129.28'  # Red Pitaya IP address
        self.redpitaya_port = 2222  # Red Pitaya SSH port

        # Initial correction values for I & Q - these will be optimized
        self.I_offset, self.Q_offset = 0.0, 0.0

        # Initial corrections for g and phi - these will be optimized
        self.g_cor, self.phi_cor = 0.0, 0.0

        # Red Pitaya output channels
        self.channel_I = 0  # DAC A for I component
        self.channel_Q = 1  # DAC B for Q component

        self.optimization_repititions = 3  # How often LO-leakage and IQ-imbalances are repeated
        self.mw_usb_com_port = "COM3"  # The USB port for the MW, opened using visa.
        self.osci_address = "TCPIP0::10.203.129.15::inst0::INSTR"  # The TCIP address for the oscilloscope

        # Frequencies
        self.qubit_LO = self.ensemble_lo
        self.qubit_IF = self.f_base

        # Important Parameters:
        self.bDoSweeps = True  # If True, performs a large sweep before and after the optimization.
        self.method = 2  # If set to 1, checks power using a channel power measurement. If set to 2, checks power using a marker.

        # Parameters for oscilloscope spectrum measurement:
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
        self.maxiter = 50  # 50 iterations should be more than enough

        # Red Pitaya specific: component to use for IF signal generation
        self.rp_component_index = 0  # Use component 0 for the IF signal


def IQ_imbalance(g, phi):
    """Calculate IQ imbalance correction matrix."""
    c = np.cos(phi)
    s = np.sin(phi)
    n = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(n * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]