import matplotlib
from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
from configuration_FM_working_oscillator import parameters

matplotlib.use("Qt5Agg")


# IN DIESER VERSION: REFERENZSIGNAL DIREKT ÜBER OSZILLATOR
# ZURÜCKSETZEN DER PHASE DES OSZILLATORS NACH JEDER ITERATION
# ZURÜCKSETZEN DER ENSEMBLE FREQUENZ NACH JEDER ITERATION


class FM_setup:

    def __init__(self, f_mod=7e3, f_dev=250e3, voltage_opx=0.01, g_cor=0.11474, phi_cor=-0.09787, I_offset=-0.01040,
                 Q_offset=-0.01000):

        self.config_parameters = parameters()

        # not sure if this is a good way of handling data, have to learn a bit more object-oriented python
        self.config_parameters.f_mod = f_mod
        self.config_parameters.f_dev = f_dev
        self.config_parameters.voltage_opx = voltage_opx
        self.config_parameters.g_cor = g_cor
        self.config_parameters.phi_cor = phi_cor
        self.config_parameters.I_offset = I_offset
        self.config_parameters.Q_offset = Q_offset

        # time differences between the segments of the chirp
        self.dt = self.config_parameters.FM_period_duration() / self.config_parameters.n_chirp_segments
        self.omega_FM = 2 * np.pi / self.config_parameters.FM_period_duration()

        self.time_vec = self.dt * np.array(range(self.config_parameters.n_chirp_segments + 1))
        # chirp times have to be specified in clock cycles not real time
        clock_cycles_vec_int = np.rint(self.time_vec / 4).astype(int)

        self.freq_vec = self.config_parameters.f_dev * np.sin(self.omega_FM * self.time_vec)

        # rates with that the frequency is changed after each segment
        # rates = (np.diff(freq_vec) / (FM_period_duration / n_segments)).astype(int).tolist()
        self.rates = np.rint(
            (np.diff(self.freq_vec) / (
                        self.config_parameters.FM_period_duration() / self.config_parameters.n_chirp_segments))).astype(
            int).tolist()

        self.units = "Hz/nsec"

        pass

    def execute_FM(self):
        # EXECUTION OF THE OPX PROGRAM
        # -----------------------------------------------------------------------------------------
        qmm = QuantumMachinesManager(self.config_parameters.opx_ip_address)
        for machine in qmm.list_open_quantum_machines():
            qm_config = qmm.get_qm(machine).get_config()
            try:
                ports = qm_config['controllers']['con1']['analog_outputs']
            except:
                continue
            if self.config_parameters.port_I in ports or self.config_parameters.port_Q in ports:
                qmm.get_qm(machine).close()

        qm = qmm.open_qm(self.config_parameters.create_config(), close_other_machines=False)

        # on the IQ-ports a sine with varying frequency is generated (chirp)
        # on the LIA port a sine with constant frequency - the FM frequency - is generated (reference signal)
        # the method with the zero pulses in the infinite_loop and the sticky parameters (in configuration_FM.py) was suggested
        # by the QM team (see their discord channel)
        with program() as prog:
            update_frequency("ensemble", self.config_parameters.f_base)
            reset_phase("LIA")
            align("ensemble", "LIA")
            play("const", "ensemble", chirp=(self.rates, self.units), continue_chirp=True)
            play("const_single", "LIA")
            with infinite_loop_():
                update_frequency("ensemble", self.config_parameters.f_base)
                reset_phase("LIA")
                align("ensemble", "LIA")
                play("zero", "ensemble", chirp=(self.rates, self.units), continue_chirp=True)
                play("zero_single", "LIA")

        my_job = qm.execute(prog)

# DEFINITION OF THE CHIRP RATES
# -----------------------------------------------------------------------------------------


fm = FM_setup()
fm.execute_FM()