import matplotlib
from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
from my_software.opx_tools.frequency_modulation.configuration_FM_working_oscillator_RF_mixer_hyperfine import parameters, IQ_imbalance
import pandas as pd

matplotlib.use("Qt5Agg")


# IN DIESER VERSION: REFERENZSIGNAL DIREKT ÜBER OSZILLATOR
# ZURÜCKSETZEN DER PHASE DES OSZILLATORS NACH JEDER ITERATION
# ZURÜCKSETZEN DER ENSEMBLE FREQUENZ NACH JEDER ITERATION


class FM_setup:

    def __init__(self, f_mod=7e3, f_dev=250e3, OPX_LO_voltage=0.01, OPX_IF_voltage=0.5, g_cor=0.11474, phi_cor=-0.09787, I_offset=-0.01040,
                 Q_offset=-0.01000):

        self.config_parameters = parameters()

        # not sure if this is a good way of handling data, have to learn a bit more object-oriented python
        self.config_parameters.f_mod = f_mod
        self.config_parameters.f_dev = f_dev
        self.config_parameters.OPX_LO_voltage = OPX_LO_voltage
        self.config_parameters.OPX_IF_voltage = OPX_IF_voltage
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

            with infinite_loop_():
                play("const_single_IF_hyperfine", "IF_hyperfine")

        my_job = qm.execute(prog)

def get_calibration_data(voltages, cal_filename="calibration_data.csv"):
    calibration_data = pd.read_csv(cal_filename, sep="\t", index_col=0)

    cal_voltages = np.array(calibration_data["voltage [V]"])
    g_cal = np.array(calibration_data["g"])
    phi_cal = np.array(calibration_data["phi"])
    I_cal = np.array(calibration_data["I"])
    Q_cal = np.array(calibration_data["Q"])

    # interpolate calibration data to get g, phi, I, Q for the now chosen voltages
    g = np.interp(voltages, cal_voltages, g_cal)
    phi = np.interp(voltages, cal_voltages, phi_cal)
    I = np.interp(voltages, cal_voltages, I_cal)
    Q = np.interp(voltages, cal_voltages, Q_cal)

    return g, phi, I, Q

if __name__ == '__main__':

    OPX_LO_voltage = 0.5
    OPX_IF_voltage = 0.1

    f_dev = 620e3
    f_mod = 6.3e3

    g, phi, I, Q = get_calibration_data(OPX_LO_voltage, cal_filename="..\IQ_calibration\calibration_2024-08-02-10-33-57.csv")
    print("g: ", g, "phi: ", phi, "I: ", I, "Q: ", Q)

    # # wenn ich die werte hier verändere, sehe ich, dass sich im Oszi was ändert (die stärke von LO durchbruch z.B. wird größer/kleiner). Die werte hier zu setzen hat also übers FM_setup schon auswirkungen
    # g = 0.03890917662120046
    # phi = -0.09894201816923673
    #
    # I = -0.005133440923152878
    # Q = -0.007058443272055322

    fm = FM_setup(OPX_LO_voltage=OPX_LO_voltage, OPX_IF_voltage=OPX_IF_voltage, f_dev=f_dev, f_mod=f_mod, g_cor=g, phi_cor=phi, I_offset=I, Q_offset=Q, )
    fm.execute_FM()