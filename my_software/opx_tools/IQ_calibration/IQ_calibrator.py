from qm.QuantumMachinesManager import QuantumMachinesManager
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opti

from configuration_IQ_calibration import *  # Import all settings from configuration.py
from auto_mixer_tools_visa import RhodeSchwarzRTO6
from microwave_control import Windfreak_MW_control

parameters = parameters()  # Create an instance of the Parameters class




class IQMixerCalibrator():
    """
    A class to calibrate an IQ mixer using an oscilloscope and a microwave source.
    """

    def __init__(self, voltage_opx):
        """Initializes the IQMixerCalibrator with configurations from 'configuration.py'."""
        self.parameters = None  # Use your configuration module
        self.oscilloscope = None
        self.mw_source = None
        self.qm = None
        self.qmm = None  # Add QuantumMachinesManager instance
        self.voltage_opx = voltage_opx

    def connect_instruments(self):
        """Connects to and configures the required instruments."""
        try:
            # 0. Configuration
            self.parameters = parameters
            self.opx_config = parameters.create_config(self.voltage_opx)
            print(self.parameters.qubit_LO)

            # 1. Connect to Quantum Machines Manager
            self.qmm = QuantumMachinesManager(self.parameters.opx_ip_address)
            for machine in self.qmm.list_open_quantum_machines():
                qm_config = self.qmm.get_qm(machine).get_config()
                try:
                    ports = qm_config['controllers']['con1']['analog_outputs']
                except:
                    continue
                if parameters.port_I in ports:
                    self.qmm.get_qm(machine).close()
                if parameters.port_Q in ports:
                    self.qmm.get_qm(machine).close()

            self.qm = self.qmm.open_qm(self.opx_config, close_other_machines=False)

            # 3. Connect to Oscilloscope
            self.oscilloscope = RhodeSchwarzRTO6(qm=self.qm)
            self.oscilloscope.method = 2
            self.oscilloscope.set_automatic_video_bandwidth(1)
            self.oscilloscope.set_automatic_bandwidth(0)
            self.oscilloscope.set_cont_off()

            # 4. Connect to Microwave Source
            self.mw_source = Windfreak_MW_control(self.parameters.mw_usb_com_port)
            self.mw_source.start_MW(self.parameters.qubit_LO, self.parameters.microwave_max_power)

        except Exception as e:
            raise RuntimeError(f"Error connecting to instruments: {e}")

    def disconnect_instruments(self):
        """Disconnects from the instruments and closes connections."""
        try:
            if self.oscilloscope:
                self.oscilloscope.set_cont_on()
                self.oscilloscope.__del__()
            if self.mw_source:
                self.mw_source.end_MW()
            if self.qm:
                self.qm.close()  # Close the Quantum Machine
            if self.qmm:
                self.qmm.close()  # Close the Quantum Machines Manager

        except Exception as e:
            print(f"Error during instrument disconnection: {e}")

    def _initial_simplex(self, center):
        """Creates the initial simplex for optimization."""
        simplex = np.zeros([3, 2])
        simplex[0, :] = [center[0] - 0.05, center[1] - 0.05]
        simplex[1, :] = [center[0] + 0.05, center[1]]
        simplex[2, :] = [center[0] - 0.05, center[1] + 0.05]
        return simplex

    def _setup_oscilloscope_measurement(self):
        """Configures the oscilloscope for power measurement."""
        if self.parameters.method == 1:  # Channel power
            self.oscilloscope.enable_measurement()
            self.oscilloscope.sets_measurement_integration_bw(10 * self.parameters.measBW)
            self.oscilloscope.disables_measurement_averaging()
        elif self.parameters.method == 2:  # Marker
            self.oscilloscope.get_single_trigger()
            self.oscilloscope.active_marker(1)
        self.oscilloscope.set_sweep_points(self.parameters.measNumPoints)
        self.oscilloscope.set_span(self.parameters.fullSpan)
        self.oscilloscope.set_bandwidth(1e6)

    def _get_signal_power(self):
        """Gets the power of the signal at the desired IF frequency."""
        self.oscilloscope.set_center_freq(self.parameters.qubit_LO + self.parameters.qubit_IF)
        if self.parameters.method == 2:  # Marker
            self.oscilloscope.set_marker_freq(1, self.parameters.qubit_LO + self.parameters.qubit_IF)
        return int(self.oscilloscope.get_amp())

    def optimize_LO_leakage(self, initial_simplex=None, x0=[0, 0]):
        """Optimizes I and Q DC offsets to minimize LO leakage."""
        self.oscilloscope.set_center_freq(self.parameters.qubit_LO)
        if self.parameters.method == 2:  # Marker
            self.oscilloscope.set_marker_freq(1, self.parameters.qubit_LO)

        start_time = time.time()
        fun_leakage = lambda x: self.oscilloscope.get_leakage(x[0], x[1])

        res_leakage = opti.minimize(
            fun_leakage,
            x0,
            method="Nelder-Mead",
            options={
                "xatol": self.parameters.xatol,
                "fatol": self.parameters.fatol,
                "maxiter": self.parameters.maxiter,
                "initial_simplex": initial_simplex
            },
        )
        print(
            f"LO Leakage Results: Found a minimum of {int(res_leakage.fun)} dBm at (I0, Q0) = ({res_leakage.x[0]:.5f}, {res_leakage.x[1]:.5f}) in"
            f" {int(time.time() - start_time)} seconds"
        )
        return res_leakage.x, res_leakage.fun

    def optimize_image(self, initial_simplex=None, x0=[0, 0]):
        """Optimizes I and Q DC offsets to maximize image rejection."""
        self.oscilloscope.set_center_freq(self.parameters.qubit_LO - self.parameters.qubit_IF)
        if self.parameters.method == 2:  # Marker
            self.oscilloscope.set_marker_freq(1, self.parameters.qubit_LO - self.parameters.qubit_IF)

        start_time = time.time()
        fun_image = lambda x: self.oscilloscope.get_image(x[0], x[1])

        res_image = opti.minimize(
            fun_image,
            x0,
            method="Nelder-Mead",
            options={
                "xatol": self.parameters.xatol,
                "fatol": self.parameters.fatol,
                "initial_simplex": initial_simplex,
                "maxiter": self.parameters.maxiter,
            },
        )
        print(
            f"Image Rejection Results: Found a minimum of {int(res_image.fun)} dBm at (I0, Q0) = ({res_image.x[0]:.5f}, {res_image.x[1]:.5f}) in "
            f"{int(time.time() - start_time)} seconds"
        )

        return res_image.x, res_image.fun

    def _perform_spectrum_sweep(self):
        """Performs a wideband spectrum sweep and plots the result."""
        self.oscilloscope.set_bandwidth(self.parameters.sweepBW)
        self.oscilloscope.set_sweep_points(self.parameters.fullNumPoints)
        self.oscilloscope.set_center_freq(self.parameters.qubit_LO)
        self.oscilloscope.set_span(self.parameters.fullSpan)
        self.oscilloscope.get_single_trigger()
        freq_vec, amp = self.oscilloscope.get_full_trace()
        return freq_vec, amp

    def _plot_spectrum(self, freq_vec, amp_before, amp_after):
        """Plots the spectrum before and after calibration."""
        plt.figure("Full Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (dBm)")
        plt.plot(freq_vec, amp_before, alpha=0.5, label="Before")
        plt.plot(freq_vec, amp_after, alpha=0.5, label="After")
        plt.title(
            f"IQ-Calibration. \n Target frequency is {(self.parameters.qubit_LO + self.parameters.qubit_IF) / 1e9:.2f} GHz."
        )
        plt.legend()
        plt.show()

    def calibrate(self):
        """Performs the full IQ mixer calibration routine."""

        if not (self.qm and self.oscilloscope and self.mw_source):
            raise RuntimeError("Instruments not connected! Call 'connect_instruments()' first.")

        self.mw_source.start_MW(self.parameters.qubit_LO, self.parameters.microwave_max_power)

        self._setup_oscilloscope_measurement()
        signal_power = self._get_signal_power()

        if self.parameters.bDoSweeps:
            freq_vec, amp_before = self._perform_spectrum_sweep()

        simplex_leakage = self._initial_simplex([0, 0])
        simplex_IQ = self._initial_simplex([0, 0])
        result_leakage = [0, 0]
        result_IQ = [0, 0]

        simplex_IQ_array = np.zeros([self.parameters.optimization_repititions, 2])
        simplex_leakage_array = np.zeros([self.parameters.optimization_repititions, 2])

        fun_IQ_array = np.zeros(self.parameters.optimization_repititions)
        fun_leakage_array = np.zeros(self.parameters.optimization_repititions)

        for i in range(self.parameters.optimization_repititions):
            print("\n############################################################")
            print(f"Optimization Step {i + 1}")
            print("############################################################")

            result_leakage, fun_leakage = self.optimize_LO_leakage(
                initial_simplex=simplex_leakage, x0=result_leakage
            )
            simplex_leakage = self._initial_simplex(result_leakage)
            simplex_leakage_array[i, :] = result_leakage
            fun_leakage_array[i] = fun_leakage

            result_IQ, fun_IQ = self.optimize_image(
                initial_simplex=simplex_IQ, x0=result_IQ
            )
            simplex_IQ = self._initial_simplex(result_IQ)
            simplex_IQ_array[i, :] = result_IQ
            fun_IQ_array[i] = fun_IQ

        if self.parameters.bDoSweeps:
            _, amp_after = self._perform_spectrum_sweep()
            self._plot_spectrum(freq_vec, amp_before, amp_after)

        # maximize suppression of image and LO leakage
        reward = np.abs(fun_IQ_array + fun_leakage_array)
        best_index = np.argmax(reward)
        best_IQ, best_leakage = simplex_IQ_array[best_index], simplex_leakage_array[best_index]

        print(best_leakage, best_IQ)

        return best_leakage, best_IQ

if __name__ == "__main__":

    voltages = np.linspace(0.001, 0.2, 50)

    g_array, phi_array = np.zeros(len(voltages)), np.zeros(len(voltages))
    I_array, Q_array = np.zeros(len(voltages)), np.zeros(len(voltages))

    for v_index, voltage in enumerate(voltages):

        exception_counter = 1
        while True:
            try:
                time.sleep(2)
                calibrator = IQMixerCalibrator(voltage)
                calibrator.connect_instruments()
                best_leakage, best_IQ = calibrator.calibrate()
                calibrator.disconnect_instruments()
                g_array[v_index], phi_array[v_index] = best_leakage
                I_array[v_index], Q_array[v_index] = best_IQ
                break
            except Exception as e:
                print(f"Calibration failed for voltage {voltage} for the {exception_counter} time. Repeating now. {e}")
                exception_counter += 1

    print(g_array, phi_array)
    np.save("g_array.npy", g_array)
    np.save("phi_array.npy", phi_array)
    np.save("I_array.npy", I_array)
    np.save("Q_array.npy", Q_array)
    np.save("voltages.npy", voltages)