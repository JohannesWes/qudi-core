from qm.QuantumMachinesManager import QuantumMachinesManager
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opti
import pandas as pd

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
                if self.parameters.port_I in ports or self.parameters.port_Q in ports:
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
        """Optimizes g and phi to maximize image rejection."""
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
            f"Image Rejection Results: Found a minimum of {int(res_image.fun)} dBm at (g, phi) = ({res_image.x[0]:.5f}, {res_image.x[1]:.5f}) in "
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
        simplex_image = self._initial_simplex([0, 0])
        result_leakage = [0, 0]
        result_image = [0, 0]

        result_image_array = np.zeros([self.parameters.optimization_repititions, 2])
        result_leakage_array = np.zeros([self.parameters.optimization_repititions, 2])

        fun_image_array = np.zeros(self.parameters.optimization_repititions)
        fun_leakage_array = np.zeros(self.parameters.optimization_repititions)

        for i in range(self.parameters.optimization_repititions):
            print("\n############################################################")
            print(f"Optimization Step {i + 1}")
            print("############################################################")

            result_leakage, fun_leakage = self.optimize_LO_leakage(
                initial_simplex=simplex_leakage, x0=result_leakage
            )

            # redefine simplex to have better starting value for the next iteration
            simplex_leakage = self._initial_simplex(result_leakage)

            result_leakage_array[i, :] = result_leakage
            fun_leakage_array[i] = fun_leakage

            result_image, fun_image = self.optimize_image(
                initial_simplex=simplex_image, x0=result_image
            )

            # redefine simplex to have better starting value for the next iteration
            simplex_image = self._initial_simplex(result_image)

            result_image_array[i, :] = result_image
            fun_image_array[i] = fun_image

        if self.parameters.bDoSweeps:
            _, amp_after = self._perform_spectrum_sweep()
            self._plot_spectrum(freq_vec, amp_before, amp_after)

        # maximize suppression of image and LO leakage
        reward = np.abs(fun_image_array + fun_leakage_array)
        best_index = np.argmax(reward)
        best_image, best_leakage = result_image_array[best_index], result_leakage_array[best_index]

        print(best_leakage, best_image)

        return best_leakage, best_image

if __name__ == "__main__":

    voltages = np.linspace(0.01, 0.5, 50)

    g_array, phi_array = np.zeros(len(voltages)), np.zeros(len(voltages))
    I_array, Q_array = np.zeros(len(voltages)), np.zeros(len(voltages))

    for v_index, voltage in enumerate(voltages):

        exception_counter = 1
        while True:
            try:
                time.sleep(1)
                calibrator = IQMixerCalibrator(voltage)
                calibrator.connect_instruments()
                best_leakage, best_image = calibrator.calibrate()
                calibrator.disconnect_instruments()
                I_array[v_index], Q_array[v_index] = best_leakage
                g_array[v_index], phi_array[v_index] = best_image
                print(f"CALIBRATION SUCCESSFUL FOR VOLTAGE {voltage}, g: {best_image[0]}, phi: {best_image[1]}, I: {best_leakage[0]}, Q: {best_leakage[1]}")

                break
            except Exception as e:
                print(f"Calibration failed for voltage {voltage} for the {exception_counter} time. Repeating now. {e}")
                exception_counter += 1



    # Save calibration data to numpy arrays
    print(g_array, phi_array)
    # np.save("g_array.npy", g_array)
    # np.save("phi_array.npy", phi_array)
    # np.save("I_array.npy", I_array)
    # np.save("Q_array.npy", Q_array)
    # np.save("voltages.npy", voltages)


    # Save calibration data to a CSV file
    calibration_df = pd.DataFrame(
        {
            "voltage [V]": voltages,
            "g": g_array,
            "phi": phi_array,
            "I": I_array,
            "Q": Q_array,
        }
    )

    # save calibration data to a csv file with filename specifying the date and time
    calibration_df.to_csv(f"calibration_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv", sep="\t")