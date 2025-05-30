import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opti
import pandas as pd
import pyrpl
import traceback

import configuration_IQ_calibration_redpitaya as config
from auto_mixer_tools_visa import RhodeSchwarzRTO6_RedPitaya
from microwave_control import Windfreak_MW_control

# Create an instance of the Parameters class from configuration
parameters = config.parameters()


class IQMixerCalibrator_RedPitaya():
    """
    A class to calibrate an IQ mixer using a Red Pitaya, oscilloscope, and microwave source.
    """

    def __init__(self, if_amplitude, lo_frequency=None):
        """
        Initializes the IQMixerCalibrator with Red Pitaya.

        Parameters:
        -----------
        if_amplitude : float
            The IF signal amplitude (0 to 1) for the Red Pitaya output
        lo_frequency : float, optional
            The LO frequency in Hz. If None, uses the default from configuration
        """
        self.parameters = None
        self.oscilloscope = None
        self.mw_source = None
        self.pyrpl = None
        self.if_amplitude = if_amplitude
        self.lo_frequency = lo_frequency if lo_frequency is not None else parameters.qubit_LO

    def connect_instruments(self):
        """Connects to and configures the required instruments."""
        try:
            # 0. Configuration
            self.parameters = parameters

            # Update LO frequency if specified
            if self.lo_frequency:
                self.parameters.qubit_LO = self.lo_frequency

            print(f"Connecting to Red Pitaya at {self.parameters.redpitaya_hostname}")
            print(f"IF frequency: {self.parameters.qubit_IF / 1e6:.3f} MHz")
            print(f"LO frequency: {self.parameters.qubit_LO / 1e9:.3f} GHz")
            print(f"IF amplitude: {self.if_amplitude}")

            # 1. Connect to Red Pitaya via pyrpl
            self.pyrpl = pyrpl.Pyrpl(
                hostname=self.parameters.redpitaya_hostname,
                config='iq_calibration_config',
                gui=False
            )

            # 2. Initialize fgen3 module
            self.fgen3 = self.pyrpl.rp.fgen3

            # Configure fgen3 for IQ signal generation
            # Use the setup method with the correct parameters
            self.fgen3.setup(
                gen_enable=True,
                output_zero=False,
                overall_dc_offset_a=0.0,
                overall_dc_offset_b=0.0,
                # Component settings using lists
                enables=[True, False, False],  # Only component 0 active
                frequencies=[self.parameters.qubit_IF, 0, 0],
                amplitudes_a=[self.if_amplitude, 0, 0],
                amplitudes_b=[self.if_amplitude, 0, 0],
                phase_offsets_a=[0.0, 0, 0],
                phase_offsets_b=[90.0, 0, 0]
            )

            # Route fgen3 outputs to physical outputs
            self.fgen3.output_to_dsp_enable_o = True
            # Configure ASG modules to route to physical outputs
            self.pyrpl.rp.asg0.output_direct = "out1"
            self.pyrpl.rp.asg0.trigger_source = "immediately"
            self.pyrpl.rp.asg1.output_direct = "out2"
            self.pyrpl.rp.asg1.trigger_source = "immediately"

            # 3. Connect to Oscilloscope
            self.oscilloscope = RhodeSchwarzRTO6_RedPitaya(
                pyrpl_instance=self.pyrpl,
                address=self.parameters.osci_address
            )
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
            if self.pyrpl:
                # Turn off signal generation
                if hasattr(self, 'fgen3'):
                    self.fgen3.gen_enable = False
                # Close pyrpl connection
                self.pyrpl.__exit__(None, None, None)

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
            f"LO Leakage Results: Found a minimum of {int(res_leakage.fun)} dBm at (I0, Q0) = ({res_leakage.x[0]:.5f}, {res_leakage.x[1]:.5f}) in "
            f"{int(time.time() - start_time)} seconds"
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

        if not (self.pyrpl and self.oscilloscope and self.mw_source):
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

        print(f"\nBest calibration values:")
        print(f"DC offsets (I, Q): {best_leakage}")
        print(f"IQ corrections (g, phi): {best_image}")

        return best_leakage, best_image


if __name__ == "__main__":
    # Define IF amplitudes to calibrate (0 to 1 scale for Red Pitaya)
    if_amplitudes = np.linspace(0.01, 1.0, 20)  # Adjusted for Red Pitaya's 0-1 amplitude range

    # Define LO frequencies to calibrate (in Hz)
    lo_frequencies = [2.87e9]  # Can add more frequencies like [2.8e9, 2.85e9, 2.87e9, 2.9e9]

    # Initialize storage for results
    results_list = []

    # Calibrate for each LO frequency and IF amplitude combination
    for lo_freq in lo_frequencies:
        print(f"\n\n{'=' * 60}")
        print(f"Starting calibration for LO frequency: {lo_freq / 1e9:.3f} GHz")
        print(f"{'=' * 60}\n")

        g_array, phi_array = np.zeros(len(if_amplitudes)), np.zeros(len(if_amplitudes))
        I_array, Q_array = np.zeros(len(if_amplitudes)), np.zeros(len(if_amplitudes))

        for amp_index, if_amplitude in enumerate(if_amplitudes):
            exception_counter = 1
            while True:
                try:
                    time.sleep(1)
                    print(f"\n{'=' * 50}")
                    print(f"Calibrating for IF amplitude: {if_amplitude:.3f}, LO: {lo_freq / 1e9:.3f} GHz")
                    print(f"{'=' * 50}")

                    calibrator = IQMixerCalibrator_RedPitaya(if_amplitude, lo_frequency=lo_freq)
                    calibrator.connect_instruments()
                    best_leakage, best_image = calibrator.calibrate()
                    calibrator.disconnect_instruments()

                    I_array[amp_index], Q_array[amp_index] = best_leakage
                    g_array[amp_index], phi_array[amp_index] = best_image

                    print(f"\nCALIBRATION SUCCESSFUL:")
                    print(f"  IF amplitude: {if_amplitude:.3f}")
                    print(f"  LO frequency: {lo_freq / 1e9:.3f} GHz")
                    print(f"  g: {best_image[0]:.5f}, phi: {best_image[1]:.5f}")
                    print(f"  I: {best_leakage[0]:.5f}, Q: {best_leakage[1]:.5f}")

                    # Store results
                    results_list.append({
                        'lo_frequency_ghz': lo_freq / 1e9,
                        'if_amplitude': if_amplitude,
                        'g': best_image[0],
                        'phi': best_image[1],
                        'I_offset': best_leakage[0],
                        'Q_offset': best_leakage[1]
                    })

                    break


                except Exception as e:
                    print(f"Calibration failed for IF amplitude {if_amplitude:.3f} (attempt {exception_counter}): {e}")
                    print("Full traceback:")
                    traceback.print_exc()  # This will print the full traceback
                    print("-" * 50)  # Visual separator
                    exception_counter += 1
                    if exception_counter > 3:
                        print(f"Skipping IF amplitude {if_amplitude:.3f} after 3 failed attempts")
                        # Fill with NaN values
                        I_array[amp_index], Q_array[amp_index] = np.nan, np.nan
                        g_array[amp_index], phi_array[amp_index] = np.nan, np.nan
                        break
                    time.sleep(5)  # Wait longer between retries

        # Save calibration data for this LO frequency
        print(f"\nCalibration complete for LO frequency {lo_freq / 1e9:.3f} GHz")
        print(f"g values: {g_array}")
        print(f"phi values: {phi_array}")

        # Save arrays for this LO frequency
        np.save(f"g_array_LO_{lo_freq / 1e9:.3f}GHz.npy", g_array)
        np.save(f"phi_array_LO_{lo_freq / 1e9:.3f}GHz.npy", phi_array)
        np.save(f"I_array_LO_{lo_freq / 1e9:.3f}GHz.npy", I_array)
        np.save(f"Q_array_LO_{lo_freq / 1e9:.3f}GHz.npy", Q_array)
        np.save(f"if_amplitudes_LO_{lo_freq / 1e9:.3f}GHz.npy", if_amplitudes)

        # Create a CSV for this LO frequency
        calibration_df_lo = pd.DataFrame({
            "if_amplitude": if_amplitudes,
            "g": g_array,
            "phi": phi_array,
            "I_offset": I_array,
            "Q_offset": Q_array,
        })
        calibration_df_lo.to_csv(
            f"calibration_redpitaya_LO_{lo_freq / 1e9:.3f}GHz_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv",
            sep="\t"
        )

    # Save all results to a comprehensive CSV file
    if results_list:
        all_results_df = pd.DataFrame(results_list)
        all_results_df.to_csv(
            f"calibration_redpitaya_all_results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv",
            sep="\t"
        )
        print(f"\n\nAll calibration results saved to CSV files")

    # Create summary plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        for lo_freq in lo_frequencies:
            # Filter results for this LO frequency
            lo_results = [r for r in results_list if r['lo_frequency_ghz'] == lo_freq / 1e9]
            if not lo_results:
                continue

            amps = [r['if_amplitude'] for r in lo_results]
            g_vals = [r['g'] for r in lo_results]
            phi_vals = [r['phi'] for r in lo_results]
            i_vals = [r['I_offset'] for r in lo_results]
            q_vals = [r['Q_offset'] for r in lo_results]

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'IQ Calibration Results - LO: {lo_freq / 1e9:.3f} GHz', fontsize=16)

            ax1.plot(amps, g_vals, 'bo-')
            ax1.set_xlabel('IF Amplitude')
            ax1.set_ylabel('g (gain imbalance)')
            ax1.grid(True)
            ax1.set_title('Gain Imbalance vs IF Amplitude')

            ax2.plot(amps, phi_vals, 'ro-')
            ax2.set_xlabel('IF Amplitude')
            ax2.set_ylabel('φ (phase imbalance) [rad]')
            ax2.grid(True)
            ax2.set_title('Phase Imbalance vs IF Amplitude')

            ax3.plot(amps, i_vals, 'go-')
            ax3.set_xlabel('IF Amplitude')
            ax3.set_ylabel('I DC Offset [V]')
            ax3.grid(True)
            ax3.set_title('I Channel DC Offset vs IF Amplitude')

            ax4.plot(amps, q_vals, 'mo-')
            ax4.set_xlabel('IF Amplitude')
            ax4.set_ylabel('Q DC Offset [V]')
            ax4.grid(True)
            ax4.set_title('Q Channel DC Offset vs IF Amplitude')

            plt.tight_layout()
            plt.savefig(f'calibration_summary_LO_{lo_freq / 1e9:.3f}GHz_{time.strftime("%Y-%m-%d-%H-%M-%S")}.png')
            plt.show()

    except Exception as e:
        print(f"Could not create summary plots: {e}")