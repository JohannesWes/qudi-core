import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opti
import pandas as pd
import pyrpl
import traceback
import logging
from datetime import datetime
import os

import configuration_IQ_calibration_redpitaya as config
from auto_mixer_tools_visa import RhodeSchwarzRTO6_RedPitaya
from microwave_control import Windfreak_MW_control
from logging_config import setup_logging, get_logger

# Setup logging
setup_logging(
    default_level=logging.WARNING,
    log_file=f'logs/iq_calibration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
logger = get_logger(__name__)

# Create an instance of the Parameters class from configuration
parameters = config.parameters()


class IQMixerCalibrator_RedPitaya():
    """
    A class to calibrate an IQ mixer using a Red Pitaya, oscilloscope, and microwave source.
    """

    def __init__(self, if_amplitude, lo_frequency=None, initial_params=None, path_result_folder=None): # Added path_result_folder
        """
        Initializes the IQMixerCalibrator with Red Pitaya.

        Parameters:
        -----------
        if_amplitude : float
            The IF signal amplitude (0 to 1) for the Red Pitaya output
        lo_frequency : float, optional
            The LO frequency in Hz. If None, uses the default from configuration
        initial_params : dict, optional
            Dictionary containing initial optimization parameters:
            {'I_offset': float, 'Q_offset': float, 'g': float, 'phi': float}
        path_result_folder : str, optional
            Path to the folder where results (like plots) should be saved.
        """
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.parameters = None
        self.oscilloscope = None
        self.mw_source = None
        self.pyrpl = None
        self.if_amplitude = if_amplitude
        self.lo_frequency = lo_frequency if lo_frequency is not None else parameters.f_LO_Hz # Ensure parameters is accessible or passed
        self.initial_params = initial_params or {'I_offset': 0.0, 'Q_offset': 0.0, 'g': 0.0, 'phi': 0.0}
        self.path_result_folder = path_result_folder if path_result_folder else "." # Store it

        self.logger.debug(
            f"Initialized calibrator with IF amplitude: {if_amplitude}, LO frequency: {self.lo_frequency}")
        self.logger.debug(f"Initial parameters: {self.initial_params}")
        if self.path_result_folder:
            self.logger.debug(f"Results will be saved in: {self.path_result_folder}")

    def connect_instruments(self):
        """Connects to and configures the required instruments."""
        try:
            # 0. Configuration
            self.parameters = parameters

            # Update LO frequency if specified
            if self.lo_frequency:
                self.parameters.f_LO_Hz = self.lo_frequency

            self.logger.info(f"Connecting to Red Pitaya at {self.parameters.redpitaya_hostname}")
            self.logger.info(f"IF frequency: {self.parameters.f_IF_Hz / 1e6:.3f} MHz")
            self.logger.info(f"LO frequency: {self.parameters.f_LO_Hz / 1e9:.3f} GHz")
            self.logger.info(f"IF amplitude: {self.if_amplitude}")

            # 1. Connect to Red Pitaya via pyrpl
            self.logger.debug("Connecting to Red Pitaya...")
            self.pyrpl = pyrpl.Pyrpl(
                hostname=self.parameters.redpitaya_hostname,
                config='iq_calibration_config',
                gui=False
            )
            self.logger.debug("Red Pitaya connected successfully")

            # 2. Initialize fgen3 module
            self.logger.debug("Initializing fgen3 module...")
            self.fgen3 = self.pyrpl.rp.fgen3

            # Configure fgen3 for IQ signal generation
            self.fgen3.setup(
                gen_enable=True,
                output_zero=False,
                overall_dc_offset_a=0.0,
                overall_dc_offset_b=0.0,
                enables=[True, False, False],
                frequencies=[self.parameters.f_IF_Hz, 0, 0],
                amplitudes_a=[self.if_amplitude, 0, 0],
                amplitudes_b=[self.if_amplitude, 0, 0],
                phase_offsets_a=[0.0, 0, 0],
                phase_offsets_b=[90.0, 0, 0]
            )
            self.logger.debug("fgen3 configured")

            # Route fgen3 outputs to physical outputs
            self.fgen3.output_to_dsp_enable_o = True
            self.pyrpl.rp.asg0.output_direct = "out1"
            self.pyrpl.rp.asg1.output_direct = "out2"
            self.logger.debug("Signal routing configured")

            # 3. Connect to Oscilloscope
            self.logger.info(f"Connecting to oscilloscope at {self.parameters.osci_address}")
            self.oscilloscope = RhodeSchwarzRTO6_RedPitaya(
                pyrpl_instance=self.pyrpl,
                address=self.parameters.osci_address
            )
            self.oscilloscope.method = 2
            self.oscilloscope.set_automatic_video_bandwidth(1)
            self.oscilloscope.set_automatic_bandwidth(0)
            self.oscilloscope.set_cont_off()
            self.logger.debug("Oscilloscope connected and configured")

            # 4. Connect to Microwave Source
            self.logger.info(f"Connecting to microwave source on {self.parameters.mw_usb_com_port}")
            self.mw_source = Windfreak_MW_control(self.parameters.mw_usb_com_port)
            self.mw_source.start_MW(self.parameters.f_LO_Hz, self.parameters.microwave_max_power)
            self.logger.debug("Microwave source connected and started")

            self.logger.info("All instruments connected successfully")

        except Exception as e:
            self.logger.error(f"Error connecting to instruments: {e}", exc_info=True)
            raise RuntimeError(f"Error connecting to instruments: {e}")

    def disconnect_instruments(self):
        """Disconnects from the instruments and closes connections."""
        self.logger.info("Disconnecting instruments...")
        try:
            if self.oscilloscope:
                try:
                    self.oscilloscope.set_cont_on()
                    del self.oscilloscope
                    self.oscilloscope = None
                except Exception as e:
                    self.logger.debug(f"Error disconnecting oscilloscope: {e}")

            if self.mw_source:
                try:
                    self.mw_source.end_MW()
                    self.mw_source = None
                except Exception as e:
                    self.logger.debug(f"Error disconnecting MW source: {e}")

            if self.pyrpl:
                try:
                    if hasattr(self, 'fgen3') and self.fgen3:
                        self.fgen3.gen_enable = False
                    # Don't call __exit__ - Pyrpl doesn't have it
                    # Just delete the reference
                    del self.pyrpl
                    self.pyrpl = None
                except Exception as e:
                    self.logger.debug(f"Error disconnecting Pyrpl: {e}")

            self.logger.info("All instruments disconnected successfully")

        except Exception as e:
            self.logger.error(f"Error during instrument disconnection: {e}", exc_info=True)

    def update_if_amplitude(self, if_amplitude):
        """Update the IF amplitude without reconnecting instruments."""
        self.if_amplitude = if_amplitude
        self.logger.info(f"Updating IF amplitude to: {if_amplitude}")

        # Update the Red Pitaya output amplitude
        if self.oscilloscope:
            self.oscilloscope.set_if_amplitude(if_amplitude)

    def _initial_simplex(self, center):
        """Creates the initial simplex for optimization."""
        simplex = np.zeros([3, 2])
        simplex[0, :] = [center[0] - 0.05, center[1] - 0.05]
        simplex[1, :] = [center[0] + 0.05, center[1]]
        simplex[2, :] = [center[0] - 0.05, center[1] + 0.05]
        self.logger.debug(f"Created initial simplex with center {center}")
        return simplex

    def _setup_oscilloscope_measurement(self):
        """Configures the oscilloscope for power measurement."""
        self.logger.debug("Setting up oscilloscope measurement...")
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
        self.logger.debug("Oscilloscope measurement setup complete")

    def _get_signal_power(self):
        """Gets the power of the signal at the desired IF frequency."""
        freq = self.parameters.f_LO_Hz - self.parameters.f_IF_Hz
        self.logger.debug(f"Getting signal power at {freq / 1e9:.3f} GHz")
        self.oscilloscope.set_center_freq(freq)
        if self.parameters.method == 2:  # Marker
            self.oscilloscope.set_marker_freq(1, freq)
        power = int(self.oscilloscope.get_amp())
        self.logger.debug(f"Signal power: {power} dBm")
        return power

    def optimize_LO_leakage(self, initial_simplex=None, x0=[0, 0]):
        """Optimizes I and Q DC offsets to minimize LO leakage."""
        self.logger.info("Starting LO leakage optimization...")
        self.oscilloscope.set_center_freq(self.parameters.f_LO_Hz)
        if self.parameters.method == 2:  # Marker
            self.oscilloscope.set_marker_freq(1, self.parameters.f_LO_Hz)

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

        elapsed = int(time.time() - start_time)
        self.logger.info(
            f"LO Leakage Results: Found a minimum of {int(res_leakage.fun)} dBm at "
            f"(I0, Q0) = ({res_leakage.x[0]:.5f}, {res_leakage.x[1]:.5f}) in {elapsed} seconds"
        )
        return res_leakage.x, res_leakage.fun

    def optimize_image(self, initial_simplex=None, x0=[0, 0]):
        """Optimizes g and phi to maximize image rejection."""
        self.logger.info("Starting image rejection optimization...")
        image_freq = self.parameters.f_LO_Hz + self.parameters.f_IF_Hz
        self.oscilloscope.set_center_freq(image_freq)
        if self.parameters.method == 2:  # Marker
            self.oscilloscope.set_marker_freq(1, image_freq)

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

        elapsed = int(time.time() - start_time)
        self.logger.info(
            f"Image Rejection Results: Found a minimum of {int(res_image.fun)} dBm at "
            f"(g, phi) = ({res_image.x[0]:.5f}, {res_image.x[1]:.5f}) in {elapsed} seconds"
        )

        return res_image.x, res_image.fun

    def _perform_spectrum_sweep(self, span_multiplier=3):
        """Performs a wideband spectrum sweep and plots the result."""
        self.logger.debug("Performing spectrum sweep...")
        self.oscilloscope.set_bandwidth(self.parameters.sweepBW)
        # Increase points for better resolution with larger span
        num_points = self.parameters.fullNumPoints * span_multiplier
        self.oscilloscope.set_sweep_points(int(num_points))
        self.oscilloscope.set_center_freq(self.parameters.f_LO_Hz)
        # Make the span larger
        span_wide = self.parameters.fullSpan * span_multiplier
        self.oscilloscope.set_span(span_wide)
        self.oscilloscope.get_single_trigger()
        freq_vec, amp = self.oscilloscope.get_full_trace()
        # Ensure numpy arrays
        freq_vec = np.array(freq_vec)
        amp = np.array(amp)
        self.logger.debug(f"Spectrum sweep complete, got {len(freq_vec)} points")
        return freq_vec, amp

    def _calculate_sfdr(self, freq_vec, amp_dbm, bandwidth_hz=None):
        """
        Calculate the Spurious Free Dynamic Range (SFDR) around the target frequency.

        Parameters:
        -----------
        freq_vec : array
            Frequency vector in Hz
        amp_dbm : array
            Amplitude vector in dBm
        bandwidth_hz : float, optional
            Bandwidth around target frequency to search for spurs.
            If None, uses 8*IF frequency to capture up to 3rd harmonics

        Returns:
        --------
        sfdr_db : float
            SFDR in dB
        target_freq : float
            Target frequency in Hz
        target_power : float
            Target signal power in dBm
        max_spur_freq : float
            Frequency of the maximum spur in Hz
        max_spur_power : float
            Power of the maximum spur in dBm
        """
        # Ensure arrays are numpy arrays
        freq_vec = np.array(freq_vec)
        amp_dbm = np.array(amp_dbm)

        target_freq = self.parameters.f_LO_Hz - self.parameters.f_IF_Hz

        # Default bandwidth: wide enough to capture harmonics up to 3*IF
        if bandwidth_hz is None:
            bandwidth_hz = 8 * self.parameters.f_IF_Hz
            self.logger.debug(f"Using SFDR bandwidth of {bandwidth_hz / 1e6:.1f} MHz (8 * IF frequency)")

        # Find indices within the bandwidth of interest
        freq_min = target_freq - bandwidth_hz / 2
        freq_max = target_freq + bandwidth_hz / 2
        band_mask = (freq_vec >= freq_min) & (freq_vec <= freq_max)

        freq_band = freq_vec[band_mask]
        amp_band = amp_dbm[band_mask]

        if len(freq_band) == 0:
            self.logger.error("No frequency points found within specified bandwidth")
            return np.nan, target_freq, np.nan, np.nan, np.nan

        # Find the target signal (should be the peak closest to target_freq)
        target_idx = np.argmin(np.abs(freq_band - target_freq))

        # Search for the actual peak around the expected target frequency (within ±5 MHz)
        search_range = 5e6  # 5 MHz
        search_mask = np.abs(freq_band - target_freq) <= search_range
        if np.any(search_mask):
            # Find the maximum within the search range
            search_indices = np.where(search_mask)[0]
            max_idx_in_search = np.argmax(amp_band[search_mask])
            target_idx = search_indices[max_idx_in_search]

        target_power = amp_band[target_idx]
        actual_target_freq = freq_band[target_idx]

        # Create a mask to exclude the target signal region (±2 MHz around the peak)
        target_exclusion = self.parameters.f_IF_Hz/2
        spur_mask = np.abs(freq_band - actual_target_freq) > target_exclusion

        if not np.any(spur_mask):
            self.logger.warning("No spurious signals found outside target exclusion zone")
            return np.inf, actual_target_freq, target_power, np.nan, -np.inf

        # Find the maximum spur
        spur_amps = amp_band[spur_mask]
        spur_freqs = freq_band[spur_mask]
        max_spur_idx = np.argmax(spur_amps)
        max_spur_power = spur_amps[max_spur_idx]
        max_spur_freq = spur_freqs[max_spur_idx]

        # Debug: Find top 5 spurs
        if len(spur_amps) > 5 and self.logger.isEnabledFor(logging.DEBUG):
            top_spur_indices = np.argsort(spur_amps)[-5:][::-1]
            self.logger.debug("Top 5 spurs found:")
            for idx in top_spur_indices:
                self.logger.debug(f"  {spur_freqs[idx] / 1e9:.4f} GHz: {spur_amps[idx]:.1f} dBm")

        # Calculate SFDR
        sfdr_db = target_power - max_spur_power

        # Identify which harmonic the spur is closest to
        lo_freq = self.parameters.f_LO_Hz
        if_freq = self.parameters.f_IF_Hz
        harmonic_freqs = {
            'LO': lo_freq,
            'LO+IF': lo_freq + if_freq,
            'LO+2IF': lo_freq + 2 * if_freq,
            'LO+3IF': lo_freq + 3 * if_freq,
            'LO-2IF': lo_freq - 2 * if_freq,
            'LO-3IF': lo_freq - 3 * if_freq,
        }

        closest_harmonic = None
        min_distance = float('inf')
        for name, freq in harmonic_freqs.items():
            distance = abs(max_spur_freq - freq)
            if distance < min_distance and distance < 5e6:  # Within 5 MHz
                min_distance = distance
                closest_harmonic = name

        self.logger.info(f"SFDR Calculation:")
        self.logger.info(f"  Bandwidth: ±{bandwidth_hz / 2e6:.1f} MHz around target")
        self.logger.info(f"  Target frequency: {actual_target_freq / 1e9:.4f} GHz")
        self.logger.info(f"  Target power: {target_power:.1f} dBm")
        self.logger.info(
            f"  Max spur frequency: {max_spur_freq / 1e9:.4f} GHz{f' (near {closest_harmonic})' if closest_harmonic else ''}")
        self.logger.info(f"  Max spur power: {max_spur_power:.1f} dBm")
        self.logger.info(f"  SFDR: {sfdr_db:.1f} dB")

        return sfdr_db, actual_target_freq, target_power, max_spur_freq, max_spur_power

    def _plot_spectrum(self, freq_vec, amp_before, amp_after):
        """Plots the spectrum before and after calibration."""
        self.logger.debug("Creating spectrum plot...")

        # Ensure numpy arrays
        freq_vec = np.array(freq_vec)
        amp_before = np.array(amp_before)
        amp_after = np.array(amp_after)

        # Calculate SFDR for the "after" spectrum
        sfdr_db, target_freq, target_power, spur_freq, spur_power = self._calculate_sfdr(freq_vec, amp_after)

        # Calculate the actual SFDR bandwidth used
        sfdr_bandwidth_mhz = (12.5 * self.parameters.f_IF_Hz) / 1e6

        fig_spectrum, ax_spectrum = plt.subplots(1, 1, figsize=(12, 8), num="Full Spectrum")
        ax_spectrum.set_xlabel("Frequency (GHz)")
        ax_spectrum.set_ylabel("Amplitude (dBm)")
        ax_spectrum.plot(freq_vec / 1e9, amp_before, alpha=0.5, label="Before")
        ax_spectrum.plot(freq_vec / 1e9, amp_after, alpha=0.5, label="After")

        # Mark the target frequency and maximum spur
        if not np.isnan(target_power):
            ax_spectrum.axvline(target_freq / 1e9, color='green', linestyle='--', alpha=0.7,
                        label=f'Target: {target_freq / 1e9:.3f} GHz')


            ax_spectrum.plot(target_freq / 1e9, target_power, 'go', markersize=10, label=f'Target: {target_power:.1f} dBm')

            # Show SFDR measurement bandwidth
            bw_half = sfdr_bandwidth_mhz/2 * 1e6 / 1e9  # Half of the SFDR bandwidth in GHz
            ax_spectrum.axvspan((target_freq / 1e9) - bw_half, (target_freq / 1e9) + bw_half,
                        alpha=0.1, color='gray', label=f'SFDR BW: ±{bw_half * 1000:.1f} MHz')

        if not np.isnan(spur_power):
            ax_spectrum.plot(spur_freq / 1e9, spur_power, 'ro', markersize=8, label=f'Max Spur: {spur_power:.1f} dBm')
            # Draw a line from spur to target to visualize SFDR
            if not np.isnan(target_power):
                ax_spectrum.plot([target_freq / 1e9, spur_freq / 1e9], [target_power, spur_power], 'r--', alpha=0.5)

        # Mark expected spur locations
        lo_freq_ghz = self.parameters.f_LO_Hz / 1e9
        if_freq_ghz = self.parameters.f_IF_Hz / 1e9
        expected_spurs = [
            (lo_freq_ghz, "LO"),
            (lo_freq_ghz + if_freq_ghz, "LO+IF"),
            (lo_freq_ghz + 2 * if_freq_ghz, "LO+2IF"),
            (lo_freq_ghz + 3 * if_freq_ghz, "LO+3IF"),
            (lo_freq_ghz - 2 * if_freq_ghz, "LO-2IF"),
            (lo_freq_ghz - 3 * if_freq_ghz, "LO-3IF"),
        ]

        for freq, label in expected_spurs:
            if freq_vec.min() <= freq * 1e9 <= freq_vec.max():
                ax_spectrum.axvline(freq, color='orange', linestyle=':', alpha=0.3)

        ax_spectrum.set_title(
            f"IQ-Calibration. Target frequency: {(self.parameters.f_LO_Hz - self.parameters.f_IF_Hz) / 1e9:.3f} GHz\n"
            f"SFDR: {sfdr_db:.1f} dB (in ±{sfdr_bandwidth_mhz / 2:.1f} MHz bandwidth)"
        )
        ax_spectrum.legend()
        ax_spectrum.grid(True, alpha=0.3)
        fig_spectrum.tight_layout()
        # save as pdf, include if_amp and lo_freq in filename
        # Use self.path_result_folder
        plot_filename = os.path.join(self.path_result_folder, f"spectrum_if_{self.if_amplitude:.3f}_lo_{self.parameters.f_LO_Hz / 1e9:.3f}_GHz_IF_{self.parameters.f_IF_Hz / 1e6:.3f}MHz.pdf")
        fig_spectrum.savefig(plot_filename)
        plt.close(fig_spectrum)
        self.logger.debug(f"Spectrum plot saved to {plot_filename}")

    def calibrate(self):
        """
        Performs the full IQ mixer calibration routine.

        Returns:
        --------
        best_leakage : array
            Best DC offset values [I_offset, Q_offset]
        best_image : array
            Best IQ correction values [g, phi]
        optimization_results : dict
            Dictionary containing all optimization results and parameters
        """
        self.logger.info("Starting IQ mixer calibration...")

        if not (self.pyrpl and self.oscilloscope and self.mw_source):
            raise RuntimeError("Instruments not connected! Call 'connect_instruments()' first.")

        self.mw_source.start_MW(self.parameters.f_LO_Hz, self.parameters.microwave_max_power)

        self._setup_oscilloscope_measurement()
        signal_power = self._get_signal_power()
        self.logger.info(f"Initial signal power: {signal_power} dBm")

        if self.parameters.bDoSweeps:
            freq_vec, amp_before = self._perform_spectrum_sweep()

        # Initialize with provided initial parameters
        result_leakage = [self.initial_params['I_offset'], self.initial_params['Q_offset']]
        result_image = [self.initial_params['g'], self.initial_params['phi']]

        # Apply initial parameters before starting optimization
        self.logger.info(f"Starting with initial parameters: "
                         f"DC offsets ({result_leakage[0]:.5f}, {result_leakage[1]:.5f}), "
                         f"IQ corrections ({result_image[0]:.5f}, {result_image[1]:.5f})")

        # Set initial DC offsets
        self.oscilloscope.get_leakage(result_leakage[0], result_leakage[1])
        # Set initial IQ corrections
        self.oscilloscope.get_image(result_image[0], result_image[1])

        simplex_leakage = self._initial_simplex(result_leakage)
        simplex_image = self._initial_simplex(result_image)

        result_image_array = np.zeros([self.parameters.optimization_repititions, 2])
        result_leakage_array = np.zeros([self.parameters.optimization_repititions, 2])

        fun_image_array = np.zeros(self.parameters.optimization_repititions)
        fun_leakage_array = np.zeros(self.parameters.optimization_repititions)

        for i in range(self.parameters.optimization_repititions):
            self.logger.info(f"Optimization Step {i + 1}/{self.parameters.optimization_repititions}")

            # Optimize LO leakage starting from the current best values
            result_leakage, fun_leakage = self.optimize_LO_leakage(
                initial_simplex=simplex_leakage, x0=result_leakage
            )

            # Update simplex around new optimum
            simplex_leakage = self._initial_simplex(result_leakage)
            result_leakage_array[i, :] = result_leakage
            fun_leakage_array[i] = fun_leakage

            # Optimize image rejection starting from the current best values
            # Note: The LO leakage optimization may have affected the image rejection
            result_image, fun_image = self.optimize_image(
                initial_simplex=simplex_image, x0=result_image
            )

            # Update simplex around new optimum
            simplex_image = self._initial_simplex(result_image)
            result_image_array[i, :] = result_image
            fun_image_array[i] = fun_image

            self.logger.info(f"Step {i + 1} results: "
                             f"LO leakage = {fun_leakage:.1f} dBm, "
                             f"Image = {fun_image:.1f} dBm")

        # Apply best calibration values before final measurements
        reward = np.abs(fun_image_array + fun_leakage_array)
        best_index = np.argmax(reward)
        best_image, best_leakage = result_image_array[best_index], result_leakage_array[best_index]

        # Apply the best calibration values
        self.oscilloscope.get_leakage(best_leakage[0], best_leakage[1])
        self.oscilloscope.get_image(best_image[0], best_image[1])

        # Measure SFDR with optimal settings
        sfdr_db = np.nan
        if self.parameters.bDoSweeps:
            _, amp_after = self._perform_spectrum_sweep()
            #self._plot_spectrum(freq_vec, amp_before, amp_after)
            # Calculate SFDR from the final spectrum
            sfdr_db, _, _, _, _ = self._calculate_sfdr(freq_vec, amp_after)
        else:
            # If not doing sweeps, still measure SFDR
            self.logger.info("Measuring SFDR without plotting...")
            # Perform a spectrum sweep just for SFDR measurement
            original_span = self.parameters.fullSpan
            # Use wider span to capture all harmonics
            sfdr_span = 10 * self.parameters.f_IF_Hz  # Wide enough for all harmonics
            self.oscilloscope.set_bandwidth(self.parameters.sweepBW)
            self.oscilloscope.set_sweep_points(4096)  # High resolution for accurate measurement
            self.oscilloscope.set_center_freq(self.parameters.f_LO_Hz - self.parameters.f_IF_Hz)
            self.oscilloscope.set_span(sfdr_span)
            self.logger.info(f"Using {sfdr_span / 1e6:.1f} MHz span for SFDR measurement")
            self.oscilloscope.get_single_trigger()
            freq_vec_sfdr, amp_sfdr = self.oscilloscope.get_full_trace()
            # Ensure numpy arrays
            freq_vec_sfdr = np.array(freq_vec_sfdr)
            amp_sfdr = np.array(amp_sfdr)
            sfdr_db, _, _, _, _ = self._calculate_sfdr(freq_vec_sfdr, amp_sfdr)
            # Restore original settings
            self.oscilloscope.set_span(original_span)

        self.logger.info(f"Best calibration values:")
        self.logger.info(f"DC offsets (I, Q): {best_leakage}")
        self.logger.info(f"IQ corrections (g, phi): {best_image}")
        self.logger.info(f"SFDR: {sfdr_db:.1f} dB")

        # Return optimization results dictionary for use as initial params
        optimization_results = {
            'I_offset': best_leakage[0],
            'Q_offset': best_leakage[1],
            'g': best_image[0],
            'phi': best_image[1],
            'sfdr_db': sfdr_db,
            'all_leakage_results': result_leakage_array,
            'all_image_results': result_image_array,
            'all_leakage_powers': fun_leakage_array,
            'all_image_powers': fun_image_array
        }

        return best_leakage, best_image, optimization_results


if __name__ == "__main__":
    # Configure logging level from command line if needed
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level', default='WARNING',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set the logging level')
    args = parser.parse_args()

    # Update logging level if specified
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("Starting IQ Mixer Calibration Script")

    # --- START OF MODIFICATIONS ---

    # Get the original IF frequency from configuration to use as a base
    original_f_IF_Hz = parameters.f_IF_Hz
    if_offset_hz = 2.158e6  # 2.158 MHz

    base_if_frequencies_to_test = [
        original_f_IF_Hz,
        original_f_IF_Hz + if_offset_hz,
        original_f_IF_Hz - if_offset_hz
    ]

    # Define IF amplitudes to calibrate (0 to 1 scale for Red Pitaya)
    if_amplitudes = np.linspace(0.01, 0.3, 20)

    # Define LO frequencies to calibrate (in Hz)
    lo_frequencies = np.linspace(2.87e9-300e6, 2.87e9+300e6, 60)

    # Create a main output directory based on current time
    current_time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    main_output_dir = f"calibration_results/{current_time_str}"
    os.makedirs(main_output_dir, exist_ok=True)
    logger.info(f"Main output directory: {main_output_dir}")

    all_results_across_all_IFs = [] # To store all results if a grand summary is needed

    for base_f_IF_Hz in base_if_frequencies_to_test:
        logger.info(f"===== Processing for base IF Frequency: {base_f_IF_Hz / 1e6:.3f} MHz =====")
        parameters.f_IF_Hz = base_f_IF_Hz # Critically update the parameter for the calibrator

        # Create a subdirectory for this specific base IF frequency's results
        if_specific_output_dir = os.path.join(main_output_dir, f"IF_{base_f_IF_Hz / 1e6:.3f}MHz")
        os.makedirs(if_specific_output_dir, exist_ok=True)
        logger.info(f"Results for this IF will be saved in: {if_specific_output_dir}")

        # Initialize storage for results for this specific base_f_IF_Hz
        results_list_for_current_if = []
        last_best_params = None # Reset for each new base IF frequency

        # Calibrate for each LO frequency and IF amplitude combination
        for lo_freq in lo_frequencies:
            logger.info(f"Starting calibration for LO frequency: {lo_freq / 1e9:.3f} GHz (Base IF: {base_f_IF_Hz / 1e6:.3f} MHz)")

            g_array, phi_array = np.zeros(len(if_amplitudes)), np.zeros(len(if_amplitudes))
            I_array, Q_array = np.zeros(len(if_amplitudes)), np.zeros(len(if_amplitudes))
            sfdr_array = np.zeros(len(if_amplitudes))

            initial_params_for_lo = last_best_params if last_best_params else None
            last_if_params = initial_params_for_lo

            # Create calibrator and connect instruments ONCE per LO frequency
            # Pass the IF-specific output directory for plots
            calibrator = IQMixerCalibrator_RedPitaya(
                if_amplitude=if_amplitudes[0], # Initial amplitude
                lo_frequency=lo_freq,
                initial_params=initial_params_for_lo,
                path_result_folder=if_specific_output_dir # Pass the specific folder
            )
            # The calibrator will use the updated parameters.f_IF_Hz

            try:
                calibrator.connect_instruments() # This will now use the updated parameters.f_IF_Hz

                for amp_index, if_amplitude_val in enumerate(if_amplitudes):
                    exception_counter = 1
                    while True:
                        try:
                            time.sleep(1)
                            logger.info(f"Calibrating for IF amplitude: {if_amplitude_val:.3f}, LO: {lo_freq / 1e9:.3f} GHz, Base IF: {base_f_IF_Hz / 1e6:.3f} MHz")

                            calibrator.update_if_amplitude(if_amplitude_val)
                            if last_if_params:
                                calibrator.initial_params = last_if_params

                            best_leakage, best_image, optimization_results = calibrator.calibrate()

                            I_array[amp_index], Q_array[amp_index] = best_leakage
                            g_array[amp_index], phi_array[amp_index] = best_image
                            sfdr_array[amp_index] = optimization_results['sfdr_db']

                            logger.info(f"CALIBRATION SUCCESSFUL (Base IF: {base_f_IF_Hz / 1e6:.3f} MHz):")
                            logger.info(f"  IF amplitude: {if_amplitude_val:.3f}")
                            logger.info(f"  LO frequency: {lo_freq / 1e9:.3f} GHz")
                            logger.info(f"  g: {best_image[0]:.5f}, phi: {best_image[1]:.5f}")
                            logger.info(f"  I: {best_leakage[0]:.5f}, Q: {best_leakage[1]:.5f}")
                            logger.info(f"  SFDR: {optimization_results['sfdr_db']:.1f} dB")

                            current_result = {
                                'base_if_frequency_mhz': base_f_IF_Hz / 1e6, # Add this new info
                                'lo_frequency_ghz': lo_freq / 1e9,
                                'if_amplitude': if_amplitude_val,
                                'g': best_image[0],
                                'phi': best_image[1],
                                'I_offset': best_leakage[0],
                                'Q_offset': best_leakage[1],
                                'sfdr_db': optimization_results['sfdr_db']
                            }
                            results_list_for_current_if.append(current_result)
                            all_results_across_all_IFs.append(current_result)


                            last_if_params = {
                                'I_offset': best_leakage[0],
                                'Q_offset': best_leakage[1],
                                'g': best_image[0],
                                'phi': best_image[1]
                            }
                            break
                        except Exception as e:
                            logger.error(
                                f"Calibration failed for IF amplitude {if_amplitude_val:.3f} (attempt {exception_counter}, Base IF: {base_f_IF_Hz / 1e6:.3f} MHz): {e}",
                                exc_info=True)
                            exception_counter += 1
                            if exception_counter > 3:
                                logger.error(f"Skipping IF amplitude {if_amplitude_val:.3f} after 3 failed attempts (Base IF: {base_f_IF_Hz / 1e6:.3f} MHz)")
                                I_array[amp_index], Q_array[amp_index] = np.nan, np.nan
                                g_array[amp_index], phi_array[amp_index] = np.nan, np.nan
                                sfdr_array[amp_index] = np.nan
                                break
                            time.sleep(5)
                if last_if_params: # Update best params for next LO freq within the same base IF
                    last_best_params = last_if_params

            finally:
                if calibrator: # Ensure calibrator was instantiated
                    calibrator.disconnect_instruments()

            # Save calibration data for this LO frequency and current base IF
            logger.info(f"Calibration complete for LO frequency {lo_freq / 1e9:.3f} GHz (Base IF: {base_f_IF_Hz / 1e6:.3f} MHz)")

            # Adjust filenames to include base IF frequency
            if_label = f"IF_{base_f_IF_Hz / 1e6:.3f}MHz"
            # np.save(os.path.join(if_specific_output_dir, f"g_array_LO_{lo_freq / 1e9:.3f}GHz_{if_label}.npy"), g_array)
            # np.save(os.path.join(if_specific_output_dir, f"phi_array_LO_{lo_freq / 1e9:.3f}GHz_{if_label}.npy"), phi_array)
            # np.save(os.path.join(if_specific_output_dir, f"I_array_LO_{lo_freq / 1e9:.3f}GHz_{if_label}.npy"), I_array)
            # np.save(os.path.join(if_specific_output_dir, f"Q_array_LO_{lo_freq / 1e9:.3f}GHz_{if_label}.npy"), Q_array)
            # np.save(os.path.join(if_specific_output_dir, f"sfdr_array_LO_{lo_freq / 1e9:.3f}GHz_{if_label}.npy"), sfdr_array)
            # np.save(os.path.join(if_specific_output_dir, f"if_amplitudes_LO_{lo_freq / 1e9:.3f}GHz_{if_label}.npy"), if_amplitudes)

            calibration_df_lo = pd.DataFrame({
                "if_amplitude": if_amplitudes,
                "g": g_array,
                "phi": phi_array,
                "I_offset": I_array,
                "Q_offset": Q_array,
                "sfdr_db": sfdr_array,
            })

            csv_filename_lo = os.path.join(if_specific_output_dir, f"calibration_redpitaya_LO_{lo_freq / 1e9:.3f}GHz_{if_label}.csv")
            calibration_df_lo.to_csv(csv_filename_lo, sep="\t", index=False)
            logger.info(f"Saved calibration data to {csv_filename_lo}")

        # Save all results for the current base_f_IF_Hz to its specific CSV file
        if results_list_for_current_if:
            all_results_df_current_if = pd.DataFrame(results_list_for_current_if)
            csv_filename_current_if = os.path.join(if_specific_output_dir, f"calibration_redpitaya_all_results_{if_label}.csv")
            all_results_df_current_if.to_csv(csv_filename_current_if, sep="\t", index=False)
            logger.info(f"All calibration results for {if_label} saved to {csv_filename_current_if}")

    # Optionally, save a grand summary CSV of all results from all IFs
    if all_results_across_all_IFs:
        grand_summary_df = pd.DataFrame(all_results_across_all_IFs)
        grand_csv_filename = os.path.join(main_output_dir, "calibration_redpitaya_GRAND_SUMMARY_all_IFs.csv")
        grand_summary_df.to_csv(grand_csv_filename, sep="\t", index=False)
        logger.info(f"Grand summary of all calibration results saved to {grand_csv_filename}")

    logger.info("IQ Mixer Calibration Script completed")