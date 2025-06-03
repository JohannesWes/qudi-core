# redpitaya_if_source.py
import numpy as np
import pyrpl
from typing import Tuple, Optional, Dict, List
from if_source_base import IFSourceBase, IQDeviceConfig, IQComponentConfig
import logging
import pandas as pd
from scipy.interpolate import interp2d, griddata


class RedPitayaIFSource(IFSourceBase):
    """Red Pitaya implementation of IF source using PyRPL with multi-frequency support."""

    def __init__(self, hostname: str, port: int = 2222, name: str = "RedPitaya"):
        super().__init__(name)
        self.hostname = hostname
        self.port = port
        self.pyrpl = None
        self.fgen3 = None
        self._current_config = None
        self._calibration_data = {}  # Dict[frequency, DataFrame]
        self._max_components = 3  # fgen3 supports 3 components

    def connect(self, **kwargs) -> None:
        """Connect to the Red Pitaya."""
        try:
            self.logger.info(f"Connecting to Red Pitaya at {self.hostname}:{self.port}")

            config_name = kwargs.get('config_name', 'iq_calibration_config')
            gui = kwargs.get('gui', False)

            self.pyrpl = pyrpl.Pyrpl(
                hostname=self.hostname,
                config=config_name,
                gui=gui
            )

            # Initialize fgen3 module
            self.fgen3 = self.pyrpl.rp.fgen3

            # Configure signal routing
            self.fgen3.output_to_dsp_enable_o = True
            self.pyrpl.rp.asg0.output_direct = "out1"
            self.pyrpl.rp.asg1.output_direct = "out2"

            self._is_connected = True
            self.logger.info("Red Pitaya connected successfully")

        except Exception as e:
            self.logger.error(f"Failed to connect to Red Pitaya: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from the Red Pitaya."""
        try:
            if self.fgen3:
                self.fgen3.gen_enable = False
                self.fgen3.output_zero = True
                # Disable all components
                for i in range(self._max_components):
                    setattr(self.fgen3, f'enable{i}', False)
                self.logger.debug("Red Pitaya output disabled")

            if self.pyrpl:
                # PyRPL doesn't have a proper disconnect method
                del self.pyrpl
                self.pyrpl = None

            self._is_connected = False
            self.logger.info("Red Pitaya disconnected")

        except Exception as e:
            self.logger.error(f"Error disconnecting Red Pitaya: {e}")

    def load_calibration_data(self, frequency: float, path: str) -> pd.DataFrame:
        """Load IQ calibration settings for a specific frequency from a CSV file."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        try:
            calibration_df = pd.read_csv(path, sep="\t", index_col=0)
            self._calibration_data[frequency] = calibration_df
            self.logger.info(f"IQ calibration data for {frequency / 1e6:.3f} MHz loaded from {path}")
            return calibration_df
        except Exception as e:
            self.logger.error(f"Failed to load IQ calibration data: {e}")
            raise RuntimeError(f"Could not load IQ calibration data from {path}") from e

    def _interpolate_calibration_parameters(self, frequency: float, lo_frequency_ghz: float,
                                            if_amplitude: float) -> Tuple[float, float, float, float]:
        """Interpolate calibration parameters for a given frequency and amplitude."""
        if frequency not in self._calibration_data:
            raise RuntimeError(f"Calibration data for {frequency / 1e6:.3f} MHz not loaded")

        calibration_data = self._calibration_data[frequency]

        # Extract the parameter ranges from calibration data
        lo_min = calibration_data['lo_frequency_ghz'].min()
        lo_max = calibration_data['lo_frequency_ghz'].max()
        if_min = calibration_data['if_amplitude'].min()
        if_max = calibration_data['if_amplitude'].max()

        # Check if requested parameters are within bounds
        if not (lo_min <= lo_frequency_ghz <= lo_max):
            raise ValueError(
                f"lo_frequency_ghz ({lo_frequency_ghz}) is outside the calibrated range [{lo_min}, {lo_max}]")

        if not (if_min <= if_amplitude <= if_max):
            raise ValueError(f"if_amplitude ({if_amplitude}) is outside the calibrated range [{if_min}, {if_max}]")

        # Log the interpolation request
        self.logger.debug(
            f"Interpolating calibration for frequency={frequency / 1e6:.3f} MHz, "
            f"lo_frequency_ghz={lo_frequency_ghz}, if_amplitude={if_amplitude}")

        # Prepare the input points for interpolation
        points = calibration_data[['lo_frequency_ghz', 'if_amplitude']].values

        # Parameters to interpolate
        parameters = ['g', 'phi', 'I_offset', 'Q_offset']
        interpolated_values = {}

        # Perform interpolation for each parameter
        for param in parameters:
            values = calibration_data[param].values

            try:
                # Using griddata for more robust interpolation
                try:
                    interpolated_value = griddata(points, values,
                                                  (lo_frequency_ghz, if_amplitude),
                                                  method='cubic')
                except:
                    # Fallback to linear interpolation if cubic fails
                    interpolated_value = griddata(points, values,
                                                  (lo_frequency_ghz, if_amplitude),
                                                  method='linear')

                # Check for NaN results
                if np.isnan(interpolated_value):
                    self.logger.warning(f"Interpolation resulted in NaN for {param}, using nearest neighbor")
                    interpolated_value = griddata(points, values,
                                                  (lo_frequency_ghz, if_amplitude),
                                                  method='nearest')

                interpolated_values[param] = float(interpolated_value)

            except Exception as e:
                self.logger.error(f"Failed to interpolate {param}: {e}")
                raise RuntimeError(f"Could not interpolate {param}") from e

        return (interpolated_values['g'], interpolated_values['phi'],
                interpolated_values['I_offset'], interpolated_values['Q_offset'])

    def configure_signal(self, config: IQDeviceConfig) -> None:
        """Configure the IQ signal parameters for all components."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        if config.num_components > self._max_components:
            raise ValueError(f"Too many components ({config.num_components}). Maximum: {self._max_components}")

        self.logger.debug(f"Configuring signal with {config.num_components} components")

        # First, ensure the generator is properly initialized
        self.fgen3.gen_enable = True
        self.fgen3.output_zero = False

        # Disable all components first
        for i in range(self._max_components):
            setattr(self.fgen3, f'enable{i}', False)

        # Configure active components
        for i, component in enumerate(config.components):
            if component.enabled:
                self.configure_component(i, component)

        # Set DC offsets (averaged across all active components)
        self.set_dc_offsets(config.dc_offset_i, config.dc_offset_q)

        self._current_config = config
        self.logger.debug("Signal configured")

    def configure_component(self, component_index: int, config: IQComponentConfig) -> None:
        """Configure a single frequency component."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        if component_index >= self._max_components:
            raise ValueError(f"Component index {component_index} exceeds maximum ({self._max_components - 1})")

        # Set frequency
        setattr(self.fgen3, f'frequency{component_index}', config.frequency)

        # Set amplitudes
        setattr(self.fgen3, f'amplitude_a{component_index}', config.amplitude_i)
        setattr(self.fgen3, f'amplitude_b{component_index}', config.amplitude_q)

        # Set phases
        setattr(self.fgen3, f'phase_offset_a{component_index}', config.phase_i)
        setattr(self.fgen3, f'phase_offset_b{component_index}', config.phase_q)

        # Enable the component
        setattr(self.fgen3, f'enable{component_index}', config.enabled)

        self.logger.debug(f"Component {component_index} configured: "
                          f"freq={config.frequency / 1e6:.3f} MHz, "
                          f"amp_i={config.amplitude_i:.3f}, amp_q={config.amplitude_q:.3f}")

    def enable_output(self, enable: bool = True) -> None:
        """Enable or disable the output."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        self.fgen3.gen_enable = enable
        self.fgen3.output_zero = not enable
        self.logger.debug(f"Output {'enabled' if enable else 'disabled'}")

    def set_dc_offsets(self, i_offset: float, q_offset: float) -> None:
        """Set DC offsets for I and Q channels."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        self.fgen3.overall_dc_offset_a = i_offset
        self.fgen3.overall_dc_offset_b = q_offset

        if self._current_config:
            self._current_config.dc_offset_i = i_offset
            self._current_config.dc_offset_q = q_offset

        self.logger.debug(f"DC offsets set: I={i_offset:.5f}, Q={q_offset:.5f}")

    def set_iq_correction(self, component_index: int, if_amplitude: float,
                          gain_imbalance: float, phase_imbalance: float) -> Tuple[float, float, float, float]:
        """Apply IQ imbalance correction for a specific component."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        if component_index >= self._max_components:
            raise ValueError(f"Component index {component_index} exceeds maximum ({self._max_components - 1})")

        # Calculate corrected amplitudes
        amp_i = if_amplitude * (1 + gain_imbalance)
        amp_q = if_amplitude * (1 - gain_imbalance)

        # Calculate corrected phases
        phase_i = 0.0
        phase_q = 90.0 + np.degrees(phase_imbalance)

        # Apply corrections
        setattr(self.fgen3, f'amplitude_a{component_index}', amp_i)
        setattr(self.fgen3, f'amplitude_b{component_index}', amp_q)
        setattr(self.fgen3, f'phase_offset_a{component_index}', phase_i)
        setattr(self.fgen3, f'phase_offset_b{component_index}', phase_q)
        setattr(self.fgen3, f'enable{component_index}', True)

        if self._current_config and component_index < len(self._current_config.components):
            comp = self._current_config.components[component_index]
            comp.amplitude_i = amp_i
            comp.amplitude_q = amp_q
            comp.phase_i = phase_i
            comp.phase_q = phase_q

        self.logger.debug(f"IQ correction applied to component {component_index}: "
                          f"g={gain_imbalance:.5f}, phi={phase_imbalance:.5f}")

        return amp_i, amp_q, phase_i, phase_q

    # Backward compatibility methods
    def set_amplitude(self, amplitude: float) -> None:
        """Set the IF signal amplitude (0 to 1) without correction. For backward compatibility."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        # Set amplitude on component 0 for backward compatibility
        self.fgen3.amplitude_a0 = amplitude
        self.fgen3.amplitude_b0 = amplitude
        self.fgen3.enable0 = True

        self.logger.debug(f"Amplitude set to {amplitude:.3f} on component 0")

    def set_corrected_amplitude(self, amplitude: float, f_lo: float = 21.158e6) -> None:
        """Set the IF signal amplitude using calibration. For backward compatibility."""
        # Use the convenience method with a single frequency
        # Assume the frequency is already set on component 0
        if hasattr(self.fgen3, 'frequency0') and self.fgen3.frequency0:
            frequency = self.fgen3.frequency0
        else:
            # Default to a common IF frequency if not set
            frequency = 21.580e6

        self.set_multi_frequency_signal([frequency], [amplitude], f_lo)

    def get_current_config(self) -> IQDeviceConfig:
        """Get the current device configuration."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        if not self._current_config:
            # Read current configuration from device
            components = []
            for i in range(self._max_components):
                if getattr(self.fgen3, f'enable{i}', False):
                    components.append(IQComponentConfig(
                        frequency=getattr(self.fgen3, f'frequency{i}'),
                        amplitude=(getattr(self.fgen3, f'amplitude_a{i}') +
                                   getattr(self.fgen3, f'amplitude_b{i}')) / 2,
                        amplitude_i=getattr(self.fgen3, f'amplitude_a{i}'),
                        amplitude_q=getattr(self.fgen3, f'amplitude_b{i}'),
                        phase_i=getattr(self.fgen3, f'phase_offset_a{i}'),
                        phase_q=getattr(self.fgen3, f'phase_offset_b{i}'),
                        enabled=True
                    ))

            self._current_config = IQDeviceConfig(
                components=components,
                dc_offset_i=self.fgen3.overall_dc_offset_a,
                dc_offset_q=self.fgen3.overall_dc_offset_b
            )

        return self._current_config

    def verify_signal_generation(self) -> dict:
        """Verify that the signal is being generated properly (for debugging)."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        status = {
            'gen_enable': self.fgen3.gen_enable,
            'output_zero': self.fgen3.output_zero,
            'dc_offset_a': self.fgen3.overall_dc_offset_a,
            'dc_offset_b': self.fgen3.overall_dc_offset_b,
            'components': []
        }

        for i in range(self._max_components):
            comp_status = {
                'index': i,
                'enabled': getattr(self.fgen3, f'enable{i}'),
                'frequency': getattr(self.fgen3, f'frequency{i}'),
                'amplitude_a': getattr(self.fgen3, f'amplitude_a{i}'),
                'amplitude_b': getattr(self.fgen3, f'amplitude_b{i}'),
                'phase_offset_a': getattr(self.fgen3, f'phase_offset_a{i}'),
                'phase_offset_b': getattr(self.fgen3, f'phase_offset_b{i}')
            }
            status['components'].append(comp_status)

        self.logger.info("Red Pitaya Signal Status:")
        self.logger.info(f"  Master enable: {status['gen_enable']}")
        self.logger.info(f"  Output zero: {status['output_zero']}")
        self.logger.info(f"  DC offsets: I={status['dc_offset_a']:.5f}, Q={status['dc_offset_b']:.5f}")

        for comp in status['components']:
            if comp['enabled']:
                self.logger.info(f"  Component {comp['index']}: "
                                 f"freq={comp['frequency'] / 1e6:.3f} MHz, "
                                 f"amp_i={comp['amplitude_a']:.3f}, amp_q={comp['amplitude_b']:.3f}")

        return status


if __name__ == "__main__":
    # Basic multi-frequency setup
    rp_source = RedPitayaIFSource("10.203.129.28")
    rp_source.connect()

    # Define frequencies and amplitudes
    frequencies = [19.422e6, 21.580e6, 23.738e6]  # Hz
    amplitudes = [0.1, 0.1, 0.1]  # 0-1 normalized
    lo_frequency = 2.6e9  # Hz

    # Specify calibration files explicitly
    calibration_files = {
        19.422e6: r"C:\Users\aj92uwef\PycharmProjects\qudi-core\my_software\redpitaya_tools\IQ_calibration\calibration_results\2025-06-02-01-41-05\IF_19.422MHz\calibration_redpitaya_all_results_IF_19.422MHz.csv",
        21.580e6: r"C:\Users\aj92uwef\PycharmProjects\qudi-core\my_software\redpitaya_tools\IQ_calibration\calibration_results\2025-06-02-01-41-05\IF_21.580MHz\calibration_redpitaya_all_results_IF_21.580MHz.csv",
        23.738e6: r"C:\Users\aj92uwef\PycharmProjects\qudi-core\my_software\redpitaya_tools\IQ_calibration\calibration_results\2025-06-02-01-41-05\IF_23.738MHz\calibration_redpitaya_all_results_IF_23.738MHz.csv"
    }

    # Configure the multi-frequency signal with calibration
    rp_source.set_multi_frequency_signal(frequencies, amplitudes, lo_frequency, calibration_files)

    # Enable output
    rp_source.enable_output(True)

    # Verify configuration
    status = rp_source.verify_signal_generation()

    # Keep the connection open for a moment to ensure settings are applied
    import time

    time.sleep(300)

    # Disconnect when done
    rp_source.disconnect()