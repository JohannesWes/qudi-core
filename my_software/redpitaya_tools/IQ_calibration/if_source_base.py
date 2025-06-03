# if_source_base.py
from abc import ABC, abstractmethod
import logging
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class IQComponentConfig:
    """Configuration for a single IQ signal component."""
    frequency: float  # Hz
    amplitude: float  # 0-1 for normalized amplitude
    amplitude_i: float = None  # Corrected I amplitude (calculated)
    amplitude_q: float = None  # Corrected Q amplitude (calculated)
    phase_i: float = 0.0  # degrees
    phase_q: float = 90.0  # degrees
    enabled: bool = True

    def __post_init__(self):
        # If corrected amplitudes not set, use base amplitude
        if self.amplitude_i is None:
            self.amplitude_i = self.amplitude
        if self.amplitude_q is None:
            self.amplitude_q = self.amplitude


@dataclass
class IQDeviceConfig:
    """Configuration for IQ signal generation with multiple components."""
    components: List[IQComponentConfig]
    dc_offset_i: float = 0.0  # V
    dc_offset_q: float = 0.0  # V

    @property
    def num_components(self) -> int:
        return len(self.components)

    @property
    def active_components(self) -> List[IQComponentConfig]:
        return [c for c in self.components if c.enabled]


class IFSourceBase(ABC):
    """Abstract base class for IF signal sources supporting multi-frequency generation."""

    def __init__(self, name: str = "IF Source"):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.name = name
        self._is_connected = False
        self._current_config = None
        self._calibration_data = {}  # Dict mapping frequency to calibration DataFrame
        self._max_components = 1  # Default, subclasses can override

    @property
    def max_components(self) -> int:
        """Maximum number of frequency components supported."""
        return self._max_components

    @abstractmethod
    def connect(self, **kwargs) -> None:
        """Connect to the IF source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the IF source."""
        pass

    @abstractmethod
    def configure_signal(self, config: IQDeviceConfig) -> None:
        """Configure the IQ signal parameters for all components."""
        pass

    @abstractmethod
    def configure_component(self, component_index: int, config: IQComponentConfig) -> None:
        """Configure a single frequency component."""
        pass

    @abstractmethod
    def enable_output(self, enable: bool = True) -> None:
        """Enable or disable the output."""
        pass

    @abstractmethod
    def set_dc_offsets(self, i_offset: float, q_offset: float) -> None:
        """Set DC offsets for I and Q channels."""
        pass

    @abstractmethod
    def set_iq_correction(self, component_index: int, if_amplitude: float,
                          gain_imbalance: float, phase_imbalance: float) -> Tuple[float, float, float, float]:
        """
        Apply IQ imbalance correction for a specific component.

        Parameters:
        -----------
        component_index : int
            Index of the frequency component (0 to max_components-1)
        if_amplitude : float
            IF signal amplitude (0 to 1)
        gain_imbalance : float
            Gain imbalance parameter (g)
        phase_imbalance : float
            Phase imbalance parameter (phi) in radians

        Returns:
        --------
        tuple : (amplitude_i, amplitude_q, phase_i, phase_q)
        """
        pass

    @abstractmethod
    def load_calibration_data(self, frequency: float, path: str) -> pd.DataFrame:
        """
        Load IQ calibration settings for a specific frequency from a CSV file.

        Parameters:
        -----------
        frequency : float
            IF frequency in Hz
        path : str
            Path to the calibration CSV file

        Returns:
        --------
        pd.DataFrame : Loaded calibration data
        """
        pass

    def set_multi_frequency_signal(self, frequencies: List[float], amplitudes: List[float],
                                   lo_frequency: float, calibration_files: Optional[Dict[float, str]] = None) -> None:
        """
        Convenience method to set up a multi-frequency signal with automatic calibration.

        Parameters:
        -----------
        frequencies : List[float]
            List of IF frequencies in Hz
        amplitudes : List[float]
            List of amplitudes (0 to 1) for each frequency
        lo_frequency : float
            LO frequency in Hz
        calibration_files : Optional[Dict[float, str]]
            Dictionary mapping frequency to calibration file path.
            If None, will look for default files based on frequency.
        """
        if len(frequencies) != len(amplitudes):
            raise ValueError("Frequencies and amplitudes lists must have the same length")

        if len(frequencies) > self.max_components:
            raise ValueError(f"Too many frequencies. Maximum supported: {self.max_components}")

        # Load calibration data for each frequency if not already loaded
        for freq in frequencies:
            if freq not in self._calibration_data:
                if calibration_files and freq in calibration_files:
                    cal_file = calibration_files[freq]
                else:
                    # Try to find default calibration file
                    cal_file = f'calibration_redpitaya_all_results_IF_{freq / 1e6:.3f}MHz.csv'

                try:
                    self.load_calibration_data(freq, cal_file)
                except Exception as e:
                    self.logger.warning(f"Could not load calibration for {freq / 1e6:.3f} MHz: {e}")

        # Create component configurations
        components = []
        dc_offsets_i = []
        dc_offsets_q = []

        for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
            component = IQComponentConfig(frequency=freq, amplitude=amp)

            # Apply calibration if available
            if freq in self._calibration_data:
                try:
                    g, phi, i_offset, q_offset = self._interpolate_calibration_parameters(
                        freq, lo_frequency / 1e9, amp
                    )

                    # Calculate corrected amplitudes and phases
                    component.amplitude_i = amp * (1 + g)
                    component.amplitude_q = amp * (1 - g)
                    component.phase_i = 0.0
                    component.phase_q = 90.0 + np.degrees(phi)

                    dc_offsets_i.append(i_offset)
                    dc_offsets_q.append(q_offset)

                except Exception as e:
                    self.logger.warning(f"Could not apply calibration for {freq / 1e6:.3f} MHz: {e}")

            components.append(component)

        # Calculate average DC offsets
        avg_dc_offset_i = np.mean(dc_offsets_i) if dc_offsets_i else 0.0
        avg_dc_offset_q = np.mean(dc_offsets_q) if dc_offsets_q else 0.0

        # Create and apply configuration
        config = IQDeviceConfig(
            components=components,
            dc_offset_i=avg_dc_offset_i,
            dc_offset_q=avg_dc_offset_q
        )

        self.configure_signal(config)
        self.logger.info(f"Configured {len(components)} frequency components with calibration")

    @abstractmethod
    def _interpolate_calibration_parameters(self, frequency: float, lo_frequency_ghz: float,
                                            if_amplitude: float) -> Tuple[float, float, float, float]:
        """
        Interpolate calibration parameters for a given frequency and amplitude.

        Parameters:
        -----------
        frequency : float
            IF frequency in Hz
        lo_frequency_ghz : float
            LO frequency in GHz
        if_amplitude : float
            IF amplitude (0 to 1)

        Returns:
        --------
        tuple : (g, phi, i_offset, q_offset)
        """
        pass

    def get_current_config(self) -> Optional[IQDeviceConfig]:
        """Get the current signal configuration."""
        return self._current_config

    @property
    def is_connected(self) -> bool:
        """Check if the source is connected."""
        return self._is_connected

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.is_connected:
            self.disconnect()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', connected={self.is_connected})"