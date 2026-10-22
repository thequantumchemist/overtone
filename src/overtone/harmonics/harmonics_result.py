"""
HarmonicsResult: Container for harmonic frequency calculation results.
"""

import json
import numpy as np
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_serializer
from typing import Optional, Union
from rich.console import Console
from rich.table import Table


class HarmonicsResult(BaseModel):
    """
    Container for harmonic frequency calculation results.
    
    Attributes
    ----------
    hessian : np.ndarray
        Hessian matrix in eV/Ang^2
    w2 : np.ndarray
        Eigenvalues of the mass-weighted Hessian
    modes : np.ndarray
        Normal mode eigenvectors (columns), normalized
    freq : np.ndarray
        Real harmonic frequencies in cm^-1
    ifreq : np.ndarray
        Imaginary harmonic frequencies in cm^-1 (for imaginary modes)
    freq_eV : np.ndarray
        Harmonic frequencies in eV
    chemical_formula : str
        Chemical formula of the molecule
    n_tr : int
        Number of rigid-body (translation/rotation) modes at the start of
        the mode list: 6, 5 for linear molecules, 0 with constrained atoms
    is_linear : bool
    projected : bool
        Whether translations/rotations were projected out
    symmetrized : bool
        Whether (H + H^T)/2 was used
    hessian_asymmetry : float
        max|H - H^T| / max|H| of the raw finite-difference Hessian, a
        diagnostic for the numerical quality of the Hessian
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    hessian: np.ndarray
    w2: np.ndarray
    modes: np.ndarray
    freq: np.ndarray
    ifreq: np.ndarray
    freq_eV: np.ndarray
    chemical_formula: str
    n_tr: int = 6
    is_linear: bool = False
    projected: bool = False
    symmetrized: bool = False
    hessian_asymmetry: float = 0.0
    
    @field_serializer('hessian', 'w2', 'modes', 'freq', 'ifreq', 'freq_eV')
    def serialize_array_field(self, value: np.ndarray):
        """Serialize numpy arrays to lists for JSON serialization."""
        from overtone.utils.serialization import serialize_array
        return serialize_array(value)
    
    def is_equivalent(
        self, 
        other: 'HarmonicsResult', 
        rtol: float = 1e-5, 
        atol: float = 1e-8
    ) -> bool:
        """
        Check if two HarmonicsResult instances are equivalent.
        
        Parameters
        ----------
        other : HarmonicsResult
            The other instance to compare with
        rtol : float, optional
            Relative tolerance for numpy array comparison (default: 1e-5)
        atol : float, optional
            Absolute tolerance for numpy array comparison (default: 1e-8)
            
        Returns
        -------
        bool
            True if all fields are equivalent within tolerance
        """
        if not isinstance(other, HarmonicsResult):
            return False
        
        # Compare string field
        if self.chemical_formula != other.chemical_formula:
            return False
        
        # Compare numpy array fields
        array_fields = ['hessian', 'w2', 'modes', 'freq', 'ifreq', 'freq_eV']
        for field in array_fields:
            if not np.allclose(
                getattr(self, field), 
                getattr(other, field), 
                rtol=rtol, 
                atol=atol
            ):
                return False
        
        return True
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality using is_equivalent with default tolerances.
        
        Parameters
        ----------
        other : object
            The other object to compare with
            
        Returns
        -------
        bool
            True if instances are equivalent
        """
        if not isinstance(other, HarmonicsResult):
            return False
        return self.is_equivalent(other)
    
    @property
    def vibrational_modes(self) -> list:
        """Indices of the vibrational modes (everything after the rigid-body modes)."""
        return list(range(self.n_tr, len(self.freq)))

    def check_active_modes(
        self,
        active_modes: list,
        min_frequency: float = 20.0,
        warn_frequency: float = 100.0,
    ) -> None:
        """
        Fail early for modes that VPT2 cannot treat.

        Raises for imaginary modes and modes below ``min_frequency`` (cm^-1):
        a fixed step in dimensionless coordinates becomes a huge Cartesian
        displacement for (near-)zero frequencies, e.g. free rotations that
        remain when too few atoms are constrained. Warns below
        ``warn_frequency``, where VPT2 is often unreliable (torsions, floppy
        modes).
        """
        import warnings

        bad = [
            (m, self.freq[m], self.ifreq[m])
            for m in active_modes
            if self.ifreq[m] > 0 or self.freq[m] < min_frequency
        ]
        if bad:
            desc = ", ".join(
                f"mode {m}: {f:.1f}" + (f" (imaginary {fi:.1f}i)" if fi >= 0.05 else "")
                for m, f, fi in bad
            )
            raise ValueError(
                f"Active modes with imaginary or very low frequency (< {min_frequency} "
                f"cm^-1): {desc}. Tighten the geometry optimisation, constrain at "
                "least three non-collinear atoms if you constrain any, or exclude "
                "these modes via active_modes."
            )
        low = [m for m in active_modes if self.freq[m] < warn_frequency]
        if low:
            warnings.warn(
                f"Modes {low} are below {warn_frequency} cm^-1. Their displacements "
                "are large and VPT2 is often unreliable for such floppy modes; "
                "consider excluding them via active_modes."
            )

    def get_real_frequencies(self) -> np.ndarray:
        """Get only real (non-imaginary) frequencies."""
        return self.freq[self.freq.real != 0]
    
    def get_imaginary_frequencies(self) -> np.ndarray:
        """Get only imaginary frequencies."""
        return self.ifreq[self.ifreq != 0]
    
    def has_imaginary_frequencies(self) -> bool:
        """Check if there are any imaginary frequencies (indicating saddle point)."""
        return np.any(self.ifreq != 0)
    
    def summary(self, show_all: bool = True, file=None) -> str:
        """
        Generate a human-readable summary of harmonic frequencies.
        
        Parameters
        ----------
        show_all : bool, optional
            If True, show all modes. If False, only show non-zero frequencies.
        file : file object, optional
            File to write table to (default: console output)
            
        Returns
        -------
        str
            Summary text
        """
        console = Console(file=file)
        table = Table(
            title=f"Harmonic Frequencies for {self.chemical_formula}"
        )

        table.add_column("Mode", justify="right", style="cyan", no_wrap=True)
        table.add_column("Frequency (1/cm)", justify="right", style="magenta")
        table.add_column("Frequency (eV)", justify="right", style="green")

        # Combine real and imaginary for display
        _freq = self.freq + 1j * self.ifreq
        
        for i, freq in enumerate(_freq):
            # Skip zero frequencies if show_all is False
            if not show_all and freq.real == 0 and freq.imag == 0:
                continue
                
            # Wavenumber
            frequency_str = ""
            if freq.real != 0:
                frequency_str = f"{freq.real:.2f}"
            if freq.imag != 0:
                frequency_str += f"{freq.imag:.2f}i"

            # Energy
            frequency_str_eV = ""
            if self.freq_eV[i].real != 0:
                frequency_str_eV = f"{self.freq_eV[i].real:.2f}"
            if self.freq_eV[i].imag != 0:
                frequency_str_eV += f"{self.freq_eV[i].imag:.2f}i"

            table.add_row(str(i + 1), frequency_str, frequency_str_eV)

        console.print(table)
        
        # Return summary statistics as string
        lines = [
            f"Harmonic Frequencies for {self.chemical_formula}",
            "=" * 60,
            f"Total modes: {len(self.freq)}",
            f"Real frequencies: {np.sum(self.freq != 0)}",
            f"Imaginary frequencies: {np.sum(self.ifreq != 0)}",
        ]
        
        if self.has_imaginary_frequencies():
            lines.append(f"\n⚠️  Warning: Imaginary frequencies detected (saddle point)")
        
        return "\n".join(lines)
    
    def dump(self, filepath: Union[str, Path]) -> None:
        """
        Save HarmonicsResult to a JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the JSON file
        """
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'HarmonicsResult':
        """
        Load HarmonicsResult from a JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file
            
        Returns
        -------
        HarmonicsResult
            Loaded HarmonicsResult instance
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        from overtone.utils.serialization import deserialize_array
        
        # Convert lists back to numpy arrays
        for key in ['hessian', 'w2', 'modes', 'freq', 'ifreq', 'freq_eV']:
            if key in data:
                data[key] = deserialize_array(data[key])
        
        return cls(**data)
