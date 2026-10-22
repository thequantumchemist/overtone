"""
VPT2Solver: Anharmonic vibrational calculations using second-order perturbation theory.

This module provides a clean interface for calculating anharmonic contributions
to vibrational frequencies given third-order (T) and fourth-order (U) force constants.
"""

import json
import numpy as np
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_serializer
from typing import List, Tuple, Optional, Union
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from overtone.anharmonics.transitions import VPT2Transition


class VPT2Result(BaseModel):
    """
    Container for VPT2 calculation results.

    Provides structured access to anharmonicity constants, frequencies,
    and optional split analysis or Coriolis contributions.

    Attributes
    ----------
    X : np.ndarray
        Anharmonicity constants (Chi matrix), shape (len_modes, len_modes)
    anharm_freq : np.ndarray
        Anharmonic fundamental frequencies in cm^-1
    method : str
        Calculation method used ('standard', 'split', 'deperturbed', 'dcpt2')
    X_3 : Optional[np.ndarray]
        Third-order contribution to X (split analysis only)
    X_4 : Optional[np.ndarray]
        Fourth-order contribution to X (split analysis only)
    X_coriolis : Optional[np.ndarray]
        Coriolis contribution to X (if computed)
    zeta : Optional[np.ndarray]
        Coriolis coupling constants (if computed)
    fermi_type1 : Optional[List[Tuple]]
        Detected type-1 Fermi resonances (deperturbed only)
    fermi_type2 : Optional[List[Tuple]]
        Detected type-2 Fermi resonances (deperturbed only)
    transitions : Optional[List[VPT2Transition]]
        Fundamentals, overtones and combination bands (sorted by energy),
        with IR intensities if dipole derivatives were available
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    X: np.ndarray
    anharm_freq: np.ndarray
    method: str
    X_3: Optional[np.ndarray] = None
    X_4: Optional[np.ndarray] = None
    X_coriolis: Optional[np.ndarray] = None
    zeta: Optional[np.ndarray] = None
    fermi_type1: Optional[List[Tuple]] = None
    fermi_type2: Optional[List[Tuple]] = None
    transitions: Optional[List[VPT2Transition]] = None
    
    @field_serializer('X', 'anharm_freq', 'X_3', 'X_4', 'X_coriolis', 'zeta')
    def serialize_array_field(self, value: Optional[np.ndarray]):
        """Serialize numpy arrays to lists for JSON serialization."""
        if value is None:
            return None
        from overtone.utils.serialization import serialize_array
        return serialize_array(value)

    # ------------------------------------------------------------------
    # Transitions / spectra
    # ------------------------------------------------------------------
    def get_transitions(
        self,
        kinds: Optional[List[str]] = None,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
    ) -> List[VPT2Transition]:
        """
        Filter transitions by kind ('fundamental', 'overtone', 'combination')
        and/or wavenumber window (cm^-1).
        """
        if self.transitions is None:
            raise ValueError("No transitions stored in this result")
        out = []
        for t in self.transitions:
            if kinds is not None and t.kind not in kinds:
                continue
            if xmin is not None and t.energy < xmin:
                continue
            if xmax is not None and t.energy > xmax:
                continue
            out.append(t)
        return out

    @property
    def has_intensities(self) -> bool:
        return bool(self.transitions) and any(
            t.intensity is not None for t in self.transitions
        )

    def transitions_table(
        self,
        kinds: Optional[List[str]] = None,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        file=None,
    ) -> None:
        """Print fundamentals/overtones/combinations as a table."""
        rows = self.get_transitions(kinds, xmin, xmax)
        table = Table(title=f"VPT2 transitions (method: {self.method})")
        table.add_column("Band", style="cyan")
        table.add_column("Type")
        table.add_column("Harmonic (1/cm)", justify="right")
        table.add_column("VPT2 (1/cm)", justify="right", style="magenta")
        if self.has_intensities:
            table.add_column("Intensity (km/mol)", justify="right", style="green")
        table.add_column("Fermi", justify="center", style="yellow")
        for t in rows:
            cells = [t.label, t.kind, f"{t.harmonic:.1f}", f"{t.energy:.1f}"]
            if self.has_intensities:
                cells.append("-" if t.intensity is None else f"{t.intensity:.4g}")
            cells.append("*" if t.resonance_affected else "")
            table.add_row(*cells)
        console = Console(file=file)
        console.print(table)
        if any(t.resonance_affected for t in rows):
            if self.method == "gvpt2":
                console.print("[yellow]* part of a Fermi resonance, treated "
                              "variationally (GVPT2).[/yellow]")
            else:
                console.print("[yellow]* part of a detected Fermi resonance. Use "
                              "method='gvpt2' for quantitative values.[/yellow]")

    def spectrum(self, **kwargs):
        """Broadened spectrum (x, y); see overtone.spectrum.broaden_transitions."""
        from overtone.spectrum import broaden_transitions
        return broaden_transitions(self.get_transitions(), **kwargs)

    def plot_spectrum(self, **kwargs):
        """Plot the spectrum; see overtone.spectrum.plot_spectrum."""
        from overtone.spectrum import plot_spectrum
        return plot_spectrum(self, **kwargs)

    def get_active_X(self, active_modes: List[int]) -> np.ndarray:
        """Extract X matrix for active modes only."""
        active_idx = np.ix_(active_modes, active_modes)
        return self.X[active_idx]

    def get_active_frequencies(self, active_modes: List[int]) -> np.ndarray:
        """Extract anharmonic frequencies for active modes only."""
        return self.anharm_freq[active_modes]

    def summary(self, active_modes: Optional[List[int]] = None, file=None) -> str:
        """
        Generate a human-readable summary of VPT2 results.
        
        Parameters
        ----------
        active_modes : Optional[List[int]], optional
            If provided, only show results for these mode indices
        file : file object, optional
            File to write output to (default: console output)
            
        Returns
        -------
        str
            Summary text
        """
        console = Console(file=file)
        
        # Determine what to display
        if active_modes is not None:
            X_show = self.get_active_X(active_modes)
            freq_show = self.get_active_frequencies(active_modes)
            mode_labels = active_modes
            title = f"VPT2 Results (method: {self.method}, active modes: {active_modes})"
        else:
            X_show = self.X
            freq_show = self.anharm_freq
            mode_labels = list(range(len(self.anharm_freq)))
            title = f"VPT2 Results (method: {self.method})"
        
        # Create frequency table
        freq_table = Table(title="Anharmonic Frequencies")
        freq_table.add_column("Mode", justify="right", style="cyan", no_wrap=True)
        freq_table.add_column("Frequency (1/cm)", justify="right", style="magenta")
        
        for mode, freq in zip(mode_labels, freq_show):
            freq_table.add_row(str(mode + 1), f"{freq:.2f}")
        
        console.print(freq_table)
        
        # Create statistics table
        stats_table = Table(title="Anharmonicity Statistics")
        stats_table.add_column("Property", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")
        
        stats_table.add_row("Max |X|", f"{np.max(np.abs(X_show)):.4f} cm⁻¹")
        stats_table.add_row("X matrix shape", str(X_show.shape))
        
        if self.method == "split" and self.X_3 is not None:
            stats_table.add_row("Max |X_3|", f"{np.max(np.abs(self.X_3)):.4f} cm⁻¹")
            stats_table.add_row("Max |X_4|", f"{np.max(np.abs(self.X_4)):.4f} cm⁻¹")
        
        if self.X_coriolis is not None:
            stats_table.add_row("Max |X_coriolis|", f"{np.max(np.abs(self.X_coriolis)):.4f} cm⁻¹")
        
        console.print(stats_table)
        
        # Fermi resonances if detected
        if self.fermi_type1 is not None or self.fermi_type2 is not None:
            fermi_table = Table(title="Fermi Resonances Detected", title_style="bold yellow")
            fermi_table.add_column("Type", style="cyan")
            fermi_table.add_column("Count", justify="right", style="yellow")
            
            type1_count = len(self.fermi_type1) if self.fermi_type1 else 0
            type2_count = len(self.fermi_type2) if self.fermi_type2 else 0
            
            fermi_table.add_row("Type 1", str(type1_count))
            fermi_table.add_row("Type 2", str(type2_count))
            
            console.print(fermi_table)
        
        # Return summary text
        lines = [
            title,
            "=" * 60,
            f"Number of modes: {len(freq_show)}",
            f"Max anharmonic frequency: {np.max(freq_show):.2f} cm⁻¹",
            f"Min anharmonic frequency: {np.min(freq_show):.2f} cm⁻¹",
        ]
        
        return "\n".join(lines)
    
    def is_equivalent(
        self,
        other: 'VPT2Result',
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> bool:
        """
        Check if two VPT2Result instances are equivalent.
        
        Parameters
        ----------
        other : VPT2Result
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
        if not isinstance(other, VPT2Result):
            return False
        
        # Compare string field
        if self.method != other.method:
            return False
        
        # Compare required numpy array fields
        if not np.allclose(self.X, other.X, rtol=rtol, atol=atol):
            return False
        if not np.allclose(self.anharm_freq, other.anharm_freq, rtol=rtol, atol=atol):
            return False
        
        # Compare optional numpy array fields
        optional_array_fields = ['X_3', 'X_4', 'X_coriolis', 'zeta']
        for field in optional_array_fields:
            self_val = getattr(self, field)
            other_val = getattr(other, field)
            
            if (self_val is None) != (other_val is None):
                return False
            
            if self_val is not None and not np.allclose(
                self_val, other_val, rtol=rtol, atol=atol
            ):
                return False
        
        # Compare optional list fields
        if self.fermi_type1 != other.fermi_type1:
            return False
        if self.fermi_type2 != other.fermi_type2:
            return False
        if (self.transitions is None) != (other.transitions is None):
            return False
        if self.transitions is not None:
            if len(self.transitions) != len(other.transitions):
                return False
            for a, b in zip(self.transitions, other.transitions):
                if a.quanta != b.quanta or not np.isclose(
                    a.energy, b.energy, rtol=rtol, atol=atol
                ):
                    return False
                if (a.intensity is None) != (b.intensity is None):
                    return False
                if a.intensity is not None and not np.isclose(
                    a.intensity, b.intensity, rtol=rtol, atol=atol
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
        if not isinstance(other, VPT2Result):
            return False
        return self.is_equivalent(other)
    
    def dump(self, filepath: Union[str, Path]) -> None:
        """
        Save VPT2Result to a JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the JSON file
        """
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'VPT2Result':
        """
        Load VPT2Result from a JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file
            
        Returns
        -------
        VPT2Result
            Loaded VPT2Result instance
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        from overtone.utils.serialization import deserialize_array
        
        # Convert lists back to numpy arrays
        array_fields = ['X', 'anharm_freq', 'X_3', 'X_4', 'X_coriolis', 'zeta']
        for key in array_fields:
            if key in data and data[key] is not None:
                data[key] = deserialize_array(data[key])
        
        return cls(**data)
