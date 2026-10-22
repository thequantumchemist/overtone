import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator, field_serializer, field_validator
from pathlib import Path
from typing import List, Optional, Union, TYPE_CHECKING, Literal
import json
from ase import Atoms

if TYPE_CHECKING:
    from overtone.harmonics.harmonics_result import HarmonicsResult


class VPT2Config(BaseModel):
    """
    Configuration container for VPT2 anharmonic calculations.
    
    Parameters
    ----------
    freq : np.ndarray
        Harmonic frequencies in cm^-1, shape (n_modes,)
    active_modes : List[int]
        Indices of active vibrational modes
    min_mode_idx : int
        Minimum mode index (typically 6 for non-linear molecules)
    max_mode_idx : int
        Maximum mode index
    len_modes : int
        Total number of modes (3 * num_atoms)
    coupling : int
        Degree of coupling (2 or 3)
    masses : Optional[np.ndarray]
        Atomic masses in amu, shape (3*num_atoms,)
    modes : Optional[np.ndarray]
        Normal mode eigenvectors
    atoms : Optional[Atoms]
        ASE Atoms object (needed for Coriolis calculations)
    method: str
        'standard', 'split', 'deperturbed' or 'dcpt2'
    coriolis : bool
        Include Coriolis contributions to X (default True; needs atoms)
    resonance_window : float
        Frequency window (cm^-1) for the Fermi-resonance test (default 200)
    resonance_K1, resonance_K2 : float
        Martin-test thresholds (cm^-1) for type-1 and type-2 resonances
        (default 1)
    max_quanta : int
        Highest total number of quanta for which transitions (fundamentals,
        overtones, combination bands) are listed; 2 covers 2v_i and v_i+v_j,
        3 adds the second-overtone region of the NIR. Default 2.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    freq: np.ndarray
    active_modes: List[int]
    min_mode_idx: int
    max_mode_idx: int
    len_modes: int
    coupling: int = 3
    masses: Optional[np.ndarray] = None
    modes: Optional[np.ndarray] = None
    atoms: Optional[Atoms] = None
    freq_array: Optional[np.ndarray] = None
    method: Literal['standard', 'split', 'deperturbed', 'dcpt2', 'gvpt2'] = 'gvpt2'
    max_quanta: int = 2
    coriolis: bool = True
    resonance_window: float = 200.0
    resonance_K1: float = 1.0
    resonance_K2: float = 1.0
    
    @field_serializer('freq', 'masses', 'modes', 'freq_array')
    def serialize_array(self, value: Optional[np.ndarray]):
        """Serialize numpy arrays to lists for JSON serialization."""
        if value is None:
            return None
        return value.tolist()
    
    @field_validator('method', mode='before')
    @classmethod
    def _method_aliases(cls, value):
        """Accept the long DCPT2 spellings used in older scripts."""
        aliases = {
            'degeneracy_corrected': 'dcpt2',
            'degenerancy_corrected': 'dcpt2',
            'vpt2': 'standard',
            'dvpt2': 'deperturbed',
            'generalized': 'gvpt2',
        }
        return aliases.get(value, value)

    @model_validator(mode='after')
    def validate_and_init(self):
        """Validate configuration parameters."""
        if len(self.freq) != self.len_modes:
            raise ValueError(
                f"freq array length ({len(self.freq)}) must match len_modes ({self.len_modes})"
            )
        if self.coupling not in [2, 3]:
            raise ValueError(f"coupling must be 2 or 3, got {self.coupling}")
        if self.max_quanta < 1:
            raise ValueError(f"max_quanta must be >= 1, got {self.max_quanta}")
        
        # Reshape freq to column vector for vectorized operations
        object.__setattr__(self, 'freq_array', self.freq.reshape(-1, 1))
        return self
    
    @classmethod
    def from_harmonics_result(
        cls,
        harmonics_result: "HarmonicsResult",
        active_modes: List[int],
        atoms: Optional[object] = None,
        coupling: int = 3,
        method: Literal['standard', 'split', 'deperturbed', 'dcpt2', 'gvpt2'] = 'gvpt2',
        max_quanta: int = 2,
        coriolis: bool = True,
        resonance_window: float = 200.0,
        resonance_K1: float = 1.0,
        resonance_K2: float = 1.0,
    ) -> "VPT2Config":
        """
        Create a VPT2Config from a HarmonicsResult and additional parameters.
        
        Parameters
        ----------
        harmonics_result : HarmonicsResult
            Result object from harmonic frequency calculation
        active_modes : List[int]
            Indices of active vibrational modes for VPT2 calculation
        atoms : Optional[object]
            ASE Atoms object (needed for Coriolis calculations)
        coupling : int
            Degree of coupling (2 or 3), default is 3
            
        Returns
        -------
        VPT2Config
            Configured VPT2Config object ready for anharmonic calculations
        """
        len_modes = len(harmonics_result.freq)
        
        # Determine min and max mode indices from active modes
        if active_modes:
            min_mode_idx = min(active_modes)
            max_mode_idx = max(active_modes)
        else:
            # Default: skip first 6 modes (translations + rotations for non-linear)
            min_mode_idx = 6
            max_mode_idx = len_modes - 1
        
        # Extract masses from modes if available
        # The modes eigenvectors already incorporate mass weighting
        masses = None
        if atoms is not None:
            masses = np.repeat(atoms.get_masses(), 3)
        
        return cls(
            freq=harmonics_result.freq,
            active_modes=active_modes,
            min_mode_idx=min_mode_idx,
            max_mode_idx=max_mode_idx,
            len_modes=len_modes,
            coupling=coupling,
            masses=masses,
            modes=harmonics_result.modes,
            atoms=atoms,
            method=method,
            max_quanta=max_quanta,
            coriolis=coriolis,
            resonance_window=resonance_window,
            resonance_K1=resonance_K1,
            resonance_K2=resonance_K2,
        )
    
    def dump(self, filepath: Union[str, Path]) -> None:
        """
        Save VPT2Config to a JSON file.
        
        The atoms object (if present) is serialized as extended XYZ format
        and included in the JSON.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the JSON file
        """
        from overtone.utils.serialization import serialize_atoms
        
        filepath = Path(filepath)
        data = self.model_dump(exclude={'atoms'})
        
        # Serialize atoms to extended XYZ format if present
        atoms_xyz = serialize_atoms(self.atoms)
        if atoms_xyz is not None:
            data['atoms_xyz'] = atoms_xyz
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'VPT2Config':
        """
        Load VPT2Config from a JSON file.
        
        The atoms object (if it was saved) is automatically reconstructed
        from the extended XYZ format stored in the JSON.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file
            
        Returns
        -------
        VPT2Config
            Loaded VPT2Config instance
        """
        from overtone.utils.serialization import deserialize_atoms
        
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        array_fields = ['freq', 'masses', 'modes', 'freq_array']
        for key in array_fields:
            if key in data and data[key] is not None:
                data[key] = np.array(data[key])
        
        # Reconstruct atoms from XYZ if present
        if 'atoms_xyz' in data:
            data['atoms'] = deserialize_atoms(data.pop('atoms_xyz'))
        
        return cls(**data)
