"""
Configuration for anharmonic derivative calculations.
"""

import json
import numpy as np
from pathlib import Path
from pydantic import BaseModel, ConfigDict, model_validator, field_serializer
from typing import List, Optional, Union, TYPE_CHECKING
from ase import Atoms

if TYPE_CHECKING:
    from overtone.harmonics.harmonics_result import HarmonicsResult


class AnharmonicDerivativesConfig(BaseModel):
    """
    Configuration container for anharmonic derivative calculations.
    
    This class holds all the parameters needed to generate displacement
    trajectories and calculate 3rd and 4th order derivatives (T and U tensors).
    
    Parameters
    ----------
    freq : np.ndarray
        Harmonic frequencies in cm^-1, shape (n_modes,)
    modes : np.ndarray
        Normal mode eigenvectors (columns), normalized
    atoms : Atoms
        ASE Atoms object
    active_modes : List[int]
        Indices of active vibrational modes
    delta : float
        Step in dimensionless reduced normal coordinates (default 0.2)
    coupling : int
        Degree of coupling: 2 or 3 (default 3)
        - coupling=2: Only calculate T[i,i,j] type derivatives
        - coupling=3: Calculate all T[i,j,k] derivatives
    stencil : int
        Finite difference stencil type (0-3, default 1)
        - 0: 2-point stencil
        - 1: 5-point stencil (Bowman's choice)
        - 2: 7-point stencil
        - 3: 9-point stencil
    masses : Optional[np.ndarray]
        Atomic masses in amu, shape (3*num_atoms,). If None, extracted from atoms.
    constrained : bool
        Whether constraints are applied (default False)
    unconstrained_index : Optional[List[int]]
        Indices of unconstrained atoms
    force_indices : Optional[List[int]]
        Indices for force calculations
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    freq: np.ndarray
    modes: np.ndarray
    atoms: Atoms
    active_modes: List[int]
    delta: float = 0.2
    coupling: int = 3
    stencil: int = 1
    masses: Optional[np.ndarray] = None
    constrained: bool = False
    unconstrained_index: Optional[List[int]] = None
    force_indices: Optional[List[int]] = None
    len_modes: Optional[int] = None
    num_atoms: Optional[int] = None
    freq_array: Optional[np.ndarray] = None
    
    @field_serializer('freq', 'modes', 'masses', 'freq_array')
    def serialize_array(self, value: Optional[np.ndarray]):
        """Serialize numpy arrays to lists for JSON serialization."""
        if value is None:
            return None
        return value.tolist()
    
    @model_validator(mode='after')
    def validate_and_init(self):
        """Validate and initialize configuration parameters."""
        if self.coupling not in [2, 3]:
            raise ValueError(f"coupling must be 2 or 3, got {self.coupling}")
        
        if self.stencil not in [0, 1, 2, 3]:
            raise ValueError(f"stencil must be 0, 1, 2, or 3, got {self.stencil}")
        
        if self.delta <= 0:
            raise ValueError(f"delta must be positive, got {self.delta}")
        
        # Extract masses if not provided
        if self.masses is None:
            if self.force_indices is not None:
                masses = np.repeat(self.atoms.get_masses() ** 0.5, 3)[
                    self.force_indices
                ].reshape(-1, 1)
            else:
                masses = np.repeat(self.atoms.get_masses() ** 0.5, 3).reshape(-1, 1)
            object.__setattr__(self, 'masses', masses)
        
        # Set unconstrained indices if not provided
        if self.unconstrained_index is None:
            object.__setattr__(self, 'unconstrained_index', [i for i in range(len(self.atoms))])
        
        if self.force_indices is None:
            object.__setattr__(self, 'force_indices', [i for i in range(3 * len(self.atoms))])
        
        # Store array shape
        object.__setattr__(self, 'len_modes', len(self.freq))
        object.__setattr__(self, 'num_atoms', len(self.unconstrained_index) if self.constrained else len(self.atoms))
        
        # Reshape freq for vectorized operations
        object.__setattr__(self, 'freq_array', self.freq.reshape(-1, 1))
        
        return self
    
    @classmethod
    def from_harmonics_result(
        cls,
        harmonics_result: "HarmonicsResult",
        atoms: object,
        active_modes: List[int],
        delta: float = 0.2,
        coupling: int = 3,
        stencil: int = 1,
        constrained: bool = False,
        unconstrained_index: Optional[List[int]] = None,
        force_indices: Optional[List[int]] = None,
    ) -> "AnharmonicDerivativesConfig":
        """
        Create configuration from a HarmonicsResult and atoms object.
        
        Parameters
        ----------
        harmonics_result : HarmonicsResult
            Result object from harmonic frequency calculation
        atoms : object
            ASE Atoms object
        active_modes : List[int]
            Indices of active vibrational modes for derivative calculation
        delta : float
            Step in dimensionless reduced normal coordinates (default 0.2)
        coupling : int
            Degree of coupling (2 or 3), default is 3
        stencil : int
            Finite difference stencil type (0-3), default is 1
        constrained : bool
            Whether constraints are applied (default False)
        unconstrained_index : Optional[List[int]]
            Indices of unconstrained atoms (auto-detected if None)
        force_indices : Optional[List[int]]
            Indices for force calculations (auto-detected if None)
            
        Returns
        -------
        AnharmonicDerivativesConfig
            Configured object ready for derivative calculations
        """
        return cls(
            freq=harmonics_result.freq,
            modes=harmonics_result.modes,
            atoms=atoms,
            active_modes=active_modes,
            delta=delta,
            coupling=coupling,
            stencil=stencil,
            masses=None,  # Will be extracted in __post_init__
            constrained=constrained,
            unconstrained_index=unconstrained_index,
            force_indices=force_indices,
        )
    
    def dump(self, filepath: Union[str, Path]) -> None:
        """
        Save AnharmonicDerivativesConfig to a JSON file.
        
        The atoms object is serialized as extended XYZ format and included
        in the JSON.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the JSON file
        """
        from overtone.utils.serialization import serialize_atoms
        
        filepath = Path(filepath)
        data = self.model_dump(exclude={'atoms'})
        
        # Serialize atoms to extended XYZ format
        data['atoms_xyz'] = serialize_atoms(self.atoms)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'AnharmonicDerivativesConfig':
        """
        Load AnharmonicDerivativesConfig from a JSON file.
        
        The atoms object is automatically reconstructed from the extended XYZ
        format stored in the JSON.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file
            
        Returns
        -------
        AnharmonicDerivativesConfig
            Loaded AnharmonicDerivativesConfig instance
        """
        from overtone.utils.serialization import deserialize_atoms
        
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        array_fields = ['freq', 'modes', 'masses', 'freq_array']
        for key in array_fields:
            if key in data and data[key] is not None:
                data[key] = np.array(data[key])
        
        # Reconstruct atoms from XYZ
        data['atoms'] = deserialize_atoms(data.pop('atoms_xyz'))
        
        return cls(**data)
