"""
Result container for anharmonic derivative calculations.
"""

import json
import numpy as np
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_serializer
from typing import Dict, Tuple, Union
from rich.console import Console
from rich.table import Table


class AnharmonicDerivativesResult(BaseModel):
    """
    Container for anharmonic derivative calculation results.
    
    This class holds the 3rd and 4th order derivatives (T and U tensors)
    calculated from displaced configurations.
    
    Attributes
    ----------
    T : np.ndarray
        3rd-order derivatives tensor (cubic force constants), shape (n_modes, n_modes, n_modes)
        Units: cm^-1
    U : np.ndarray
        4th-order derivatives tensor (quartic force constants), shape (n_modes, n_modes, n_modes, n_modes)
        Units: cm^-1
    V : Dict[Tuple[int, ...], float]
        Energy dictionary mapping displacement keys to energies in cm^-1
    required_calculations : np.ndarray
        Array of displacement combinations needed, shape (n_configs, n_modes)
    number_of_calculations : int
        Total number of configurations evaluated
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    T: np.ndarray
    U: np.ndarray
    V: Dict[Tuple[int, ...], float]
    required_calculations: np.ndarray
    number_of_calculations: int
    
    @field_serializer('T', 'U', 'required_calculations')
    def serialize_array_field(self, value: np.ndarray):
        """Serialize numpy arrays to lists for JSON serialization."""
        from overtone.utils.serialization import serialize_array
        return serialize_array(value)
    
    @field_serializer('V')
    def serialize_dict(self, value: Dict[Tuple[int, ...], float]):
        """Serialize dict with tuple keys to dict with string keys for JSON."""
        return {str(k): v for k, v in value.items()}
    
    def get_nonzero_T_count(self) -> int:
        """Get count of non-zero T tensor elements."""
        return np.count_nonzero(self.T)
    
    def get_nonzero_U_count(self) -> int:
        """Get count of non-zero U tensor elements."""
        return np.count_nonzero(self.U)
    
    def get_T_statistics(self) -> Dict[str, float]:
        """Get statistics for T tensor."""
        nonzero_T = self.T[self.T != 0]
        if len(nonzero_T) == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        
        return {
            "min": float(np.min(np.abs(nonzero_T))),
            "max": float(np.max(np.abs(nonzero_T))),
            "mean": float(np.mean(np.abs(nonzero_T))),
            "std": float(np.std(np.abs(nonzero_T))),
        }
    
    def get_U_statistics(self) -> Dict[str, float]:
        """Get statistics for U tensor."""
        nonzero_U = self.U[self.U != 0]
        if len(nonzero_U) == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        
        return {
            "min": float(np.min(np.abs(nonzero_U))),
            "max": float(np.max(np.abs(nonzero_U))),
            "mean": float(np.mean(np.abs(nonzero_U))),
            "std": float(np.std(np.abs(nonzero_U))),
        }
    
    def summary(self):
        """
        Display a summary of the derivative calculation results.
        """
        console = Console()
        
        console.print("\n[bold cyan]Anharmonic Derivatives Summary[/bold cyan]")
        console.print(f"Number of configurations calculated: {self.number_of_calculations}")
        
        # T tensor statistics
        table_T = Table(title="3rd Order Derivatives (T tensor)")
        table_T.add_column("Property", style="cyan")
        table_T.add_column("Value", justify="right", style="magenta")
        
        T_stats = self.get_T_statistics()
        table_T.add_row("Total elements", f"{self.T.size:,}")
        table_T.add_row("Non-zero elements", f"{self.get_nonzero_T_count():,}")
        table_T.add_row("Min |T|", f"{T_stats['min']:.6f} cm⁻¹")
        table_T.add_row("Max |T|", f"{T_stats['max']:.6f} cm⁻¹")
        table_T.add_row("Mean |T|", f"{T_stats['mean']:.6f} cm⁻¹")
        table_T.add_row("Std |T|", f"{T_stats['std']:.6f} cm⁻¹")
        
        console.print(table_T)
        
        # U tensor statistics
        table_U = Table(title="4th Order Derivatives (U tensor)")
        table_U.add_column("Property", style="cyan")
        table_U.add_column("Value", justify="right", style="magenta")
        
        U_stats = self.get_U_statistics()
        table_U.add_row("Total elements", f"{self.U.size:,}")
        table_U.add_row("Non-zero elements", f"{self.get_nonzero_U_count():,}")
        table_U.add_row("Min |U|", f"{U_stats['min']:.6f} cm⁻¹")
        table_U.add_row("Max |U|", f"{U_stats['max']:.6f} cm⁻¹")
        table_U.add_row("Mean |U|", f"{U_stats['mean']:.6f} cm⁻¹")
        table_U.add_row("Std |U|", f"{U_stats['std']:.6f} cm⁻¹")
        
        console.print(table_U)
        console.print()
    
    def is_equivalent(
        self,
        other: 'AnharmonicDerivativesResult',
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> bool:
        """
        Check if two AnharmonicDerivativesResult instances are equivalent.
        
        Parameters
        ----------
        other : AnharmonicDerivativesResult
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
        if not isinstance(other, AnharmonicDerivativesResult):
            return False
        
        # Compare integer field
        if self.number_of_calculations != other.number_of_calculations:
            return False
        
        # Compare numpy array fields
        if not np.allclose(self.T, other.T, rtol=rtol, atol=atol):
            return False
        if not np.allclose(self.U, other.U, rtol=rtol, atol=atol):
            return False
        if not np.allclose(self.required_calculations, other.required_calculations, rtol=rtol, atol=atol):
            return False
        
        # Compare V dictionary
        if set(self.V.keys()) != set(other.V.keys()):
            return False
        for key in self.V:
            if not np.isclose(self.V[key], other.V[key], rtol=rtol, atol=atol):
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
        if not isinstance(other, AnharmonicDerivativesResult):
            return False
        return self.is_equivalent(other)
    
    def dump(self, filepath: Union[str, Path]) -> None:
        """
        Save AnharmonicDerivativesResult to a JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the JSON file
        """
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'AnharmonicDerivativesResult':
        """
        Load AnharmonicDerivativesResult from a JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file
            
        Returns
        -------
        AnharmonicDerivativesResult
            Loaded AnharmonicDerivativesResult instance
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        from overtone.utils.serialization import deserialize_array
        
        # Convert lists back to numpy arrays
        for key in ['T', 'U', 'required_calculations']:
            if key in data:
                data[key] = deserialize_array(data[key])
        
        # Convert V dict with string keys back to tuple keys
        if 'V' in data:
            data['V'] = {eval(k): v for k, v in data['V'].items()}
        
        return cls(**data)
