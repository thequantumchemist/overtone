"""
Serialization utilities for overtone classes.

This module provides utilities for serializing and deserializing objects
that are not natively JSON-serializable, such as ASE Atoms objects and
complex numbers.
"""

from io import StringIO
from typing import Optional, Any
from ase import Atoms
import numpy as np


def serialize_atoms(atoms: Optional[Atoms]) -> Optional[str]:
    """
    Serialize an ASE Atoms object to extended XYZ format string.
    
    Parameters
    ----------
    atoms : Optional[Atoms]
        ASE Atoms object to serialize. If None, returns None.
        
    Returns
    -------
    Optional[str]
        Extended XYZ format string representation of the atoms,
        or None if atoms is None.
        
    Examples
    --------
    >>> from ase import Atoms
    >>> atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> xyz_str = serialize_atoms(atoms)
    >>> print(xyz_str)
    3
    Properties=species:S:1:pos:R:3 pbc="F F F"
    H 0.0 0.0 0.0
    ...
    """
    if atoms is None:
        return None
    
    from ase.io import write
    
    atoms_io = StringIO()
    write(atoms_io, atoms, format='extxyz')
    return atoms_io.getvalue()


def deserialize_atoms(xyz_str: Optional[str]) -> Optional[Atoms]:
    """
    Deserialize an ASE Atoms object from extended XYZ format string.
    
    Parameters
    ----------
    xyz_str : Optional[str]
        Extended XYZ format string. If None, returns None.
        
    Returns
    -------
    Optional[Atoms]
        Reconstructed ASE Atoms object, or None if xyz_str is None.
        
    Examples
    --------
    >>> xyz_str = "3\\nProperties=species:S:1:pos:R:3\\nH 0.0 0.0 0.0\\n..."
    >>> atoms = deserialize_atoms(xyz_str)
    >>> print(atoms.get_chemical_formula())
    H2O
    """
    if xyz_str is None:
        return None
    
    from ase.io import read
    
    atoms_io = StringIO(xyz_str)
    return read(atoms_io, format='extxyz')


def serialize_complex(obj: Any) -> Any:
    """
    Recursively convert complex numbers to JSON-serializable dicts.
    
    Complex numbers are converted to dictionaries with 'real' and 'imag' keys.
    This function works recursively on lists and nested structures.
    
    Parameters
    ----------
    obj : Any
        Object to serialize. Can be a complex number, list, or other type.
        
    Returns
    -------
    Any
        Serialized object with complex numbers converted to dicts
        
    Examples
    --------
    >>> serialize_complex(complex(1.5, 2.3))
    {'real': 1.5, 'imag': 2.3}
    
    >>> serialize_complex([1.0, complex(2.0, 3.0), 4.0])
    [1.0, {'real': 2.0, 'imag': 3.0}, 4.0]
    """
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    elif isinstance(obj, list):
        return [serialize_complex(item) for item in obj]
    else:
        return obj


def deserialize_complex(obj: Any) -> Any:
    """
    Recursively convert complex number dicts back to complex numbers.
    
    Dictionaries with 'real' and 'imag' keys are converted back to
    complex numbers. This function works recursively on lists and
    nested structures.
    
    Parameters
    ----------
    obj : Any
        Object to deserialize. Can be a dict, list, or other type.
        
    Returns
    -------
    Any
        Deserialized object with dicts converted back to complex numbers
        
    Examples
    --------
    >>> deserialize_complex({'real': 1.5, 'imag': 2.3})
    (1.5+2.3j)
    
    >>> deserialize_complex([1.0, {'real': 2.0, 'imag': 3.0}, 4.0])
    [1.0, (2+3j), 4.0]
    """
    if isinstance(obj, dict) and 'real' in obj and 'imag' in obj:
        return complex(obj['real'], obj['imag'])
    elif isinstance(obj, list):
        return [deserialize_complex(item) for item in obj]
    else:
        return obj


def serialize_array(value: np.ndarray) -> Any:
    """
    Serialize a numpy array to a JSON-serializable format.
    
    Handles both real and complex numpy arrays. Complex arrays are
    serialized with complex numbers converted to dicts.
    
    Parameters
    ----------
    value : np.ndarray
        Numpy array to serialize
        
    Returns
    -------
    Any
        List representation of the array, with complex numbers
        converted to dicts if present
        
    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> serialize_array(arr)
    [1.0, 2.0, 3.0]
    
    >>> arr_complex = np.array([1+2j, 3+4j])
    >>> serialize_array(arr_complex)
    [{'real': 1.0, 'imag': 2.0}, {'real': 3.0, 'imag': 4.0}]
    """
    list_value = value.tolist()
    
    # Check if the array contains complex numbers
    if np.iscomplexobj(value):
        return serialize_complex(list_value)
    
    return list_value


def deserialize_array(data: Any) -> np.ndarray:
    """
    Deserialize a JSON representation back to a numpy array.
    
    Handles both real and complex arrays. Complex number dicts are
    converted back to complex numbers before creating the array.
    
    Parameters
    ----------
    data : Any
        List representation of an array, possibly with complex number dicts
        
    Returns
    -------
    np.ndarray
        Reconstructed numpy array
        
    Examples
    --------
    >>> deserialize_array([1.0, 2.0, 3.0])
    array([1., 2., 3.])
    
    >>> data_complex = [{'real': 1.0, 'imag': 2.0}, {'real': 3.0, 'imag': 4.0}]
    >>> deserialize_array(data_complex)
    array([1.+2.j, 3.+4.j])
    """
    # First deserialize any complex numbers
    deserialized = deserialize_complex(data)
    # Then convert to numpy array
    return np.array(deserialized)
