"""
Coordinate transformation utilities for VPT2 calculations.
"""

import numpy as np


class CoordinateTransformer:
    """
    Handles coordinate transformations between different coordinate systems.
    
    Coordinate systems:
    - Cartesian (R): Standard Cartesian coordinates in Angstroms
    - Normal (Q): Normal mode coordinates in [sqrt(amu) * Å]
    - Reduced Normal (q): Dimensionless normal coordinates in [sqrt(kg) * m]
    
    Parameters
    ----------
    freq : np.ndarray
        Harmonic frequencies in cm^-1, shape (n_modes,) or (n_modes, 1)
    modes : np.ndarray
        Normal mode eigenvectors (columns), normalized
    masses : np.ndarray
        Atomic masses in sqrt(amu), shape (3*num_atoms,) or (3*num_atoms, 1)
    """
    
    # Physical constants
    C_M_PER_S = 2.99792458e8          # Speed of light in m/s
    H_J_S = 6.62607015e-34       # Planck's constant in J·s
    INV_CM_TO_INV_M = 1e2        # Conversion factor
    AMU_TO_KG = 1.66053906660e-27      # Atomic mass unit to kg
    A_TO_M = 1e-10               # Angstrom to meter
    EPSILON = 1e-20              # Small constant to avoid division by zero
    
    def __init__(self, freq: np.ndarray, modes: np.ndarray, masses: np.ndarray):
        """
        Initialize the coordinate transformer.
        
        Parameters
        ----------
        freq : np.ndarray
            Harmonic frequencies in cm^-1
        modes : np.ndarray
            Normal mode eigenvectors (columns)
        masses : np.ndarray
            Atomic masses in sqrt(amu)
        """
        self.modes = modes
        self.masses = masses
        
        # Ensure freq is a column vector for vectorized operations
        if freq.ndim == 1:
            self.freq_array = freq.reshape(-1, 1)
        else:
            self.freq_array = freq
        
        # Ensure masses is a column vector
        if masses.ndim == 1:
            self.masses = masses.reshape(-1, 1)
    
    def norm_to_cart(self, Q: np.ndarray) -> np.ndarray:
        """
        Transform from normal mode coordinates to Cartesian coordinates.
        
        Parameters
        ----------
        Q : np.ndarray
            Normal mode coordinates in [sqrt(amu) * Å]
            
        Returns
        -------
        np.ndarray
            Cartesian coordinates in Angstroms
        """
        r = (self.modes @ Q) / self.masses
        return r
    
    def cart_to_norm(self, R: np.ndarray) -> np.ndarray:
        """
        Transform from Cartesian to normal mode coordinates.
        
        Parameters
        ----------
        R : np.ndarray
            Cartesian coordinates in Angstroms
            
        Returns
        -------
        np.ndarray
            Normal mode coordinates in [sqrt(amu) * Å]
        """
        Q = self.modes.T @ (R * self.masses)
        return Q
    
    def norm_to_reduced_norm(self, Q: np.ndarray) -> np.ndarray:
        """
        Transform from normal to reduced normal coordinates.
        
        Q is in [sqrt(amu) * Å], convert to dimensionless [sqrt(kg) * m]
        
        Parameters
        ----------
        Q : np.ndarray
            Normal mode coordinates in [sqrt(amu) * Å]
            
        Returns
        -------
        np.ndarray
            Reduced normal coordinates (dimensionless)
        """
        q = (
            2 * np.pi
            * np.sqrt(self.C_M_PER_S / self.H_J_S)
            * np.sqrt(self.freq_array * self.INV_CM_TO_INV_M)
            * Q
            * np.sqrt(self.AMU_TO_KG)
            * self.A_TO_M
        )
        return q
    
    def reduced_norm_to_norm(self, q: np.ndarray) -> np.ndarray:
        """
        Transform from reduced normal to normal coordinates.
        
        Parameters
        ----------
        q : np.ndarray
            Reduced normal coordinates (dimensionless)
            
        Returns
        -------
        np.ndarray
            Normal mode coordinates in [sqrt(amu) * Å]
        """
        Q = q / (
            2 * np.pi
            * np.sqrt(self.C_M_PER_S / self.H_J_S)
            * np.sqrt(self.freq_array * self.INV_CM_TO_INV_M)
            + self.EPSILON
        )
        Q /= np.sqrt(self.AMU_TO_KG) * self.A_TO_M
        
        # Handle NaN values (e.g., for zero frequencies)
        return np.nan_to_num(Q)
    
    def cart_to_reduced_norm(self, R: np.ndarray) -> np.ndarray:
        """
        Transform from Cartesian to reduced normal coordinates.
        
        Parameters
        ----------
        R : np.ndarray
            Cartesian coordinates in Angstroms
            
        Returns
        -------
        np.ndarray
            Reduced normal coordinates (dimensionless)
        """
        return self.norm_to_reduced_norm(self.cart_to_norm(R))
    
    def reduced_norm_to_cart(self, q: np.ndarray) -> np.ndarray:
        """
        Transform from reduced normal to Cartesian coordinates.
        
        Parameters
        ----------
        q : np.ndarray
            Reduced normal coordinates (dimensionless)
            
        Returns
        -------
        np.ndarray
            Cartesian coordinates in Angstroms
        """
        return self.norm_to_cart(self.reduced_norm_to_norm(q))
