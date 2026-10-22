"""
Solver for calculating anharmonic derivatives from displaced configurations.
"""

import numpy as np
from typing import List, Tuple, Dict
from ase import Atoms
from ase.io import write

from overtone.anharmonics.derivatives_config import AnharmonicDerivativesConfig
from overtone.anharmonics.derivatives_result import AnharmonicDerivativesResult
from overtone.anharmonics.coordinate_transforms import CoordinateTransformer
from overtone.anharmonics.dipole_derivatives import (
    DipoleDerivativesCalculator,
    DipoleDerivativesResult,
)
from overtone.utils.numerical_derivatives import (
    NumericalDerivativesCalculator,
    CartesianForceDerivativesCalculator
)


class AnharmonicDerivativesSolver:
    """
    Solver for generating displacement trajectories and calculating
    3rd and 4th order derivatives (T and U tensors) from ASE Atoms.
    
    This class specializes the general NumericalDerivativesCalculator to work
    with ASE Atoms objects and vibrational normal modes. It handles:
    1. Generation of displaced atomic configurations along normal modes
    2. Extraction of energies from computed trajectories
    3. Coordinate transformations between Cartesian and normal mode spaces
    
    The actual numerical differentiation is delegated to NumericalDerivativesCalculator.
    
    Parameters
    ----------
    config : AnharmonicDerivativesConfig
        Configuration object with all necessary parameters
    """
    
    # Physical constants
    EV_TO_INV_CM = 8065.543937
    
    def __init__(self, config: AnharmonicDerivativesConfig):
        """
        Initialize the solver with configuration.
        
        Parameters
        ----------
        config : AnharmonicDerivativesConfig
            Configuration object
        """
        self.config = config
        
        # Initialize coordinate transformer
        self.transformer = CoordinateTransformer(
            config.freq,
            config.modes,
            config.masses
        )
        
        # Initialize general numerical derivatives calculator
        self.numerical_calculator = NumericalDerivativesCalculator(
            n_variables=config.len_modes,
            delta=config.delta,
            stencil=config.stencil
        )
        
        # Setup derivative indices
        self.numerical_calculator.setup_derivative_indices(
            active_indices=config.active_modes,
            derivative_orders=[3, 4],
            coupling=config.coupling
        )
        
        # Initialize force-based derivatives calculator
        self.force_calculator = CartesianForceDerivativesCalculator(
            n_variables=config.len_modes,
            delta=config.delta,
            modes=config.modes,
            masses=config.masses,
            frequencies=config.freq,
            stencil=config.stencil
        )
        
        # Setup derivative indices for force calculator
        self.force_calculator.setup_derivative_indices(
            active_indices=config.active_modes,
            derivative_orders=[3, 4],
            coupling=config.coupling
        )
        
        # Dipole derivatives (only used when intensities are requested)
        self.dipole_calculator = DipoleDerivativesCalculator(
            n_modes=config.len_modes,
            active_modes=config.active_modes,
            delta=config.delta,
            stencil=config.stencil,
        )

        # Storage for required calculations
        self.required_calculations = None
        self.required_calculations_forces = None
    
    
    def generate_displacement_trajectory(self) -> Tuple[List[Atoms], np.ndarray]:
        """
        Generate trajectory of displaced configurations needed for derivatives.
        
        Uses the general numerical derivatives calculator to determine required
        displacements, then creates the corresponding displaced atomic configurations.
        
        Returns
        -------
        trajectory : List[Atoms]
            List of ASE Atoms objects with displaced geometries
        displacement_keys : np.ndarray
            Array of displacement combinations, shape (n_configs, n_modes)
            Each row represents the displacement multiples for each mode
        """
        # Get displacement keys from numerical calculator
        self.required_calculations = self.numerical_calculator.get_displacement_keys()
        
        # Generate displaced geometries
        trajectory = self._create_displaced_trajectory(self.required_calculations)
        
        return trajectory, self.required_calculations
    
    def create_displaced_trajectory(self, displacement_keys: np.ndarray) -> List[Atoms]:
        """Displaced geometries for arbitrary displacement keys."""
        return self._create_displaced_trajectory(np.atleast_2d(displacement_keys))

    def _create_displaced_trajectory(self, displacement_keys: np.ndarray) -> List[Atoms]:
        """
        Create displaced atomic configurations from displacement keys.
        
        Parameters
        ----------
        displacement_keys : np.ndarray
            Array of displacement combinations, shape (n_configs, n_modes)
            
        Returns
        -------
        List[Atoms]
            List of displaced atomic configurations
        """
        n_configs = displacement_keys.shape[0]
        
        # Convert displacement keys to reduced normal coordinates
        q = displacement_keys.T * self.config.delta
        
        # Transform to Cartesian displacements
        R = self.transformer.reduced_norm_to_cart(q).T
        
        # Reshape to (n_configs, n_atoms, 3)
        R = R.reshape(n_configs, self.config.num_atoms, 3)
        
        # Get reference positions
        pos = self.config.atoms.get_positions()[self.config.unconstrained_index]
        
        # Create displaced configurations
        trajectory = []
        for r in R:
            atoms = self.config.atoms.copy()
            atoms.positions[self.config.unconstrained_index] = pos + r
            trajectory.append(atoms)
        
        return trajectory
    
    
    def calculate_derivatives(
        self, 
        trajectory_with_energies: List[Atoms],
        displacement_keys: np.ndarray = None
    ) -> AnharmonicDerivativesResult:
        """
        Calculate T and U derivative tensors from trajectory with computed energies.
        
        Parameters
        ----------
        trajectory_with_energies : List[Atoms]
            List of Atoms objects with energies attached (from get_potential_energy())
        displacement_keys : np.ndarray, optional
            Displacement keys corresponding to trajectory. If None, uses keys from
            generate_displacement_trajectory()
            
        Returns
        -------
        AnharmonicDerivativesResult
            Result object containing T, U tensors and related data
        """
        if displacement_keys is None:
            if self.required_calculations is None:
                raise ValueError(
                    "No displacement keys provided. Either pass displacement_keys "
                    "or call generate_displacement_trajectory() first."
                )
            displacement_keys = self.required_calculations
        
        # Build energy dictionary
        V = self._build_energy_dict(trajectory_with_energies, displacement_keys)
        
        # Calculate derivatives using general calculator
        print("Calculating derivatives...")
        derivatives = self.numerical_calculator.calculate_derivatives(V)
        print("Finished calculating derivatives")
        
        T = derivatives['order_3']
        U = derivatives['order_4']
        
        return AnharmonicDerivativesResult(
            T=T,
            U=U,
            V=V,
            required_calculations=displacement_keys,
            number_of_calculations=len(trajectory_with_energies)
        )
    
    
    def _build_energy_dict(
        self, 
        trajectory: List[Atoms], 
        keys: np.ndarray
    ) -> Dict[Tuple[int, ...], float]:
        """
        Build dictionary mapping displacement keys to energies.
        
        This extracts energies from ASE Atoms objects and converts to cm^-1.
        
        Parameters
        ----------
        trajectory : List[Atoms]
            Trajectory with computed energies
        keys : np.ndarray
            Displacement keys
            
        Returns
        -------
        Dict[Tuple[int, ...], float]
            Energy dictionary in cm^-1
        """
        V = {}
        for i, atoms in enumerate(trajectory):
            energy_eV = atoms.get_potential_energy()
            energy_inv_cm = energy_eV * self.EV_TO_INV_CM
            V[tuple(keys[i])] = energy_inv_cm
        
        return V
    
    def save_trajectory(self, trajectory: List[Atoms], filename: str):
        """
        Save displacement trajectory to file.
        
        Parameters
        ----------
        trajectory : List[Atoms]
            Trajectory to save
        filename : str
            Output filename (e.g., 'displacements.traj')
        """
        write(filename, trajectory)
    
    def generate_displacement_trajectory_for_forces(self) -> Tuple[List[Atoms], np.ndarray]:
        """
        Generate trajectory of displaced configurations for force-based calculation.
        
        This generates displacements required for calculating derivatives from
        analytical Cartesian forces. The displacement pattern differs from
        energy-based calculation because forces reduce the order of finite differences.
        
        Returns
        -------
        trajectory : List[Atoms]
            List of ASE Atoms objects with displaced geometries
        displacement_keys : np.ndarray
            Array of displacement combinations, shape (n_configs, n_modes)
        """
        # Get displacement keys from force calculator
        self.required_calculations_forces = self.force_calculator.get_displacement_keys()
        
        # Generate displaced geometries
        trajectory = self._create_displaced_trajectory(self.required_calculations_forces)
        
        return trajectory, self.required_calculations_forces
    
    def calculate_derivatives_from_forces(
        self,
        trajectory_with_forces: List[Atoms],
        displacement_keys: np.ndarray = None
    ) -> AnharmonicDerivativesResult:
        """
        Calculate T and U derivative tensors from trajectory with analytical forces.
        
        This method uses analytical Cartesian forces instead of energies, which:
        - Requires one less order of finite differentiation
        - Typically provides better numerical accuracy
        - Leverages analytical gradients from quantum chemistry methods
        
        Parameters
        ----------
        trajectory_with_forces : List[Atoms]
            List of Atoms objects with forces attached (from get_forces())
        displacement_keys : np.ndarray, optional
            Displacement keys corresponding to trajectory. If None, uses keys from
            generate_displacement_trajectory_for_forces()
            
        Returns
        -------
        AnharmonicDerivativesResult
            Result object containing T, U tensors and related data
            
        Examples
        --------
        >>> # Generate displacement trajectory
        >>> solver = AnharmonicDerivativesSolver(config)
        >>> trajectory, keys = solver.generate_displacement_trajectory_for_forces()
        >>> 
        >>> # Compute forces for each displaced geometry
        >>> for atoms in trajectory:
        >>>     atoms.calc = my_calculator
        >>>     atoms.get_forces()  # Computes and caches forces
        >>> 
        >>> # Calculate derivatives from forces
        >>> result = solver.calculate_derivatives_from_forces(trajectory)
        """
        if displacement_keys is None:
            if self.required_calculations_forces is None:
                raise ValueError(
                    "No displacement keys provided. Either pass displacement_keys "
                    "or call generate_displacement_trajectory_for_forces() first."
                )
            displacement_keys = self.required_calculations_forces
        
        # Build force dictionary
        F = self._build_force_dict(trajectory_with_forces, displacement_keys)
        
        # Calculate derivatives using force calculator
        print("Calculating derivatives from forces...")
        derivatives = self.force_calculator.calculate_derivatives(F)
        print("Finished calculating derivatives")
        
        # Convert derivatives from eV to cm^-1
        # The force-based method computes derivatives in eV units
        # (since ASE forces are in eV/Angstrom)
        T = derivatives['order_3'] * self.EV_TO_INV_CM
        U = derivatives['order_4'] * self.EV_TO_INV_CM
        
        # Build energy dictionary for compatibility (using V=0 as placeholder)
        V = {tuple(key): 0.0 for key in displacement_keys}
        
        return AnharmonicDerivativesResult(
            T=T,
            U=U,
            V=V,
            required_calculations=displacement_keys,
            number_of_calculations=len(trajectory_with_forces)
        )
    
    def _build_force_dict(
        self,
        trajectory: List[Atoms],
        keys: np.ndarray
    ) -> Dict[Tuple[int, ...], np.ndarray]:
        """
        Build dictionary mapping displacement keys to Cartesian force arrays.
        
        This extracts forces from ASE Atoms objects and converts units.
        Forces are kept in eV/Angstrom (ASE default).
        
        Parameters
        ----------
        trajectory : List[Atoms]
            Trajectory with computed forces
        keys : np.ndarray
            Displacement keys
            
        Returns
        -------
        Dict[Tuple[int, ...], np.ndarray]
            Force dictionary with Cartesian forces in eV/Angstrom
        """
        F = {}
        for i, atoms in enumerate(trajectory):
            # Get forces for unconstrained atoms only
            forces = atoms.get_forces()[self.config.unconstrained_index]
            F[tuple(keys[i])] = forces
        
        return F

    # ------------------------------------------------------------------
    # Dipole derivatives (IR intensities, on request only)
    # ------------------------------------------------------------------
    def get_dipole_displacement_keys(self) -> np.ndarray:
        """
        Displacement keys at which dipoles are needed for intensities.

        For the energy route (coupling >= 2) and the force route these keys are
        a subset of the force-field trajectory, so no extra electronic
        structure calculations are needed; see missing_dipole_keys().
        """
        return self.dipole_calculator.get_displacement_keys()

    def missing_dipole_keys(self, available_keys: np.ndarray) -> np.ndarray:
        """Dipole keys that are not part of ``available_keys``."""
        have = {tuple(k) for k in np.asarray(available_keys)}
        need = self.get_dipole_displacement_keys()
        missing = [k for k in need if tuple(k) not in have]
        return np.array(missing, dtype=int).reshape(-1, need.shape[1])

    def calculate_dipole_derivatives(
        self,
        trajectory_with_dipoles: List[Atoms],
        displacement_keys: np.ndarray,
    ) -> DipoleDerivativesResult:
        """
        Dipole derivatives from a trajectory whose structures carry dipoles.

        Only structures whose key is needed are queried, via
        atoms.get_dipole_moment() (ASE units: e*Angstrom).

        Parameters
        ----------
        trajectory_with_dipoles : List[Atoms]
        displacement_keys : np.ndarray
            Keys corresponding to the trajectory, shape (n_configs, n_modes)
        """
        need = {tuple(k) for k in self.get_dipole_displacement_keys()}
        dipoles = {}
        for atoms, key in zip(trajectory_with_dipoles, displacement_keys):
            key = tuple(int(x) for x in key)
            if key in need and key not in dipoles:
                try:
                    dipoles[key] = np.asarray(atoms.get_dipole_moment(), float)
                except Exception as err:  # PropertyNotImplementedError etc.
                    raise RuntimeError(
                        "IR intensities need dipole moments, but the calculator / "
                        "stored structures do not provide them "
                        f"({type(err).__name__}: {err}). Use a calculator that "
                        "implements 'dipole' and run with intensities=True."
                    ) from err
        return self.dipole_calculator.calculate(dipoles)

