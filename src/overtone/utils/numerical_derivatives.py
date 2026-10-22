"""
General numerical derivatives calculator using finite differences.

This module provides a general-purpose class for calculating higher-order
numerical derivatives of arbitrary functions using finite difference methods.

It also provides a specialized calculator that computes normal mode derivatives
from analytical Cartesian forces, which is more efficient than using numerical
derivatives of energies.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from overtone.utils.stencils import get_stencil


def _symmetrize_iij(T: np.ndarray, active_indices: List[int]) -> None:
    """
    Fill all index permutations of the computed T[i,i,j] elements.

    With coupling=2 only T[i,i,j] is evaluated, but the VPT2 expressions
    also access the same constant as T[i,j,i] and T[i,j,j] (the k = i, j
    terms of the three-index sum, and the DCPT2 single-pole terms). Without
    this, those two-mode contributions silently become zero.
    """
    for i in active_indices:
        for j in active_indices:
            if i != j:
                T[i, j, i] = T[j, i, i] = T[i, i, j]


class NumericalDerivativesCalculator:
    """
    General-purpose calculator for higher-order numerical derivatives.
    
    This class computes derivatives of a function f(x) with respect to
    multiple variables using finite difference stencils. It can handle:
    - Arbitrary order derivatives (3rd, 4th, or higher)
    - Mixed partial derivatives
    - Different coupling schemes
    - Various finite difference stencils
    
    The function values are stored in a dictionary keyed by displacement
    tuples, allowing efficient lookup during recursive differentiation.
    
    Parameters
    ----------
    n_variables : int
        Number of variables in the function
    delta : float
        Step size for finite differences
    stencil : int, optional
        Stencil order (0, 1, or 2), default: 1
        
    Examples
    --------
    >>> # Calculate 3rd derivative of f(x,y,z) with respect to x, y, z
    >>> calc = NumericalDerivativesCalculator(n_variables=3, delta=0.1)
    >>> # Define which derivatives to calculate
    >>> calc.setup_derivative_indices(
    ...     active_indices=[0, 1, 2],
    ...     derivative_orders=[3, 4],
    ...     coupling=3
    ... )
    >>> # Get displacement keys
    >>> keys = calc.get_displacement_keys()
    >>> # Evaluate function at each displacement
    >>> values = {tuple(key): f(*displacement_to_coords(key)) for key in keys}
    >>> # Calculate derivatives
    >>> derivatives = calc.calculate_derivatives(values)
    """
    
    def __init__(self, n_variables: int, delta: float, stencil: int = 1):
        """
        Initialize the numerical derivatives calculator.
        
        Parameters
        ----------
        n_variables : int
            Number of variables
        delta : float
            Step size for finite differences
        stencil : int, optional
            Stencil order (0, 1, or 2), default: 1
        """
        self.n_variables = n_variables
        self.delta = delta
        self.stencil = stencil
        
        # Get finite difference stencil
        self.displacements, self.coefficients, self.dfac = get_stencil(stencil)
        
        # Storage for derivative specifications
        self.active_indices = None
        self.derivative_orders = None
        self.coupling = None
        self.required_displacements = []
    
    def setup_derivative_indices(
        self,
        active_indices: List[int],
        derivative_orders: List[int],
        coupling: int = 3
    ):
        """
        Setup which derivatives to calculate.
        
        Parameters
        ----------
        active_indices : List[int]
            Indices of active variables (e.g., [6, 7, 8] for modes 6, 7, 8)
        derivative_orders : List[int]
            Orders of derivatives to calculate (e.g., [3, 4] for T and U tensors)
        coupling : int, optional
            Coupling level:
            - 3: Full coupling (all mixed derivatives)
            - 2: Reduced coupling (limited mixed derivatives)
            - 1: Diagonal only
            Default: 3
        """
        self.active_indices = active_indices
        self.derivative_orders = derivative_orders
        self.coupling = coupling
        
        # Determine required displacements
        self._determine_required_displacements()
    
    def _determine_required_displacements(self):
        """
        Determine all unique displacement combinations needed for derivatives.
        
        This performs a "dry run" through the recursive differentiation to
        find all displacement keys that will be needed.
        """
        self.required_displacements = []
        arr = np.zeros(self.n_variables, dtype=int)
        
        # For each derivative order
        for order in self.derivative_orders:
            if order == 4:
                # Fourth-order: need f_iijj terms
                for i in self.active_indices:
                    for j in self.active_indices:
                        self._dry_recursive_differentiation([i, i, j, j], 4, arr)
                        
            elif order == 3:
                # Third-order: depends on coupling
                if self.coupling >= 3:
                    # Full coupling: all T_ijk
                    for i in self.active_indices:
                        for j in self.active_indices:
                            for k in self.active_indices:
                                self._dry_recursive_differentiation([i, j, k], 3, arr)
                                
                elif self.coupling == 2:
                    # Reduced coupling: T_iij and T_jji
                    for i in self.active_indices:
                        for j in self.active_indices:
                            self._dry_recursive_differentiation([i, i, j], 3, arr)
                            self._dry_recursive_differentiation([j, j, i], 3, arr)
                            
                elif self.coupling == 1:
                    # Diagonal only: T_iii
                    for i in self.active_indices:
                        self._dry_recursive_differentiation([i, i, i], 3, arr)
        
        # Convert to array and remove duplicates
        self.required_displacements = np.array(self.required_displacements)
        self.required_displacements = np.unique(self.required_displacements, axis=0)
    
    def _dry_recursive_differentiation(
        self,
        indices: List[int],
        order: int,
        arr: np.ndarray
    ):
        """
        Recursively determine displacement combinations (dry run).
        
        Parameters
        ----------
        indices : List[int]
            Variable indices for this derivative term
        order : int
            Order of differentiation (counts down to 1)
        arr : np.ndarray
            Current displacement array (modified in place)
        """
        prevd = 0
        for d in self.displacements:
            arr[indices[order - 1]] += d - prevd
            prevd = d
            
            if order != 1:
                self._dry_recursive_differentiation(indices, order - 1, arr)
            else:
                # Base case: record this displacement combination
                self.required_displacements.append(arr.copy())
        
        # Reset displacement for this variable
        arr[indices[order - 1]] -= prevd
    
    def keys_for(self, index_lists: List[List[int]]) -> np.ndarray:
        """
        Displacement keys needed for an arbitrary set of mixed derivatives.

        Parameters
        ----------
        index_lists : list of index lists, e.g. [[6], [6, 7]] for
            d/dq6 and d^2/dq6 dq7

        Returns
        -------
        np.ndarray
            Unique displacement keys, shape (n_keys, n_variables)
        """
        saved = self.required_displacements
        self.required_displacements = []
        arr = np.zeros(self.n_variables, dtype=int)
        for idx in index_lists:
            self._dry_recursive_differentiation(list(idx), len(idx), arr)
        keys = np.unique(np.array(self.required_displacements), axis=0)
        self.required_displacements = saved
        return keys

    def derivative(self, indices: List[int], values: Dict[Tuple[int, ...], np.ndarray]):
        """
        Mixed derivative d^n f / dx_i1 ... dx_in by nested finite differences.

        ``values`` may hold scalars or arrays (e.g. dipole vectors); the
        derivative has the same shape as the values.
        """
        arr = np.zeros(self.n_variables, dtype=int)
        return self._recursive_differentiation(list(indices), len(indices), arr, values)

    def get_displacement_keys(self) -> np.ndarray:
        """
        Get the displacement keys needed for derivative calculations.
        
        Returns
        -------
        np.ndarray
            Array of displacement keys, shape (n_displacements, n_variables)
            Each row represents displacement multiples for each variable
            
        Raises
        ------
        ValueError
            If setup_derivative_indices() has not been called
        """
        if len(self.required_displacements) == 0:
            raise ValueError(
                "No displacement keys available. Call setup_derivative_indices() first."
            )
        return self.required_displacements
    
    def calculate_derivatives(
        self,
        function_values: Dict[Tuple[int, ...], float]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate derivatives from function values.
        
        Parameters
        ----------
        function_values : Dict[Tuple[int, ...], float]
            Dictionary mapping displacement keys to function values
            Keys are tuples of integers (displacement multiples)
            Values are function evaluations at those displacements
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with derivative tensors:
            - 'order_3': 3rd-order derivatives, shape (n_var, n_var, n_var)
            - 'order_4': 4th-order derivatives, shape (n_var, n_var, n_var, n_var)
            - Other orders as specified
            
        Raises
        ------
        ValueError
            If setup_derivative_indices() has not been called
        """
        if self.active_indices is None:
            raise ValueError("Call setup_derivative_indices() first")
        
        results = {}
        
        # Calculate each order of derivatives
        if 3 in self.derivative_orders:
            results['order_3'] = self._calculate_3rd_order(function_values)
        
        if 4 in self.derivative_orders:
            results['order_4'] = self._calculate_4th_order(function_values)
        
        return results
    
    def _calculate_3rd_order(
        self,
        V: Dict[Tuple[int, ...], float]
    ) -> np.ndarray:
        """
        Calculate 3rd-order derivatives.
        
        Parameters
        ----------
        V : Dict[Tuple[int, ...], float]
            Function values dictionary
            
        Returns
        -------
        np.ndarray
            3rd-order derivative tensor, shape (n_var, n_var, n_var)
        """
        T = np.zeros((self.n_variables, self.n_variables, self.n_variables))
        arr = np.zeros(self.n_variables, dtype=int)
        
        if self.coupling >= 3:
            # Full coupling
            for i in self.active_indices:
                for j in self.active_indices:
                    for k in self.active_indices:
                        T[i, j, k] = self._recursive_differentiation(
                            [i, j, k], 3, arr, V
                        )
                        
        elif self.coupling == 2:
            # Reduced coupling
            for i in self.active_indices:
                for j in self.active_indices:
                    T[i, i, j] = self._recursive_differentiation(
                        [i, i, j], 3, arr, V
                    )
                    T[j, j, i] = self._recursive_differentiation(
                        [j, j, i], 3, arr, V
                    )
            _symmetrize_iij(T, self.active_indices)
                    
        elif self.coupling == 1:
            # Diagonal only
            for i in self.active_indices:
                T[i, i, i] = self._recursive_differentiation(
                    [i, i, i], 3, arr, V
                )
        
        return T
    
    def _calculate_4th_order(
        self,
        V: Dict[Tuple[int, ...], float]
    ) -> np.ndarray:
        """
        Calculate 4th-order derivatives.
        
        Parameters
        ----------
        V : Dict[Tuple[int, ...], float]
            Function values dictionary
            
        Returns
        -------
        np.ndarray
            4th-order derivative tensor, shape (n_var, n_var, n_var, n_var)
        """
        U = np.zeros((self.n_variables, self.n_variables, self.n_variables, self.n_variables))
        arr = np.zeros(self.n_variables, dtype=int)
        
        # Calculate U[i,i,j,j] terms
        for i in self.active_indices:
            for j in self.active_indices:
                U[i, i, j, j] = self._recursive_differentiation(
                    [i, i, j, j], 4, arr, V
                )
        
        return U
    
    def _recursive_differentiation(
        self,
        indices: List[int],
        order: int,
        arr: np.ndarray,
        V: Dict[Tuple[int, ...], float]
    ) -> float:
        """
        Recursively calculate finite difference derivative.
        
        This implements a general n-th order finite difference formula.
        
        Parameters
        ----------
        indices : List[int]
            Variable indices for this derivative term
        order : int
            Order of differentiation (counts down to 1)
        arr : np.ndarray
            Current displacement array (modified in place)
        V : Dict[Tuple[int, ...], float]
            Function values dictionary
            
        Returns
        -------
        float
            Derivative value
        """
        ans = 0.0
        prevd = 0
        
        for d, a in zip(self.displacements, self.coefficients):
            arr[indices[order - 1]] += d - prevd
            prevd = d
            
            if order != 1:
                # Recursive case: differentiate again
                ans += a * self._recursive_differentiation(indices, order - 1, arr, V)
            else:
                # Base case: look up function value
                ans += a * V[tuple(arr)]
        
        # Reset displacement for this variable
        arr[indices[order - 1]] -= prevd
        
        # Apply finite difference formula
        return ans / (self.dfac * self.delta)
    
    def evaluate_function_at_displacements(
        self,
        function: Callable,
        base_point: np.ndarray,
        displacement_to_coords: Callable[[np.ndarray, np.ndarray], np.ndarray] = None
    ) -> Dict[Tuple[int, ...], float]:
        """
        Evaluate function at all required displacement points.
        
        This is a convenience method for simple cases where you can directly
        evaluate the function.
        
        Parameters
        ----------
        function : Callable
            Function to evaluate, should accept array of coordinates
        base_point : np.ndarray
            Base point coordinates
        displacement_to_coords : Callable, optional
            Function to convert displacement keys to coordinate perturbations
            If None, assumes displacements are directly in coordinate space
            Signature: (displacement_key, base_point) -> coords
            
        Returns
        -------
        Dict[Tuple[int, ...], float]
            Dictionary mapping displacement keys to function values
        """
        def default_displacement_to_coords(key, base):
            return base + key * self.delta
        
        if displacement_to_coords is None:
            displacement_to_coords = default_displacement_to_coords
        
        values = {}
        for key in self.required_displacements:
            coords = displacement_to_coords(key, base_point)
            values[tuple(key)] = function(coords)
        
        return values


class CartesianForceDerivativesCalculator:
    """
    Calculator for normal mode derivatives using analytical Cartesian forces.
    
    This class computes 3rd and 4th order derivatives in normal mode coordinates
    by transforming analytical Cartesian forces to normal mode gradients and then
    applying finite differences. This is more efficient and accurate than numerical
    derivatives of energies since:
    - Uses analytical forces (typically more accurate than numerical gradients)
    - Requires one less order of finite differentiation
    - Better numerical convergence
    
    The workflow is:
    1. Evaluate Cartesian forces F(R) at displaced geometries along normal modes
    2. Transform forces to normal mode gradients: ∂V/∂Q = -Σ F_a × (∂R_a/∂Q)
    3. Apply finite differences to gradients to get 3rd/4th derivatives
    
    Parameters
    ----------
    n_variables : int
        Number of normal mode coordinates
    delta : float
        Step size for finite differences in reduced normal coordinates
    modes : np.ndarray
        Normal mode eigenvectors (columns), shape (3*n_atoms, n_modes)
    frequencies : np.ndarray
        Harmonic frequencies in cm^-1 (required for reduced normal coordinates)
        Shape (n_modes,) or (n_modes, 1)
    masses : np.ndarray
        Square root of atomic masses in amu, repeated for each Cartesian direction
        Shape (3*n_atoms,) or (3*n_atoms, 1)
    stencil : int, optional
        Stencil order (0, 1, or 2), default: 1
        
    Examples
    --------
    >>> # Calculate 3rd/4th derivatives using forces
    >>> calc = CartesianForceDerivativesCalculator(
    ...     n_variables=9,
    ...     delta=0.1,
    ...     modes=normal_modes,
    ...     masses=sqrt_masses,
    ...     frequencies=harmonic_freqs
    ... )
    >>> calc.setup_derivative_indices(
    ...     active_indices=[6, 7, 8],
    ...     derivative_orders=[3, 4],
    ...     coupling=3
    ... )
    >>> # Get displacement keys
    >>> keys = calc.get_displacement_keys()
    >>> # Evaluate forces at each displacement
    >>> forces = {tuple(key): atoms.get_forces() for key, atoms in zip(keys, trajectory)}
    >>> # Calculate derivatives
    >>> derivatives = calc.calculate_derivatives(forces)
    """
    
    def __init__(
        self,
        n_variables: int,
        delta: float,
        modes: np.ndarray,
        masses: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        stencil: int = 1
    ):
        """
        Initialize the force-based derivatives calculator.
        
        Parameters
        ----------
        n_variables : int
            Number of normal mode coordinates
        delta : float
            Step size for finite differences
        modes : np.ndarray
            Normal mode eigenvectors (columns)
        masses : np.ndarray
            Square root of atomic masses in amu
        frequencies : Optional[np.ndarray]
            Harmonic frequencies in cm^-1 (required for reduced normal coordinates)
        stencil : int, optional
            Stencil order (0, 1, or 2), default: 1
        """
        self.n_variables = n_variables
        self.delta = delta
        self.stencil = stencil
        self.modes = modes
        
        # Ensure masses is a column vector
        if masses.ndim == 1:
            self.masses = masses.reshape(-1, 1)
        else:
            self.masses = masses
        
        # Get finite difference stencil
        self.displacements, self.coefficients, self.dfac = get_stencil(stencil)
        
        # Precompute Jacobian matrix for force transformation
        # For reduced normal coordinates q: ∂R_a/∂q_i = (L_ai / masses_a) * (1/sqrt(omega_i))
        # where omega_i = 2π√(c·freq_i)·√(amu)·Å_to_m
        if frequencies is not None:
            # Physical constants for conversion
            C_M_PER_S = 2.99792458e8          # Speed of light in m/s
            H_J_S = 6.62607015e-34       # Planck's constant in J·s
            INV_CM_TO_INV_M = 1e2        # Conversion factor
            AMU_TO_KG = 1.66053906660e-27      # Atomic mass unit to kg
            A_TO_M = 1e-10               # Angstrom to meter
            
            # Compute omega = 2π√(c/h) * sqrt(freq) * sqrt(amu) * Å_to_m
            freq_array = frequencies.reshape(-1, 1) if frequencies.ndim == 1 else frequencies
            # Add small epsilon to avoid division by zero for inactive modes (freq=0)
            EPSILON = 1e-20
            omega = (
                2 * np.pi
                * np.sqrt(C_M_PER_S / H_J_S)
                * np.sqrt(freq_array * INV_CM_TO_INV_M + EPSILON)
                * np.sqrt(AMU_TO_KG)
                * A_TO_M
            )
            
            # Jacobian: ∂R/∂q = (L / masses) / sqrt(omega)
            # Broadcasting: modes (3N, M) / masses (3N, 1) / omega.T (1, M) = (3N, M)
            jacobian = self.modes / self.masses / omega.T
            
            # Set Jacobian columns to zero for inactive modes (freq ≈ 0)
            # This prevents NaN propagation for modes that are never used
            inactive_mask = (freq_array.flatten() < 1e-6)
            jacobian[:, inactive_mask] = 0.0
            self.jacobian = jacobian
        else:
            # Fallback for normal (not reduced) coordinates: ∂R_a/∂Q_i = L_ai / masses_a
            self.jacobian = self.modes / self.masses  # Shape: (3*n_atoms, n_modes)
        
        # Storage for derivative specifications
        self.active_indices = None
        self.derivative_orders = None
        self.coupling = None
        self.required_displacements = []
    
    def setup_derivative_indices(
        self,
        active_indices: List[int],
        derivative_orders: List[int],
        coupling: int = 3
    ):
        """
        Setup which derivatives to calculate.
        
        Parameters
        ----------
        active_indices : List[int]
            Indices of active normal modes
        derivative_orders : List[int]
            Orders of derivatives to calculate (e.g., [3, 4])
        coupling : int, optional
            Coupling level (1=diagonal, 2=reduced, 3=full), default: 3
        """
        self.active_indices = active_indices
        self.derivative_orders = derivative_orders
        self.coupling = coupling
        
        # Determine required displacements
        self._determine_required_displacements()
    
    def _determine_required_displacements(self):
        """
        Determine all unique displacement combinations needed.
        
        For force-based calculation, we need displacements for gradients:
        - 3rd derivatives: need gradients at 2nd-order displaced points
        - 4th derivatives: need gradients at 3rd-order displaced points
        """
        self.required_displacements = []
        arr = np.zeros(self.n_variables, dtype=int)
        
        # For each derivative order requested
        for order in self.derivative_orders:
            if order == 4:
                # Fourth-order: need gradients at f_iij displacement patterns
                for i in self.active_indices:
                    for j in self.active_indices:
                        # Calculate ∂³V/∂Q_i²∂Q_j from gradients
                        self._dry_recursive_differentiation([i, i, j], 3, arr)
                        
            elif order == 3:
                # Third-order: depends on coupling
                if self.coupling >= 3:
                    # Full coupling: all T_ijk
                    for i in self.active_indices:
                        for j in self.active_indices:
                            for k in self.active_indices:
                                self._dry_recursive_differentiation([i, j], 2, arr)
                                
                elif self.coupling == 2:
                    # Reduced coupling: T_iij and T_jji
                    for i in self.active_indices:
                        for j in self.active_indices:
                            self._dry_recursive_differentiation([i, j], 2, arr)
                            
                elif self.coupling == 1:
                    # Diagonal only: T_iii
                    for i in self.active_indices:
                        self._dry_recursive_differentiation([i, i], 2, arr)
        
        # Convert to array and remove duplicates
        self.required_displacements = np.array(self.required_displacements)
        self.required_displacements = np.unique(self.required_displacements, axis=0)
    
    def _dry_recursive_differentiation(
        self,
        indices: List[int],
        order: int,
        arr: np.ndarray
    ):
        """
        Recursively determine displacement combinations (dry run).
        
        Parameters
        ----------
        indices : List[int]
            Variable indices for this derivative term
        order : int
            Order of differentiation
        arr : np.ndarray
            Current displacement array
        """
        prevd = 0
        for d in self.displacements:
            arr[indices[order - 1]] += d - prevd
            prevd = d
            
            if order != 1:
                self._dry_recursive_differentiation(indices, order - 1, arr)
            else:
                # Base case: record this displacement
                self.required_displacements.append(arr.copy())
        
        # Reset displacement
        arr[indices[order - 1]] -= prevd
    
    def get_displacement_keys(self) -> np.ndarray:
        """
        Get the displacement keys needed for derivative calculations.
        
        Returns
        -------
        np.ndarray
            Array of displacement keys, shape (n_displacements, n_variables)
        """
        if len(self.required_displacements) == 0:
            raise ValueError(
                "No displacement keys available. Call setup_derivative_indices() first."
            )
        return self.required_displacements
    
    def _transform_forces_to_normal_gradient(
        self,
        forces_cartesian: np.ndarray
    ) -> np.ndarray:
        """
        Transform Cartesian forces to normal mode gradients.
        
        Uses chain rule: ∂V/∂Q_i = -Σ_a F_a × (∂R_a/∂Q_i)
        where ∂R_a/∂Q_i = L_ai / masses_a
        
        Parameters
        ----------
        forces_cartesian : np.ndarray
            Cartesian forces, shape (n_atoms, 3) or flattened (3*n_atoms,)
            
        Returns
        -------
        np.ndarray
            Normal mode gradients ∂V/∂Q, shape (n_modes,)
        """
        # Flatten forces if needed
        if forces_cartesian.ndim == 2:
            F = forces_cartesian.flatten().reshape(-1, 1)
        else:
            F = forces_cartesian.reshape(-1, 1)
        
        # Transform: ∂V/∂Q = -F^T @ (∂R/∂Q) = -F^T @ jacobian
        gradient_normal = -(F.T @ self.jacobian).flatten()
        
        return gradient_normal
    
    def calculate_derivatives(
        self,
        force_values: Dict[Tuple[int, ...], np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate derivatives from Cartesian forces.
        
        Parameters
        ----------
        force_values : Dict[Tuple[int, ...], np.ndarray]
            Dictionary mapping displacement keys to Cartesian force arrays
            Forces should be in shape (n_atoms, 3) or (3*n_atoms,)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with derivative tensors:
            - 'order_3': 3rd-order derivatives
            - 'order_4': 4th-order derivatives
        """
        if self.active_indices is None:
            raise ValueError("Call setup_derivative_indices() first")
        
        # Transform all forces to normal mode gradients
        gradient_values = {}
        for key, forces in force_values.items():
            gradient_values[key] = self._transform_forces_to_normal_gradient(forces)
        
        results = {}
        
        # Calculate each order of derivatives
        if 3 in self.derivative_orders:
            results['order_3'] = self._calculate_3rd_order(gradient_values)
        
        if 4 in self.derivative_orders:
            results['order_4'] = self._calculate_4th_order(gradient_values)
        
        return results
    
    def _calculate_3rd_order(
        self,
        gradients: Dict[Tuple[int, ...], np.ndarray]
    ) -> np.ndarray:
        """
        Calculate 3rd-order derivatives from normal mode gradients.
        
        Uses finite differences of gradients: ∂V/∂Q → ∂²V/∂Q² → ∂³V/∂Q³
        
        Parameters
        ----------
        gradients : Dict[Tuple[int, ...], np.ndarray]
            Normal mode gradient vectors
            
        Returns
        -------
        np.ndarray
            3rd-order derivative tensor, shape (n_var, n_var, n_var)
        """
        T = np.zeros((self.n_variables, self.n_variables, self.n_variables))
        arr = np.zeros(self.n_variables, dtype=int)
        
        if self.coupling >= 3:
            # Full coupling: all T_ijk
            for i in self.active_indices:
                for j in self.active_indices:
                    for k in self.active_indices:
                        # ∂³V/∂Q_i∂Q_j∂Q_k = ∂/∂Q_j∂Q_k (∂V/∂Q_i)
                        T[i, j, k] = self._recursive_gradient_differentiation(
                            gradient_index=i,
                            diff_indices=[j, k],
                            order=2,
                            arr=arr,
                            gradients=gradients
                        )
                        
        elif self.coupling == 2:
            # Reduced coupling
            for i in self.active_indices:
                for j in self.active_indices:
                    T[i, i, j] = self._recursive_gradient_differentiation(
                        gradient_index=i,
                        diff_indices=[i, j],
                        order=2,
                        arr=arr,
                        gradients=gradients
                    )
                    T[j, j, i] = self._recursive_gradient_differentiation(
                        gradient_index=j,
                        diff_indices=[j, i],
                        order=2,
                        arr=arr,
                        gradients=gradients
                    )
                    _symmetrize_iij(T, self.active_indices)
            
        elif self.coupling == 1:
            # Diagonal only
            for i in self.active_indices:
                T[i, i, i] = self._recursive_gradient_differentiation(
                    gradient_index=i,
                    diff_indices=[i, i],
                    order=2,
                    arr=arr,
                    gradients=gradients
                )
        
        return T
    
    def _calculate_4th_order(
        self,
        gradients: Dict[Tuple[int, ...], np.ndarray]
    ) -> np.ndarray:
        """
        Calculate 4th-order derivatives from normal mode gradients.
        
        Uses finite differences: ∂V/∂Q → ∂²V/∂Q² → ∂³V/∂Q³ → ∂⁴V/∂Q⁴
        
        Parameters
        ----------
        gradients : Dict[Tuple[int, ...], np.ndarray]
            Normal mode gradient vectors
            
        Returns
        -------
        np.ndarray
            4th-order derivative tensor, shape (n_var, n_var, n_var, n_var)
        """
        U = np.zeros((self.n_variables, self.n_variables, 
                      self.n_variables, self.n_variables))
        arr = np.zeros(self.n_variables, dtype=int)
        
        # Calculate U[i,i,j,j] terms
        for i in self.active_indices:
            for j in self.active_indices:
                # ∂⁴V/∂Q_i²∂Q_j² = ∂/∂Q_i∂Q_j² (∂V/∂Q_i)
                U[i, i, j, j] = self._recursive_gradient_differentiation(
                    gradient_index=i,
                    diff_indices=[i, j, j],
                    order=3,
                    arr=arr,
                    gradients=gradients
                )
        
        return U
    
    def _recursive_gradient_differentiation(
        self,
        gradient_index: int,
        diff_indices: List[int],
        order: int,
        arr: np.ndarray,
        gradients: Dict[Tuple[int, ...], np.ndarray]
    ) -> float:
        """
        Recursively calculate finite difference derivative of gradient component.
        
        Computes: ∂^n/∂Q_i1...∂Q_in (∂V/∂Q_gradient_index)
        
        Parameters
        ----------
        gradient_index : int
            Which component of gradient to differentiate
        diff_indices : List[int]
            Variables to differentiate with respect to
        order : int
            Order of differentiation (counts down to 1)
        arr : np.ndarray
            Current displacement array
        gradients : Dict[Tuple[int, ...], np.ndarray]
            Gradient vectors at different displacements
            
        Returns
        -------
        float
            Derivative value
        """
        ans = 0.0
        prevd = 0
        
        for d, a in zip(self.displacements, self.coefficients):
            arr[diff_indices[order - 1]] += d - prevd
            prevd = d
            
            if order != 1:
                # Recursive case: differentiate again
                ans += a * self._recursive_gradient_differentiation(
                    gradient_index, diff_indices, order - 1, arr, gradients
                )
            else:
                # Base case: look up gradient component
                gradient_vector = gradients[tuple(arr)]
                ans += a * gradient_vector[gradient_index]
        
        # Reset displacement
        arr[diff_indices[order - 1]] -= prevd
        
        # Apply finite difference formula
        return ans / (self.dfac * self.delta)
