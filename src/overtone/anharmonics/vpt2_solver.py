import warnings

import numpy as np
from typing import List, Optional, Tuple, Union

from overtone.anharmonics.vpt2_config import VPT2Config
from overtone.anharmonics.vpt2_result import VPT2Result
from overtone.anharmonics.derivatives_result import AnharmonicDerivativesResult
from overtone.anharmonics.dipole_derivatives import DipoleDerivativesResult
from overtone.anharmonics.transitions import TransitionCalculator, VPT2Transition

class VPT2Solver:
    """
    Solver for anharmonic vibrational frequencies using VPT2 theory.

    This class implements various VPT2 methods for calculating anharmonic
    corrections given third-order (T) and fourth-order (U) force constants.

    Attributes
    ----------
    config : VPT2Config
        Configuration object containing frequencies and mode information
    derivatives : AnharmonicDerivativesResult
        Result object containing T and U force constants
    T : np.ndarray
        Third-order force constants, shape (len_modes, len_modes, len_modes)
    U : np.ndarray
        Fourth-order force constants, shape (len_modes, len_modes, len_modes, len_modes)
    """

    # Physical constants
    c_cm_per_s = 2.99792458e10  # Speed of light in cm/s
    c_m_per_s = 2.99792458e8  # Speed of light in m/s
    h_J_s = 6.62607015e-34  # Planck's constant in J·s
    amu_to_kg = 1.66053906660e-27  # amu to kg conversion
    A_to_m = 1e-10  # Angstrom to meter conversion

    # VPT2 scaling constants (from perturbation theory)
    FOURTH_ORDER_DIAGONAL_SCALE = 1.0 / 16.0
    FOURTH_ORDER_OFFDIAG_SCALE = 0.25
    THIRD_ORDER_DIAGONAL_SCALE = 1.0 / 16.0
    THIRD_ORDER_OFFDIAG_SCALE = 0.25
    THIRD_ORDER_TRIPLE_SCALE = 0.5
    DEGENERACY_SELF_TERM_SCALE = 5.0 / 3.0
    FUNDAMENTAL_DIAGONAL_FACTOR = 2.0
    FUNDAMENTAL_OFFDIAG_FACTOR = 0.5

    def __init__(
        self,
        config: VPT2Config,
        derivatives: Union[AnharmonicDerivativesResult, Tuple[np.ndarray, np.ndarray]],
        dipole_derivatives: Optional[DipoleDerivativesResult] = None,
    ):
        """
        Initialize VPT2Solver with configuration and force constants.

        Parameters
        ----------
        config : VPT2Config
            Configuration object with frequencies and mode information
        derivatives : AnharmonicDerivativesResult or tuple of (T, U)
            Either a result object from AnharmonicDerivativesSolver containing
            T and U tensors, or a tuple of (T, U) arrays directly.
            T: Third-order force constants in cm^-1, shape (len_modes, len_modes, len_modes)
            U: Fourth-order force constants in cm^-1, shape (len_modes, len_modes, len_modes, len_modes)
        dipole_derivatives : DipoleDerivativesResult, optional
            Only needed for IR intensities. Without it, transitions are
            listed with energies only.
        """
        self.config = config
        self.dipole_derivatives = dipole_derivatives
        
        # Accept either AnharmonicDerivativesResult or tuple of (T, U)
        if isinstance(derivatives, AnharmonicDerivativesResult):
            self.derivatives = derivatives
            self.T = derivatives.T
            self.U = derivatives.U
        else:
            # Legacy support: accept tuple of (T, U)
            self.derivatives = None
            self.T, self.U = derivatives
        self.X = None
        self.anharm_freq = None

        # Convenience attributes
        self.freq = config.freq
        self.active_modes = config.active_modes
        self.len_modes = config.len_modes
        self.min_mode_idx = config.min_mode_idx
        self.max_mode_idx = config.max_mode_idx
        self.coupling = config.coupling

        self._check_modes()

        # Pre-compute frequently used values
        self._freq_squared = self.freq**2
        self._active_freq = self.freq[self.active_modes]
        self._active_freq_sq = self._freq_squared[self.active_modes]

    # ========================================================================
    # Helper Methods
    # ========================================================================

    #: harmonic frequencies closer than this (cm^-1) count as degenerate
    DEGENERACY_THRESHOLD = 1.0

    def _check_modes(self):
        """Refuse imaginary/zero active modes, warn about degenerate ones."""
        w = np.asarray(self.freq)[self.active_modes]
        bad = [m for m, f in zip(self.active_modes, w) if not f > 1e-3]
        if bad:
            raise ValueError(
                f"Active modes {bad} have zero or imaginary harmonic frequencies. "
                "VPT2 needs a minimum: tighten the geometry optimisation, check "
                "the harmonic step, or exclude these modes via active_modes."
            )
        pairs = [
            (a, b)
            for n, a in enumerate(self.active_modes)
            for b in self.active_modes[n + 1 :]
            if abs(self.freq[a] - self.freq[b]) < self.DEGENERACY_THRESHOLD
        ]
        if pairs:
            warnings.warn(
                f"Degenerate harmonic modes {pairs} (within "
                f"{self.DEGENERACY_THRESHOLD} cm^-1). The VPT2 expressions used "
                "here are those for non-degenerate modes (asymmetric tops); "
                "results for degenerate modes of linear and symmetric/spherical "
                "tops are approximate."
            )

    def _use_coriolis(self, requested: bool) -> bool:
        return bool(requested and self.config.coriolis and self.config.atoms is not None)

    @staticmethod
    def _energy_denominator_3mode(w_i: float, w_j: float, w_k: float) -> float:
        """
        Calculate the energy denominator for 3-mode coupling.

        D_ijk = (ω_i + ω_j - ω_k)(ω_i + ω_j + ω_k)(ω_i - ω_j + ω_k)(ω_i - ω_j - ω_k)

        Parameters
        ----------
        w_i, w_j, w_k : float
            Frequencies in cm^-1

        Returns
        -------
        float
            Energy denominator
        """
        return (
            (w_i + w_j - w_k)
            * (w_i + w_j + w_k)
            * (w_i - w_j + w_k)
            * (w_i - w_j - w_k)
        )

    @staticmethod
    def _resonance_corrected_term(
        sign_energy: float, sqrt_energy: float, coupling_term: float
    ) -> float:
        """
        Calculate resonance-corrected contribution for degeneracy-corrected VPT2.

        Returns: sign(sign_energy) * (sqrt((sqrt_energy/2)² + C) - |sqrt_energy/2|)

        In degeneracy-corrected VPT2, the sign and sqrt can use different energy expressions.

        Parameters
        ----------
        sign_energy : float
            Energy difference used for determining the sign
        sqrt_energy : float
            Energy difference used in the sqrt expression
        coupling_term : float
            Coupling term (numerator contribution)

        Returns
        -------
        float
            Resonance-corrected value
        """
        half_energy = 0.5 * sqrt_energy
        return np.sign(sign_energy) * (
            np.sqrt(half_energy**2 + coupling_term) - np.abs(half_energy)
        )

    def _calculate_diagonal_third_order(self, i: int) -> float:
        """
        Calculate third-order contribution to X[i,i].

        Parameters
        ----------
        i : int
            Mode index

        Returns
        -------
        float
            Third-order contribution
        """
        w_i_sq = self._freq_squared[i]

        # Vectorized calculation over all active modes
        numerator = (8 * w_i_sq - 3 * self._active_freq_sq) * self.T[
            i, i, self.active_modes
        ] ** 2
        denominator = self._active_freq * (4 * w_i_sq - self._active_freq_sq)

        return (
            -np.sum(numerator / denominator) * self.THIRD_ORDER_DIAGONAL_SCALE
        )

    def _calculate_offdiag_third_order_2mode(self, i: int, j: int) -> float:
        """
        Calculate third-order contribution to X[i,j] from 2-mode terms.

        Parameters
        ----------
        i, j : int
            Mode indices

        Returns
        -------
        float
            2-mode third-order contribution
        """
        coupling_product = (
            self.T[i, i, self.active_modes] * self.T[j, j, self.active_modes]
        )
        return (
            -np.sum(coupling_product / self._active_freq)
            * self.THIRD_ORDER_OFFDIAG_SCALE
        )

    def _calculate_offdiag_third_order_3mode(self, i: int, j: int) -> float:
        """
        Calculate third-order contribution to X[i,j] from 3-mode coupling.

        Parameters
        ----------
        i, j : int
            Mode indices

        Returns
        -------
        float
            3-mode third-order contribution
        """
        w_i, w_j = self.freq[i], self.freq[j]
        w_i_sq, w_j_sq = self._freq_squared[i], self._freq_squared[j]
        w_m = self._active_freq
        w_m_sq = self._active_freq_sq

        # Vectorized calculation over all active modes
        dijk = (
            (w_i + w_j - w_m)
            * (w_i + w_j + w_m)
            * (w_i - w_j + w_m)
            * (w_i - w_j - w_m)
        )

        numerator = (
            w_m
            * (w_i_sq + w_j_sq - w_m_sq)
            * self.T[i, j, self.active_modes] ** 2
        )
        return np.sum(numerator / (2 * dijk))

    # ========================================================================
    # Main Calculation Methods
    # ========================================================================

    def calculate(self) -> VPT2Result:
        """
        Calculate anharmonicity constants, fundamentals, overtones and
        combination bands (and IR intensities if dipole derivatives were
        given) with the method set in the configuration.

        Returns
        -------
        VPT2Result
        """
        ctype = self.config.method
        if ctype == "standard":
            result = self.calculate_anharmonic(add_coriolis=True)
        elif ctype == "split":
            result = self.calculate_anharmonic_split()
        elif ctype == "deperturbed":
            result = self.calculate_anharmonic_deperturbed(add_coriolis=True)
        elif ctype == "gvpt2":
            result = self.calculate_anharmonic_deperturbed(add_coriolis=True)
            result.method = "gvpt2"
        elif ctype == "dcpt2":
            result = self.calculate_anharmonic_degeneracy_corrected(
                add_coriolis=True
            )
        else:
            raise ValueError(f"Unknown calculation type: {ctype}")

        result.transitions = self.calculate_transitions(
            result.X, max_quanta=self.config.max_quanta,
            variational=(ctype == "gvpt2"),
        )
        if ctype == "gvpt2":
            # fundamentals after the variational treatment
            for t in result.transitions:
                if t.kind == "fundamental":
                    result.anharm_freq[next(iter(t.quanta))] = t.energy
        return result

    def calculate_transitions(
        self, X: np.ndarray, max_quanta: int = 2, variational: bool = False
    ) -> List[VPT2Transition]:
        """
        Fundamentals, overtones and combination bands from the X matrix.

        Intensities are added only if dipole derivatives were passed to the
        solver. Transitions involved in a detected Fermi resonance are
        flagged (resonance_affected=True).

        Parameters
        ----------
        X : np.ndarray
            Anharmonicity constants
        max_quanta : int
            Highest total number of quanta (2: overtones + binary
            combinations; 3: adds 3v_i, 2v_i+v_j, v_i+v_j+v_k)
        """
        ft1, ft2 = self.resonance_check()
        mu1 = mu2 = None
        if self.dipole_derivatives is not None:
            mu1 = self.dipole_derivatives.mu1
            mu2 = self.dipole_derivatives.mu2
        calc = TransitionCalculator(
            self.freq, X, self.active_modes,
            T=self.T, mu1=mu1, mu2=mu2,
            fermi_type1=ft1, fermi_type2=ft2,
        )
        return calc.calculate(max_quanta=max_quanta, variational=variational)

    def calculate_anharmonic(self, add_coriolis: bool = True) -> VPT2Result:
        """
        Calculate anharmonicity constants using standard VPT2.

        Based on: Yu & Bowman, "Vibrational second-order perturbation theory
        using local monomer normal modes", Molecular Physics, 2015.
        Implements Eq. 10-13 from the paper.

        Parameters
        ----------
        add_coriolis : bool, optional
            Whether to include Coriolis coupling corrections (default: True)

        Returns
        -------
        VPT2Result
            Result object containing X matrix, frequencies, and optional Coriolis data
        """
        X = np.zeros((self.len_modes, self.len_modes))

        # Calculate diagonal elements
        for i in self.active_modes:
            fourth_order_term = (
                self.U[i, i, i, i] * self.FOURTH_ORDER_DIAGONAL_SCALE
            )
            third_order_term = self._calculate_diagonal_third_order(i)
            X[i, i] = fourth_order_term + third_order_term

        # Calculate off-diagonal elements
        for i in self.active_modes:
            for j in self.active_modes:
                if i == j:
                    continue

                # Fourth-order and 2-mode third-order contributions
                fourth_order = (
                    self.U[i, i, j, j]
                    * self.FOURTH_ORDER_OFFDIAG_SCALE
                )
                third_2mode = (
                    self._calculate_offdiag_third_order_2mode(i, j)
                )
                X[i, j] = fourth_order + third_2mode

                # phi_ijk^2 term. With coupling=2 only phi_iij-type constants
                # are non-zero, so this reduces to the k = i, j terms, which
                # are two-mode terms and must be kept.
                X[i, j] += self._calculate_offdiag_third_order_3mode(i, j)

        X_coriolis = None
        if self._use_coriolis(add_coriolis):
            X_coriolis = self.calculate_anharmonic_coriolis()
            X = X + X_coriolis

        anharm_freq = self.calculate_fundamental_frequencies(X)

        return VPT2Result(
            X=X,
            anharm_freq=anharm_freq,
            method="standard",
            X_coriolis=X_coriolis,
        )

    def calculate_anharmonic_split(self) -> VPT2Result:
        """
        Calculate anharmonicity constants with 3rd and 4th order contributions split.

        Based on Yu & Bowman, 2015 Eq. 10, 11, 13.
        Useful for analyzing the relative importance of different perturbative orders.

        Returns
        -------
        VPT2Result
            Result object with X_3, X_4, and X_total split out
        """
        X_3 = np.zeros((self.len_modes, self.len_modes))
        X_4 = np.zeros((self.len_modes, self.len_modes))

        # Diagonal contributions
        for i in self.active_modes:
            X_3[i, i] = self._calculate_diagonal_third_order(i)
            X_4[i, i] = self.U[i, i, i, i] * self.FOURTH_ORDER_DIAGONAL_SCALE

        # Off-diagonal contributions
        for i in self.active_modes:
            for j in self.active_modes:
                if j != i:
                    # Third-order: 2-mode and 3-mode coupling
                    X_3[i, j] = self._calculate_offdiag_third_order_2mode(i, j)
                    X_3[i, j] += self._calculate_offdiag_third_order_3mode(
                        i, j
                    )

                    # Fourth-order
                    X_4[i, j] = (
                        self.U[i, i, j, j] * self.FOURTH_ORDER_OFFDIAG_SCALE
                    )

        print("Third order:")
        print(X_3[self.min_mode_idx :, self.min_mode_idx :])
        print("Fourth order:")
        print(X_4[self.min_mode_idx :, self.min_mode_idx :])

        X = X_3 + X_4
        anharm_freq = self.calculate_fundamental_frequencies(X)

        return VPT2Result(
            X=X,
            anharm_freq=anharm_freq,
            method="split",
            X_3=X_3,
            X_4=X_4,
        )

    def calculate_anharmonic_deperturbed(
        self,
        add_coriolis: bool = True,
        threshold: Optional[float] = None,
        K1: Optional[float] = None,
        K2: Optional[float] = None,
    ) -> VPT2Result:
        """
        Calculate deperturbed (DVPT2) anharmonicity constants.

        Based on: Martin, Lee, Taylor & Francois, J. Chem. Phys. 103,
        2589 (1995), Appendix A (Eqs. A1-A9, A14-A21).

        The cubic contributions are written as partial fractions,

            X_ii:  -phi_iik^2 [ 1/(32(2w_i+w_k)) - 1/(32(2w_i-w_k)) + 1/(8 w_k) ]
            X_ij:  -phi_ijk^2/8 [ 1/(w_i+w_j+w_k) + 1/(-w_i+w_j+w_k)
                                 + 1/(w_i-w_j+w_k) - 1/(w_i+w_j-w_k) ]

        and every pole that belongs to a detected Fermi resonance is
        dropped. Each resonance is removed from *all* X elements in which
        it appears (e.g. 2w_i ~ w_k is removed from X_ii and X_ik), and X
        stays symmetric.

        The returned fundamentals of resonant modes are deperturbed values.
        Physical values require diagonalising the resonant polyad
        (Martin et al. Eqs. A10/A11, i.e. GVPT2).

        Returns
        -------
        VPT2Result
            Result object with X matrix and detected Fermi resonances
        """
        fermi_type1, fermi_type2 = self.resonance_check(
            threshold=threshold, K1=K1, K2=K2
        )
        res1 = set(fermi_type1)  # (high, low): w_high ~ 2 w_low
        res2 = set(fermi_type2)  # (high, low1, low2), low1 < low2

        def is_type2(high, a, b):
            return (high, min(a, b), max(a, b)) in res2

        w = self.freq
        T, U = self.T, self.U
        X = np.zeros((self.len_modes, self.len_modes))

        # Diagonal elements
        for i in self.active_modes:
            X[i, i] = U[i, i, i, i] / 16.0
            for k in self.active_modes:
                phi2 = T[i, i, k] ** 2
                X[i, i] -= phi2 * (
                    1.0 / (32.0 * (2 * w[i] + w[k])) + 1.0 / (8.0 * w[k])
                )
                # resonant pole 2w_i - w_k  <->  w_k ~ 2 w_i
                if k == i or (k, i) not in res1:
                    X[i, i] += phi2 / (32.0 * (2 * w[i] - w[k]))

        # Off-diagonal elements
        for i in self.active_modes:
            for j in self.active_modes:
                if j == i:
                    continue
                X[i, j] = U[i, i, j, j] / 4.0
                X[i, j] -= np.sum(
                    T[i, i, self.active_modes]
                    * T[j, j, self.active_modes]
                    / (4.0 * self._active_freq)
                )
                for k in self.active_modes:
                    phi2_8 = T[i, j, k] ** 2 / 8.0
                    if phi2_8 == 0.0:
                        continue
                    X[i, j] -= phi2_8 / (w[i] + w[j] + w[k])

                    # pole (-w_i + w_j + w_k): w_i ~ w_j + w_k
                    if k == j:
                        resonant = (i, j) in res1
                    else:
                        resonant = k != i and is_type2(i, j, k)
                    if not resonant:
                        X[i, j] -= phi2_8 / (-w[i] + w[j] + w[k])

                    # pole (w_i - w_j + w_k): w_j ~ w_i + w_k
                    if k == i:
                        resonant = (j, i) in res1
                    else:
                        resonant = k != j and is_type2(j, i, k)
                    if not resonant:
                        X[i, j] -= phi2_8 / (w[i] - w[j] + w[k])

                    # pole (w_i + w_j - w_k): w_k ~ w_i + w_j
                    resonant = k not in (i, j) and is_type2(k, i, j)
                    if not resonant:
                        X[i, j] += phi2_8 / (w[i] + w[j] - w[k])

        X_coriolis = None
        if self._use_coriolis(add_coriolis):
            X_coriolis = self.calculate_anharmonic_coriolis()
            X = X + X_coriolis

        anharm_freq = self.calculate_fundamental_frequencies(X)

        return VPT2Result(
            X=X,
            anharm_freq=anharm_freq,
            method="deperturbed",
            X_coriolis=X_coriolis,
            fermi_type1=fermi_type1,
            fermi_type2=fermi_type2,
        )

    def calculate_anharmonic_degeneracy_corrected(
        self, add_coriolis: bool = True
    ) -> VPT2Result:
        """
        Calculate anharmonicity corrections using degeneracy-corrected VPT2.

        Based on: Kuhler, Truhlar & Isaacson, "General method for removing
        resonance singularities in quantum mechanical perturbation theory",
        J. Chem. Phys. 104, 4664 (1996), Eqs. 13-37 (DCPT2).

        This method replaces pole singularities with square root expressions
        to handle near-degenerate and degenerate modes.

        Parameters
        ----------
        add_coriolis : bool, optional
            Whether to include Coriolis coupling corrections (default: True)

        Returns
        -------
        np.ndarray
            Degeneracy-corrected anharmonicity constants
        """
        X = np.zeros((self.len_modes, self.len_modes))

        # Diagonal elements
        for i in self.active_modes:
            w_i = self.freq[i]
            w_i_sq = self._freq_squared[i]

            # Fourth-order term
            X[i, i] = self.U[i, i, i, i] * self.FOURTH_ORDER_DIAGONAL_SCALE

            # Self-interaction third-order term
            X[i, i] -= self.FOURTH_ORDER_DIAGONAL_SCALE * (
                self.DEGENERACY_SELF_TERM_SCALE * self.T[i, i, i] ** 2 / w_i
            )

            # Cross-mode contributions (resonance-corrected)
            for m in self.active_modes:
                if i != m:
                    w_m = self.freq[m]
                    w_m_sq = self._freq_squared[m]

                    sign_energy = -(8 * w_i_sq - 3 * w_m_sq) * (2 * w_i - w_m)
                    sqrt_energy = 2 * w_i - w_m
                    coupling = np.abs(
                        -(self.T[i, i, m] ** 2) * (8 * w_i_sq - 3 * w_m_sq)
                    ) / (16 * w_m * (2 * w_i + w_m))

                    X[i, i] += self._resonance_corrected_term(
                        sign_energy, sqrt_energy, coupling
                    )

        # Off-diagonal elements
        for i in self.active_modes:
            for j in self.active_modes:
                if j != i:
                    w_i, w_j = self.freq[i], self.freq[j]

                    # Fourth-order and 2-mode third-order
                    X[i, j] = (
                        self.U[i, i, j, j] * self.FOURTH_ORDER_OFFDIAG_SCALE
                    )
                    coupling_product = (
                        self.T[i, i, self.active_modes]
                        * self.T[j, j, self.active_modes]
                    )
                    X[i, j] -= np.sum(
                        coupling_product / (4 * self._active_freq)
                    )

                    # Single-pole terms (m = i and m = j)
                    energy_2wi_wj = 2 * w_i - w_j
                    coupling_i = (
                        self.T[i, i, j] ** 2 * w_i / (2 * (2 * w_i + w_j))
                    )
                    X[i, j] += self._resonance_corrected_term(
                        -energy_2wi_wj, energy_2wi_wj, coupling_i
                    )

                    energy_2wj_wi = 2 * w_j - w_i
                    coupling_j = (
                        self.T[i, j, j] ** 2 * w_j / (2 * (2 * w_j + w_i))
                    )
                    X[i, j] += self._resonance_corrected_term(
                        -energy_2wj_wi, energy_2wj_wi, coupling_j
                    )

                    # Three-pole terms (m ≠ i, j)
                    for m in self.active_modes:
                        if m != i and m != j:
                            w_m = self.freq[m]
                            T_ijm_sq = self.T[i, j, m] ** 2

                            # No-pole term
                            X[i, j] -= T_ijm_sq / (8 * (w_i + w_j + w_m))

                            # Three resonance-corrected poles
                            coupling_term = T_ijm_sq / 8
                            energy_1 = w_i - w_j - w_m
                            energy_2 = -w_i + w_j - w_m
                            energy_3 = w_i + w_j - w_m
                            X[i, j] += self._resonance_corrected_term(
                                energy_1, energy_1, coupling_term
                            )
                            X[i, j] += self._resonance_corrected_term(
                                energy_2, energy_2, coupling_term
                            )
                            X[i, j] += self._resonance_corrected_term(
                                energy_3, energy_3, coupling_term
                            )

        X_coriolis = None
        if self._use_coriolis(add_coriolis):
            X_coriolis = self.calculate_anharmonic_coriolis()
            X = X + X_coriolis

        anharm_freq = self.calculate_fundamental_frequencies(X)

        return VPT2Result(
            X=X,
            anharm_freq=anharm_freq,
            method="dcpt2",
            X_coriolis=X_coriolis,
        )

    def calculate_fundamental_frequencies(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate fundamental vibrational frequencies from anharmonicity constants.

        The fundamental frequency for mode i is given by:
        ν_i = ω_i + 2X_ii + (1/2)∑_{j≠i} X_ij

        Parameters
        ----------
        X : np.ndarray
            Anharmonicity constants matrix

        Returns
        -------
        np.ndarray
            Anharmonic fundamental frequencies in cm^-1
        """
        v = np.zeros_like(self.freq)

        # Vectorized calculation for active modes
        for i in self.active_modes:
            diagonal_contrib = self.FUNDAMENTAL_DIAGONAL_FACTOR * X[i, i]

            # Off-diagonal contribution (vectorized)
            offdiag_sum = np.sum(X[i, self.active_modes]) - X[i, i]
            offdiag_contrib = self.FUNDAMENTAL_OFFDIAG_FACTOR * offdiag_sum

            v[i] = self.freq[i] + diagonal_contrib + offdiag_contrib

        return v

    def resonance_check(
        self,
        threshold: Optional[float] = None,
        K1: Optional[float] = None,
        K2: Optional[float] = None,
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Detect Fermi resonances with the Martin test.

        Based on: Martin et al., J. Chem. Phys. 103, 2589 (1995),
        Eqs. A24/A25, combined with the frequency window used by Barone,
        J. Chem. Phys. 122, 014108 (2005).

        Type 1 (w_i ~ 2 w_j):     |w_i - 2 w_j| < threshold and
                                  |phi_jji|^4 / (256 |2w_j - w_i|^3) >= K1
        Type 2 (w_i ~ w_j + w_k): |w_i - w_j - w_k| < threshold and
                                  |phi_ijk|^4 / (64 |w_j + w_k - w_i|^3) >= K2

        Parameters
        ----------
        threshold : float, optional
            Frequency window in cm^-1 (default: config.resonance_window)
        K1, K2 : float, optional
            Martin-test thresholds in cm^-1 (default: config.resonance_K1/K2)

        Returns
        -------
        fermi_type1 : list of (i, j), meaning w_i ~ 2 w_j
        fermi_type2 : list of (i, j, k), meaning w_i ~ w_j + w_k, with
                      i, j, k distinct and j < k
        """
        threshold = self.config.resonance_window if threshold is None else threshold
        K1 = self.config.resonance_K1 if K1 is None else K1
        K2 = self.config.resonance_K2 if K2 is None else K2
        w = self.freq
        T = self.T
        ftype1, ftype2 = [], []
        modes = list(self.active_modes)

        for i in modes:
            for j in modes:
                if j == i:
                    continue
                delta = 2 * w[j] - w[i]
                if abs(delta) < threshold:
                    if T[j, j, i] ** 4 / (256 * abs(delta) ** 3) >= K1:
                        ftype1.append((i, j))
                for k in modes:
                    if k <= j or k == i:
                        continue
                    delta = w[j] + w[k] - w[i]
                    if abs(delta) < threshold:
                        if T[i, j, k] ** 4 / (64 * abs(delta) ** 3) >= K2:
                            ftype2.append((i, j, k))

        return ftype1, ftype2

    # ========================================================================
    # Coriolis Coupling Methods
    # ========================================================================

    def _principal_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Principal moments (amu*A^2) and axes of the equilibrium structure.

        The inertia tensor is built about the centre of mass and
        diagonalised. Rotational constants and Coriolis zetas must refer to
        this same frame, otherwise sum_a B_a (zeta^a)^2 is not invariant to
        how the molecule happens to be oriented and placed in space.
        """
        if self.config.atoms is None:
            raise ValueError("Atoms object not provided in configuration")
        masses = self.config.atoms.get_masses()
        R = self.config.atoms.positions - self.config.atoms.get_center_of_mass()
        inertia = -np.einsum("a,ai,aj->ij", masses, R, R)
        inertia[np.diag_indices(3)] += np.sum(masses * np.sum(R**2, axis=1))
        moments, axes = np.linalg.eigh(inertia)  # columns = principal axes
        return moments, axes

    def calculate_zeta(self) -> np.ndarray:
        """
        Calculate Coriolis zeta coupling constants in the principal axis frame.

        Based on: I.M Mills, "Critical Evaluation of Chemical and Physical
        Structural Information", 1974, p. 278, eq. 3.

        Returns
        -------
        np.ndarray
            Coriolis coupling constants, shape (3, len_modes, len_modes),
            axis order matching calculate_rotational_constants()
        """
        if self.config.modes is None:
            raise ValueError("Normal modes not provided in configuration")

        L = self.config.modes
        num_atoms = L.shape[0] // 3
        _, axes = self._principal_axes()

        zeta_lab = np.zeros((3, self.len_modes, self.len_modes))
        for r in self.active_modes:
            for s in self.active_modes:
                for k in range(num_atoms):
                    lkr = L[k * 3 : (k + 1) * 3, r]
                    lks = L[k * 3 : (k + 1) * 3, s]
                    zeta_lab[:, r, s] += np.cross(lkr, lks)

        # rotate the zeta vectors into the principal axis frame
        zeta = np.einsum("ap,ars->prs", axes, zeta_lab)
        zeta[np.abs(zeta) < 1e-6] = 0
        return zeta

    def calculate_inertia(self) -> np.ndarray:
        """
        Principal moments of inertia about the centre of mass.

        Returns
        -------
        np.ndarray
            Diagonal inertia tensor in amu*A^2 (principal axis frame), shape (3, 3)
        """
        moments, _ = self._principal_axes()
        return np.diag(moments)

    def calculate_rotational_constants(self) -> np.ndarray:
        """
        Calculate equilibrium rotational constants in cm^-1.

        Returns
        -------
        np.ndarray
            Rotational constants B_e for each principal axis, shape (3,).
            Axes with a vanishing moment (linear molecules) get B = 0; the
            corresponding zetas vanish as well.
        """
        I = np.diag(self.calculate_inertia()) * self.A_to_m**2 * self.amu_to_kg
        beta_e = np.zeros(3)
        for d in range(3):
            if I[d] > 1e-8 * I.max():
                beta_e[d] = self.h_J_s / (8 * np.pi**2 * self.c_cm_per_s * I[d])
        return beta_e

    def calculate_anharmonic_coriolis(self) -> np.ndarray:
        """
        Calculate Coriolis contribution to anharmonicity constants.

        X_ij += sum_a B_a (zeta_ij^a)^2 (w_i/w_j + w_j/w_i)
        (Martin et al. 1995, Eq. A2; Yu & Bowman 2015, Eq. 11)

        Returns
        -------
        np.ndarray
            Coriolis contribution to X matrix, shape (len_modes, len_modes)
        """
        beta = self.calculate_rotational_constants()
        zeta = self.calculate_zeta()
        w = self.freq

        X = np.zeros((self.len_modes, self.len_modes))
        for i in self.active_modes:
            for j in self.active_modes:
                if i != j:
                    X[i, j] = (
                        (w[i] ** 2 + w[j] ** 2)
                        / (w[i] * w[j])
                        * np.sum(beta * zeta[:, i, j] ** 2)
                    )
        return X
