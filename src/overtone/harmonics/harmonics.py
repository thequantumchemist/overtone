"""
Harmonic analysis: finite-difference Hessian, normal modes and frequencies.

Three options control how the normal modes are obtained (all on by default):

symmetrize_hessian
    Replace H by (H + H^T)/2. Finite differences of forces give a slightly
    asymmetric H (step size, SCF noise). With False the legacy behaviour is
    kept, where numpy.linalg.eigh silently uses only the lower triangle.

project_translations_rotations
    Build the 3 translations and the 3 rotations about the principal axes
    (2 for linear molecules) in mass-weighted Cartesian coordinates and
    diagonalise the Hessian only in the orthogonal complement. The rigid-body
    modes get exactly zero frequency and come first in the mode list, and all
    vibrational modes are free of translation/rotation contamination. This
    matters for loosely converged geometries, noisy (e.g. ML) potentials and
    for dipole derivatives of ions. Not possible (and switched off) when
    atoms are constrained, because the rigid-body motions are not free then.

linear
    None detects linear molecules from the principal moments of inertia
    (I_min / I_max < linear_tolerance). True/False overrides the detection.
"""

import warnings
from typing import List, Optional

import numpy as np
from ase import Atoms

from overtone.harmonics.atom_constraints import AtomConstraints
from overtone.harmonics.harmonics_result import HarmonicsResult

# sqrt(eV / (amu * Angstrom^2)) -> hbar*omega in eV
_EV_PER_SQRT_EIGENVALUE = 6.582119569e-16 * 1e10 * np.sqrt(1.602176634e-19 / 1.66053906660e-27)
EV_TO_INV_CM = 8065.543937


def _principal_frame(masses: np.ndarray, positions: np.ndarray):
    com = masses @ positions / masses.sum()
    r = positions - com
    inertia = -np.einsum("a,ai,aj->ij", masses, r, r)
    inertia[np.diag_indices(3)] += np.sum(masses * np.sum(r**2, axis=1))
    moments, axes = np.linalg.eigh(inertia)
    return r, moments, axes


def is_linear_molecule(atoms: Atoms, tolerance: float = 1e-5) -> bool:
    """True if the smallest principal moment is negligible (I_min/I_max < tolerance)."""
    if len(atoms) < 2:
        return False
    _, moments, _ = _principal_frame(atoms.get_masses(), atoms.get_positions())
    return bool(moments[0] < tolerance * moments[2])


def rigid_body_basis(atoms: Atoms, linear: bool) -> np.ndarray:
    """
    Orthonormal translation/rotation vectors in mass-weighted Cartesians.

    Returns
    -------
    np.ndarray, shape (3N, 6) or (3N, 5) for linear molecules
    """
    masses = atoms.get_masses()
    sqm = np.sqrt(masses)
    r, moments, axes = _principal_frame(masses, atoms.get_positions())
    n = len(atoms)
    vecs = []
    for alpha in range(3):  # translations
        v = np.zeros((n, 3))
        v[:, alpha] = sqm
        vecs.append(v.ravel())
    rot_axes = range(1, 3) if linear else range(3)  # skip the molecular axis
    for k in rot_axes:  # rotations about principal axes
        v = sqm[:, None] * np.cross(axes[:, k], r)
        vecs.append(v.ravel())
    B = np.array(vecs).T
    # Gram-Schmidt via QR (vectors are orthogonal already up to round-off)
    Q, _ = np.linalg.qr(B)
    return Q


class Harmonics:
    """
    Harmonic frequencies from finite differences of forces.

    Parameters
    ----------
    atoms : ase.Atoms
        Equilibrium structure (should be tightly optimised).
    delta : float
        Cartesian displacement for the central differences, in Angstrom.
        Default 0.005. Smaller steps reduce the anharmonic truncation error
        (relevant for torsions) but need tight SCF convergence.
    constrain_index : list of int, optional
        Atoms that are kept fixed (e.g. a surface). Only the remaining atoms
        enter the Hessian. Disables the rigid-body projection.
    symmetrize_hessian : bool
        Use (H + H^T)/2 (default True). See module docstring.
    project_translations_rotations : bool
        Eckart projection of the rigid-body motions (default True).
    linear : bool, optional
        Force linear (True) or non-linear (False); None = detect (default).
    linear_tolerance : float
        Threshold for I_min/I_max in the automatic detection (default 1e-5).
    """

    def __init__(
        self,
        atoms: Atoms,
        delta: float = 0.005,
        constrain_index: Optional[List[int]] = None,
        symmetrize_hessian: bool = True,
        project_translations_rotations: bool = True,
        linear: Optional[bool] = None,
        linear_tolerance: float = 1e-5,
    ):
        self.atoms = atoms
        self.num_atoms = len(atoms)
        self.delta = delta
        self.constraints = AtomConstraints(
            num_atoms=self.num_atoms, constrain_index=constrain_index
        )
        self.symmetrize_hessian = symmetrize_hessian
        self.project = project_translations_rotations
        if self.project and self.constraints.constrained:
            warnings.warn(
                "Constrained atoms: translations/rotations are not free, "
                "so the rigid-body projection is switched off."
            )
            self.project = False
        self.linear = (
            is_linear_molecule(atoms, linear_tolerance) if linear is None else bool(linear)
        )

    # ------------------------------------------------------------------
    def calculate_displacements(self) -> List[Atoms]:
        """
        Structures for the central-difference Hessian.

        Order: reference, then for every unconstrained atom and x, y, z the
        +delta and -delta displacement.
        """
        trajectory = [self.atoms.copy()]
        atoms = self.atoms.copy()
        for i in self.constraints.unconstrained_index:
            for d in range(3):
                atoms.positions[i, d] += self.delta
                trajectory.append(atoms.copy())
                atoms.positions[i, d] -= 2 * self.delta
                trajectory.append(atoms.copy())
                atoms.positions[i, d] += self.delta
        return trajectory

    def calculate_hessian(self, trajectory: List[Atoms]) -> HarmonicsResult:
        """
        Hessian from the forces of the trajectory from calculate_displacements().
        """
        n = 3 * self.constraints.num_eff_atoms
        hessian = np.zeros((n, n))
        idx = self.constraints.force_indices
        for J, j in enumerate(range(1, len(trajectory), 2)):
            fp = trajectory[j].get_forces().flat[idx]
            fm = trajectory[j + 1].get_forces().flat[idx]
            hessian[:, J] = (fm - fp) / (2 * self.delta)  # eV/A^2
        return self.analyse_hessian(hessian)

    def analyse_hessian(self, hessian: np.ndarray) -> HarmonicsResult:
        """
        Normal modes and frequencies from a Cartesian Hessian (eV/Angstrom^2).

        Can also be used with a Hessian from another program (e.g. a
        higher level of theory than the anharmonic part).
        """
        hessian = np.asarray(hessian, float)
        scale = np.max(np.abs(hessian)) or 1.0
        asymmetry = float(np.max(np.abs(hessian - hessian.T)) / scale)
        H = 0.5 * (hessian + hessian.T) if self.symmetrize_hessian else hessian

        idx = self.constraints.force_indices
        inv_sqm = np.repeat(self.atoms.get_masses() ** -0.5, 3)[idx]
        Hmw = inv_sqm[:, None] * H * inv_sqm

        if self.project:
            T = rigid_body_basis(self.atoms, self.linear)
            n_tr = T.shape[1]
            # orthonormal basis of the vibrational space
            P = np.eye(Hmw.shape[0]) - T @ T.T
            pw, pv = np.linalg.eigh(P)
            D = pv[:, pw > 0.5]
            w2_vib, C = np.linalg.eigh(D.T @ Hmw @ D)
            modes = np.hstack([T, D @ C])
            w2 = np.concatenate([np.zeros(n_tr), w2_vib])
        else:
            n_tr = 0 if self.constraints.constrained else (5 if self.linear else 6)
            w2, modes = np.linalg.eigh(Hmw)  # legacy: lower triangle if not symmetric

        # phase convention: largest component of every mode is positive
        for i in range(modes.shape[1]):
            k = np.argmax(np.abs(modes[:, i]))
            if modes[k, i] < 0:
                modes[:, i] *= -1

        freq_ev = _EV_PER_SQRT_EIGENVALUE * w2.astype(complex) ** 0.5
        if self.project:
            freq_ev[:n_tr] = 0.0
        return HarmonicsResult(
            hessian=hessian,
            w2=w2,
            modes=modes,
            freq=freq_ev.real * EV_TO_INV_CM,
            ifreq=freq_ev.imag * EV_TO_INV_CM,
            freq_eV=freq_ev,
            chemical_formula=self.atoms.get_chemical_formula(),
            n_tr=n_tr,
            is_linear=self.linear,
            projected=self.project,
            symmetrized=self.symmetrize_hessian,
            hessian_asymmetry=asymmetry,
        )
