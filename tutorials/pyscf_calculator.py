"""
Minimal ASE calculator for molecular DFT/HF with PySCF.

Provides energy, analytic forces and the dipole moment, which is all
OVERTONE needs (including IR intensities). Install with `pip install pyscf`.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree


class PySCFCalculator(Calculator):
    """
    Parameters
    ----------
    xc : str
        Exchange-correlation functional, e.g. 'b3lyp', or 'hf' for Hartree-Fock
    basis : str
        Basis set, e.g. 'aug-cc-pvtz'
    charge, spin : int
        Total charge and 2S
    conv_tol : float
        SCF energy convergence in Hartree. Finite differences amplify SCF
        noise, so keep this tight (default 1e-11).
    grid_level : int
        DFT integration grid (0-9). A fine grid (default 4) avoids grid noise
        in the numerical third and fourth derivatives.
    """

    implemented_properties = ["energy", "forces", "dipole"]

    def __init__(self, xc="b3lyp", basis="aug-cc-pvtz", charge=0, spin=0,
                 conv_tol=1e-11, grid_level=4, **kwargs):
        super().__init__(**kwargs)
        self.xc, self.basis = xc, basis
        self.charge, self.spin = charge, spin
        self.conv_tol, self.grid_level = conv_tol, grid_level

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        from pyscf import dft, gto, scf

        super().calculate(atoms, properties, system_changes)
        mol = gto.M(
            atom=[(s, p) for s, p in zip(self.atoms.get_chemical_symbols(),
                                         self.atoms.positions)],
            unit="Angstrom", basis=self.basis, charge=self.charge, spin=self.spin,
            verbose=0,
        )
        if self.xc.lower() == "hf":
            mf = scf.RHF(mol) if self.spin == 0 else scf.UHF(mol)
        else:
            mf = dft.RKS(mol) if self.spin == 0 else dft.UKS(mol)
            mf.xc = self.xc
            mf.grids.level = self.grid_level
        mf.conv_tol = self.conv_tol
        mf.conv_tol_grad = np.sqrt(self.conv_tol)
        energy = mf.kernel()
        if not mf.converged:
            raise RuntimeError("PySCF SCF did not converge")
        self.results["energy"] = energy * Hartree
        if "forces" in properties:
            self.results["forces"] = -mf.nuc_grad_method().kernel() * Hartree / Bohr
        # dipole in e*Angstrom (ASE convention), origin-independent for neutrals
        self.results["dipole"] = mf.dip_moment(unit="AU", verbose=0) * Bohr
