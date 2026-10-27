"""ASE calculators used as controllable test doubles."""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT

from overtone.harmonics.harmonics import EV_TO_INV_CM


class QFFCalculator(Calculator):
    """V = 1/2 sum w q^2 + 1/6 sum T qqq + 1/24 sum U qqqq  (cm^-1 -> eV)."""

    implemented_properties = ["energy", "forces", "dipole"]

    def __init__(self, ref, transformer, w, T, U, active, **kw):
        super().__init__(**kw)
        self.ref, self.tr = ref, transformer
        self.w, self.T, self.U, self.active = w, T, U, active
        n = len(ref) * 3
        # q = J (R - R0), J constant
        self.J = transformer.cart_to_reduced_norm(np.eye(n))[active]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        q = self.J @ (self.atoms.positions - self.ref).ravel()
        w, T, U = self.w, self.T, self.U
        V = (0.5 * np.sum(w * q**2) + np.einsum("ijk,i,j,k", T, q, q, q) / 6
             + np.einsum("ijkl,i,j,k,l", U, q, q, q, q) / 24)
        g = (w * q + np.einsum("ijk,j,k->i", T, q, q) / 2
             + np.einsum("ijkl,j,k,l->i", U, q, q, q) / 6)
        self.results["energy"] = V / EV_TO_INV_CM
        self.results["forces"] = -(self.J.T @ g).reshape(-1, 3) / EV_TO_INV_CM
        self.results["dipole"] = np.zeros(3)


class ChargedEMT(Calculator):
    """EMT energies/forces plus a fixed-point-charge dipole (e*Angstrom)."""

    implemented_properties = ["energy", "forces", "dipole"]

    def __init__(self, charges, with_dipole=True, **kw):
        super().__init__(**kw)
        self.charges = np.asarray(charges, float)
        self.with_dipole = with_dipole
        self._emt = EMT()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if "dipole" in properties and not self.with_dipole:
            raise RuntimeError("this calculator cannot provide dipoles")
        a = self.atoms.copy()
        a.calc = self._emt
        self.results["energy"] = a.get_potential_energy()
        self.results["forces"] = a.get_forces()
        if self.with_dipole:
            # nonlinear in the geometry, so second derivatives are non-zero
            r = self.atoms.positions - self.atoms.positions.mean(0)
            self.results["dipole"] = self.charges @ (r + 0.1 * r**2)


WATER_CHARGES = [-0.8, 0.4, 0.4]


class Crash(Exception):
    pass


def counting_emt(counter, crash_after=None):
    """EMT class that counts calculations in counter["n"] and optionally crashes."""
    class Counting(EMT):
        def calculate(self, *args, **kwargs):
            counter["n"] += 1
            if crash_after is not None and counter["n"] > crash_after:
                raise Crash("simulated crash")
            super().calculate(*args, **kwargs)
    return Counting
