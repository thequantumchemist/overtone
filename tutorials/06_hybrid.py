"""
Tutorial 6: hybrid scheme with two levels of theory.

Harmonic frequencies are sensitive to the level of theory, anharmonic
corrections much less. A common strategy therefore computes the Hessian at a
high level and the quartic force field at a lower level (see Barnes et al.,
J. Mol. Model. 22, 285 (2016)). Here the Hessian comes from B3LYP (PySCF)
and the anharmonic part from GFN2-xTB.

    pip install "overtone-vpt2[tutorial]"
    python 06_hybrid.py         # about 20 seconds

The structure is optimised at the high level. The low-level force field is
expanded around this geometry, where its own gradient is not exactly zero.
This approximation is standard in hybrid schemes.
"""

# --8<-- [start:imports]
from ase.build import molecule
from ase.optimize import BFGS
from pyscf_calculator import PySCFCalculator
from tblite.ase import TBLite

from overtone import calculate_anharmonics
# --8<-- [end:imports]


# --8<-- [start:calculators]
def high_level():
    return PySCFCalculator(xc="b3lyp", basis="def2-tzvp")


def low_level():
    return TBLite(method="GFN2-xTB", verbosity=0, accuracy=0.01)
# --8<-- [end:calculators]

# --8<-- [start:hybrid]
atoms = molecule("H2O")
atoms.calc = high_level()
BFGS(atoms).run(fmax=1e-4)

hybrid = calculate_anharmonics(
    atoms,
    low_level,
    "water_hybrid",
    setup_harmonic_calculator=high_level,
    derivative_source="forces",
)
# --8<-- [end:hybrid]

# --8<-- [start:compare]
low = molecule("H2O")
low.calc = low_level()
BFGS(low).run(fmax=1e-4)
pure_low = calculate_anharmonics(
    low, low_level, "water_gfn2", derivative_source="forces"
)

EXPERIMENT = [1595, 3657, 3756]
print(f"\n{'':10s} {'bend':>8} {'sym. str.':>10} {'asym. str.':>11}")
for name, r in (("GFN2-xTB", pure_low), ("hybrid", hybrid)):
    f = sorted(t.energy for t in r.transitions if t.kind == "fundamental")
    print(f"{name:10s} {f[0]:8.1f} {f[1]:10.1f} {f[2]:11.1f}")
print(f"{'experiment':10s} {EXPERIMENT[0]:8d} {EXPERIMENT[1]:10d} {EXPERIMENT[2]:11d}")
# --8<-- [end:compare]
