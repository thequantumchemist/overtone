"""
Tutorial 2: every step of calculate_anharmonics, called by hand.

Use this when you want full control, e.g. to inspect the force constants,
to reuse a Hessian from another program, or to run the single points
somewhere else.
"""
from ase.build import molecule
from ase.optimize import BFGS
from tblite.ase import TBLite

from overtone import (AnharmonicDerivativesConfig, AnharmonicDerivativesSolver,
                      Harmonics, VPT2Config, VPT2Solver)


def calc():
    return TBLite(method="GFN2-xTB", verbosity=0, accuracy=0.01)


# 1. optimise
atoms = molecule("H2O")
atoms.calc = calc()
BFGS(atoms).run(fmax=1e-4)

# 2. harmonic analysis (forces at +-0.005 A, symmetrised, Eckart-projected)
# --8<-- [start:harmonics]
harmonics = Harmonics(atoms, delta=0.005, symmetrize_hessian=True,
                      project_translations_rotations=True)
displaced = harmonics.calculate_displacements()
for a in displaced:
    a.calc = calc()
    a.get_forces()
harm = harmonics.calculate_hessian(displaced)
# --8<-- [end:harmonics]
harm.summary()
modes = harm.vibrational_modes

# 3. quartic force field from forces (dimensionless step 0.2, 5-point stencil)
# --8<-- [start:forcefield]
config = AnharmonicDerivativesConfig.from_harmonics_result(
    harmonics_result=harm, atoms=atoms, active_modes=modes,
    delta=0.2, stencil=1, coupling=3)
solver = AnharmonicDerivativesSolver(config)
traj, keys = solver.generate_displacement_trajectory_for_forces()
# --8<-- [end:forcefield]
print(f"{len(traj)} anharmonic single points")
for a in traj:
    a.calc = calc()
    a.get_forces()
    a.get_dipole_moment()           # only needed for intensities
# --8<-- [start:derivatives]
derivatives = solver.calculate_derivatives_from_forces(traj)
dipoles = solver.calculate_dipole_derivatives(traj, keys)
# --8<-- [end:derivatives]

# 4. VPT2: compare all methods on the same force field
# --8<-- [start:vpt2]
for method in ("standard", "deperturbed", "dcpt2", "gvpt2"):
    vpt2 = VPT2Config.from_harmonics_result(harm, modes, atoms=atoms, method=method)
    result = VPT2Solver(vpt2, derivatives, dipole_derivatives=dipoles).calculate()
    print(f"{method:12s}", [round(float(f), 1) for f in result.anharm_freq[modes]])
# --8<-- [end:vpt2]

result.transitions_table()
