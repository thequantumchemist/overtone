from ase.calculators.emt import EMT

from overtone import Harmonics


def run_harmonics(atoms, calc_factory=EMT, **kw):
    """Finite-difference Hessian of atoms with a fresh calculator per displacement."""
    h = Harmonics(atoms.copy(), delta=0.005, **kw)
    traj = h.calculate_displacements()
    for a in traj:
        a.calc = calc_factory()
        a.get_forces()
    return h.calculate_hessian(traj)
