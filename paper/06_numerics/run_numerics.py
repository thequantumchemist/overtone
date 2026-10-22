"""
Numerical analysis behind the default settings (Supporting Information).

1. Number of single points per stencil and derivative route.
2. Noise amplification. An exact quartic force field (water, EMT modes) is
   evaluated with random energy noise; scatter of the VPT2 fundamentals.
3. Truncation error. GFN2-xTB fundamentals of water and methanol as a
   function of the anharmonic step.

Runs locally in a few minutes (no quantum chemistry program needed).
    python run_numerics.py
"""

import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
from ase.build import molecule
from ase.calculators.calculator import all_changes

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "tests"))
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

import common  # noqa: E402
from overtone import CoordinateTransformer, calculate_anharmonics, estimate_cost  # noqa: E402
from test_harmonics_and_pipeline import (  # noqa: E402
    QFFCalculator, _harmonics, _model_force_field, _optimised)

HARTREE = 27.211386245988
OUT = HERE / "results"


def costs():
    rows = []
    for name in ("H2O", "CH4", "C2H4", "CH3NO2"):
        atoms = molecule(name)
        for src in ("energy", "forces"):
            row = {"molecule": name, "route": src}
            for st in (0, 1, 2, 3):
                row[f"stencil {st}"] = estimate_cost(atoms, derivative_source=src, stencil=st)["total"]
            rows.append(row)
    common.write_csv(rows, OUT / "costs.csv")


def noise():
    atoms = _optimised("H2O")
    harm = _harmonics(atoms)
    act = harm.vibrational_modes
    w = harm.freq[act]
    tr = CoordinateTransformer(harm.freq, harm.modes, np.repeat(atoms.get_masses() ** 0.5, 3))
    T, U = _model_force_field(w)

    class Noisy(QFFCalculator):
        def __init__(self, *a, sigma=0.0, rng=None, **k):
            super().__init__(*a, **k)
            self.sigma, self.rng = sigma, rng

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] += self.rng.normal(0, self.sigma)

    rows = []
    for sigma_eh in (1e-9, 1e-8, 1e-7):
        for delta in (0.1, 0.2, 0.3, 0.5):
            vals = []
            for seed in range(8):
                rng = np.random.default_rng(seed)
                setup = lambda: Noisy(atoms.positions.copy(), tr, w, T, U, act,
                                      sigma=sigma_eh * HARTREE, rng=rng)
                with contextlib.redirect_stdout(io.StringIO()):
                    r = calculate_anharmonics(atoms.copy(), setup, OUT / "tmp", stencil=1,
                                              coriolis=False, harmonic_delta=0.001,
                                              anharmonic_delta=delta, method="standard")
                vals.append(r.anharm_freq[act])
            rows.append({"noise (Eh)": sigma_eh, "delta": delta,
                         "max std of fundamentals (cm-1)": round(float(np.std(vals, 0).max()), 3)})
            print(rows[-1], flush=True)
    common.write_csv(rows, OUT / "noise.csv")


def truncation():
    rows = []
    setup = common.xtb()
    for name in ("H2O", "CH3OH"):
        atoms = common.optimise(molecule(name), setup, fmax=1e-5, logfile=None)
        res = {}
        for delta in (0.05, 0.1, 0.2, 0.3, 0.5):
            with contextlib.redirect_stdout(io.StringIO()):
                res[delta] = calculate_anharmonics(atoms.copy(), setup, OUT / "tmp",
                                                   derivative_source="forces",
                                                   anharmonic_delta=delta,
                                                   method="standard").anharm_freq[6:]
        ref = res[0.1]
        for delta, v in res.items():
            d = np.abs(v - ref)
            rows.append({"molecule": name, "delta": delta,
                         "max |nu - nu(0.1)| all modes": round(float(d.max()), 2),
                         "max |nu - nu(0.1)| without lowest mode": round(float(d[1:].max()), 2)})
            print(rows[-1], flush=True)
    common.write_csv(rows, OUT / "truncation.csv")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    costs()
    noise()
    truncation()
