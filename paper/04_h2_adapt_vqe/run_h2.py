"""
H2 / STO-3G: anharmonic stretching frequency from CCSD (PySCF) and from
ADAPT-VQE on a simulated quantum computer (Qiskit Aer statevector).

For two electrons in STO-3G, CCSD is exact (equal to FCI), and ADAPT-VQE
with a UCCSD operator pool converges to the same exact energy. Any
difference between the two therefore measures the remaining ADAPT-VQE
convergence error and how strongly the finite differences amplify it.
ADAPT-VQE is run with the default and with a tight optimizer tolerance.

Requirements (separate environment, see ase_quantum_vqe's README):
    pip install overtone-vpt2
    pip install git+https://github.com/thequantumchemist/ase_quantum_vqe.git
    (qiskit==1.4.4, qiskit-algorithms==0.3.1, qiskit-nature==0.7.2, ...)

Usage:
    python run_h2.py              # both methods, default step 0.2
    python run_h2.py --scan       # additionally scan the step size
    python run_h2.py --n-jobs 6   # parallel numerical forces
"""

import argparse
import contextlib
import csv
import io
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import BFGS

from pyscf import cc, scf

from overtone import calculate_anharmonics
from qiskit_algorithms.optimizers import SLSQP
from ase_quantum_vqe.qiskit_vqe_calculator import QiskitVQECalculator
from ase_quantum_vqe.utils.pyscf import PySCFCalculator

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Tight PySCF convergence for the CCSD reference. The defaults (SCF 1e-9 Eh,
# CCSD 1e-7 Eh) scatter the anharmonic shift by about +-0.3 cm-1, because the
# finite differences amplify convergence errors. PySCFCalculator does not
# expose these thresholds, but they are class attributes in PySCF.
scf.hf.SCF.conv_tol = 1e-12
cc.ccsd.CCSD.conv_tol = 1e-12
cc.ccsd.CCSD.conv_tol_normt = 1e-9

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
HERE = Path(__file__).parent
RESULTS = HERE / "results"
STEPS_SCAN = [0.1, 0.2, 0.3, 0.4]   # dimensionless anharmonic steps for --scan
DEFAULT_STEP = 0.2


def make_calculators(n_jobs):
    """
    CCSD reference and ADAPT-VQE with two optimizer tolerances.

    'ADAPT-VQE (default)' uses Qiskit's SLSQP default (ftol = 1e-6), as in the
    first version of the paper. Finite differences amplify the remaining
    convergence error, so the anharmonic shift then depends on the step.
    'ADAPT-VQE (tight)' converges the optimizer to ftol = 1e-12.
    """
    return {
        "CCSD": lambda: PySCFCalculator(
            basis="sto-3g", method="ccsd", charge=0, spin=0, n_jobs=n_jobs, coreorb=0
        ),
        "ADAPT-VQE (default)": lambda: QiskitVQECalculator(
            basis="sto3g", backend="aer", n_jobs=n_jobs, charge=0, spin=0, maxiter=500
        ),
        "ADAPT-VQE (tight)": lambda: QiskitVQECalculator(
            basis="sto3g", backend="aer", n_jobs=n_jobs, charge=0, spin=0, maxiter=500,
            optimizer=SLSQP(maxiter=500, ftol=1e-12), vqe_eigenvalue=1e-12,
        ),
    }


def optimised_h2(setup):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.735]])
    atoms.calc = setup()
    BFGS(atoms, logfile=None).run(fmax=1e-4)
    return atoms


def run(name, setup, atoms, step):
    tag = name.replace(" ", "_").replace("(", "").replace(")", "")
    directory = RESULTS / f"{tag}_step{step}"
    with contextlib.redirect_stdout(io.StringIO()):
        result = calculate_anharmonics(
            atoms.copy(),
            setup,
            directory,
            derivative_source="energy",  # forces of both calculators are numerical
            anharmonic_delta=step,
            stencil=1,
            save_results=True,
        )
    mode = 5  # H2 is linear: 5 rigid-body modes, then the stretch
    fund = [t for t in result.transitions if t.kind == "fundamental"][0]
    return {
        "method": name,
        "step": step,
        "r_e (A)": round(float(atoms.get_distance(0, 1)), 6),
        "harmonic": round(fund.harmonic, 2),
        "anharmonic": round(fund.energy, 2),
        "shift": round(fund.harmonic - fund.energy, 2),
        "overtone_2v": round([t for t in result.transitions if t.kind == "overtone"][0].energy, 2),
        "X11": round(float(result.X[mode, mode]), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--scan", action="store_true", help="scan the anharmonic step")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    steps = STEPS_SCAN if args.scan else [DEFAULT_STEP]
    rows = []
    for name, setup in make_calculators(args.n_jobs).items():
        atoms = optimised_h2(setup)
        print(f"{name}: r_e = {atoms.get_distance(0, 1):.6f} A, "
              f"E = {atoms.get_potential_energy():.8f} eV")
        for step in steps:
            row = run(name, setup, atoms, step)
            rows.append(row)
            print(f"  step {step}: harmonic {row['harmonic']:.2f}, "
                  f"fundamental {row['anharmonic']:.2f}, shift {row['shift']:.2f} cm-1")

    # deviations of ADAPT-VQE from CCSD at equal step
    ref = {r["step"]: r for r in rows if r["method"] == "CCSD"}
    for r in rows:
        c = ref[r["step"]]
        r["d_harmonic"] = round(r["harmonic"] - c["harmonic"], 2)
        r["d_shift"] = round(r["shift"] - c["shift"], 2)
        r["d_shift_percent"] = round(100 * (r["shift"] - c["shift"]) / c["shift"], 2)

    out = RESULTS / "h2_summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary written to {out}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
