"""
Hybrid scheme for water: Hessian at a high level, quartic force field at a
low level (Barnes, Schindler, Compagnon and Allouche, J. Mol. Model. 22, 285
(2016)). Compared with the full high-level and full low-level calculation,
with the MP2/aug-cc-pVTZ result of 01_water and with experiment.

    high level : B3LYP/def2-TZVP (PySCF, default) or ORCA (--orca)
    low level  : GFN2-xTB (tblite)

    python run_hybrid.py            # about 20 min on one core, faster in parallel
    python run_hybrid.py --orca     # B3LYP with ORCA instead of PySCF
    python run_hybrid.py --test     # GFN1-xTB as "high level", seconds

In the hybrid run the intensities use dipole derivatives of the low level
(the dipoles are evaluated on the displaced structures of the force field).
"""

import contextlib
import io
import sys
from pathlib import Path

from ase.build import molecule

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "tutorials"))
import common  # noqa: E402

from overtone import VPT2Result, calculate_anharmonics, estimate_cost  # noqa: E402

EXPERIMENT = {"v2": 1595, "v1": 3657, "v3": 3756, "2v2": 3152, "v1+v2": 5235,
              "v2+v3": 5331, "2v1": 7202, "v1+v3": 7250, "2v3": 7445}

# MP2/aug-cc-pVTZ GVPT2 values of 01_water (production run), used if the
# result file of 01_water is not available. See paper/RESULTS.md.
MP2_FALLBACK = {"v2": 1577.8, "v1": 3654.3, "v3": 3767.8, "2v2": 3123.9,
                "v1+v2": 5215.1, "v2+v3": 5325.4, "2v1": 7227.0,
                "v1+v3": 7267.2, "2v3": 7443.3}
MP2_FILE = HERE.parent / "01_water" / "results" / "anharmonics" / "vpt2_result.json"


def calculators(args):
    if args.test:
        from tblite.ase import TBLite
        high = lambda: TBLite(method="GFN1-xTB", verbosity=0, accuracy=0.01)
    elif args.orca:
        high = common.orca("EnGrad B3LYP def2-TZVP VeryTightSCF DefGrid3",
                           nprocs=args.nprocs, command=common.ORCA5, label="water_b3lyp")
    else:
        from pyscf_calculator import PySCFCalculator
        high = lambda: PySCFCalculator(xc="b3lyp", basis="def2-tzvp")
    return high, common.xtb()


def bands(result):
    modes = common.water_modes(result)
    return {label: common.band(result, common.quanta_of(label, modes)) for label in EXPERIMENT}


def main():
    p = common.parser(__doc__.split("\n")[1])
    p.add_argument("--orca", action="store_true", help="B3LYP with ORCA instead of PySCF")
    args = p.parse_args()
    high, low = calculators(args)
    out = HERE / ("results_test" if args.test else "results")
    atoms = molecule("H2O")
    kw = dict(derivative_source="forces", intensities=True, max_quanta=3, save_results=True)

    if args.dry_run:
        c = estimate_cost(atoms, derivative_source="forces")
        print(f"full high level: {c['total']} high-level single points")
        print(f"hybrid: {c['harmonic']} high-level + {c['anharmonic']} low-level single points")
        return

    high_atoms = common.optimise(atoms, high, logfile=str(out / "opt_high.log"),
                                 save=out / "optimised_high.traj")
    low_atoms = common.optimise(atoms, low, logfile=str(out / "opt_low.log"),
                                save=out / "optimised_low.traj")
    with contextlib.redirect_stdout(io.StringIO()):
        runs = {
            "high": calculate_anharmonics(high_atoms, high, out / "high", **kw),
            "hybrid": calculate_anharmonics(high_atoms, low, out / "hybrid",
                                            setup_harmonic_calculator=high, **kw),
            "low": calculate_anharmonics(low_atoms, low, out / "low", **kw),
        }
    if MP2_FILE.exists():
        mp2 = {k: t.energy for k, t in bands(VPT2Result.load(MP2_FILE)).items()}
        print(f"MP2 values from {MP2_FILE}")
    else:
        mp2 = MP2_FALLBACK
        print("MP2 values from the stored production results (01_water not found)")

    b = {name: bands(r) for name, r in runs.items()}
    cost = estimate_cost(atoms, derivative_source="forces")
    rows = []
    for label, nu in EXPERIMENT.items():
        rows.append({
            "band": label,
            "MP2": round(mp2[label], 1),
            "B3LYP": round(b["high"][label].energy, 1),
            "hybrid": round(b["hybrid"][label].energy, 1),
            "GFN2-xTB": round(b["low"][label].energy, 1),
            "exp": nu,
            "B3LYP harmonic": round(b["high"][label].harmonic, 1),
            "hybrid harmonic": round(b["hybrid"][label].harmonic, 1),
        })
    rows.append({"band": "high-level single points", "MP2": "", "B3LYP": cost["total"],
                 "hybrid": cost["harmonic"], "GFN2-xTB": 0, "exp": "",
                 "B3LYP harmonic": "", "hybrid harmonic": ""})
    common.write_csv(rows, out / "hybrid_comparison.csv")
    common.write_latex(rows, ["band", "MP2", "B3LYP", "hybrid", "GFN2-xTB", "exp"],
                       out / "hybrid_comparison.tex")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
