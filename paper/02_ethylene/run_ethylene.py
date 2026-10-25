"""
Ethylene, CCSD(T)/cc-pVDZ, against Martin, Lee, Taylor and Francois,
J. Chem. Phys. 103, 2589 (1995). Tests the treatment of Fermi resonances.

ORCA has no analytic CCSD(T) gradients. The Hessian is therefore computed
with numerical gradients (separate harmonic calculator, 73 structures), and
the quartic force field from energies only (about 18,500 single points).

    python run_ethylene.py --dry-run    # number of single points
    python run_ethylene.py              # ORCA
    python run_ethylene.py --test       # GFN2-xTB stand-in
"""

import contextlib
import io
import sys
from pathlib import Path

from ase.build import molecule

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

if getattr(common, "COMMON_VERSION", 0) < 3:
    raise SystemExit("paper/common.py is outdated. Copy common.py from the same "
                     "release as this script.")

from overtone import calculate_anharmonics, recompute_vpt2  # noqa: E402

HERE = Path(__file__).parent

# Martin et al. (1995), CCSD(T)/cc-pVDZ column
HARMONIC = {  # Table I
    "v1": 3163, "v2": 1671, "v3": 1358, "v4": 1030, "v5": 3233, "v6": 1229,
    "v7": 949, "v8": 909, "v9": 3259, "v10": 820, "v11": 3144, "v12": 1461,
}
FUNDAMENTAL = {  # Table II, final values (resonances treated variationally)
    "v1": 3015.7, "v2": 1615.5, "v3": 1328.9, "v4": 1004.0, "v5": 3084.0,
    "v6": 1205.4, "v7": 930.8, "v8": 892.8, "v9": 3106.9, "v10": 813.6,
    "v11": 2973.6, "v12": 1420.8,
}
DEPERTURBED = {"v2": 1624.850, "v11": 2999.826}  # Table II footnotes b and c
UNCORRECTED = {"v2": 1630.4}                      # Table II footnote b
X_DIAG = {  # Table IV, CCSD(T)/cc-pVDZ
    "v1": -14.328, "v2": -1.329, "v3": -1.178, "v4": -1.918, "v5": -16.484,
    "v6": -0.578, "v7": 0.872, "v8": 3.224, "v9": -16.449, "v10": 3.540,
    "v11": -14.316, "v12": -3.005,
}


def main():
    args = common.parser(__doc__.split("\n")[1]).parse_args()
    if args.test:
        setup = setup_harm = common.xtb()
        out = HERE / "results_test"
        source = "forces"
    else:
        # anharmonic part: energies only. Frozen core is the ORCA default,
        # as in Martin et al. (carbon 1s doubly occupied).
        setup = common.orca("CCSD(T) cc-pVDZ VeryTightSCF", nprocs=args.nprocs,
                            command=common.ORCA5, label="ethylene")
        # harmonic part and optimisation: numerical CCSD(T) gradients.
        # Check that your ORCA version writes the .engrad file with NumGrad.
        setup_harm = common.orca("EnGrad NumGrad CCSD(T) cc-pVDZ VeryTightSCF",
                                 nprocs=args.nprocs, command=common.ORCA5,
                                 label="ethylene_grad")
        out = HERE / "results"
        source = "energy"

    atoms = molecule("C2H4")
    if args.dry_run:
        calculate_anharmonics(atoms, setup, out, derivative_source=source, dry_run=True)
        return

    atoms = common.optimise(atoms, setup_harm, logfile=str(HERE / "opt.log"),
                            save=out / "optimised.traj")
    with contextlib.redirect_stdout(io.StringIO()):
        res = calculate_anharmonics(
            atoms, setup, out, setup_harmonic_calculator=setup_harm,
            engine=common.engine(args.shephex), derivative_source=source,
            method="gvpt2", max_quanta=2, save_results=True,
        )
        std = recompute_vpt2(out, method="standard")
        dep = recompute_vpt2(out, method="deperturbed")
        dc = recompute_vpt2(out, method="dcpt2")

    fg, fs, fd, fc = (common.fundamentals(r) for r in (res, std, dep, dc))
    modes = common.assign_modes({m: t.harmonic for m, t in fg.items()}, HARMONIC)

    rows = []
    for label in HARMONIC:
        m = modes[label]
        rows.append({
            "mode": label,
            "harmonic": round(fg[m].harmonic, 1),
            "Martin harmonic": HARMONIC[label],
            "VPT2": round(fs[m].energy, 1),
            "deperturbed": round(fd[m].energy, 1),
            "DCPT2": round(fc[m].energy, 1),
            "GVPT2": round(fg[m].energy, 1),
            "Martin": FUNDAMENTAL[label],
            "Martin deperturbed": DEPERTURBED.get(label, ""),
            "GVPT2-Martin": round(fg[m].energy - FUNDAMENTAL[label], 1),
            "X_ii": round(float(res.X[m, m]), 3),
            "Martin X_ii": X_DIAG[label],
            "mixing": fg[m].mixing or "",
        })
    common.write_csv(rows, out / "ethylene_fundamentals.csv")
    common.write_latex(rows, ["mode", "harmonic", "Martin harmonic", "VPT2", "GVPT2",
                              "Martin", "GVPT2-Martin"], out / "ethylene_fundamentals.tex")
    print("resonances type 1:", res.fermi_type1, " type 2:", res.fermi_type2)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
