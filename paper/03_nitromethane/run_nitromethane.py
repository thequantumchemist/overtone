"""
Nitromethane, B3LYP/cc-pVTZ, staggered and eclipsed conformers, against
Cornaton, Ringholm, Louant and Ruud, Phys. Chem. Chem. Phys. 18, 4201 (2016).
Fundamentals, harmonic and anharmonic IR intensities, overtones and
combination bands. As in that work, the nearly free methyl torsion is
excluded from the anharmonic treatment (14 instead of 15 modes).

    python run_nitromethane.py              # ORCA, both conformers
    python run_nitromethane.py --test       # GFN2-xTB stand-in
    python run_nitromethane.py --conformer staggered
"""

import contextlib
import io
import sys
from pathlib import Path

from ase.build import molecule
from ase.constraints import FixInternals

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from overtone import calculate_anharmonics, recompute_vpt2  # noqa: E402

HERE = Path(__file__).parent

# Cornaton et al. (2016), Table 2 (cm^-1) and Table 5 (km/mol)
LABELS = [f"v{i}" for i in range(1, 15)]
CORNATON = {
    "staggered": {
        "harmonic": [3193, 3161, 3074, 1632, 1478, 1464, 1427, 1399, 1136, 1109, 924, 662, 616, 481],
        "anharmonic": [3038, 3006, 2953, 1584, 1427, 1414, 1388, 1358, 1106, 1084, 898, 645, 605, 475],
        "I_harm": [0.57, 2.25, 0.71, 298.07, 11.30, 41.14, 54.95, 52.25, 1.22, 11.83, 15.56, 17.19, 5.37, 0.82],
        "I_anharm": [1.29, 3.14, 1.07, 251.14, 11.33, 38.85, 41.37, 68.51, 1.43, 8.53, 18.63, 14.45, 5.08, 0.97],
    },
    "eclipsed": {
        "harmonic": [3194, 3161, 3076, 1632, 1477, 1466, 1427, 1398, 1136, 1109, 925, 657, 624, 482],
        "anharmonic": [3039, 3005, 2956, 1583, 1427, 1416, 1388, 1356, 1107, 1081, 898, 640, 614, 475],
        "I_harm": [2.15, 1.78, 2.36, 580.62, 94.00, 9.70, 38.31, 2.24, 1.26, 30.97, 5.10, 39.17, 7.44, 1.76],
        "I_anharm": [3.94, 2.63, 2.65, 550.36, 92.23, 9.50, 40.69, 0.47, 1.34, 30.22, 6.34, 37.49, 6.84, 1.90],
    },
}
EXPERIMENT = [3080, 3045, 2974, 1583, 1434, 1410, 1397, 1380, 1131, 1096, 918, 657, 603, 475]
# Combination and overtone bands CALCULATED by Cornaton et al. (GVPT2, Sec. 4.2.2).
# They are not experimental values: Cornaton et al. relate them qualitatively
# to weak features of the experimental gas-phase spectrum.
COMBINATIONS = {"v4+v7": 2955, "v4+v8": 2933, "v4+v11": 2471, "2v7": 2768}

C, N, H_PLANE, O = 0, 1, 2, 5  # atom indices in ase.build.molecule("CH3NO2")
DIHEDRAL = {"staggered": 90.0, "eclipsed": 0.0}  # H-C-N-O of one methyl H


def conformer(name):
    """Exact Cs structure. Staggered: one C-H perpendicular to the NO2 plane.
    Eclipsed: one C-H in the NO2 plane."""
    atoms = molecule("CH3NO2")
    atoms.set_dihedral(H_PLANE, C, N, O, DIHEDRAL[name], mask=[0, 0, 1, 1, 1, 0, 0])
    return atoms


def quanta_of(label, modes):
    q = {}
    for part in label.split("+"):
        n = int(part[0]) if part[0].isdigit() else 1
        name = part[1:] if part[0].isdigit() else part
        q[modes[name]] = q.get(modes[name], 0) + n
    return q


def run(name, args):
    if args.test:
        setup = common.xtb()
        out = HERE / f"results_test_{name}"
    else:
        # VeryTightSCF and DefGrid3: SCF and grid noise are amplified by the
        # finite differences. Cornaton et al. used no RI approximation. ORCA
        # may switch on RIJCOSX for hybrids by default; check the output and
        # switch it off if so, to keep the comparison clean.
        setup = common.orca("EnGrad B3LYP cc-pVTZ VeryTightSCF DefGrid3",
                            nprocs=args.nprocs, command=common.ORCA5,
                            label=f"nitromethane_{name}")
        out = HERE / f"results_{name}"

    atoms = conformer(name)
    if args.dry_run:
        calculate_anharmonics(atoms, setup, out, derivative_source="forces",
                              exclude_below=100, dry_run=True)
        return
    fix = FixInternals(dihedrals_deg=[[DIHEDRAL[name], [H_PLANE, C, N, O]]])
    atoms = common.optimise(atoms, setup, constraint=fix, logfile=str(HERE / f"opt_{name}.log"),
                            save=out / "optimised.traj")

    with contextlib.redirect_stdout(io.StringIO()):
        res = calculate_anharmonics(
            atoms, setup, out, engine=common.engine(args.shephex),
            derivative_source="forces", exclude_below=100,  # methyl torsion
            intensities=True, max_quanta=2, method="gvpt2", save_results=True,
        )
        std = recompute_vpt2(out, method="standard")

    ref = CORNATON[name]
    fg, fs = common.fundamentals(res), common.fundamentals(std)
    modes = common.assign_modes({m: t.harmonic for m, t in fg.items()},
                                dict(zip(LABELS, ref["harmonic"])))
    rows = []
    for i, label in enumerate(LABELS):
        m = modes[label]
        rows.append({
            "mode": label,
            "harmonic": round(fg[m].harmonic, 1), "Cornaton harmonic": ref["harmonic"][i],
            "VPT2": round(fs[m].energy, 1), "GVPT2": round(fg[m].energy, 1),
            "Cornaton GVPT2": ref["anharmonic"][i], "exp": EXPERIMENT[i],
            "I_harm": round(float(_harm_intensity(fs[m])), 2),  # unmixed moment
            "Cornaton I_harm": ref["I_harm"][i],
            "I_GVPT2": round(fg[m].intensity, 2), "Cornaton I_anharm": ref["I_anharm"][i],
        })
    common.write_csv(rows, out / f"nitromethane_{name}.csv")
    common.write_latex(rows, ["mode", "harmonic", "Cornaton harmonic", "GVPT2",
                              "Cornaton GVPT2", "exp", "I_harm", "Cornaton I_harm"],
                       out / f"nitromethane_{name}.tex")
    combos = []
    for label, nu_exp in COMBINATIONS.items():
        t = common.band(res, quanta_of(label, modes))
        combos.append({"band": label, "GVPT2": round(t.energy, 1), "Cornaton GVPT2": nu_exp,
                       "intensity": None if t.intensity is None else round(t.intensity, 3)})
    common.write_csv(combos, out / f"nitromethane_{name}_combinations.csv")
    for r in rows + combos:
        print(r)
    return res


def _harm_intensity(t):
    """Double-harmonic intensity of a fundamental (harmonic wavenumber)."""
    from overtone.anharmonics.transitions import INTENSITY_CONSTANT
    import numpy as np
    M = np.asarray(t.transition_dipole)
    return INTENSITY_CONSTANT * t.harmonic * M @ M


def main():
    p = common.parser(__doc__.split("\n")[1])
    p.add_argument("--conformer", choices=["staggered", "eclipsed", "both"], default="both")
    args = p.parse_args()
    names = ["staggered", "eclipsed"] if args.conformer == "both" else [args.conformer]
    results = {n: run(n, args) for n in names}
    if args.dry_run or len(results) < 2:
        return

    # equal-weight average of both rotamers, as in Cornaton et al.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from overtone.spectrum import broaden_transitions

    fig, ax = plt.subplots(figsize=(8, 4))
    x = None
    total = 0
    for n, r in results.items():
        x, y = broaden_transitions(r.transitions, shape="lorentzian", fwhm=10.0,
                                   xmin=400, xmax=3300, resolution=1.0)
        total = total + 0.5 * y
        ax.plot(x, 0.5 * y, lw=0.8, label=n)
    ax.plot(x, total, color="k", lw=1.5, label="average")
    ax.set_xlim(3300, 400)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (km mol$^{-1}$ / cm$^{-1}$)")
    ax.legend(frameon=False)
    out = HERE / ("results_test" if args.test else "results")
    out.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "nitromethane_spectrum.png", dpi=200)
    print(f"written {out / 'nitromethane_spectrum.png'}")


if __name__ == "__main__":
    main()
