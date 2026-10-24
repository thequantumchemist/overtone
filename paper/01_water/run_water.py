"""
Water, MP2/aug-cc-pVTZ: fundamentals against Barone (2005) and the full
mid- and near-IR band list (up to three quanta) against experiment.

Experimental band centres are those compiled in J. Chem. Theory Comput. 10,
5426 (2014), Table 6. Experimental band intensities are optional and are
summed from a HITRAN line list (--hitran H2O.par, download from hitran.org).

    python run_water.py                 # ORCA, ~200 single points
    python run_water.py --test          # GFN2-xTB stand-in, seconds
    python run_water.py --hitran H2O.par
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from ase.build import molecule

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from overtone import calculate_anharmonics, recompute_vpt2  # noqa: E402

HERE = Path(__file__).parent

# MP2/aug-cc-pVTZ VPT2 values of Barone, J. Chem. Phys. 122, 014108 (2005),
# Tables IV and V (frozen core). Checked against the paper.
BARONE_2005 = {
    "v2": {"harmonic": 1628.0, "anharmonic": 1578.0},
    "v1": {"harmonic": 3822.0, "anharmonic": 3653.0},
    "v3": {"harmonic": 3948.0, "anharmonic": 3767.0},
}

# Experimental band centres (cm^-1), JCTC 10, 5426 (2014), Table 6
EXPERIMENT = {
    "v2": 1595, "v1": 3657, "v3": 3756,
    "2v2": 3152, "v1+v2": 5235, "v2+v3": 5331,
    "2v1": 7202, "v1+v3": 7250, "2v3": 7445,
}


def quanta_of(label, modes):
    """'v1+v3' -> {mode index: quanta} using modes = {'v1': idx, ...}."""
    q = {}
    for part in label.split("+"):
        n = int(part[0]) if part[0].isdigit() else 1
        name = part[1:] if part[0].isdigit() else part
        q[modes[name]] = q.get(modes[name], 0) + n
    return q


def vvv(label):
    """'v1+v3' -> (1, 0, 1) as used by HITRAN."""
    v = [0, 0, 0]
    for part in label.split("+"):
        n = int(part[0]) if part[0].isdigit() else 1
        v[int(part[-1]) - 1] += n
    return tuple(v)


def main():
    p = common.parser(__doc__.split("\n")[1])
    p.add_argument("--hitran", help="HITRAN .par line list for experimental intensities")
    args = p.parse_args()

    if args.test:
        setup = common.xtb()
        out = HERE / "results_test"
    else:
        # VeryTightSCF: finite differences amplify SCF noise (see SI).
        # Density relaxed: ORCA then prints the dipole of the unrelaxed and
        # afterwards of the relaxed MP2 density. ASE reads the last block,
        # i.e. the relaxed MP2 dipole (checked with ORCA 5.0.4: 1.865 D for
        # water). Without this block ASE would read a non-relaxed dipole.
        setup = common.orca("EnGrad MP2 aug-cc-pVTZ VeryTightSCF",
                            blocks="%mp2 Density relaxed end",
                            nprocs=args.nprocs, command=common.ORCA5, label="water")
        out = HERE / "results"

    atoms = molecule("H2O")
    if args.dry_run:
        calculate_anharmonics(atoms, setup, out, derivative_source="forces", dry_run=True)
        return

    atoms = common.optimise(atoms, setup, logfile=str(HERE / "opt.log"),
                            save=out / "optimised.traj")
    with contextlib.redirect_stdout(io.StringIO()):
        res = calculate_anharmonics(
            atoms, setup, out, engine=common.engine(args.shephex),
            derivative_source="forces", intensities=True, max_quanta=3,
            method="gvpt2", save_results=True,
        )
        std = recompute_vpt2(out, method="standard", max_quanta=3)

    # modes: bend < symmetric stretch < antisymmetric stretch
    f = common.fundamentals(res)
    order = sorted(f, key=lambda m: f[m].harmonic)
    modes = {"v2": order[0], "v1": order[1], "v3": order[2]}

    hitran = common.hitran_band_intensities(args.hitran) if args.hitran else {}

    rows = []
    for label, nu_exp in EXPERIMENT.items():
        q = quanta_of(label, modes)
        g, s = common.band(res, q), common.band(std, q)
        rows.append({
            "band": label,
            "harmonic": round(g.harmonic, 1),
            "VPT2": round(s.energy, 1),
            "GVPT2": round(g.energy, 1),
            "exp": nu_exp,
            "GVPT2-exp": round(g.energy - nu_exp, 1),
            "I_GVPT2 (km/mol)": None if g.intensity is None else round(g.intensity, 4),
            "I_exp (km/mol)": round(hitran[vvv(label)], 4) if vvv(label) in hitran else None,
            "Barone harmonic": BARONE_2005.get(label, {}).get("harmonic"),
            "Barone VPT2": BARONE_2005.get(label, {}).get("anharmonic"),
        })
    common.write_csv(rows, out / "water_bands.csv")
    common.write_latex(rows, ["band", "harmonic", "VPT2", "GVPT2", "exp", "GVPT2-exp",
                              "I_GVPT2 (km/mol)"], out / "water_bands.tex")
    for r in rows:
        print(r)

    # figure: computed spectrum with experimental band positions
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    res.plot_spectrum(ax=axes[0], xmin=1200, xmax=4200, show_harmonic=True)
    res.plot_spectrum(ax=axes[1], xmin=2900, xmax=7800, log=True,
                      kinds=["overtone", "combination"], shape="gaussian",
                      fwhm={"overtone": 30, "combination": 30})
    for ax in axes:
        lo, hi = sorted(ax.get_xlim())
        for label, nu in EXPERIMENT.items():
            if lo <= nu <= hi:
                ax.axvline(nu, color="tab:purple", ls="--", lw=0.8)
    axes[0].set_title("H$_2$O fundamentals (dashed: experiment)", fontsize=10)
    axes[1].set_title("H$_2$O overtones and combination bands", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "water_spectrum.png", dpi=200)
    print(f"written {out / 'water_spectrum.png'}")


if __name__ == "__main__":
    main()
