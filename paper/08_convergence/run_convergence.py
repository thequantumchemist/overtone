"""
Convergence with respect to the finite-difference stencil, the coupling level
and the step size, at the MP2/aug-cc-pVTZ level of 01_water (SI).

    stencil  0, 1, 2       x  coupling 2, 3     (delta = 0.2)
    delta    0.1, 0.2, 0.3                      (stencil 1, coupling 3)

water        : stencil and step. For water the coupling cannot matter: its only
               three-mode constant phi_123 vanishes by symmetry (A1 x A1 x B2).
formaldehyde : coupling. Its type-2 Fermi resonance runs through a
               three-mode constant, so coupling 2 misses it.

Every setting runs in its own directory and is restartable. For water the
optimised structure of 01_water is reused if available.

    python run_convergence.py --molecule water --dry-run
    python run_convergence.py --molecule water
    python run_convergence.py --molecule formaldehyde
    python run_convergence.py --molecule formaldehyde --test   # GFN2-xTB stand-in
"""

import contextlib
import io
import sys
from pathlib import Path

from ase.build import molecule
from ase.io import read

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

if getattr(common, "COMMON_VERSION", 0) < 3:
    raise SystemExit("paper/common.py is outdated. Copy common.py from the same "
                     "release as this script.")

from overtone import calculate_anharmonics, estimate_cost  # noqa: E402

SETTINGS = ([{"stencil": s, "coupling": c, "anharmonic_delta": 0.2}
             for s in (0, 1, 2) for c in (2, 3)]
            + [{"stencil": 1, "coupling": 3, "anharmonic_delta": d} for d in (0.1, 0.3)])
WATER_BANDS = ["v2", "v1", "v3", "2v2", "v1+v2", "v2+v3", "2v1", "v1+v3", "2v3"]
MOLECULES = {"water": "H2O", "formaldehyde": "H2CO"}


def band_values(res, molecule_name):
    """Water: the nine bands of the paper. Otherwise: all fundamentals, as
    mode 1, 2, ... in the order of increasing harmonic frequency."""
    if molecule_name == "water":
        modes = common.water_modes(res)
        return {b: common.band(res, common.quanta_of(b, modes)).energy for b in WATER_BANDS}
    f = common.fundamentals(res)
    order = sorted(f, key=lambda m: f[m].harmonic)
    return {f"mode {i + 1}": f[m].energy for i, m in enumerate(order)}


def main():
    p = common.parser(__doc__.split("\n")[1])
    p.add_argument("--molecule", choices=list(MOLECULES), default="water")
    args = p.parse_args()
    if args.test:
        setup = common.xtb()
        out = HERE / f"results_test_{args.molecule}"
        geometry = None
    else:
        setup = common.orca("EnGrad MP2 aug-cc-pVTZ VeryTightSCF",
                            blocks="%mp2 Density relaxed end",
                            nprocs=args.nprocs, command=common.ORCA5, label=args.molecule)
        out = HERE / f"results_{args.molecule}"
        # 01_water geometry: optimised.traj (new code) or, for runs with the
        # older code, the first structure of the saved harmonic displacements
        geometry = None
        if args.molecule == "water":
            w = HERE.parent / "01_water" / "results"
            for candidate in (w / "optimised.traj",
                              w / "harmonics" / "harmonic_displacements.traj"):
                if candidate.exists():
                    geometry = candidate
                    break

    atoms = molecule(MOLECULES[args.molecule])
    if args.dry_run:
        total = 0
        for st in SETTINGS:
            n = estimate_cost(atoms, derivative_source="forces", stencil=st["stencil"],
                              coupling=st["coupling"])["total"]
            total += n
            print(f"stencil {st['stencil']}, coupling {st['coupling']}, "
                  f"delta {st['anharmonic_delta']}: {n} single points")
        print(f"total {total} single points")
        return

    if geometry is not None and geometry.exists():
        atoms = read(geometry, index=0)  # first structure = optimised geometry
        print(f"using the optimised structure of 01_water: {geometry}")
    else:
        atoms = common.optimise(atoms, setup, logfile=str(out / "opt.log"),
                                save=out / "optimised.traj")

    rows = []
    for st in SETTINGS:
        name = f"stencil{st['stencil']}_coupling{st['coupling']}_delta{st['anharmonic_delta']}"
        with contextlib.redirect_stdout(io.StringIO()):
            res = calculate_anharmonics(atoms.copy(), setup, out / name,
                                        engine=common.engine(args.shephex),
                                        derivative_source="forces", max_quanta=2,
                                        save_results=True, **st)
        n = estimate_cost(atoms, derivative_source="forces", stencil=st["stencil"],
                          coupling=st["coupling"])["total"]
        bands = band_values(res, args.molecule)
        row = {**st, "single points": n, **{b: round(v, 2) for b, v in bands.items()},
               "Fermi resonances": f"{res.fermi_type1} {res.fermi_type2}"}
        rows.append(row)
        print(row, flush=True)

    # deviations from the production setting (stencil 1, coupling 3, delta 0.2)
    ref = next(r for r in rows if (r["stencil"], r["coupling"], r["anharmonic_delta"]) == (1, 3, 0.2))
    names = list(band_values(res, args.molecule))
    fund = names[:3] if args.molecule == "water" else names
    for r in rows:
        r["max |dev| fundamentals"] = round(max(abs(r[b] - ref[b]) for b in fund), 2)
        r["max |dev| all bands"] = round(max(abs(r[b] - ref[b]) for b in names), 2)
    common.write_csv(rows, out / "convergence.csv")
    common.write_latex(rows, ["stencil", "coupling", "anharmonic_delta", "single points",
                              "max |dev| fundamentals", "max |dev| all bands"],
                       out / "convergence.tex", fmt="{:.2f}")


if __name__ == "__main__":
    main()
