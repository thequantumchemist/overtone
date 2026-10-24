"""
Active learning of the quartic force field with FALCON.

For each system, a fully exact reference run is compared with FALCON runs at
several uncertainty thresholds. The Hessian is always computed exactly
(hybrid scheme). Its displaced structures and the optimisation trajectory
serve as initial training data. FALCON then decides for every anharmonic
displacement whether an exact calculation is needed.

    water    : MP2/aug-cc-pVTZ   (ORCA 5.0.4)    3 modes
    methane  : B3LYP/cc-pVDZ     (ORCA 6.0.1)    9 modes
    benzene  : GFN2-xTB (tblite)                30 modes, D6h
    pyridine : GFN2-xTB (tblite)                27 modes, C2v

Benzene and pyridine have almost the same size but very different symmetry.
SOAP is invariant to symmetry operations, so symmetry-equivalent displaced
structures look identical to the model. The difference between the two
molecules shows how much FALCON gains from symmetry and how much from
learning.

Requirements: pip install falcon-md dscribe scikit-learn. The SKLEARN_GPR
model is part of the upcoming FALCON release.

Two evaluations per threshold:
    mixed : the active-learning run itself. Each energy comes either from
            the exact method or from the ML model.
    final : the trained model then evaluates every displacement (no exact
            calls). All stencil points come from one smooth surface.

    python run_falcon.py --system water
    python run_falcon.py --system methane --thresholds 0.001 0.0005 0.00025
    python run_falcon.py --system pyridine --quick # small grid, local check
    python run_falcon.py --system water --test     # GFN2-xTB as "exact" method
"""

import contextlib
import io
import sys
from pathlib import Path

from ase.build import molecule
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from overtone import calculate_anharmonics, estimate_cost  # noqa: E402

HERE = Path(__file__).parent

# settings of the original calculations (see the SI)
# train_start: number of initial calls that are always exact. Water and
# methane use 50, as in the production runs. With the exact Hessian
# structures as initial training data, fewer are needed (benzene, pyridine).
#
# modelsize / max_clusters: FALCON splits the training data into clusters of
# about modelsize structures, each with its own GPR. With max_clusters=1 (the
# FALCON default) one model is trained on all data, and every retraining gets
# slower. Benzene and pyridine use the values of the FALCON tutorial.
SYSTEMS = {
    "water": dict(formula="H2O", simpleinput="EnGrad MP2 aug-cc-pVTZ VeryTightSCF",
                  command=common.ORCA5, train_start=50, modelsize=500, max_clusters=1),
    "methane": dict(formula="CH4", simpleinput="EnGrad B3LYP cc-pVDZ VeryTightSCF",
                    command=common.ORCA6, train_start=50, modelsize=500, max_clusters=1),
    "benzene": dict(formula="C6H6", xtb=True, train_start=10, modelsize=250, max_clusters=20,
                    thresholds=[0.005]),
    "pyridine": dict(formula="C5H5N", xtb=True, train_start=10, modelsize=250, max_clusters=20,
                     thresholds=[0.005]),
}
# Thresholds: the total-energy uncertainty grows with the number of atoms.
# For the larger molecules a looser threshold of 5 meV is used. The final
# evaluation (trained model everywhere) avoids the noise of mixed stencils.
DEFAULT_THRESHOLDS = [0.001, 0.0005, 0.00025]
STENCIL, SOURCE = 1, "energy"           # same grid for reference and FALCON runs


def energy_only(model_class):
    """
    Model that does not predict forces.

    OVERTONE's energy route only needs energies. FALCON nevertheless asks the
    model for forces at every point, and without trained forces they come
    from numerical differentiation (6 N extra predictions, about 100 ms per
    point for pyridine). FALCON uses the force uncertainty only if
    accuracy_f is set, which it is not here, so skipping the forces changes
    no decision and no energy.
    """
    import numpy as np

    class EnergyOnly(model_class):
        def predict_forces(self, atoms):
            return np.zeros((len(atoms), 3))

        def predict_uncertainty_forces(self, atoms):
            return np.zeros((len(atoms), 3))

    EnergyOnly.__name__ = f"EnergyOnly{model_class.__name__}"
    return EnergyOnly


def make_falcon_factory(exact_setup, run_dir, threshold, opt_traj, train_start,
                        modelsize=500, max_clusters=1):
    """
    One FALCON instance for the whole anharmonic run, created on first use.
    Training data: optimisation trajectory + exact harmonic displacements.
    """
    from falcon_md.otf_calculator import FALCON
    try:
        from falcon_md.models import SKLEARN_GPR  # upcoming FALCON release
    except ImportError:
        try:
            from sklearn_models import SKLEARN_GPR  # local copy, if available
        except ImportError as err:
            raise ImportError(
                "SKLEARN_GPR not found. It is part of the upcoming FALCON release "
                "(pip install --upgrade falcon-md)."
            ) from err

    state = {}

    def setup():
        if "calc" not in state:
            harm = read(run_dir / "harmonics" / "harmonic_displacements.traj", ":")
            training = list(opt_traj) + list(harm)
            model = energy_only(SKLEARN_GPR)(atoms_example=training[0], descriptor="soap",
                                             n_jobs=1)
            state["n0"] = len(training)
            state["calc"] = FALCON(
                model=model, calc=exact_setup(), training_data=training,
                accuracy_e=threshold, td_filename=str(run_dir / "falcon_training.traj"),
                train_start=train_start, modelsize=modelsize, max_clusters=max_clusters,
            )
        return state["calc"]

    def n_exact():
        return len(state["calc"].training_data) - state["n0"] if "calc" in state else 0

    def freeze():
        """Prediction only: no further exact calculations or retraining."""
        calc = state["calc"]
        calc.accuracy_e = float("inf")          # instance attributes, not the class
        calc.accuracy_f = float("inf")
        calc.train_every = 1e12
        calc.log_test = lambda step: False

    return setup, n_exact, freeze


def main():
    p = common.parser(__doc__.split("\n")[1])
    p.add_argument("--system", choices=list(SYSTEMS), required=True)
    p.add_argument("--thresholds", type=float, nargs="+", default=None,
                   help="FALCON accuracy_e in eV (default: per system)")
    p.add_argument("--train-start", type=int, default=None,
                   help="initial calls that are always exact (default: per system)")
    p.add_argument("--modelsize", type=int, default=None, help="FALCON modelsize (default: per system)")
    p.add_argument("--max-clusters", type=int, default=None, help="FALCON max_clusters (default: per system)")
    p.add_argument("--quick", action="store_true",
                   help="small grid (stencil 0, coupling 2) for a local check only")
    args = p.parse_args()

    sysdef = SYSTEMS[args.system]
    train_start = args.train_start if args.train_start is not None else sysdef["train_start"]
    modelsize = args.modelsize if args.modelsize is not None else sysdef["modelsize"]
    if args.thresholds is None:
        args.thresholds = sysdef.get("thresholds", DEFAULT_THRESHOLDS)
    max_clusters = args.max_clusters if args.max_clusters is not None else sysdef["max_clusters"]
    if sysdef.get("xtb"):
        exact = common.xtb()
        base = HERE / (f"results_quick_{args.system}" if args.quick else f"results_{args.system}")
    elif args.test:
        exact = common.xtb()
        base = HERE / f"results_test_{args.system}"
    else:
        exact = common.orca(sysdef["simpleinput"], nprocs=args.nprocs,
                            command=sysdef["command"], label=args.system)
        base = HERE / f"results_{args.system}"
    base.mkdir(parents=True, exist_ok=True)

    atoms = molecule(sysdef["formula"])
    stencil, coupling = (0, 2) if args.quick else (STENCIL, 3)
    cost = estimate_cost(atoms, derivative_source=SOURCE, stencil=stencil, coupling=coupling)
    forces_cost = estimate_cost(atoms, derivative_source="forces", stencil=stencil,
                                coupling=coupling)["total"]
    if args.dry_run:
        print(cost)
        return

    opt_log = base / "opt.traj"
    from ase.optimize import BFGS
    if not (base / "opt.done").exists():  # restart: keep a finished optimisation
        a = atoms.copy()
        a.calc = exact()
        BFGS(a, logfile=str(base / "opt.log"), trajectory=str(opt_log)).run(fmax=1e-4)
        (base / "opt.done").touch()
    opt_traj = read(opt_log, ":")
    atoms = opt_traj[-1]

    common_kw = dict(derivative_source=SOURCE, stencil=stencil, coupling=coupling,
                     method="gvpt2", save_results=True)
    if sysdef.get("xtb"):
        # cheap exact method: no restart files (hundreds of thousands of points)
        from overtone.engines import SerialEngine
        common_kw["engine"] = SerialEngine(restart=False)
    rows = []
    with contextlib.redirect_stdout(io.StringIO()):
        ref = calculate_anharmonics(atoms.copy(), exact, base / "exact", **common_kw)
    f_ref = common.fundamentals(ref)
    order = sorted(f_ref, key=lambda m: f_ref[m].harmonic)
    n_ref = cost["harmonic"] + cost["anharmonic"]
    ref_values = [f_ref[m].energy for m in order]
    rows.append({"threshold (eV)": "exact", "evaluation": "exact",
                 "exact calculations": n_ref, "fraction (%)": 100.0,
                 "fraction of force route (%)": round(100 * n_ref / forces_cost, 1),
                 "train_start": "", "max |dev|": 0.0, "mean |dev|": 0.0,
                 **{f"mode {i + 1}": round(v, 2) for i, v in enumerate(ref_values)}})

    for thr in args.thresholds:
        run_dir = base / f"falcon_{thr}"
        setup_ml, n_exact, freeze = make_falcon_factory(exact, run_dir, thr, opt_traj,
                                                        train_start, modelsize, max_clusters)
        with contextlib.redirect_stdout(io.StringIO()):
            res = calculate_anharmonics(atoms.copy(), setup_ml, run_dir,
                                        setup_harmonic_calculator=exact, **common_kw)
        n = cost["harmonic"] + n_exact()
        freeze()
        with contextlib.redirect_stdout(io.StringIO()):
            final = calculate_anharmonics(atoms.copy(), setup_ml, run_dir / "final",
                                          setup_harmonic_calculator=exact, **common_kw)
        assert cost["harmonic"] + n_exact() == n, "frozen model made exact calls"
        for kind, r in (("mixed", res), ("final", final)):
            f = common.fundamentals(r)
            values = [f[m].energy for m in order]
            dev = [abs(v - ref) for v, ref in zip(values, ref_values)]
            rows.append({"threshold (eV)": thr, "evaluation": kind,
                         "exact calculations": n, "fraction (%)": round(100 * n / n_ref, 1),
                         "fraction of force route (%)": round(100 * n / forces_cost, 1),
                         "train_start": train_start,
                         "max |dev|": round(max(dev), 2),
                         "mean |dev|": round(sum(dev) / len(dev), 2),
                         **{f"mode {i + 1}": round(v, 2) for i, v in enumerate(values)}})
            print({k: rows[-1][k] for k in ("threshold (eV)", "evaluation", "exact calculations",
                                             "fraction (%)", "max |dev|", "mean |dev|")},
                  flush=True)

    common.write_csv(rows, base / f"falcon_{args.system}.csv")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
