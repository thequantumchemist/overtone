"""
Active learning of the quartic force field with FALCON.

For each system, a fully exact reference run is compared with FALCON runs at
several uncertainty thresholds. The Hessian is always computed exactly
(hybrid scheme). Its displaced structures and the optimisation trajectory
serve as initial training data. FALCON then decides for every anharmonic
displacement whether an exact calculation is needed.

    water   : MP2/aug-cc-pVTZ   (ORCA 5.0.4)
    methane : B3LYP/cc-pVDZ     (ORCA 6.0.1)

Requirements: pip install falcon-md dscribe scikit-learn. The SKLEARN_GPR
model is part of the upcoming FALCON release.

Two evaluations per threshold:
    mixed : the active-learning run itself. Each energy comes either from
            the exact method or from the ML model.
    final : the trained model then evaluates every displacement (no exact
            calls). All stencil points come from one smooth surface.

    python run_falcon.py --system water
    python run_falcon.py --system methane --thresholds 0.001 0.0005 0.00025
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
SYSTEMS = {
    "water": dict(formula="H2O", simpleinput="EnGrad MP2 aug-cc-pVTZ VeryTightSCF",
                  command=common.ORCA5),
    "methane": dict(formula="CH4", simpleinput="EnGrad B3LYP cc-pVDZ VeryTightSCF",
                    command=common.ORCA6),
}
FALCON_SETTINGS = dict(train_start=50)   # as in the original runs
STENCIL, SOURCE = 1, "energy"           # same grid for reference and FALCON runs


def make_falcon_factory(exact_setup, run_dir, threshold, opt_traj):
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
            model = SKLEARN_GPR(atoms_example=training[0], descriptor="soap", n_jobs=1)
            state["n0"] = len(training)
            state["calc"] = FALCON(
                model=model, calc=exact_setup(), training_data=training,
                accuracy_e=threshold, td_filename=str(run_dir / "falcon_training.traj"),
                **FALCON_SETTINGS,
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
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.001, 0.0005, 0.00025], help="FALCON accuracy_e in eV")
    args = p.parse_args()

    sysdef = SYSTEMS[args.system]
    if args.test:
        exact = common.xtb()
        base = HERE / f"results_test_{args.system}"
    else:
        exact = common.orca(sysdef["simpleinput"], nprocs=args.nprocs,
                            command=sysdef["command"], label=args.system)
        base = HERE / f"results_{args.system}"
    base.mkdir(parents=True, exist_ok=True)

    atoms = molecule(sysdef["formula"])
    cost = estimate_cost(atoms, derivative_source=SOURCE, stencil=STENCIL)
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

    common_kw = dict(derivative_source=SOURCE, stencil=STENCIL, method="gvpt2",
                     save_results=True)
    rows = []
    with contextlib.redirect_stdout(io.StringIO()):
        ref = calculate_anharmonics(atoms.copy(), exact, base / "exact", **common_kw)
    f_ref = common.fundamentals(ref)
    order = sorted(f_ref, key=lambda m: f_ref[m].harmonic)
    n_ref = cost["harmonic"] + cost["anharmonic"]
    rows.append({"threshold (eV)": "exact", "evaluation": "exact",
                 "exact calculations": n_ref,
                 "fraction (%)": 100.0,
                 **{f"mode {i + 1}": round(f_ref[m].energy, 2) for i, m in enumerate(order)}})

    for thr in args.thresholds:
        run_dir = base / f"falcon_{thr}"
        setup_ml, n_exact, freeze = make_falcon_factory(exact, run_dir, thr, opt_traj)
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
            rows.append({"threshold (eV)": thr, "evaluation": kind,
                         "exact calculations": n, "fraction (%)": round(100 * n / n_ref, 1),
                         **{f"mode {i + 1}": round(f[m].energy, 2) for i, m in enumerate(order)}})
            print(rows[-1], flush=True)

    common.write_csv(rows, base / f"falcon_{args.system}.csv")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
