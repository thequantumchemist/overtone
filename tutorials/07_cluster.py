"""
Tutorial 7: the single points on a Slurm cluster with shephex.

The quickstart calculation of water, with every single point submitted as a
Slurm job. The script does not wait for the jobs: it submits the single points
of the current stage and stops. Run it again when the jobs have finished, and
it continues with the next stage. Finished single points are never repeated.

    pip install "overtone-vpt2[tutorial,cluster]"
    export OMP_NUM_THREADS=1    # passed on to the jobs
    python 07_cluster.py        # submits the harmonic single points
    python 07_cluster.py        # when finished: submits the anharmonic ones
    python 07_cluster.py        # when finished: VPT2 and the spectrum
"""
# --8<-- [start:imports]
from ase.build import molecule
from ase.optimize import BFGS
from shephex.executor import SlurmExecutor
from tblite.ase import TBLite

from overtone import CalculationsPending, calculate_anharmonics
from overtone.engines import ShephexEngine
# --8<-- [end:imports]


def setup_calculator():
    return TBLite(method="GFN2-xTB", verbosity=0, accuracy=0.01)


if __name__ == "__main__":  # shephex only submits from the main script
    atoms = molecule("H2O")
    atoms.calc = setup_calculator()
    BFGS(atoms, logfile=None).run(fmax=1e-4)

    # --8<-- [start:executor]
    executor = SlurmExecutor(
        directory="slurm",        # the job scripts
        environment="current",    # run the jobs in this Python environment
        array_limit=20,           # at most 20 single points at once
        time="00:10:00",          # per single point
        ntasks=1,
        cpus_per_task=1,
        # partition="...",        # and any other sbatch option
    )
    engine = ShephexEngine(executor)
    # --8<-- [end:executor]

    # --8<-- [start:workflow]
    try:
        result = calculate_anharmonics(
            atoms, setup_calculator, "water_cluster",
            derivative_source="forces",
            intensities=True,
            max_quanta=3,
            engine=engine,
        )
    except CalculationsPending as pending:
        print(pending)            # rerun the script when the jobs have finished
    else:
        ax = result.plot_spectrum(xmin=1000, xmax=8000, log=True)
        ax.figure.savefig("water_cluster.png", dpi=150)
    # --8<-- [end:workflow]
