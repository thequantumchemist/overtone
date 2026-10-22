from pathlib import Path
from time import sleep

from ase import Atoms
from ase.io import read, write
from rich.console import Console
from rich.progress import Progress

try:
    SHEPHEX_AVAILABLE = True
    from shephex import Experiment
    from shephex.executor import Executor, LocalExecutor
    from shephex.study import Study
    from shephex.where import path_where
except ModuleNotFoundError:
    SHEPHEX_AVAILABLE = False
    Experiment = object
    Executor = object
    LocalExecutor = object
    Study = object
    path_where = object


from overtone.engines import BaseEngine
from typing import Callable, Sequence

OUTPUT_NAME = "calc_{}.traj"


class ShephexEngine(BaseEngine):
    def __init__(self, executor: Executor | None = None):
        if not SHEPHEX_AVAILABLE:
            raise ImportError("Shephex is not available. Please install shephex to use ShephexEngine.")
        self.executor = executor if executor is not None else LocalExecutor()

    def calculate_trajectory(
        self,
        trajectory: list[Atoms],
        setup_calculator: Callable,
        trajectory_path: Path | None = None,
        output_directory: Path | str | None = None,
        overwrite: bool = False,
        properties: Sequence[str] = ("energy",),
    ) -> list[Atoms]:
        
        if output_directory is None:
            output_directory = Path("overtone_hex_output")
            output_directory.mkdir(exist_ok=True)
        
        if trajectory_path is None:
            trajectory_path = output_directory / "trajectory.traj"
        self._execute_shephex(
            trajectory,
            setup_calculator,
            trajectory_path,
            output_directory=output_directory,
            overwrite=overwrite,
            compute_dipole="dipole" in properties,
        )

        calculated_trajectory = self._gather_shephex(
            directory=output_directory if output_directory is not None else "."
        )
        return calculated_trajectory

    def _execute_shephex(
        self,
        trajectory: list[Atoms],
        setup_calculator: Callable,
        trajectory_path: Path,
        output_directory: Path | str | None = Path("."),
        overwrite: bool = False,
        compute_dipole: bool = False,
    ):
        def calculate_energy(trajectory_path, index, compute_dipole=False):
            from ase.io import read, write

            atoms = read(trajectory_path, index=index)
            atoms.calc = setup_calculator()
            forces = atoms.get_forces()
            energy = atoms.get_potential_energy()
            if compute_dipole:
                # stored in the written .traj together with energy/forces
                atoms.get_dipole_moment()

            write(OUTPUT_NAME.format(index), atoms)
            return energy, forces

        if not trajectory_path.exists() or overwrite:
            write(trajectory_path.resolve().as_posix(), trajectory)

        study = Study(output_directory, avoid_duplicates=True)
        # compute_dipole is only added as an option when requested, so the
        # experiment identities (and thus caching) of energy-only runs are
        # unchanged.
        extra = {"compute_dipole": True} if compute_dipole else {}
        for index in range(len(trajectory)):
            experiment = Experiment(
                function=calculate_energy,
                trajectory_path=trajectory_path.resolve().as_posix(),
                index=index,
                root_path=output_directory,
                **extra,
            )
            study.add_experiment(experiment)

        calculations = study.get_experiments(status="pending")
        self.executor.execute(calculations)

    def _gather_shephex(self, directory: Path | str):
        paths, options = path_where(directory=directory)
        trajectory = {}

        with Progress() as progress:
            task1 = progress.add_task(
                "[red]Gathering structures...", total=len(paths)
            )

            while len(trajectory) < len(paths):
                failure = False
                for path, option_set in zip(paths, options):
                    try:
                        if option_set["index"] in trajectory:
                            continue
                        atoms = read(
                            path / OUTPUT_NAME.format(option_set["index"])
                        )
                        trajectory[option_set["index"]] = atoms
                    except FileNotFoundError:
                        failure = True

                progress.update(task1, completed=len(trajectory))
                if failure:
                    sleep(1)

        console = Console()
        console.print(
            f"[green]Gathered {len(trajectory)} structures from {directory}"
        )

        trajectory = [trajectory[i] for i in range(len(trajectory))]

        return trajectory
