import shutil
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

from ase import Atoms
from ase.io import read, write

try:
    SHEPHEX_AVAILABLE = True
    from shephex import Experiment
    from shephex.executor import Executor, LocalExecutor
    from shephex.experiment import ExperimentResult
    from shephex.study import Study
except ModuleNotFoundError:
    SHEPHEX_AVAILABLE = False
    Experiment = object
    Executor = object
    LocalExecutor = object
    ExperimentResult = object
    Study = object


from overtone.engines import BaseEngine
from overtone.engines.exceptions import CalculationsFailed, CalculationsPending

OUTPUT_NAME = "calc_{}.traj"


class ShephexEngine(BaseEngine):
    """Runs every single point as a shephex experiment.

    The engine does not wait for the experiments. Each call submits the ones
    that have not been run yet and returns the trajectory if all of them have
    finished. Otherwise it raises:

    * ``CalculationsFailed`` if any single point failed,
    * ``CalculationsPending`` if some are still queued or running (with an
      asynchronous executor such as shephex's ``SlurmExecutor``). Rerun the
      script when they have finished; finished single points are not repeated.

    With ``LocalExecutor`` the experiments run during the call, so a single
    call finishes the trajectory.

    Parameters
    ----------
    executor
        shephex executor, default ``LocalExecutor()``.
    retry_failed
        Remove failed experiments and run them again, with the calculator
        given in this call. By default they are reported, not rerun.
    """

    def __init__(self, executor: Executor | None = None, retry_failed: bool = False):
        if not SHEPHEX_AVAILABLE:
            raise ImportError("Shephex is not available. Please install shephex to use ShephexEngine.")
        self.executor = executor if executor is not None else LocalExecutor()
        self.retry_failed = retry_failed

    def calculate_trajectory(
        self,
        trajectory: list[Atoms],
        setup_calculator: Callable,
        trajectory_path: Path | str | None = None,
        output_directory: Path | str | None = None,
        overwrite: bool = False,
        properties: Sequence[str] = ("energy",),
        **kwargs,
    ) -> list[Atoms]:
        output_directory = Path(
            output_directory if output_directory is not None else "overtone_hex_output"
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        if trajectory_path is None:
            trajectory_path = output_directory / "trajectory.traj"
        trajectory_path = Path(trajectory_path)
        if not trajectory_path.exists() or overwrite:
            write(trajectory_path.resolve().as_posix(), trajectory)

        # compute_dipole is only added as an option when requested, so the
        # experiment identities (and thus caching) of energy-only runs are
        # unchanged; their rows have compute_dipole None.
        options = {
            "trajectory_path": trajectory_path.resolve().as_posix(),
            "compute_dipole": True if "dipole" in properties else None,
        }

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

        # The experiments are identified by their options only, so the
        # calculator is captured by calculate_energy instead of being an option.
        study = Study(output_directory, avoid_duplicates=True)
        if self.retry_failed:
            study = self._remove_failed(study, options, output_directory)
        for index in range(len(trajectory)):
            study.add_experiment(Experiment(
                function=calculate_energy,
                trajectory_path=options["trajectory_path"],
                index=index,
                root_path=output_directory,
                **({"compute_dipole": True} if options["compute_dipole"] else {}),
            ))

        rows = self._rows(study, options, len(trajectory))
        pending = [identifier for identifier, status in rows.values() if status == "pending"]
        if pending:
            self.executor.execute([
                Experiment.load(self._experiment_directory(output_directory, identifier))
                for identifier in pending
            ])
            # the executor updates the statuses on disk, not in this study
            study = Study(output_directory, avoid_duplicates=True)
            rows = self._rows(study, options, len(trajectory))

        return self._gather(rows, output_directory)

    @staticmethod
    def _experiment_directory(output_directory: Path, identifier: str) -> Path:
        return output_directory / f"{identifier}-{Experiment.extension}"

    @staticmethod
    def _rows(study: Study, options: dict, n_structures: int) -> dict[int, tuple[str, str]]:
        """(identifier, status) of the experiment of each structure.

        Only experiments with exactly these options count: other trajectories,
        or energy-only runs when dipoles are requested, may share the directory.
        """
        rows = {}
        for row in study.table.table:
            if getattr(row, "trajectory_path", None) != options["trajectory_path"]:
                continue
            if getattr(row, "compute_dipole", None) != options["compute_dipole"]:
                continue
            if row.index < n_structures:
                rows[row.index] = (row.identifier, str(row.status))
        missing = set(range(n_structures)) - rows.keys()
        if missing:  # every structure was added to the study before this
            raise RuntimeError(f"No shephex experiment for structures {sorted(missing)}")
        return rows

    def _remove_failed(self, study: Study, options: dict, output_directory: Path) -> Study:
        failed = [
            row.identifier for row in study.table.table
            if getattr(row, "trajectory_path", None) == options["trajectory_path"]
            and getattr(row, "compute_dipole", None) == options["compute_dipole"]
            and str(row.status) == "failed"
        ]
        for identifier in failed:
            shutil.rmtree(self._experiment_directory(output_directory, identifier))
        return Study(output_directory, avoid_duplicates=True) if failed else study

    def _gather(self, rows: dict[int, tuple[str, str]], output_directory: Path) -> list[Atoms]:
        statuses = {index: status for index, (_, status) in rows.items()}
        failed = {
            index: self._error(self._experiment_directory(output_directory, identifier))
            for index, (identifier, status) in rows.items() if status == "failed"
        }
        if failed:
            raise CalculationsFailed(output_directory, len(rows), failed)
        unfinished = Counter(s for s in statuses.values() if s != "completed")
        if unfinished:
            raise CalculationsPending(output_directory, len(rows), unfinished)

        return [
            read(self._experiment_directory(output_directory, rows[index][0])
                 / OUTPUT_NAME.format(index))
            for index in range(len(rows))
        ]

    @staticmethod
    def _error(experiment_directory: Path) -> str:
        try:
            error = ExperimentResult.load(experiment_directory / "shephex").result
        except Exception:
            return "no error recorded"
        error = getattr(error, "message", error)
        if isinstance(error, BaseException):
            return f"{type(error).__name__}: {error}"
        return str(error)
