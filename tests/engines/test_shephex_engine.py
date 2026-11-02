"""
ShephexEngine with shephex's LocalExecutor: every single point becomes a
shephex experiment in its own directory, and the results are gathered back
in input order. A run must give exactly what SerialEngine gives.

Tests that need shephex are skipped when it is not installed (they request
the ``local_executor`` fixture); the ImportError test runs without it.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT

from helpers import queue
from helpers.calculators import WATER_CHARGES, ChargedEMT
from overtone import (
    CalculationsFailed,
    CalculationsPending,
    calculate_anharmonics,
    estimate_cost,
)
from overtone.engines import SerialEngine, ShephexEngine, shephex_engine

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

N_STRUCTURES = 3


def charged_emt():
    return ChargedEMT(WATER_CHARGES)


def experiment_directories(directory):
    return sorted(Path(directory).glob("*-exp"))


@pytest.fixture(scope="module")
def local_executor():
    """shephex's LocalExecutor class; skips the requesting test without shephex."""
    return pytest.importorskip("shephex.executor").LocalExecutor


@pytest.fixture(scope="module")
def structures():
    """Distinct water geometries, so a mix-up of the order is visible."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(N_STRUCTURES):
        atoms = molecule("H2O")
        atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
        out.append(atoms)
    return out


def copies(structures):
    return [atoms.copy() for atoms in structures]


# ---------------------------------------------------------------- engine level


@pytest.fixture(scope="module")
def engine_run(local_executor, structures, tmp_path_factory):
    directory = tmp_path_factory.mktemp("engine")
    trajectory_path = directory / "input.traj"
    result = ShephexEngine(local_executor()).calculate_trajectory(
        copies(structures), EMT, trajectory_path=trajectory_path,
        output_directory=directory, properties=("energy", "forces"),
    )
    return SimpleNamespace(result=result, directory=directory,
                           trajectory_path=trajectory_path)


@pytest.fixture(scope="module")
def rerun(local_executor, structures, engine_run):
    """A second run in the same directory; records what is handed to the executor."""
    executed = []

    class RecordingExecutor(local_executor):
        def execute(self, experiments, *args, **kwargs):
            executed.extend(experiments)
            return super().execute(experiments, *args, **kwargs)

    result = ShephexEngine(RecordingExecutor()).calculate_trajectory(
        copies(structures), EMT, trajectory_path=engine_run.trajectory_path,
        output_directory=engine_run.directory, properties=("energy", "forces"),
    )
    return SimpleNamespace(result=result, executed=executed)


@pytest.fixture(scope="module")
def dipole_run(local_executor, structures, tmp_path_factory):
    directory = tmp_path_factory.mktemp("dipole")
    return ShephexEngine(local_executor()).calculate_trajectory(
        copies(structures), charged_emt, output_directory=directory,
        properties=("energy", "forces", "dipole"),
    )


def reference(atoms, calculator, prop):
    atoms = atoms.copy()
    atoms.calc = calculator()
    return atoms.calc.get_property(prop, atoms)


def test_one_structure_per_input(engine_run):
    assert len(engine_run.result) == N_STRUCTURES


@pytest.mark.parametrize("index", range(N_STRUCTURES))
def test_structures_are_returned_in_input_order(engine_run, structures, index):
    assert np.allclose(engine_run.result[index].positions, structures[index].positions,
                       atol=1e-12)


@pytest.mark.parametrize("index", range(N_STRUCTURES))
def test_energy_matches_calculator(engine_run, structures, index):
    assert np.isclose(engine_run.result[index].get_potential_energy(),
                      reference(structures[index], EMT, "energy"), atol=1e-12)


@pytest.mark.parametrize("index", range(N_STRUCTURES))
def test_forces_match_calculator(engine_run, structures, index):
    assert np.allclose(engine_run.result[index].get_forces(),
                       reference(structures[index], EMT, "forces"), atol=1e-12)


def test_no_dipole_unless_requested(engine_run):
    assert all("dipole" not in atoms.calc.results for atoms in engine_run.result)


@pytest.mark.parametrize("index", range(N_STRUCTURES))
def test_dipole_calculated_when_requested(dipole_run, structures, index):
    assert np.allclose(dipole_run[index].get_dipole_moment(),
                       reference(structures[index], charged_emt, "dipole"), atol=1e-12)


def test_input_trajectory_written_to_trajectory_path(engine_run):
    assert engine_run.trajectory_path.is_file()


def test_one_experiment_directory_per_structure(engine_run):
    assert len(experiment_directories(engine_run.directory)) == N_STRUCTURES


def test_each_structure_written_in_its_experiment_directory(engine_run):
    written = sorted(p.name for p in engine_run.directory.glob("*-exp/calc_*.traj"))
    assert written == sorted(f"calc_{i}.traj" for i in range(N_STRUCTURES))


def test_rerun_executes_nothing(rerun):
    assert rerun.executed == []


@pytest.mark.parametrize("index", range(N_STRUCTURES))
def test_rerun_returns_finished_results(rerun, engine_run, index):
    assert np.isclose(rerun.result[index].get_potential_energy(),
                      engine_run.result[index].get_potential_energy(), atol=0)


def test_default_trajectory_path_is_in_output_directory(local_executor, structures, tmp_path):
    ShephexEngine(local_executor()).calculate_trajectory(
        copies(structures), EMT, output_directory=tmp_path)
    assert (tmp_path / "trajectory.traj").is_file()


def test_default_output_directory_in_working_directory(local_executor, structures,
                                                       tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ShephexEngine(local_executor()).calculate_trajectory(copies(structures), EMT)
    assert len(experiment_directories(tmp_path / "overtone_hex_output")) == N_STRUCTURES


def test_default_executor_is_local(local_executor):
    assert isinstance(ShephexEngine().executor, local_executor)


def test_import_error_without_shephex(monkeypatch):
    monkeypatch.setattr(shephex_engine, "SHEPHEX_AVAILABLE", False)
    with pytest.raises(ImportError, match="shephex"):
        ShephexEngine()


@pytest.mark.xfail(strict=True, reason=(
    "experiments are identified by (trajectory_path, index) only, so a changed "
    "trajectory in the same directory returns the old results, even with overwrite=True"))
def test_changed_structures_are_recalculated(local_executor, structures, tmp_path):
    engine = ShephexEngine(local_executor())
    engine.calculate_trajectory(copies(structures), EMT, output_directory=tmp_path)
    moved = copies(structures)
    for atoms in moved:
        atoms.positions[0, 0] += 0.1
    result = engine.calculate_trajectory(moved, EMT, output_directory=tmp_path,
                                         overwrite=True)
    assert np.allclose(result[0].positions, moved[0].positions)


def test_output_directory_may_be_a_string(local_executor, structures, tmp_path):
    result = ShephexEngine(local_executor()).calculate_trajectory(
        copies(structures), EMT, output_directory=str(tmp_path))
    assert len(result) == N_STRUCTURES


# ------------------------------------------------- failed and pending experiments

FAILING_INDEX = 1


class FailingEMT(EMT):
    """EMT that fails for one of the structures."""

    def calculate(self, atoms=None, *args, **kwargs):
        if np.allclose(atoms.positions, FailingEMT.bad_positions):
            raise RuntimeError("SCF did not converge")
        super().calculate(atoms, *args, **kwargs)


@pytest.fixture
def failing_emt(structures):
    FailingEMT.bad_positions = structures[FAILING_INDEX].positions
    return FailingEMT


@pytest.fixture
def failed_run(local_executor, structures, failing_emt, tmp_path):
    with pytest.raises(CalculationsFailed) as error:
        ShephexEngine(local_executor()).calculate_trajectory(
            copies(structures), failing_emt, output_directory=tmp_path)
    return SimpleNamespace(error=error.value, directory=tmp_path)


def test_failed_calculation_raises(failed_run):
    assert list(failed_run.error.failed) == [FAILING_INDEX]


def test_failed_calculation_reports_the_error(failed_run):
    assert "SCF did not converge" in str(failed_run.error)


def test_failed_calculation_is_not_rerun_by_default(local_executor, structures, failed_run):
    with pytest.raises(CalculationsFailed):
        ShephexEngine(local_executor()).calculate_trajectory(
            copies(structures), EMT, output_directory=failed_run.directory)


def test_retry_failed_reruns_with_the_new_calculator(local_executor, structures, failed_run):
    result = ShephexEngine(local_executor(), retry_failed=True).calculate_trajectory(
        copies(structures), EMT, output_directory=failed_run.directory)
    assert np.isclose(result[FAILING_INDEX].get_potential_energy(),
                      reference(structures[FAILING_INDEX], EMT, "energy"), atol=1e-12)


def test_retry_failed_keeps_one_experiment_per_structure(local_executor, structures,
                                                         failed_run):
    ShephexEngine(local_executor(), retry_failed=True).calculate_trajectory(
        copies(structures), EMT, output_directory=failed_run.directory)
    assert len(experiment_directories(failed_run.directory)) == N_STRUCTURES


@pytest.fixture
def submit_only_executor(local_executor):
    return queue.submit_only_executor()


@pytest.fixture
def pending_run(submit_only_executor, structures, tmp_path):
    with pytest.raises(CalculationsPending) as error:
        ShephexEngine(submit_only_executor()).calculate_trajectory(
            copies(structures), EMT, output_directory=tmp_path)
    return SimpleNamespace(error=error.value, directory=tmp_path)


def test_unfinished_calculations_raise_pending(pending_run):
    assert pending_run.error.statuses == {"submitted": N_STRUCTURES}


def test_pending_calculations_are_not_resubmitted(submit_only_executor, structures,
                                                  pending_run):
    executed = []

    class RecordingExecutor(submit_only_executor):
        def execute(self, experiments, *args, **kwargs):
            executed.extend(experiments)
            return super().execute(experiments, *args, **kwargs)

    with pytest.raises(CalculationsPending):
        ShephexEngine(RecordingExecutor()).calculate_trajectory(
            copies(structures), EMT, output_directory=pending_run.directory)
    assert executed == []


def test_rerun_after_jobs_finish_returns_results(local_executor, structures, pending_run):
    queue.finish_submitted(pending_run.directory)
    result = ShephexEngine(local_executor()).calculate_trajectory(
        copies(structures), EMT, output_directory=pending_run.directory)
    assert np.allclose([a.get_potential_energy() for a in result],
                       [reference(a, EMT, "energy") for a in structures], atol=1e-12)


def test_other_trajectories_in_directory_are_ignored(local_executor, submit_only_executor,
                                                     structures, tmp_path):
    with pytest.raises(CalculationsPending):
        ShephexEngine(submit_only_executor()).calculate_trajectory(
            copies(structures), EMT, output_directory=tmp_path,
            trajectory_path=tmp_path / "other.traj")
    result = ShephexEngine(local_executor()).calculate_trajectory(
        copies(structures), EMT, output_directory=tmp_path)
    assert len(result) == N_STRUCTURES


@pytest.mark.xfail(strict=False, raises=ValueError, reason=(
    "shephex bug: Study treats an experiment whose options are a subset of another's "
    "(energy-only vs compute_dipole=True) as a duplicate, and fails to load the directory "
    "depending on the order of the (random) experiment directories, so not strict"))
def test_dipole_run_does_not_reuse_energy_only_results(local_executor, structures, tmp_path):
    engine = ShephexEngine(local_executor())
    engine.calculate_trajectory(copies(structures), charged_emt, output_directory=tmp_path)
    result = engine.calculate_trajectory(copies(structures), charged_emt,
                                         output_directory=tmp_path,
                                         properties=("energy", "dipole"))
    assert all("dipole" in atoms.calc.results for atoms in result)


# --------------------------------------------------------------- workflow level


def run(atoms, directory, setup_calculator, **kw):
    return calculate_anharmonics(atoms.copy(), setup_calculator, directory, stencil=0, **kw)


@pytest.fixture(scope="module")
def water(optimised_emt):
    return optimised_emt("H2O")


@pytest.fixture(scope="module")
def shephex_workflow(local_executor, water, tmp_path_factory):
    directory = tmp_path_factory.mktemp("shephex")
    result = run(water, directory, EMT, engine=ShephexEngine(local_executor()))
    return SimpleNamespace(result=result, directory=directory)


@pytest.fixture(scope="module")
def serial_workflow(water, tmp_path_factory):
    return run(water, tmp_path_factory.mktemp("serial"), EMT, engine=SerialEngine())


@pytest.fixture(scope="module")
def shephex_intensities(local_executor, charged_water, tmp_path_factory):
    return run(charged_water, tmp_path_factory.mktemp("shephex_ir"), charged_emt,
               engine=ShephexEngine(local_executor()), intensities=True)


@pytest.fixture(scope="module")
def serial_intensities(charged_water, tmp_path_factory):
    return run(charged_water, tmp_path_factory.mktemp("serial_ir"), charged_emt,
               engine=SerialEngine(), intensities=True)


def test_vpt2_frequencies_match_serial_engine(shephex_workflow, serial_workflow):
    assert np.allclose(shephex_workflow.result.anharm_freq, serial_workflow.anharm_freq,
                       atol=1e-10)


def test_intensities_match_serial_engine(shephex_intensities, serial_intensities):
    shephex = [t.intensity for t in shephex_intensities.transitions]
    serial = [t.intensity for t in serial_intensities.transitions]
    assert np.allclose(shephex, serial, atol=1e-10)


@pytest.mark.parametrize("stage", ["harmonic", "anharmonic"])
def test_workflow_experiments_land_in_stage_directory(shephex_workflow, water, stage):
    expected = estimate_cost(water, stencil=0)[stage]
    assert len(experiment_directories(shephex_workflow.directory / f"{stage}s")) == expected


def test_workflow_stops_at_unfinished_stage(submit_only_executor, water, tmp_path):
    with pytest.raises(CalculationsPending):
        run(water, tmp_path, EMT, engine=ShephexEngine(submit_only_executor()))
    assert experiment_directories(tmp_path / "harmonics")
    assert not experiment_directories(tmp_path / "anharmonics")


def test_workflow_continues_on_rerun(local_executor, submit_only_executor, water,
                                     serial_workflow, tmp_path):
    with pytest.raises(CalculationsPending):
        run(water, tmp_path, EMT, engine=ShephexEngine(submit_only_executor()))
    queue.finish_submitted(tmp_path / "harmonics")
    with pytest.raises(CalculationsPending):  # the anharmonic stage is submitted now
        run(water, tmp_path, EMT, engine=ShephexEngine(submit_only_executor()))
    queue.finish_submitted(tmp_path / "anharmonics")
    result = run(water, tmp_path, EMT, engine=ShephexEngine(local_executor()))
    assert np.allclose(result.anharm_freq, serial_workflow.anharm_freq, atol=1e-10)
