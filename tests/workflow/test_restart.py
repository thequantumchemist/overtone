"""
Restart: an interrupted run, started again in the same directory, must only
calculate the missing structures and give exactly the uninterrupted result.
"""
import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest
from ase.calculators.emt import EMT

from helpers.calculators import Crash, counting_emt
from overtone import calculate_anharmonics, estimate_cost
from overtone.engines import SerialEngine

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

CRASH_AFTER = 40


@pytest.fixture(scope="module")
def water(optimised_emt):
    return optimised_emt("H2O")


def run(atoms, directory, factory):
    return calculate_anharmonics(atoms.copy(), factory, directory, stencil=0,
                                 derivative_source="forces")


def counted_run(atoms, directory, **kw):
    """Run and return the result, the number of calculations and the printed log."""
    counter = {"n": 0}
    # class scope cannot use capsys, so capture the log directly
    with contextlib.redirect_stdout(io.StringIO()) as out:
        res = run(atoms, directory, counting_emt(counter, **kw))
    return SimpleNamespace(result=res, n=counter["n"], log=out.getvalue())


class TestInterruptedRun:
    """Crash after CRASH_AFTER calculations, then run twice more in the same directory."""

    @pytest.fixture(scope="class")
    def directory(self, tmp_path_factory):
        return tmp_path_factory.mktemp("run")

    @pytest.fixture(scope="class")
    def crashed(self, water, directory):
        with pytest.raises(Crash):
            counted_run(water, directory, crash_after=CRASH_AFTER)
        return len(list(directory.rglob("job_*.traj")))

    @pytest.fixture(scope="class")
    def resumed(self, water, directory, crashed):
        return counted_run(water, directory)

    @pytest.fixture(scope="class")
    def repeated(self, water, directory, resumed):
        return counted_run(water, directory)

    def test_crash_keeps_finished_structures(self, crashed):
        assert crashed == CRASH_AFTER

    def test_resume_calculates_only_missing_structures(self, water, resumed):
        total = estimate_cost(water, derivative_source="forces", stencil=0)["total"]
        assert resumed.n == total - CRASH_AFTER

    def test_resume_is_reported(self, resumed):
        assert "Restart: reused" in resumed.log

    def test_resumed_result_equals_uninterrupted(self, water, resumed, tmp_path_factory):
        reference = run(water, tmp_path_factory.mktemp("reference"), EMT)
        assert np.allclose(resumed.result.anharm_freq, reference.anharm_freq, atol=1e-10)

    def test_finished_run_calculates_nothing(self, repeated):
        assert repeated.n == 0


def test_changed_structures_are_recalculated(water, tmp_path):
    run(water, tmp_path, EMT)
    moved = water.copy()
    moved.positions[0, 0] += 1e-4          # different reference geometry
    rerun = counted_run(moved, tmp_path)
    assert rerun.n == estimate_cost(moved, derivative_source="forces", stencil=0)["total"]


def test_restart_can_be_switched_off(water, tmp_path):
    engine = SerialEngine(restart=False)
    calculate_anharmonics(water.copy(), EMT, tmp_path, engine=engine, stencil=0)
    assert not list(tmp_path.rglob("job_*.traj"))
