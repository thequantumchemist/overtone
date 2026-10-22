"""
Restart: an interrupted run, started again in the same directory, must only
calculate the missing structures and give exactly the uninterrupted result.
"""
import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from overtone import calculate_anharmonics, estimate_cost
from overtone.engines import SerialEngine


class Crash(Exception):
    pass


def counting_emt(counter, crash_after=None):
    class Counting(EMT):
        def calculate(self, *args, **kwargs):
            counter["n"] += 1
            if crash_after is not None and counter["n"] > crash_after:
                raise Crash("simulated crash")
            super().calculate(*args, **kwargs)
    return Counting


@pytest.fixture(scope="module")
def water():
    atoms = molecule("H2O")
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-7)
    return atoms


def run(atoms, directory, factory):
    with contextlib.redirect_stdout(io.StringIO()) as out, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = calculate_anharmonics(atoms.copy(), factory, Path(directory), stencil=0,
                                    derivative_source="forces")
    return res, out.getvalue()


def test_interrupted_run_resumes_without_recalculation(water, tmp_path):
    total = estimate_cost(water, derivative_source="forces", stencil=0)["total"]
    reference, _ = run(water, tmp_path / "reference", EMT)

    # first attempt crashes in the middle of the anharmonic part
    counter = {"n": 0}
    with pytest.raises(Crash):
        run(water, tmp_path / "run", counting_emt(counter, crash_after=40))
    done = len(list((tmp_path / "run").rglob("job_*.traj")))
    assert done == 40

    # restart in the same directory: only the missing structures are calculated
    counter2 = {"n": 0}
    resumed, log = run(water, tmp_path / "run", counting_emt(counter2))
    assert counter2["n"] == total - 40
    assert "Restart: reused" in log
    assert np.allclose(resumed.anharm_freq, reference.anharm_freq, atol=1e-10)

    # a third run calculates nothing at all
    counter3 = {"n": 0}
    run(water, tmp_path / "run", counting_emt(counter3))
    assert counter3["n"] == 0


def test_changed_structures_are_recalculated(water, tmp_path):
    run(water, tmp_path, EMT)
    moved = water.copy()
    moved.positions[0, 0] += 1e-4          # different reference geometry
    counter = {"n": 0}
    run(moved, tmp_path, counting_emt(counter))
    assert counter["n"] == estimate_cost(moved, derivative_source="forces", stencil=0)["total"]


def test_restart_can_be_switched_off(water, tmp_path):
    engine = SerialEngine(restart=False)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculate_anharmonics(water.copy(), EMT, tmp_path, engine=engine, stencil=0)
    assert not list(tmp_path.rglob("job_*.traj"))
