"""calculate_anharmonics(dry_run=True) only reports the cost."""
import pytest
from ase.calculators.emt import EMT

from helpers.calculators import counting_emt
from overtone import calculate_anharmonics, estimate_cost


@pytest.fixture
def water(optimised_emt):
    return optimised_emt("H2O")


def test_returns_cost_estimate(water, tmp_path):
    cost = calculate_anharmonics(water, EMT, tmp_path / "run", stencil=2, dry_run=True)
    assert cost == estimate_cost(water, stencil=2)


def test_prints_number_of_single_points(water, tmp_path, capsys):
    cost = calculate_anharmonics(water, EMT, tmp_path / "run", dry_run=True)
    assert f"= {cost['total']} single points" in capsys.readouterr().out


def test_calculates_nothing(water, tmp_path):
    counter = {"n": 0}
    calculate_anharmonics(water, counting_emt(counter), tmp_path / "run", dry_run=True)
    assert counter["n"] == 0


def test_creates_no_directory(water, tmp_path):
    calculate_anharmonics(water, EMT, tmp_path / "run", dry_run=True)
    assert not (tmp_path / "run").exists()


def test_exclude_below_gives_upper_bound(water, tmp_path, capsys):
    calculate_anharmonics(water, EMT, tmp_path / "run", exclude_below=100, dry_run=True)
    assert "upper bound" in capsys.readouterr().out


def test_constrained_atoms_have_no_rigid_body_modes(water, tmp_path):
    cost = calculate_anharmonics(water, EMT, tmp_path / "run", constrain_index=[0],
                                 dry_run=True)
    assert cost["n_modes"] == 6  # 3 * 2 free atoms


def test_unknown_derivative_source_raises(water, tmp_path):
    with pytest.raises(ValueError, match="derivative_source"):
        calculate_anharmonics(water, EMT, tmp_path / "run", derivative_source="hessian")


@pytest.mark.xfail(strict=True, reason="derivative_source is only validated after the "
                   "harmonic analysis; estimate_cost treats any value other than "
                   "'energy' as 'forces', so a dry run accepts it")
def test_dry_run_rejects_unknown_derivative_source(water, tmp_path):
    with pytest.raises(ValueError, match="derivative_source"):
        calculate_anharmonics(water, EMT, tmp_path / "run", derivative_source="hessian",
                              dry_run=True)
