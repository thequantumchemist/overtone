from overtone import estimate_cost


def test_estimate_cost_matches_actual_run_size(optimised_emt):
    atoms = optimised_emt("H2O")
    cost = estimate_cost(atoms, derivative_source="energy", stencil=1, coupling=3)
    assert cost["n_modes"] == 3 and cost["harmonic"] == 19
    co2_cost = estimate_cost(optimised_emt("CO2"), stencil=0)
    assert co2_cost["n_modes"] == 4
