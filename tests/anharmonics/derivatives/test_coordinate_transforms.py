import numpy as np

from overtone import CoordinateTransformer


def test_column_frequencies_give_same_transform():
    rng = np.random.default_rng(0)
    modes = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    freq, masses = np.linspace(500, 3000, 6), rng.uniform(1, 4, 6)
    q = rng.normal(size=(6, 2))
    flat = CoordinateTransformer(freq, modes, masses).reduced_norm_to_cart(q)
    column = CoordinateTransformer(freq[:, None], modes, masses).reduced_norm_to_cart(q)
    assert np.allclose(column, flat)
