# OVERTONE test suite

## Running

```bash
OMP_NUM_THREADS=1 uv run --extra test pytest            # everything, about 1 min
OMP_NUM_THREADS=1 uv run --extra test pytest -m "not slow"   # a few seconds
OMP_NUM_THREADS=1 uv run --extra test pytest tests/harmonics # one area
```

`OMP_NUM_THREADS=1` matters: competing BLAS threads make the suite about ten
times slower. Tests that need tblite or PySCF are skipped when those are not
installed (tblite is part of the `test` extra).

Markers:

- `slow`: takes more than a few seconds (the CH3NO2 quartic force field
  pipeline and the CH3OH tblite runs). Deselect with `-m "not slow"`.
- `tutorial`: runs a script from `tutorials/`. Deselect with `-m "not tutorial"`.

## Layout

Unit tests mirror the package; end-to-end and documentation checks have their
own directories.

| Directory | Tests |
|---|---|
| `harmonics/` | Hessian projection, symmetrisation, linear molecules, constraints |
| `anharmonics/vpt2/` | standard and deperturbed VPT2 vs exact variational levels |
| `anharmonics/gvpt2/` | GVPT2 with and without Fermi resonances |
| `anharmonics/transitions/` | term values, transition moments, IR intensities |
| `anharmonics/coriolis/` | frame invariance of the Coriolis terms |
| `spectrum/` | broadening and plotting |
| `workflow/` | `calculate_anharmonics`: restart, cost estimate, recompute, calculator usage |
| `integration/` | full pipeline against an exact quartic force field, and GFN2-xTB validation |
| `docs/` | the manual documents the API; the tutorials run |
| `helpers/` | shared code, not tests (see below) |

## Conventions

**One claim per test.** Each test checks one property, so a failure says
exactly what broke. An `np.allclose` over an array, or a pair of values that
only make sense together, is one claim. Loops over cases become
`pytest.mark.parametrize`.

**Expensive work goes in fixtures.** When several claims are made about the
same computation, compute it once in a `scope="module"` (or `"class"`) fixture
and give each claim its own test:

```python
@pytest.fixture(scope="module")
def dyad():
    ...  # solve once
    return SimpleNamespace(result=..., exact=...)


def test_fermi_type1_detected(dyad):
    assert dyad.result.fermi_type1 == [(0, 1)]


@pytest.mark.parametrize("band", BANDS)
def test_energy_matches_exact(dyad, band):
    ...
```

Module-scoped fixtures cannot use `tmp_path`, `capsys` or `monkeypatch`; use
`tmp_path_factory.mktemp(...)`, `contextlib.redirect_stdout` and
`pytest.MonkeyPatch.context()` instead. Fixtures that return `Atoms` hand out
copies, so a test cannot change the input of another.

**Helpers vs fixtures.** Plain code that tests import goes in `helpers/`
(importable as `helpers.*` because `tests/` is on the pytest `pythonpath`):

- `helpers/models.py`: model quartic force fields, the exact variational
  solver, `solve()` for VPT2 on a model
- `helpers/calculators.py`: ASE calculators used as test doubles
  (`QFFCalculator`, `ChargedEMT`, `counting_emt`)
- `helpers/harmonics.py`, `helpers/units.py`

Fixtures shared across directories go in `tests/conftest.py`
(`optimised_emt`, `charged_water`); fixtures used by one file stay in that file.

**Output and warnings.** pytest captures stdout, so tests do not silence it.
Use `capsys` to check printed output. Silence an expected warning with
`@pytest.mark.filterwarnings`, or check it with `pytest.warns`.

## Coverage

```bash
OMP_NUM_THREADS=1 uv run --extra test --with pytest-cov pytest --cov=overtone --cov-report=term-missing
```
