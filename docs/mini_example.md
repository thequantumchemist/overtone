## Minimal example

The anharmonic IR spectrum of water with GFN2-xTB, taken from the
[quickstart tutorial](tutorials/01-quickstart.md):

```python
--8<-- "tutorials/01_quickstart.py:imports"


--8<-- "tutorials/01_quickstart.py:calculator"


--8<-- "tutorials/01_quickstart.py:optimise"

--8<-- "tutorials/01_quickstart.py:workflow"
--8<-- "tutorials/01_quickstart.py:results"
```

Use `dry_run=True` to see the number of single points before starting a large
calculation, and `recompute_vpt2(directory, method=...)` to repeat only the
VPT2 step of a finished run with another method.

## Important settings

| Parameter | Default | Meaning |
|---|---|---|
| `method` | `"gvpt2"` | VPT2 variant (`"standard"`, `"deperturbed"`, `"dcpt2"`, `"gvpt2"`) |
| `derivative_source` | `"energy"` | differentiate energies or analytic forces (fewer points) |
| `anharmonic_delta` | `0.2` | step in dimensionless normal coordinates, not in Angstrom |
| `stencil` | `1` | finite-difference stencil (0 to 3) |
| `coupling` | `3` | 3 includes all cubic three-mode couplings |
| `harmonic_delta` | `0.005` | Cartesian step for the Hessian in Angstrom |
| `intensities` | `False` | compute IR intensities (needs dipoles) |
| `max_quanta` | `2` | highest total number of quanta in the band list |

Finite differences amplify numerical noise of the energies. Converge SCF
calculations tightly (e.g. ORCA `VeryTightSCF`). The [manual](manual.md)
describes every function, class and parameter with its default value.
