# OVERTONE

![OVERTONE](https://raw.githubusercontent.com/thequantumchemist/overtone/main/assets/overtone_banner.png)

OVERTONE computes anharmonic vibrational spectra with second-order vibrational
perturbation theory (VPT2). It works with any calculator of the Atomic
Simulation Environment (ASE), from quantum chemistry programs such as ORCA or
PySCF to machine-learning potentials, on-the-fly active learning with FALCON,
and variational quantum algorithms on (simulated) quantum computers.

OVERTONE provides fundamentals, overtones and combination bands, IR
intensities on request, and plotted spectra including the near-IR.

## Authors

Basila Bhanu Pattani Ameerjan, Mads-Peter Verner Christiansen and Wilke Dononelli

## Features

* Harmonic analysis with symmetrised Hessian and Eckart projection of translations and rotations
* Quartic force field from energies or forces by finite differences in dimensionless normal coordinates
* VPT2 variants: GVPT2 (default), standard VPT2, deperturbed VPT2 and DCPT2
* Automatic detection of Fermi resonances and variational treatment, including intensity borrowing
* Coriolis contributions, linear molecules, constrained atoms (e.g. adsorbates on surfaces)
* Overtones and combination bands up to any number of quanta
* IR intensities (only computed on request, e.g. not for energy-only ML potentials)
* Lorentzian and Gaussian spectra, log scale for the near-IR, experimental reference overlay
* Hybrid schemes with separate calculators for the Hessian and the anharmonic part
* Runs on a single node or distributed over a queueing system with Shephex

## Requirements

* Python 3.11 or later
* NumPy, SciPy, ASE (3.23 or later), pydantic, rich, matplotlib

Optional extras: `cluster` (shephex), `falcon` (falcon-md), `tutorial` (tblite) and `test`.

## Installation

```
pip install overtone-vpt2
```

or the latest version from GitHub

```
pip install git+https://github.com/thequantumchemist/overtone.git
```

For development, clone the repository and install it in editable mode.

```
git clone https://github.com/thequantumchemist/overtone.git
cd overtone
pip install -e ".[test]"
pytest
```

## Minimal example

```python
from ase.build import molecule
from ase.optimize import BFGS
from tblite.ase import TBLite
from overtone import calculate_anharmonics

def setup_calculator():
    return TBLite(method="GFN2-xTB", verbosity=0, accuracy=0.01)

atoms = molecule("H2O")
atoms.calc = setup_calculator()
BFGS(atoms).run(fmax=1e-4)

result = calculate_anharmonics(atoms, setup_calculator, "water",
                               derivative_source="forces", intensities=True, max_quanta=3)
result.transitions_table()
result.plot_spectrum(log=True)
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
calculations tightly (e.g. ORCA `VeryTightSCF`). The docstring of
`calculate_anharmonics` describes every parameter.

## Manual

The [manual](https://github.com/thequantumchemist/overtone/blob/main/docs/manual.md) ([PDF](https://github.com/thequantumchemist/overtone/blob/main/docs/manual.pdf)) describes the installation
and every function, class and parameter of OVERTONE with its default value.

## Tutorial

The `tutorials` folder contains six examples, from a one-call workflow with
GFN2-xTB to Fermi resonances, hybrid schemes and a comparison with the
experimental near-IR spectrum of water.
The `paper` folder contains all calculations of the publication.

## License

This project is licensed under the GPL-3.0 License. See the LICENSE file for details.

## Citation

Please cite the OVERTONE paper (in preparation) when using the code.
