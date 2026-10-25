"""
Tutorial 1: anharmonic IR spectrum of water in one call (GFN2-xTB, seconds).

    pip install "overtone-vpt2[tutorial]"
    python 01_quickstart.py
"""
# --8<-- [start:imports]
from ase.build import molecule
from ase.optimize import BFGS
from tblite.ase import TBLite

from overtone import calculate_anharmonics
# --8<-- [end:imports]


# --8<-- [start:calculator]
def setup_calculator():
    # tight SCF convergence: finite differences amplify numerical noise
    return TBLite(method="GFN2-xTB", verbosity=0, accuracy=0.01)
# --8<-- [end:calculator]


# --8<-- [start:optimise]
atoms = molecule("H2O")
atoms.calc = setup_calculator()
BFGS(atoms).run(fmax=1e-4)          # VPT2 needs a tightly optimised minimum
# --8<-- [end:optimise]

# --8<-- [start:workflow]
result = calculate_anharmonics(
    atoms, setup_calculator, "water_quickstart",
    derivative_source="forces",      # analytic forces need fewer points
    intensities=True,                # needs dipoles from the calculator
    max_quanta=3,                    # fundamentals, overtones, combinations
    save_results=True,               # keep JSON files for later analysis
)
# --8<-- [end:workflow]
# --8<-- [start:results]
ax = result.plot_spectrum(xmin=1000, xmax=8000, log=True)
# --8<-- [end:results]
ax.figure.savefig("water_quickstart.png", dpi=150)
