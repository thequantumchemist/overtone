"""
Tutorial 3: water at B3LYP/def2-TZVP (PySCF) compared with the experimental
band positions, including the near-IR overtones and combination bands.

    pip install "overtone[tutorial]" pyscf
    python 03_water_vs_experiment.py      # about 5 minutes on a laptop

Expect all bands about 1-2 % too low, typical for B3LYP. The largest
deviations occur for the stretching overtones near 7200-7450 cm^-1, where
Darling-Dennison coupling (not included) mixes the states.
"""
from ase.build import molecule
from ase.optimize import BFGS

from overtone import calculate_anharmonics
from pyscf_calculator import PySCFCalculator

# experimental band centres, J. Chem. Theory Comput. 10, 5426 (2014)
EXPERIMENT = {"v2": 1595, "2v2": 3152, "v1": 3657, "v3": 3756, "v1+v2": 5235,
              "v2+v3": 5331, "2v1": 7202, "v1+v3": 7250, "2v3": 7445}


def setup():
    return PySCFCalculator(xc="b3lyp", basis="def2-tzvp")


atoms = molecule("H2O")
atoms.calc = setup()
BFGS(atoms).run(fmax=1e-4)
result = calculate_anharmonics(atoms, setup, "water_b3lyp", derivative_source="forces",
                               intensities=True, max_quanta=2)
result.transitions_table()

ax = result.plot_spectrum(xmin=1200, xmax=7800, log=True, shape="gaussian",
                          fwhm={"fundamental": 20, "overtone": 30, "combination": 30})
for nu in EXPERIMENT.values():
    ax.axvline(nu, color="tab:purple", ls="--", lw=0.8)
ax.set_title("H2O, B3LYP/def2-TZVP GVPT2 (dashed: experiment)")
ax.figure.savefig("water_vs_experiment.png", dpi=150)
