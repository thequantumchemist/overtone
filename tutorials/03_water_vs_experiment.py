"""
Tutorial 3: water compared with the experimental band positions, including
the near-IR overtones and combination bands (GFN2-xTB, seconds).

    pip install "overtone-vpt2[tutorial]"
    python 03_water_vs_experiment.py

GFN2-xTB places the OH stretches 200-300 cm^-1 too low. For quantitative
results, replace `setup` by the B3LYP/def2-TZVP calculator from PySCF shown
in the comment below (about 3 minutes on a laptop); all bands then lie
within about 2 % of experiment.
"""
# --8<-- [start:imports]
from ase.build import molecule
from ase.optimize import BFGS
from tblite.ase import TBLite

from overtone import calculate_anharmonics
# --8<-- [end:imports]

# --8<-- [start:experiment]
# experimental band centres, J. Chem. Theory Comput. 10, 5426 (2014)
EXPERIMENT = {"v2": 1595, "2v2": 3152, "v1": 3657, "v3": 3756, "v1+v2": 5235,
              "v2+v3": 5331, "2v1": 7202, "v1+v3": 7250, "2v3": 7445}
# --8<-- [end:experiment]


# --8<-- [start:calculator]
def setup():
    return TBLite(method="GFN2-xTB", verbosity=0, accuracy=0.01)
# --8<-- [end:calculator]
# With PySCF (pip install pyscf), for results within about 2 % of experiment:
#
#     from pyscf_calculator import PySCFCalculator
#
#     def setup():
#         return PySCFCalculator(xc="b3lyp", basis="def2-tzvp")


# --8<-- [start:workflow]
atoms = molecule("H2O")
atoms.calc = setup()
BFGS(atoms).run(fmax=1e-4)
result = calculate_anharmonics(atoms, setup, "water_vs_experiment",
                               derivative_source="forces", intensities=True,
                               max_quanta=2)
# --8<-- [end:workflow]

# --8<-- [start:plot]
ax = result.plot_spectrum(xmin=1200, xmax=7800, log=True, shape="gaussian",
                          fwhm={"fundamental": 20, "overtone": 30, "combination": 30})
for nu in EXPERIMENT.values():
    ax.axvline(nu, color="tab:purple", ls="--", lw=0.8)
ax.set_title("H2O, GVPT2 (dashed: experiment)")
ax.figure.savefig("water_vs_experiment.png", dpi=150)
# --8<-- [end:plot]
