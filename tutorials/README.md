# OVERTONE Tutorials

This directory contains example scripts demonstrating how to compute anharmonic
vibrational spectra with **OVERTONE**. The tutorials are explained in the
[documentation](https://thequantumchemist.github.io/overtone/tutorials/), which includes its code directly from these scripts.
All scripts run as part of the test suite (`tests/test_tutorials.py`).

- `01_quickstart.py`
  Anharmonic IR spectrum of water in one call, using GFN2-xTB (fast, a few seconds).
  [Documentation](https://thequantumchemist.github.io/overtone/tutorials/01-quickstart/)

- `02_step_by_step.py`
  Every step of the workflow called by hand, and a comparison of all VPT2 variants (fast, a few seconds).
  [Documentation](https://thequantumchemist.github.io/overtone/tutorials/02-step-by-step/)

- `03_water_vs_experiment.py`
  Water at the B3LYP/def2-TZVP level with PySCF, compared with the experimental mid- and near-IR band positions (medium, about 5 minutes on a laptop).
  Described below.

- `04_fermi_resonances.py`
  Formaldehyde with a Fermi resonance, comparing all VPT2 variants (fast, a few seconds).
  [Documentation](https://thequantumchemist.github.io/overtone/tutorials/04-fermi-resonances/)

- `05_analysis.py`
  Reloading, recomputing, exporting and plotting the results of tutorial 1 (fast, seconds).
  [Documentation](https://thequantumchemist.github.io/overtone/tutorials/05-analysis/)

- `06_hybrid.py`
  Hessian from B3LYP (PySCF), anharmonic part from GFN2-xTB (medium, about 2 minutes).
  Described below.

- `pyscf_calculator.py`
  A small ASE calculator for PySCF with analytic forces and dipole moments, used by tutorials 3 and 6.

The documentation keeps the code in sync by including sections of the scripts
marked with `# --8<-- [start:name]` and `# --8<-- [end:name]` comments.
Keep these markers when editing the scripts.

---

## Prerequisites

Please install OVERTONE as introduced in the main `README.md`. The tutorials additionally need
[tblite](https://github.com/tblite/tblite) for GFN2-xTB and, for tutorials 3 and 6, [PySCF](https://pyscf.org).

```bash
pip install "overtone-vpt2[tutorial]"
pip install pyscf
```

---

The PySCF tutorials are not yet part of the documentation and are described here.

## Tutorial 3: `03_water_vs_experiment.py`
### Comparison with the Experimental Near-IR Spectrum

This tutorial demonstrates:
- Using a DFT calculator (PySCF) instead of GFN2-xTB
- The importance of tight SCF convergence and a fine integration grid
- Overtones and combination bands in the near-IR region
- Comparing computed bands with experimental band positions

**Run with:**
```bash
python 03_water_vs_experiment.py
```

Output files:
- `water_b3lyp/` - all intermediate results
- `water_vs_experiment.png` - computed spectrum with the experimental band positions as dashed lines

## Detailed Description: `03_water_vs_experiment.py`

### Goal of the Script

The goal of `03_water_vs_experiment.py` is to compare a **realistic calculation with experiment**,
including the near-IR region where overtones and combination bands dominate.

---

### Step 1: A DFT calculator with tight settings

```python
def setup():
    return PySCFCalculator(xc="b3lyp", basis="def2-tzvp")
```

The calculator in `pyscf_calculator.py` converges the SCF to 10⁻¹¹ Hartree and uses a fine
integration grid. Grid and SCF noise would otherwise be amplified by the finite differences.

### Step 2: Spectrum and experimental band positions

```python
result = calculate_anharmonics(atoms, setup, "water_b3lyp", derivative_source="forces",
                               intensities=True, max_quanta=2)
ax = result.plot_spectrum(xmin=1200, xmax=7800, log=True, shape="gaussian", ...)
for nu in EXPERIMENT.values():
    ax.axvline(nu, color="tab:purple", ls="--", lw=0.8)
```

The experimental band positions are taken from J. Chem. Theory Comput. 10, 5426 (2014).

### What to expect

All bands lie about 1-2 % below experiment, which is typical for B3LYP.
The largest deviations occur for the stretching overtones near 7200-7450 cm⁻¹.
There the states are mixed by Darling-Dennison resonances, which OVERTONE does not include.

### Scientific Motivation

GFN2-xTB is fast but places the OH stretches of water 200-300 cm⁻¹ too low.
A DFT calculation with tight settings brings all bands within 1-2 % of experiment.
The remaining deviations in the stretching overtone region show where VPT2 reaches its limits.

---

## Tutorial 6: `06_hybrid.py`
### Hybrid Scheme with Two Levels of Theory

This tutorial demonstrates:
- A separate calculator for the harmonic analysis (`setup_harmonic_calculator`)
- Harmonic frequencies from B3LYP and anharmonic corrections from GFN2-xTB
- The large improvement over GFN2-xTB alone at almost the same cost

**Run with:**
```bash
python 06_hybrid.py
```

Output:
- Fundamentals of GFN2-xTB, of the hybrid scheme and of experiment, printed to the terminal

### Scientific Motivation

Harmonic frequencies depend strongly on the level of theory, anharmonic
corrections much less. Combining a B3LYP Hessian with a GFN2-xTB force field
reduces the errors of the water fundamentals from up to 315 cm⁻¹ (GFN2-xTB alone)
to below 20 cm⁻¹, while the expensive anharmonic part stays cheap.

---

## License

This tutorial and all example scripts are provided under the GPL-3.0 License.
