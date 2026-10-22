# OVERTONE Tutorials

This directory contains example scripts demonstrating how to compute anharmonic
vibrational spectra with **OVERTONE**.

Installation and general setup are described in the main project `README.md`.
This document focuses exclusively on the functionality and execution of the
tutorials contained in this directory.

The directory contains six tutorial scripts, designed to gradually introduce the main concepts of **OVERTONE**:
1. The complete workflow in a single function call.
2. Full control by calling every step of the workflow yourself.
3. A comparison of a computed near-infrared spectrum with experiment.
4. Fermi resonances and their variational treatment with GVPT2.
5. Analysis of a finished run without new calculations.
6. A hybrid scheme with two levels of theory.

---

## Tutorial Overview

- `01_quickstart.py`
  Anharmonic IR spectrum of water in one call, using GFN2-xTB (fast, a few seconds).

- `02_step_by_step.py`
  Every step of the workflow called by hand, and a comparison of all VPT2 variants (fast, a few seconds).

- `03_water_vs_experiment.py`
  Water at the B3LYP/def2-TZVP level with PySCF, compared with the experimental mid- and near-IR band positions (medium, about 5 minutes on a laptop).

- `04_fermi_resonances.py`
  Formaldehyde with a Fermi resonance, comparing all VPT2 variants (fast, a few seconds).

- `05_analysis.py`
  Reloading, recomputing, exporting and plotting the results of tutorial 1 (fast, seconds).

- `06_hybrid.py`
  Hessian from B3LYP (PySCF), anharmonic part from GFN2-xTB (medium, about 2 minutes).

- `pyscf_calculator.py`
  A small ASE calculator for PySCF with analytic forces and dipole moments, used by tutorial 3.

---

## Prerequisites

Please install OVERTONE as introduced in the main `README.md`. The tutorials additionally need
[tblite](https://github.com/tblite/tblite) for GFN2-xTB and, for tutorial 3, [PySCF](https://pyscf.org).

```bash
pip install "overtone[tutorial]"
pip install pyscf
```

---

## Tutorial 1: `01_quickstart.py`
### Anharmonic Spectrum in One Call

This tutorial demonstrates:
- Geometry optimization of water with GFN2-xTB
- The complete OVERTONE workflow with the function `calculate_anharmonics`
- Fundamentals, overtones and combination bands up to three quanta
- IR intensities computed on request
- Plotting of the spectrum on a logarithmic scale

**Run with:**
```bash
python 01_quickstart.py
```

Output files:
- `water_quickstart/` - all intermediate results as JSON files (harmonic analysis, force field, dipoles, VPT2)
- `water_quickstart.png` - the computed spectrum

---

## Tutorial 2: `02_step_by_step.py`
### Full Control over Every Step

This tutorial demonstrates:
- The harmonic analysis with symmetrized Hessian and Eckart projection
- Generation of the displaced structures in dimensionless normal coordinates
- Assembly of the cubic and quartic force constants from forces
- Dipole derivatives for the IR intensities
- A comparison of standard VPT2, deperturbed VPT2, DCPT2 and GVPT2 on the same force field

Use this level of control when you want to inspect the force constants, reuse
a Hessian from another program, or run the single points elsewhere.

**Run with:**
```bash
python 02_step_by_step.py
```

Output:
- Frequencies and band table printed to the terminal

---

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

---

## Tutorial 4: `04_fermi_resonances.py`
### Fermi Resonances and GVPT2

This tutorial demonstrates:
- Automatic detection of Fermi resonances
- How standard VPT2 fails near a resonance
- Repeating the VPT2 step with all variants from saved files, without new single points
- State mixing and intensity borrowing in GVPT2

**Run with:**
```bash
python 04_fermi_resonances.py
```

Output files:
- `formaldehyde/` - all intermediate results
- `formaldehyde_resonance.png` - CH stretching region with standard VPT2 and GVPT2

---

## Tutorial 5: `05_analysis.py`
### Analysis of a Finished Run

This tutorial demonstrates:
- Reloading saved results
- Selecting bands by type and wavenumber range
- Repeating the VPT2 step with other settings, e.g. without Coriolis terms
- Exporting the broadened spectrum as a data file
- Plotting a spectrum together with reference bands

**Run with** (after tutorial 1):
```bash
python 05_analysis.py
```

Output files:
- `water_spectrum.dat` - broadened spectrum as two columns
- `water_vs_reference.png` - spectrum with mirrored reference bands

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

---

## Recommended Workflow

1. Run `01_quickstart.py` to understand the basic OVERTONE workflow.
2. Run `02_step_by_step.py` to see what happens inside the workflow and how the VPT2 variants differ.
3. Run `03_water_vs_experiment.py` to compare a realistic calculation with experiment.
4. Run `04_fermi_resonances.py` to see how GVPT2 treats Fermi resonances.
5. Run `05_analysis.py` to reuse and analyse saved results.
6. Run `06_hybrid.py` to combine two levels of theory.

---

# Detailed Tutorial Descriptions

This section provides a **step-by-step explanation of each tutorial script**,
describing the workflow, the main components, and how the individual parts
interact during an anharmonic calculation with **OVERTONE**.

---

## Detailed Description: `01_quickstart.py`

### Goal of the Script

The goal of `01_quickstart.py` is to demonstrate the **core idea of OVERTONE**
in the simplest possible setting:

Any ASE calculator provides energies, forces and dipole moments, and OVERTONE
turns them into an anharmonic vibrational spectrum.

This script is optimized for clarity and speed rather than accuracy.

---

### Step 1: Define the calculator

```python
def setup_calculator():
    return TBLite(method="GFN2-xTB", verbosity=0, accuracy=0.01)
```

OVERTONE needs a function that returns a fresh calculator. Every displaced structure gets its own calculator.
The SCF is converged tightly (`accuracy=0.01`), because the finite differences amplify numerical noise.

### Step 2: Optimize the structure

```python
atoms = molecule("H2O")
atoms.calc = setup_calculator()
BFGS(atoms).run(fmax=1e-4)
```

VPT2 expands the potential around a minimum. The geometry must therefore be optimized tightly.

### Step 3: Run the complete workflow

```python
result = calculate_anharmonics(
    atoms, setup_calculator, "water_quickstart",
    derivative_source="forces",      # analytic forces need fewer points
    intensities=True,                # needs dipoles from the calculator
    max_quanta=3,                    # fundamentals, overtones, combinations
    save_results=True,               # keep JSON files for later analysis
)
```

This single call performs the harmonic analysis, generates the displaced structures,
computes all single points, assembles the quartic force field and solves the VPT2 equations.
By default OVERTONE uses GVPT2, which treats Fermi resonances variationally.

With `dry_run=True` the function only reports how many single points a calculation will need.
For energy-only calculators, such as many machine-learning potentials, keep `intensities=False`.

### Step 4: Inspect and plot the results

```python
result.transitions_table()
ax = result.plot_spectrum(xmin=1000, xmax=8000, log=True)
```

The table lists every band with its harmonic and anharmonic wavenumber and its intensity.
Bands that take part in a Fermi resonance are marked. The logarithmic scale makes the
weak overtones and combination bands of the near-IR visible next to the strong fundamentals.

---

## Detailed Description: `02_step_by_step.py`

### Goal of the Script

The goal of `02_step_by_step.py` is to show **what happens inside**
`calculate_anharmonics`. Every step is called by hand.

---

### Step 1: Harmonic analysis

```python
harmonics = Harmonics(atoms, delta=0.005, symmetrize_hessian=True,
                      project_translations_rotations=True)
displaced = harmonics.calculate_displacements()
...
harm = harmonics.calculate_hessian(displaced)
```

The Hessian is obtained from central differences of the forces with a Cartesian step of 0.005 Å.
It is symmetrized, and translations and rotations are projected out. Both options can be switched off.

### Step 2: Displacements and quartic force field

```python
config = AnharmonicDerivativesConfig.from_harmonics_result(
    harmonics_result=harm, atoms=atoms, active_modes=modes,
    delta=0.2, stencil=1, coupling=3)
solver = AnharmonicDerivativesSolver(config)
traj, keys = solver.generate_displacement_trajectory_for_forces()
```

The structures are displaced along the normal modes in steps of `delta = 0.2`.
This step is given in dimensionless normal coordinates, not in Ångström.
The five-point stencil (`stencil=1`) and all three-mode couplings (`coupling=3`) are the defaults.

### Step 3: Force constants and dipole derivatives

```python
derivatives = solver.calculate_derivatives_from_forces(traj)
dipoles = solver.calculate_dipole_derivatives(traj, keys)
```

The cubic and quartic force constants are assembled from the forces.
The dipole derivatives come from the same displaced structures, so intensities need no extra calculations.

### Step 4: Compare the VPT2 variants

```python
for method in ("standard", "deperturbed", "dcpt2", "gvpt2"):
    vpt2 = VPT2Config.from_harmonics_result(harm, modes, atoms=atoms, method=method)
    result = VPT2Solver(vpt2, derivatives, dipole_derivatives=dipoles).calculate()
```

All variants use the same force field. They differ only in the treatment of Fermi resonances.
Water has no strong resonance in its fundamentals, so standard VPT2, deperturbed VPT2 and GVPT2 agree.

---

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

---

## Scientific Motivation

The tutorials illustrate, in a progressive manner, how anharmonic vibrational spectra
can be obtained from any electronic structure method available in ASE.

### Tutorial 1: Water — From a Calculator to a Spectrum

Water is the simplest molecule with a rich vibrational spectrum. Its three fundamentals are
accompanied by overtones and combination bands that reach far into the near-IR.
The tutorial shows that a complete anharmonic spectrum requires only a calculator and a single function call.

### Tutorial 2: Inside the Workflow

Anharmonic calculations involve several numerical choices, such as the step size,
the finite-difference stencil and the treatment of Fermi resonances.
By calling every step separately, the tutorial makes these choices visible
and shows how the different VPT2 variants behave on the same force field.

### Tutorial 3: Water — Comparison with Experiment

GFN2-xTB is fast but places the OH stretches of water 200-300 cm⁻¹ too low.
A DFT calculation with tight settings brings all bands within 1-2 % of experiment.
The remaining deviations in the stretching overtone region show where VPT2 reaches its limits.

### Tutorial 4: Formaldehyde — Fermi Resonances

In formaldehyde, one CH stretching fundamental lies close to a combination band
of two lower modes. The two states mix. Standard VPT2 cannot describe this
and gives a misleading position. GVPT2 treats the coupled states variationally.
It also shows how the combination band borrows intensity from the fundamental,
which is essential for interpreting real spectra.

### Tutorial 5: Reusing Results

Single points are the expensive part of every calculation. Once they are
stored, the VPT2 step can be repeated with other settings in seconds, and
spectra can be replotted or exported for comparison with measurements.

### Tutorial 6: Water — A Hybrid Scheme

Harmonic frequencies depend strongly on the level of theory, anharmonic
corrections much less. Combining a B3LYP Hessian with a GFN2-xTB force field
reduces the errors of the water fundamentals from up to 315 cm⁻¹ (GFN2-xTB alone)
to below 20 cm⁻¹, while the expensive anharmonic part stays cheap.

---

## License

This tutorial and all example scripts are provided under the GPL-3.0 License.
