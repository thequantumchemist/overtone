# 2. Full control over every step

Anharmonic calculations involve several numerical choices, such as the step
size, the finite-difference stencil and the treatment of Fermi resonances. By
calling every step of `calculate_anharmonics` separately, this tutorial makes
these choices visible and shows how the different VPT2 variants behave on the
same force field.

This tutorial demonstrates:

- The harmonic analysis with symmetrized Hessian and Eckart projection
- Generation of the displaced structures in dimensionless normal coordinates
- Assembly of the cubic and quartic force constants from forces
- Dipole derivatives for the IR intensities
- A comparison of standard VPT2, deperturbed VPT2, DCPT2 and GVPT2 on the same force field

Use this level of control when you want to inspect the force constants, reuse
a Hessian from another program, or run the single points elsewhere.

```bash
python 02_step_by_step.py
```

The frequencies and the band table are printed to the terminal.

## Step 1: Harmonic analysis

```python
--8<-- "tutorials/02_step_by_step.py:harmonics"
```

The Hessian is obtained from central differences of the forces with a
Cartesian step of 0.005 Å. It is symmetrized, and translations and rotations
are projected out. Both options can be switched off.

## Step 2: Displacements and quartic force field

```python
--8<-- "tutorials/02_step_by_step.py:forcefield"
```

The structures are displaced along the normal modes in steps of `delta = 0.2`.
This step is given in dimensionless normal coordinates, not in Ångström. The
five-point stencil (`stencil=1`) and all three-mode couplings (`coupling=3`)
are the defaults.

## Step 3: Force constants and dipole derivatives

```python
--8<-- "tutorials/02_step_by_step.py:derivatives"
```

The cubic and quartic force constants are assembled from the forces. The
dipole derivatives come from the same displaced structures, so intensities
need no extra calculations.

## Step 4: Compare the VPT2 variants

```python
--8<-- "tutorials/02_step_by_step.py:vpt2"
```

All variants use the same force field. They differ only in the treatment of
Fermi resonances. Water has no strong resonance in its fundamentals, so
standard VPT2, deperturbed VPT2 and GVPT2 agree.

## Full script

```python title="02_step_by_step.py"
--8<-- "tutorials/02_step_by_step.py"
```
