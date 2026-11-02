# 6. Hybrid scheme with two levels of theory

Harmonic frequencies depend strongly on the level of theory, anharmonic
corrections much less. A common strategy therefore computes the Hessian at a
high level and the cubic and quartic force constants at a lower level (see
Barnes et al., J. Mol. Model. 22, 285 (2016)). The harmonic analysis of water
needs 19 single points, the anharmonic force field 181. In this tutorial only
the 19 harmonic ones use B3LYP; the rest use GFN2-xTB.

This tutorial demonstrates:

- A separate calculator for the harmonic analysis (`setup_harmonic_calculator`)
- Harmonic frequencies from B3LYP and anharmonic corrections from GFN2-xTB
- The large improvement over GFN2-xTB alone at a fraction of the cost of B3LYP

The tutorial needs [PySCF](https://pyscf.org), which is part of the
`tutorial` extra.

```bash
python 06_hybrid.py         # about 20 seconds
```

## Step 1: Two calculators

```python
--8<-- "tutorials/06_hybrid.py:calculators"
```

`PySCFCalculator` is a small ASE calculator in
[`tutorials/pyscf_calculator.py`](https://github.com/thequantumchemist/overtone/blob/main/tutorials/pyscf_calculator.py)
with analytic forces and dipole moments. It converges the SCF tightly and
uses a fine integration grid, because the finite differences amplify
numerical noise.

## Step 2: The hybrid calculation

```python
--8<-- "tutorials/06_hybrid.py:hybrid"
```

The structure is optimized at the high level. `setup_harmonic_calculator`
gives the calculator for the harmonic analysis; the second argument, as
always, the calculator for the displaced structures of the anharmonic force
field. The low-level force field is expanded around the high-level geometry,
where its own gradient is not exactly zero. This approximation is standard in
hybrid schemes.

## Step 3: Comparison with GFN2-xTB alone

```python
--8<-- "tutorials/06_hybrid.py:compare"
```

```text
               bend  sym. str.  asym. str.
GFN2-xTB     1498.1     3439.7      3441.0
hybrid       1577.6     3642.8      3747.6
experiment     1595       3657        3756
```

The hybrid scheme reduces the errors of the fundamentals from up to 315 cm⁻¹
with GFN2-xTB alone to below 20 cm⁻¹. It is even closer to experiment than
the full B3LYP calculation in [tutorial 3](03-water-vs-experiment.md). Such
close agreement is fortunate and should not be expected in general.

## Output files

- `water_hybrid/`, `water_gfn2/`: all intermediate results
- the fundamentals, printed to the terminal

## Full script

```python title="06_hybrid.py"
--8<-- "tutorials/06_hybrid.py"
```
