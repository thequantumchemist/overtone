# Tutorials

The tutorials are example scripts that show how to compute anharmonic
vibrational spectra with **OVERTONE**. They gradually introduce the main
concepts:

| Tutorial | Content | Run time |
|---|---|---|
| [1. Quickstart](01-quickstart.md) | Anharmonic IR spectrum of water in one call, using GFN2-xTB | a few seconds |
| [2. Step by step](02-step-by-step.md) | Every step of the workflow called by hand, and a comparison of all VPT2 variants | a few seconds |
| [4. Fermi resonances](04-fermi-resonances.md) | Formaldehyde with a Fermi resonance, comparing all VPT2 variants | a few seconds |
| [5. Analysis](05-analysis.md) | Reloading, recomputing, exporting and plotting the results of tutorial 1 | seconds |

The scripts are in the
[`tutorials`](https://github.com/thequantumchemist/overtone/tree/main/tutorials)
folder of the repository, which also contains two tutorials with PySCF that
are not yet part of this site: a comparison of water at the B3LYP level with
the experimental near-IR spectrum (`03_water_vs_experiment.py`) and a hybrid
scheme with two levels of theory (`06_hybrid.py`).

## Prerequisites

Please install OVERTONE as described on the [home page](../index.md#installation).
The tutorials additionally need [tblite](https://github.com/tblite/tblite) for GFN2-xTB.

```bash
pip install "overtone-vpt2[tutorial]"
```

## Recommended order

1. Run `01_quickstart.py` to understand the basic OVERTONE workflow.
2. Run `02_step_by_step.py` to see what happens inside the workflow and how the VPT2 variants differ.
3. Run `04_fermi_resonances.py` to see how GVPT2 treats Fermi resonances.
4. Run `05_analysis.py` to reuse and analyse saved results.

All scripts are run as part of the test suite, so the code shown here is the
code that is tested.
