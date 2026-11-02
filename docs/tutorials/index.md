# Tutorials

The tutorials are example scripts that show how to compute anharmonic
vibrational spectra with **OVERTONE**. They gradually introduce the main
concepts:

| Tutorial | Content | Run time |
|---|---|---|
| [1. Quickstart](01-quickstart.md) | Anharmonic IR spectrum of water in one call, using GFN2-xTB | a few seconds |
| [2. Step by step](02-step-by-step.md) | Every step of the workflow called by hand, and a comparison of all VPT2 variants | a few seconds |
| [3. Comparison with experiment](03-water-vs-experiment.md) | Water up to two quanta compared with the experimental mid- and near-IR bands, with GFN2-xTB or B3LYP | seconds (minutes with B3LYP) |
| [4. Fermi resonances](04-fermi-resonances.md) | Formaldehyde with a Fermi resonance, comparing all VPT2 variants | a few seconds |
| [5. Analysis](05-analysis.md) | Reloading, recomputing, exporting and plotting the results of tutorial 1 | seconds |
| [6. Hybrid scheme](06-hybrid.md) | Hessian from B3LYP (PySCF), anharmonic force field from GFN2-xTB | about 20 seconds |
| [7. Running on a cluster](07-cluster.md) | The quickstart with every single point submitted to Slurm through shephex, in stages | minutes, depending on the queue |

The scripts are in the
[`tutorials`](https://github.com/thequantumchemist/overtone/tree/main/tutorials)
folder of the repository.

## Prerequisites

Please install OVERTONE as described on the [home page](../index.md#installation).
The tutorials additionally need [tblite](https://github.com/tblite/tblite) for GFN2-xTB
and [PySCF](https://pyscf.org) for DFT (tutorial 6, and optionally tutorial 3),
and tutorial 7 needs [shephex](https://pypi.org/project/shephex/) (the `cluster` extra).

```bash
pip install "overtone-vpt2[tutorial]"
```

The tutorial molecules are small, and tblite runs much faster on them with a
single thread: set `export OMP_NUM_THREADS=1` before running the scripts (see
[troubleshooting](../manual.md#14-recommendations-and-troubleshooting)).

## Recommended order

1. Run `01_quickstart.py` to understand the basic OVERTONE workflow.
2. Run `02_step_by_step.py` to see what happens inside the workflow and how the VPT2 variants differ.
3. Run `03_water_vs_experiment.py` to compare with experiment and exchange the calculator.
4. Run `04_fermi_resonances.py` to see how GVPT2 treats Fermi resonances.
5. Run `05_analysis.py` to reuse and analyse saved results.
6. Run `06_hybrid.py` to combine two levels of theory.
7. Run `07_cluster.py` on a Slurm cluster to distribute the single points.

All scripts are run as part of the test suite, so the code shown here is the
code that is tested.
