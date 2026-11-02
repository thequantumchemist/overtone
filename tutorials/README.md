# OVERTONE Tutorials

This directory contains example scripts demonstrating how to compute anharmonic
vibrational spectra with **OVERTONE**. The tutorials are explained in the
[documentation](https://thequantumchemist.github.io/overtone/tutorials/), which includes its code directly from these scripts.
All scripts run as part of the test suite (`tests/docs/test_tutorials.py`).

- `01_quickstart.py`
  Anharmonic IR spectrum of water in one call, using GFN2-xTB (fast, a few seconds).
  [Documentation](https://thequantumchemist.github.io/overtone/tutorials/01-quickstart/)

- `02_step_by_step.py`
  Every step of the workflow called by hand, and a comparison of all VPT2 variants (fast, a few seconds).
  [Documentation](https://thequantumchemist.github.io/overtone/tutorials/02-step-by-step/)

- `03_water_vs_experiment.py`
  Water up to two quanta compared with the experimental mid- and near-IR band positions, with GFN2-xTB (fast, seconds) or B3LYP/def2-TZVP from PySCF (a few minutes).
  [Documentation](https://thequantumchemist.github.io/overtone/tutorials/03-water-vs-experiment/)

- `04_fermi_resonances.py`
  Formaldehyde with a Fermi resonance, comparing all VPT2 variants (fast, a few seconds).
  [Documentation](https://thequantumchemist.github.io/overtone/tutorials/04-fermi-resonances/)

- `05_analysis.py`
  Reloading, recomputing, exporting and plotting the results of tutorial 1 (fast, seconds).
  [Documentation](https://thequantumchemist.github.io/overtone/tutorials/05-analysis/)

- `06_hybrid.py`
  Hessian from B3LYP (PySCF), anharmonic part from GFN2-xTB (about 20 seconds).
  [Documentation](https://thequantumchemist.github.io/overtone/tutorials/06-hybrid/)

- `07_cluster.py`
  The quickstart with every single point submitted to Slurm through shephex; rerun the script until it finishes (needs a Slurm cluster and the `cluster` extra).
  [Documentation](https://thequantumchemist.github.io/overtone/tutorials/07-cluster/)

- `pyscf_calculator.py`
  A small ASE calculator for PySCF with analytic forces and dipole moments, used by tutorials 3 and 6.

The documentation keeps the code in sync by including sections of the scripts
marked with `# --8<-- [start:name]` and `# --8<-- [end:name]` comments.
Keep these markers when editing the scripts.

---

## Prerequisites

Please install OVERTONE as introduced in the main `README.md`. The tutorials additionally need
[tblite](https://github.com/tblite/tblite) for GFN2-xTB and, for tutorial 6 and the B3LYP variant of tutorial 3, [PySCF](https://pyscf.org).
Both are part of the `tutorial` extra:

```bash
pip install "overtone-vpt2[tutorial]"
```

---

## License

This tutorial and all example scripts are provided under the GPL-3.0 License.
