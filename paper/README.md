# Calculations of the OVERTONE paper

| Folder | System and level | Single points | Compared with |
|---|---|---|---|
| `01_water` | H2O, MP2/aug-cc-pVTZ | 200 | Barone 2005, experiment (JCTC 10, 5426), HITRAN |
| `02_ethylene` | C2H4, CCSD(T)/cc-pVDZ | 18,534 (+ numerical gradients for the Hessian) | Martin et al. 1995 |
| `03_nitromethane` | CH3NO2, B3LYP/cc-pVTZ, staggered conformer | 5,264 (upper bound) | Cornaton et al. 2016 |
| `04_h2_adapt_vqe` | H2, CCSD and ADAPT-VQE/STO-3G | small | exact (CCSD = FCI for H2) |
| `05_active_learning` | H2O MP2, CH4 B3LYP/cc-pVDZ with FALCON | see SI | exact reference runs |
| `06_numerics` | model force field, GFN2-xTB | small | noise and step analysis (SI) |
| `07_hybrid` | H2O, B3LYP/def2-TZVP (PySCF), hybrid with GFN2-xTB | 200 + 19 B3LYP | full B3LYP, MP2 (01_water), experiment |
| `07_hybrid` | H2O, B3LYP/def2-TZVP Hessian + GFN2-xTB force field | 19 B3LYP + 181 GFN2-xTB | full B3LYP, MP2, experiment |
| `08_convergence` | H2O and H2CO, MP2/aug-cc-pVTZ, stencil x coupling x step | 1,744 (H2O) + 7,264 (H2CO) | production setting (SI) |
| `figures` | workflow and result figures | none | `python workflow_figure.py`, `python results_figures.py` |

Every script has `--test` (GFN2-xTB instead of ORCA, runs in minutes),
`--dry-run` (number of single points) and `--nprocs`. Set the ORCA paths with
the environment variables `ORCA_COMMAND` (ORCA 5.0.4) and `ORCA6_COMMAND`
(ORCA 6.0.1). Results (CSV tables, LaTeX table bodies, figures) appear in
`results*/` inside each folder.

Run everything with `bash run_all.sh` (one job each) or submit the jobs with
`sbatch submit.slurm <folder> <script> [arguments]`.

## Figures

| Figure | Script | Data |
|---|---|---|
| Fig. 1, workflow | `figures/workflow_figure.py` | `figures/water_b3lyp_bands.json` (inset spectrum, B3LYP/def2-TZVP) |
| Water spectrum | `figures/results_figures.py` | `figures/data/water_mp2_bands.json` (01_water), `figures/data/water_hitran_bands.json` |
| Formaldehyde, GVPT2 | `figures/results_figures.py` | `figures/data/formaldehyde_gfn2_*.json` (tutorial 4) |
| TOC graphic | `../assets/overtone_banner.py --toc` | none |
| README banner | `../assets/overtone_banner.py` | none |

Run the scripts inside their folders, e.g. `cd figures && python results_figures.py`.
To update a figure after a new calculation, replace its data file and run the script again.
