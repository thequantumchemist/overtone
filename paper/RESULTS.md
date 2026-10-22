# Results collected from the production runs

Values to be inserted into the manuscript when we revise it together.

## 01_water, MP2/aug-cc-pVTZ (ORCA 5.0.4, received from WD)

No Fermi resonance detected, so VPT2 = GVPT2. Wavenumbers in cm^-1,
intensities in km/mol. Experiment from J. Chem. Theory Comput. 10, 5426 (2014).

| Band | Harmonic | VPT2 | GVPT2 | Exp. | GVPT2 - exp | Intensity |
|---|---|---|---|---|---|---|
| v2 | 1628.1 | 1577.8 | 1577.8 | 1595 | -17.2 | 69.5 |
| v1 | 3821.7 | 3654.3 | 3654.3 | 3657 | -2.7 | 5.3 |
| v3 | 3947.4 | 3767.8 | 3767.8 | 3756 | 11.8 | 72.1 |
| 2v2 | 3256.1 | 3123.9 | 3123.9 | 3152 | -28.1 | 0.6 |
| v1+v2 | 5449.8 | 5215.1 | 5215.1 | 5235 | -19.9 | 0.1 |
| v2+v3 | 5575.5 | 5325.4 | 5325.4 | 5331 | -5.6 | 4.0 |
| 2v1 | 7643.4 | 7227.0 | 7227.0 | 7202 | 25.0 | 0.3 |
| v1+v3 | 7769.1 | 7267.2 | 7267.2 | 7250 | 17.2 | 2.3 |
| 2v3 | 7894.9 | 7443.3 | 7443.3 | 7445 | -1.7 | 0.0 |

Notes for the manuscript
* Agreement with Barone (2005), MP2/aug-cc-pVTZ: fundamentals within 1.3 cm^-1,
  harmonic within 0.6 cm^-1 (Barone values still to be verified in the paper).
* Use the table layout with both VPT2 and GVPT2 columns (identical here),
  as the bridge to the Fermi resonances in ethylene.
* DONE: weakest intensities with two significant digits in the manuscript
  (2v2 0.60, v1+v2 0.069, 2v1 0.32, 2v3 0.017 km/mol, from vpt2_result.json).
* DONE: water figure regenerated from the MP2 result
  (paper/figures/data/water_mp2_bands.json, results_figures.py).
* DONE: water table in the manuscript updated (only the Barone values remain to be verified).
* ORCA prints unrelaxed and relaxed MP2 dipoles. ASE reads the last block,
  the relaxed one (1.865 D for water). Confirmed with ORCA 5.0.4.

## 07_hybrid, water (computed by Claude with PySCF, same settings as run_hybrid.py)

B3LYP/def2-TZVP with PySCF (SCF 1e-11 Eh, grid level 4), GFN2-xTB with tblite
(accuracy 0.01), force route, GVPT2, up to three quanta. Already in the
manuscript (Table tab:hybrid).

| Band | MP2 | B3LYP | Hybrid | GFN2-xTB | Exp. |
|---|---|---|---|---|---|
| v2 | 1577.8 | 1566.8 | 1577.6 | 1498.1 | 1595 |
| v1 | 3654.3 | 3618.1 | 3642.8 | 3439.7 | 3657 |
| v3 | 3767.8 | 3709.4 | 3747.6 | 3441.0 | 3756 |
| 2v2 | 3123.9 | 3097.2 | 3140.9 | 2976.9 | 3152 |
| v1+v2 | 5215.1 | 5174.2 | 5195.6 | 4919.3 | 5235 |
| v2+v3 | 5325.4 | 5260.1 | 5300.8 | 4913.5 | 5331 |
| 2v1 | 7227.0 | 7153.9 | 7217.0 | 6782.3 | 7202 |
| v1+v3 | 7267.2 | 7168.0 | 7267.4 | 6686.5 | 7250 |
| 2v3 | 7443.3 | 7325.4 | 7425.8 | 6781.8 | 7445 |
| MAE | 14.4 | 61.1 | 19.1 | 353.8 | |

Anharmonic shifts omega - nu (bend, sym., antisym. stretch):
MP2 50.3, 167.4, 179.6. B3LYP 49.8, 167.3, 181.2. GFN2-xTB (in hybrid) 39.0, 142.6, 143.0.

Hybrid intensities use the GFN2-xTB dipoles: v2 128.0, v1 5.6, v3 14.4 km/mol.


## TOC graphic

assets/overtone_banner.py --toc gives overtone_toc.pdf/.png (3.25 x 1.75 in, JCTC).
Included in the manuscript via the tocentry environment.

## HITRAN2024 band intensities of water (km/mol)

Summed from the HITRAN2024 H2(16)O line list 1000-8000 cm^-1 (80,110 lines,
downloaded 2026-10-22) with common.hitran_band_intensities. In the manuscript.

| Band | HITRAN | MP2 | MP2/HITRAN |
|---|---|---|---|
| v2 | 69.05 | 69.46 | 1.01 |
| v1 | 3.257 | 5.316 | 1.63 |
| v3 | 47.02 | 72.06 | 1.53 |
| 2v2 | 0.5009 | 0.6025 | 1.20 |
| v1+v2 | 0.2267 | 0.0686 | 0.30 |
| v2+v3 | 5.073 | 3.992 | 0.79 |
| 2v1 | 0.4878 | 0.3231 | 0.66 |
| v1+v3 | 3.318 | 2.260 | 0.68 |
| 2v3 | 0.02591 | 0.01707 | 0.66 |

Sum of 2v1, v1+v3, 2v3: HITRAN 3.83, MP2 2.60 km/mol. HITRAN also lists 2v2+v3
(three quanta) with 0.27 km/mol. Citation: Gordon et al., JQSRT 353, 109807 (2026).
