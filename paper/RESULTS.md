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
  harmonic within 0.6 cm^-1. Barone values verified (JCP 122, 014108, Tables IV, V).
* Barone used the Martin resonance test with a default threshold of 10 cm^-1
  (OVERTONE default: 1 cm^-1). Possible remark for the SI.
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

## H2, comparison with experiment (HITRAN2024, quadrupole lines of Roueff et al. 2019)

Vibrational levels from line positions + lower-state energies: G(1) = 4161.17, G(2) = 8087.00 cm^-1.
2v - 2*v (= 2 X11 in VPT2): experiment -235.3, CCSD -247.3, ADAPT-VQE tight -247.4,
ADAPT-VQE default -243.3 cm^-1 (step dependent, closer only by accident).
STO-3G overestimates both levels by 14 %. Total HITRAN H2 intensity 1e-25 cm/molecule
(quadrupole), no intensity comparison possible. In the manuscript (ADAPT-VQE section).

## 06_numerics (cluster run, complete)

costs.csv and noise.csv identical to the local run (fixed random seeds).
truncation.csv, largest change relative to delta = 0.1 (cm^-1):
H2O 0.02 / 0.01 / 0.03 / 0.25 for delta 0.05 / 0.2 / 0.3 / 0.5.
CH3OH all modes 0.15 / 2.26 / 11.06 / 62.08, without torsion 0.03 / 0.19 / 0.90 / 5.50.
In the SI (Table si_truncation) and the main text.

## 03_nitromethane, staggered (B3LYP/cc-pVTZ, ORCA, received from WD)

No Fermi resonance detected (VPT2 = GVPT2). Torsion excluded, as in Cornaton et al.
(they also used 14 modes without derivatives along the methyl rotation).

| Mode | harm | Cornaton harm | GVPT2 | Cornaton GVPT2 | exp | I_harm | Cornaton I_harm | I_GVPT2 | Cornaton I_anharm |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 3193.5 | 3193 | 3046.5 | 3038 | 3080 | 0.57 | 0.57 | 0.54 | 1.29 |
| v2 | 3161.6 | 3161 | 3015.9 | 3006 | 3045 | 2.25 | 2.25 | 2.15 | 3.14 |
| v3 | 3074.5 | 3074 | 2962.1 | 2953 | 2974 | 0.71 | 0.71 | 0.69 | 1.07 |
| v4 | 1631.8 | 1632 | 1596.3 | 1584 | 1583 | 297.75 | 298.07 | 291.28 | 251.14 |
| v5 | 1477.3 | 1478 | 1430.8 | 1427 | 1434 | 11.26 | 11.3 | 10.9 | 11.33 |
| v6 | 1464.4 | 1464 | 1414.7 | 1414 | 1410 | 40.68 | 41.14 | 39.3 | 38.85 |
| v7 | 1427.0 | 1427 | 1394.3 | 1388 | 1397 | 54.92 | 54.95 | 53.66 | 41.37 |
| v8 | 1398.6 | 1399 | 1364.3 | 1358 | 1380 | 52.08 | 52.25 | 50.8 | 68.51 |
| v9 | 1136.0 | 1136 | 1108.7 | 1106 | 1131 | 1.21 | 1.22 | 1.18 | 1.43 |
| v10 | 1109.3 | 1109 | 1084.7 | 1084 | 1096 | 11.82 | 11.83 | 11.56 | 8.53 |
| v11 | 924.6 | 924 | 903.0 | 898 | 918 | 15.56 | 15.56 | 15.19 | 18.63 |
| v12 | 661.7 | 662 | 648.4 | 645 | 657 | 17.09 | 17.19 | 16.74 | 14.45 |
| v13 | 616.0 | 616 | 607.1 | 605 | 603 | 5.39 | 5.37 | 5.31 | 5.08 |
| v14 | 480.9 | 481 | 474.5 | 475 | 475 | 0.82 | 0.82 | 0.81 | 0.97 |

Combination bands (Cornaton values are CALCULATED, not experimental):
v4+v7 2980.1 (Cornaton 2955), v4+v8 2955.8 (2933), v4+v11 2492.0 (2471), 2v7 2782.2 (2768).

Notes for the manuscript
* Harmonic frequencies within 1 cm^-1, harmonic intensities within about 1 %:
  independent validation of the intensity code at the same level of theory.
* GVPT2 fundamentals 0-12 cm^-1 above Cornaton (smaller anharmonic shifts).
  Cornaton et al. found for methanimine that numerical anharmonic corrections
  are on average about 10 % smaller than their analytic ones. Same direction.
* Mean absolute deviation from experiment: OVERTONE 12.6, Cornaton 15.4 cm^-1.
* Torsion test (GFN2-xTB): including the torsion couplings changes the other
  fundamentals by mostly < 3.5 cm^-1, in both directions. Not the cause.
* Anharmonic intensities of fundamentals: ours are harmonic-level (no second-
  order corrections), Cornaton's include them.
* Eclipsed conformer: ORCA stopped during the optimisation. To be rerun.
