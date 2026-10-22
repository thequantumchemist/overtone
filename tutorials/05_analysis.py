"""
Tutorial 5: analysis of a finished run (run 01_quickstart.py first).

All intermediate results are stored as JSON files. They can be reloaded,
the VPT2 step can be repeated with other settings, and spectra can be
replotted or exported, all without new single points.
"""
from pathlib import Path

import numpy as np

from overtone import HarmonicsResult, VPT2Result, recompute_vpt2

run = Path("water_quickstart")
if not (run / "anharmonics" / "vpt2_result.json").exists():
    raise SystemExit("Run 01_quickstart.py first.")

# 1. reload the saved results
harmonics = HarmonicsResult.load(run / "harmonics" / "harmonics_result.json")
result = VPT2Result.load(run / "anharmonics" / "vpt2_result.json")
harmonics.summary()
print(f"Hessian asymmetry before symmetrisation: {harmonics.hessian_asymmetry:.1e}")

# 2. only the near-IR bands
result.transitions_table(kinds=["overtone", "combination"], xmin=4000)

# 3. repeat only the VPT2 step with other settings
standard = recompute_vpt2(run, method="standard")
no_coriolis = recompute_vpt2(run, coriolis=False)
modes = harmonics.vibrational_modes
print("GVPT2       ", np.round(result.anharm_freq[modes], 1))
print("standard    ", np.round(standard.anharm_freq[modes], 1))
print("no Coriolis ", np.round(no_coriolis.anharm_freq[modes], 1))

# 4. export the broadened spectrum, e.g. for comparison with a measurement
x, y = result.spectrum(shape="gaussian", fwhm=20, xmin=1000, xmax=8000, resolution=1.0)
np.savetxt("water_spectrum.dat", np.column_stack([x, y]),
           header="wavenumber (cm-1)   intensity (km/mol per cm-1)")

# 5. compare with reference bands (here the experimental fundamentals;
#    the intensities only set the relative heights of the reference)
reference = {"positions": [1595, 3657, 3756], "intensities": [60, 3, 45],
             "label": "experiment (fundamentals)"}
ax = result.plot_spectrum(xmin=1200, xmax=4200, reference=reference, by_kind=False)
ax.figure.savefig("water_vs_reference.png", dpi=150)
print("written water_spectrum.dat and water_vs_reference.png")
