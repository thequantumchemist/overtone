"""
Tutorial 4: Fermi resonances and GVPT2 (formaldehyde, GFN2-xTB, seconds).

A Fermi resonance couples a fundamental with an overtone or combination band
of nearly the same energy. Standard VPT2 then breaks down. This tutorial
computes the force field once and compares all VPT2 variants on it.
"""
from ase.build import molecule
from ase.optimize import BFGS
from tblite.ase import TBLite

from overtone import calculate_anharmonics, recompute_vpt2


def setup_calculator():
    return TBLite(method="GFN2-xTB", verbosity=0, accuracy=0.01)


atoms = molecule("H2CO")
atoms.calc = setup_calculator()
BFGS(atoms).run(fmax=1e-4)

# one calculation (GVPT2 is the default) ...
gvpt2 = calculate_anharmonics(atoms, setup_calculator, "formaldehyde",
                              derivative_source="forces", intensities=True,
                              save_results=True)
print("Fermi resonances, type 1 (w_i ~ 2 w_j):", gvpt2.fermi_type1)
print("Fermi resonances, type 2 (w_i ~ w_j + w_k):", gvpt2.fermi_type2)

# ... and the other variants from the saved files, without new single points
results = {"gvpt2": gvpt2}
for method in ("standard", "deperturbed", "dcpt2"):
    results[method] = recompute_vpt2("formaldehyde", method=method)


def fundamentals(result):
    return {next(iter(t.quanta)): t for t in result.transitions if t.kind == "fundamental"}


print(f"\n{'mode':>5} {'harmonic':>9} {'standard':>9} {'deperturbed':>12} {'DCPT2':>8} {'GVPT2':>8}  mixing (GVPT2)")
f = {m: fundamentals(r) for m, r in results.items()}
for mode in sorted(f["gvpt2"], key=lambda m: f["gvpt2"][m].harmonic):
    g = f["gvpt2"][mode]
    print(f"{g.label:>5} {g.harmonic:9.1f} {f['standard'][mode].energy:9.1f} "
          f"{f['deperturbed'][mode].energy:12.1f} {f['dcpt2'][mode].energy:8.1f} "
          f"{g.energy:8.1f}  {g.mixing or ''}")

# resonance partners share their intensity (intensity borrowing)
print("\nBands in a Fermi resonance:")
for t in gvpt2.transitions:
    if t.resonance_affected:
        print(f"  {t.label:8s} {t.energy:8.1f} cm-1  {t.intensity:8.3f} km/mol  {t.mixing}")

ax = results["standard"].plot_spectrum(xmin=2400, xmax=3100, label="standard VPT2",
                                       color="0.6", by_kind=False, sticks=False)
results["gvpt2"].plot_spectrum(ax=ax, xmin=2400, xmax=3100, label="GVPT2", by_kind=False)
ax.figure.savefig("formaldehyde_resonance.png", dpi=150)
