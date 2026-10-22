"""
Workflow figure of OVERTONE (Figure 1 of the paper).

    python workflow_figure.py    ->  workflow.pdf, workflow.png

The inset spectrum uses real OVERTONE output (water, B3LYP/def2-TZVP, GVPT2,
tutorials/03_water_vs_experiment.py), stored in water_b3lyp_bands.json.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                     "mathtext.fontset": "dejavusans"})

# colours: one per step, light fill + dark edge
C = {
    "calc": ("#e3f2fd", "#1565c0"),
    "harm": ("#e0f2f1", "#00796b"),
    "qff": ("#fff3e0", "#e65100"),
    "vpt2": ("#f3e5f5", "#6a1b9a"),
    "spec": ("#e8f5e9", "#2e7d32"),
    "noise": ("#ffebee", "#c62828"),
}

W, H = 7.0, 3.35                       # inches, JCTC double column
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

BOX_W, BOX_H, GAP, Y0 = 1.245, 2.35, 0.155, 0.82
xs = [0.07 + i * (BOX_W + GAP) for i in range(5)]


def box(x, key, number, title):
    fill, edge = C[key]
    ax.add_patch(FancyBboxPatch((x, Y0), BOX_W, BOX_H,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1.2))
    ax.add_patch(FancyBboxPatch((x, Y0 + BOX_H - 0.32), BOX_W, 0.32,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                fc=edge, ec=edge, lw=1.2))
    ax.text(x + 0.07, Y0 + BOX_H - 0.16, f"{number}", color="white",
            fontsize=8.0, fontweight="bold", va="center")
    ax.text(x + 0.21, Y0 + BOX_H - 0.16, title, color="white", fontsize=6.8,
            fontweight="bold", va="center")


def lines(x, y, items, size=6.4, color="#212121", dy=0.155):
    for i, s in enumerate(items):
        ax.text(x, y - i * dy, s, fontsize=size, color=color, va="top")


def arrow(x0, x1, y, color="#455a64"):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.6, color=color,
                                 shrinkA=0, shrinkB=0))


def inset(x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.patch.set_alpha(0)
    for s in a.spines.values():
        s.set_linewidth(0.6)
    a.tick_params(labelsize=5, length=1.5, pad=1)
    return a


# ---------------------------------------------------------------- 1 calculators
box(xs[0], "calc", 1, "Any ASE calculator")
chips = [("Quantum chemistry", "ORCA, PySCF, ..."),
         ("Semi-empirical", "GFN2-xTB"),
         ("Active learning", "FALCON"),
         ("Quantum computing", "ADAPT-VQE")]
for i, (t, sub) in enumerate(chips):
    y = Y0 + BOX_H - 0.55 - i * 0.43
    ax.add_patch(FancyBboxPatch((xs[0] + 0.08, y - 0.17), BOX_W - 0.16, 0.36,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                fc="white", ec=C["calc"][1], lw=0.6))
    ax.text(xs[0] + 0.14, y + 0.08, t, fontsize=6.4, fontweight="bold",
            color=C["calc"][1], va="center")
    ax.text(xs[0] + 0.14, y - 0.07, sub, fontsize=6.0, color="#424242", va="center")
lines(xs[0] + 0.08, Y0 + 0.2, ["energies, forces, dipoles"], size=5.9, color="#424242")

# ---------------------------------------------------------------- 2 harmonics
box(xs[1], "harm", 2, "Harmonic analysis")
a = inset(xs[1] + 0.12, Y0 + 1.03, BOX_W - 0.24, 0.9)
r = np.linspace(-0.55, 1.6, 300)
morse = (1 - np.exp(-1.6 * r)) ** 2
a.plot(r, morse, color="#37474f", lw=0.9)
a.plot(r, 2.56 * r**2, color=C["harm"][1], lw=0.9, ls="--")
hw = 0.12                              # level spacing at the bottom of the well
for n in range(4):
    e = hw * (n + 0.5) - hw**2 / 4 * (n + 0.5) ** 2   # Morse levels, D = 1
    rr = r[morse <= e]
    a.plot([rr.min(), rr.max()], [e, e], color="#78909c", lw=0.6)
a.set_ylim(0, 0.6)
a.set_xticks([])
a.set_yticks([])
a.set_xlabel("$q_i$", fontsize=6, labelpad=0)
lines(xs[1] + 0.08, Y0 + 0.88, [
    "Hessian from forces",
    "symmetrised",
    "Eckart projection",
    "linear molecules,",
    "constrained atoms",
    r"$\rightarrow\ \omega_i$, normal modes",
], size=6.1, dy=0.13)

# ---------------------------------------------------------------- 3 force field
box(xs[2], "qff", 3, "Quartic force field")
a = inset(xs[2] + 0.2, Y0 + 1.03, 0.85, 0.9)
steps = np.array([-2, -1, 1, 2])
pts = set()
for di in steps:                       # nested 5-point stencils for phi_iij
    for dj in steps:
        for dk in steps:
            pts.add((di + dj, dk))
pts = np.array(sorted(pts))
gx, gy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-3, 3, 200))
V = 0.5 * gx**2 + 1.2 * gy**2 + 0.035 * gx**3 + 0.09 * gx * gy**2
a.contour(gx, gy, V, levels=[1, 3, 6, 10, 15], colors="#ffcc80", linewidths=0.5)
a.scatter(pts[:, 0], pts[:, 1], s=2.4, color=C["qff"][1], lw=0, zorder=2)
a.scatter([0], [0], s=14, marker="*", color="#212121", zorder=3)
a.set_xlim(-5, 5)
a.set_ylim(-3, 3)
a.set_xticks([])
a.set_yticks([])
a.set_xlabel(r"$q_i$  (steps of $\delta$)", fontsize=5.6, labelpad=0)
a.set_ylabel("$q_j$", fontsize=6, labelpad=0)
lines(xs[2] + 0.08, Y0 + 0.88, [
    r"steps in $q$, $\delta = 0.2$",
    r"$\phi_{ijk}$, $\phi_{iijj}$ from E or F",
    r"$\mu_i$, $\mu_{ij}$ on request",
    "serial or Shephex/SLURM",
    "dry run, restart",
], size=6.1, dy=0.13)

# ---------------------------------------------------------------- 4 VPT2
box(xs[3], "vpt2", 4, "VPT2")
a = inset(xs[3] + 0.3, Y0 + 1.05, 0.66, 0.86)
M = np.full((6, 6), np.nan)
blocks = [(0, 2), (2, 5), (5, 6)]      # resonance polyads
for s, e in blocks:
    M[s:e, s:e] = 0.35
for i in range(6):
    M[i, i] = 1.0
a.imshow(M, cmap="Purples", vmin=0, vmax=1.15)
a.set_xticks([])
a.set_yticks([])
a.set_title("polyads", fontsize=5.6, pad=1.5)
lines(xs[3] + 0.08, Y0 + 0.88, [
    r"$X_{ij}$ incl. Coriolis",
    "Fermi resonance test",
    "GVPT2 (default)",
    "deperturbed, DCPT2,",
    "standard VPT2",
    "intensity borrowing",
], size=6.1, dy=0.13)

# ---------------------------------------------------------------- 5 spectrum
box(xs[4], "spec", 5, "Spectrum")
bands = json.load(open(HERE / "water_b3lyp_bands.json"))
a = inset(xs[4] + 0.15, Y0 + 1.03, BOX_W - 0.25, 0.9)
x = np.linspace(1000, 7800, 1500)
y = np.zeros_like(x)
colors = {"fundamental": "#1565c0", "overtone": "#c62828", "combination": "#2e7d32"}
for b in bands:
    y += b["intensity"] * (12 / np.pi) / ((x - b["energy"]) ** 2 + 12**2)
    a.vlines(b["energy"], 1e-4, b["intensity"] * (1 / (np.pi * 12)),
             color=colors[b["kind"]], lw=0.8)
a.plot(x, y + 1e-4, color="#212121", lw=0.6)
a.set_yscale("log")
a.set_ylim(1e-4, 40)
a.set_xlim(7800, 1000)
a.set_yticks([])
a.set_xticks([2000, 4000, 6000])
a.set_xticklabels(["2000", "4000", "6000"])
a.minorticks_off()
a.text(0.98, 0.95, "cm$^{-1}$", transform=a.transAxes, fontsize=5.4, ha="right", va="top")
for i, (label, c) in enumerate(zip(["fundamentals", "overtones", "combination bands"],
                                   colors.values())):
    yy = Y0 + 0.82 - i * 0.13
    ax.plot([xs[4] + 0.08, xs[4] + 0.2], [yy, yy], color=c, lw=1.4)
    ax.text(xs[4] + 0.25, yy, label, fontsize=6.1, color="#212121", va="center")
lines(xs[4] + 0.08, Y0 + 0.88 - 3 * 0.13, [
    "IR intensities",
    "mid- and near-IR",
], size=6.1, dy=0.13)

# ---------------------------------------------------------------- arrows
for i in range(4):
    arrow(xs[i] + BOX_W + 0.01, xs[i + 1] - 0.01, Y0 + BOX_H / 2)

# ---------------------------------------------------------------- noise bar
fill, edge = C["noise"]
x0, x1 = xs[0], xs[4] + BOX_W
ax.add_patch(FancyBboxPatch((x0, 0.1), x1 - x0, 0.56,
                            boxstyle="round,pad=0,rounding_size=0.08",
                            fc=fill, ec=edge, lw=1.0))
ax.text(x0 + 0.12, 0.38, "Numerical noise", fontsize=7.4, fontweight="bold",
        color=edge, va="center")
ax.text(x0 + 1.32, 0.48,
        r"Energy errors $\sigma$ (SCF convergence, ML prediction, VQE optimizer) enter the "
        r"fourth derivatives as $\sigma/\delta^4$.",
        fontsize=6.4, color="#212121", va="center")
ax.text(x0 + 1.32, 0.27,
        r"Controlled by tight convergence, the step $\delta = 0.2$, and one consistent "
        r"surface per finite-difference stencil.",
        fontsize=6.4, color="#212121", va="center")
for xi in (xs[0], xs[2]):              # links from calculators and force field
    ax.add_patch(FancyArrowPatch((xi + BOX_W / 2, Y0 - 0.01), (xi + BOX_W / 2, 0.67),
                                 arrowstyle="-", lw=0.9, ls=(0, (2, 1.5)), color=edge))

fig.savefig(HERE / "workflow.pdf")
fig.savefig(HERE / "workflow.png", dpi=300)
print("written workflow.pdf and workflow.png")
