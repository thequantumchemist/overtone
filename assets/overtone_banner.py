"""
OVERTONE software banner (README, slides).

    python overtone_banner.py   ->  overtone_banner.png (1536 x 1024), overtone_banner.pdf
"""
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb

HERE = Path(__file__).parent
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
rng = np.random.default_rng(7)

W, H, DPI = 12, 8, 128
fig = plt.figure(figsize=(W, H), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.0)
ax.axis("off")


def glow_line(x, y, color, lw=2.0, layers=6, alpha=1.0, z=3):
    for i in range(layers, 0, -1):
        ax.plot(x, y, color=color, lw=lw + 4.5 * i, alpha=0.035 * alpha,
                solid_capstyle="round", zorder=z)
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=z + 1)


# ---------------------------------------------------------------- background
gy, gx = np.mgrid[0:1:400j, 0:1.5:600j]
c0, c1, c2 = map(np.array, map(to_rgb, ("#06122b", "#0d2f57", "#3a1a5e")))
t = np.clip(0.55 * gx / 1.5 + 0.45 * (1 - gy), 0, 1)[..., None]
bg = (1 - t) * c0 + t * c1
bg = bg + 0.35 * np.exp(-((gx - 1.25) ** 2 + (gy - 0.85) ** 2) / 0.12)[..., None] * (c2 - bg)
vign = 1 - 0.35 * (((gx - 0.75) / 0.9) ** 2 + ((gy - 0.5) / 0.7) ** 2)
ax.imshow(np.clip(bg * vign[..., None], 0, 1), extent=[0, 1.5, 0, 1], origin="lower",
          zorder=0, aspect="auto")

# faint equations
eqs = [
    r"$G(\mathbf{v})=\sum_r\omega_r v_r+\sum_r X_{rr}(v_r^2+v_r)+\sum_{r<s}X_{rs}\,v_rv_s$",
    r"$X_{ii}=\frac{\phi_{iiii}}{16}-\sum_k\frac{\phi_{iik}^2(8\omega_i^2-3\omega_k^2)}{16\,\omega_k(4\omega_i^2-\omega_k^2)}$",
    r"$V(q)=\frac{1}{2}\sum_i\omega_i q_i^2+\frac{1}{6}\sum\phi_{ijk}q_iq_jq_k+\frac{1}{24}\sum\phi_{ijkl}q_iq_jq_kq_l$",
    r"$\nu_i=\omega_i+2X_{ii}+\frac{1}{2}\sum_{j\neq i}X_{ij}$",
    r"$A_{0f}\propto\tilde\nu_{0f}\,|\langle 0|\boldsymbol{\mu}|f\rangle|^2$",
    r"$\frac{\partial E}{\partial q}\approx\frac{-E_{+2}+8E_{+1}-8E_{-1}+E_{-2}}{12\,\delta}$",
]
pos = [(0.62, 0.90), (0.98, 0.78), (0.04, 0.05), (1.12, 0.27), (0.05, 0.93), (0.98, 0.04)]
for e, (x, y) in zip(eqs, pos):
    ax.text(x, y, e, color="#9ecbff", alpha=0.12, fontsize=13, zorder=1)

# finite-difference stencil grid (faint)
for i in range(-4, 5):
    for j in (-2, -1, 1, 2):
        ax.scatter(1.33 + 0.018 * i, 0.86 + 0.025 * j, s=6, color="#ffb74d",
                   alpha=0.18, lw=0, zorder=1)

# ---------------------------------------------------------------- overtone spectrum
x = np.linspace(0.0, 1.5, 3000)
omega, chi = 0.30, 0.012               # "fundamental" and anharmonicity (axis units)
peaks = [(n * omega - chi * n * (n + 1) + 0.08, 0.36 * 0.42 ** (n - 1)) for n in range(1, 6)]
combos = [(0.49, 0.035), (0.73, 0.02), (0.95, 0.013), (1.14, 0.008)]
y = np.full_like(x, 0.135)
for p, h in peaks + combos:
    y += h * 0.012**2 / ((x - p) ** 2 + 0.012**2)
pts = np.array([x, y]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
cmap = LinearSegmentedColormap.from_list("ot", ["#00e5ff", "#7c4dff", "#ff4081", "#ffab40"])
for lw, a in ((18, 0.04), (11, 0.07), (6, 0.12), (3, 0.3)):
    lc = LineCollection(segs, cmap=cmap, lw=lw, alpha=a, zorder=2)
    lc.set_array(x)
    ax.add_collection(lc)
lc = LineCollection(segs, cmap=cmap, lw=1.6, zorder=3)
lc.set_array(x)
ax.add_collection(lc)
for n, (p, h) in enumerate(peaks, start=1):
    label = r"$\nu$" if n == 1 else rf"${n}\nu$"
    ax.text(p, 0.135 + h + 0.025, label, color="white", alpha=0.75, fontsize=13,
            ha="center", zorder=4,
            path_effects=[pe.withStroke(linewidth=3, foreground="#06122b", alpha=0.6)])

# ---------------------------------------------------------------- water molecule
def sphere(cx, cy, r, base, z=6):
    yy, xx = np.mgrid[-1:1:200j, -1:1:200j]
    rr = np.sqrt(xx**2 + yy**2)
    light = np.clip(1.05 - 0.9 * np.sqrt((xx + 0.35) ** 2 + (yy - 0.4) ** 2), 0, 1)
    rgb = np.array(to_rgb(base))
    img = np.empty((200, 200, 4))
    img[..., :3] = np.clip(rgb * (0.35 + 0.75 * light[..., None]) + 0.45 * light[..., None] ** 6, 0, 1)
    img[..., 3] = np.clip((1 - rr) * 40, 0, 1)
    # halo
    halo = np.zeros((200, 200, 4))
    halo[..., :3] = rgb
    halo[..., 3] = 0.35 * np.exp(-((rr - 0.5) ** 2) / 0.08) * (rr < 1)
    ax.imshow(halo, extent=[cx - 1.6 * r, cx + 1.6 * r, cy - 1.6 * r, cy + 1.6 * r],
              zorder=z - 1, origin="lower")
    ax.imshow(img, extent=[cx - r, cx + r, cy - r, cy + r], zorder=z, origin="lower")


O = (0.24, 0.74)
H1 = (0.12, 0.635)
H2 = (0.36, 0.635)
for Hx in (H1, H2):
    glow_line([O[0], Hx[0]], [O[1], Hx[1]], "#b3e5fc", lw=6, layers=4, alpha=0.9, z=4)
sphere(*O, 0.065, "#e53935")
sphere(*H1, 0.042, "#eceff1")
sphere(*H2, 0.042, "#eceff1")
# vibration arrows (symmetric stretch)
for Hx, d in ((H1, (-0.06, -0.05)), (H2, (0.06, -0.05))):
    ax.annotate("", xy=(Hx[0] + d[0], Hx[1] + d[1]), xytext=(Hx[0] + 0.3 * d[0], Hx[1] + 0.3 * d[1]),
                arrowprops=dict(arrowstyle="-|>", color="#00e5ff", lw=2.2, mutation_scale=16),
                zorder=7)

# ---------------------------------------------------------------- Morse logo mark
r = np.linspace(-0.9, 4.0, 600)
morse = (1 - np.exp(-1.25 * r)) ** 2
mx, my, sx, sy = 0.53, 0.47, 0.045, 0.25
msk = morse < 1.12
glow_line(mx + sx * r[msk], my + sy * morse[msk], "#e1f5fe", lw=2.2, layers=5, z=5)
hw = 0.30
for n in range(6):
    e = hw * (n + 0.5) - hw**2 / 4 * (n + 0.5) ** 2
    rr = r[morse <= e]
    col = cmap(n / 6)
    glow_line(mx + sx * np.array([rr.min(), rr.max()]), [my + sy * e] * 2, col,
              lw=1.6, layers=3, z=5)

# ---------------------------------------------------------------- title
title = ax.text(0.73, 0.635, "OVERTONE", color="white", fontsize=62, fontweight="bold",
                va="center", zorder=8)
title.set_path_effects([pe.withStroke(linewidth=22, foreground="#7c4dff", alpha=0.05),
                        pe.withStroke(linewidth=14, foreground="#00e5ff", alpha=0.07),
                        pe.withStroke(linewidth=7, foreground="#00e5ff", alpha=0.12),
                        pe.withStroke(linewidth=3, foreground="#e1f5fe", alpha=0.25),
                        pe.Normal()])
ax.text(0.735, 0.548, "Anharmonic vibrational spectra",
        color="#cfe8ff", fontsize=15, va="center", zorder=8)
ax.text(0.735, 0.508, "from quantum chemistry and machine learning",
        color="#cfe8ff", fontsize=15, va="center", zorder=8)
ax.text(0.735, 0.452, "VPT2  \u00b7  GVPT2  \u00b7  overtones  \u00b7  combinations  \u00b7  IR / NIR intensities",
        color="#8fb8e6", fontsize=11.5, va="center", zorder=8)

fig.savefig(HERE / "overtone_banner.png", dpi=DPI)
fig.savefig(HERE / "overtone_banner.pdf")
print("written overtone_banner.png and overtone_banner.pdf")
