"""
Result figures of the paper from saved OVERTONE results (data/*.json).

    python results_figures.py

water_spectrum.pdf     water, MP2/aug-cc-pVTZ GVPT2 (paper/01_water), mid- and
                       near-IR, compared with HITRAN2024 band intensities.
formaldehyde_gvpt2.pdf Fermi resonance in formaldehyde (GFN2-xTB, tutorial 4),
                       standard VPT2 compared with GVPT2.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt

from overtone import VPT2Result, VPT2Transition, plot_spectrum

HERE = Path(__file__).parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "mathtext.fontset": "dejavusans"})

# experimental band centres, J. Chem. Theory Comput. 10, 5426 (2014)
WATER_EXP = {r"$\nu_2$": 1595, r"$2\nu_2$": 3152, r"$\nu_1$": 3657, r"$\nu_3$": 3756,
             r"$\nu_1{+}\nu_2$": 5235, r"$\nu_2{+}\nu_3$": 5331,
             r"$2\nu_1$": 7202, r"$\nu_1{+}\nu_3$": 7250, r"$2\nu_3$": 7445}


KIND_COLOR = {"fundamental": "#1565c0", "overtone": "#c62828", "combination": "#2e7d32"}
HITRAN_COLOR = "#9575cd"


def _peak(intensity, shape, fwhm):
    """Height of a unit-area line of the given shape times the intensity."""
    import numpy as np
    if shape == "lorentzian":
        return intensity / (np.pi * fwhm / 2)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return intensity / (sigma * np.sqrt(2 * np.pi))


def water(width_cm=8.4):
    """Single-column figure: (a) fundamentals, (b) near-IR, with HITRAN2024 behind."""
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from overtone.spectrum import broaden, broaden_transitions

    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    bands = [VPT2Transition(**b) for b in json.load(open(HERE / "data" / "water_mp2_bands.json"))]
    h = json.load(open(HERE / "data" / "water_hitran_bands.json"))
    w = width_cm / 2.54
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(w, 1.25 * w))

    # ---------------- (a) fundamentals, Lorentzian 10 cm-1, HITRAN mirrored
    shape, fwhm, lo, hi = "lorentzian", 10.0, 1300, 4100
    x = np.arange(lo, hi, 0.5)
    _, y_exp = broaden(h["positions"], h["intensities"], x=x, shape=shape, fwhm=fwhm)
    a1.fill_between(x, 0, -y_exp, color=HITRAN_COLOR, alpha=0.35, lw=0, zorder=1)
    a1.plot(x, -y_exp, color=HITRAN_COLOR, lw=0.5, zorder=1)
    _, y_h = broaden_transitions([t for t in bands if t.kind == "fundamental"], x=x,
                                 shape=shape, fwhm=fwhm, use_harmonic=True)
    a1.plot(x, y_h, color="0.65", lw=0.6, zorder=2)
    _, y = broaden_transitions(bands, x=x, shape=shape, fwhm=fwhm)
    a1.plot(x, y, color="black", lw=0.8, zorder=3)
    for t in bands:
        if lo < t.energy < hi:
            a1.vlines(t.energy, 0, _peak(t.intensity, shape, fwhm), color=KIND_COLOR[t.kind],
                      lw=1.0, zorder=4)
    a1.axhline(0, color="0.3", lw=0.4, zorder=2)
    a1.set_xlim(hi, lo)
    a1.set_ylabel("Intensity (km mol$^{-1}$ cm)")
    a1.text(0.02, 0.95, "(a)", transform=a1.transAxes, va="top", fontweight="bold")

    # ---------------- (b) near-IR, Gaussian 30 cm-1, log scale, HITRAN behind
    shape, fwhm, lo, hi = "gaussian", 30.0, 4800, 7850
    x = np.arange(lo, hi, 0.5)
    nir = [i for i, p in enumerate(h["positions"]) if lo < p < hi]
    _, y_exp = broaden([h["positions"][i] for i in nir], [h["intensities"][i] for i in nir],
                       x=x, shape=shape, fwhm=fwhm)
    floor = 1e-5
    a2.fill_between(x, floor, np.maximum(y_exp, floor), color=HITRAN_COLOR, alpha=0.35,
                    lw=0, zorder=1)
    a2.plot(x, np.maximum(y_exp, floor), color=HITRAN_COLOR, lw=0.5, zorder=1)
    nir_bands = [t for t in bands if t.kind != "fundamental" and lo < t.energy < hi]
    _, y = broaden_transitions(nir_bands, x=x, shape=shape, fwhm=fwhm)
    a2.plot(x, np.maximum(y, floor), color="black", lw=0.8, zorder=3)
    for t in nir_bands:
        a2.vlines(t.energy, floor, _peak(t.intensity, shape, fwhm), color=KIND_COLOR[t.kind],
                  lw=1.0, zorder=4)
    a2.set_yscale("log")
    a2.set_ylim(floor, 2.0)
    a2.set_xlim(hi, lo)
    a2.set_xlabel("Wavenumber (cm$^{-1}$)")
    a2.set_ylabel("Intensity (km mol$^{-1}$ cm)")
    a2.text(0.02, 0.95, "(b)", transform=a2.transAxes, va="top", fontweight="bold")
    names = {"2v3": r"$2\nu_3$", "v1+v3": r"$\nu_1{+}\nu_3$", "2v1": r"$2\nu_1$",
             "v2+v3": r"$\nu_2{+}\nu_3$", "v1+v2": r"$\nu_1{+}\nu_2$"}
    for i in nir:
        lab = h["labels"][i]
        side = "left" if lab in ("v1+v2", "2v1") else "right"
        a2.text(h["positions"][i], 0.9, names[lab], rotation=90, fontsize=5.5,
                ha=side, va="top", color="0.25")

    # ---------------- legend explaining every element
    handles = [Line2D([], [], color="0.65", lw=0.6, label="harmonic"),
               Line2D([], [], color="black", lw=0.8, label="GVPT2"),
               Patch(facecolor=HITRAN_COLOR, alpha=0.35, edgecolor=HITRAN_COLOR,
                     lw=0.5, label="HITRAN2024"),
               Line2D([], [], color=KIND_COLOR["fundamental"], lw=1.2, label="fundamentals"),
               Line2D([], [], color=KIND_COLOR["overtone"], lw=1.2, label="overtones"),
               Line2D([], [], color=KIND_COLOR["combination"], lw=1.2,
                      label="combination bands")]
    a1.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.62), ncol=2,
              frameon=False, fontsize=5.8, handlelength=1.4, columnspacing=1.0)

    fig.tight_layout(h_pad=0.6)
    fig.savefig(HERE / "water_spectrum.pdf")
    fig.savefig(HERE / "water_spectrum.png", dpi=600)


def formaldehyde():
    g = VPT2Result.load(HERE / "data" / "formaldehyde_gfn2_gvpt2.json")
    s = VPT2Result.load(HERE / "data" / "formaldehyde_gfn2_standard.json")
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    kw = dict(xmin=2400, xmax=2850, by_kind=False, fwhm=8)
    s.plot_spectrum(ax=ax, label="standard VPT2", color="0.65", sticks=False, **kw)
    g.plot_spectrum(ax=ax, label="GVPT2", **kw)
    for t in g.transitions:
        if 2400 <= t.energy <= 2850 and t.intensity and t.intensity > 1:
            lab = {"v11": r"$\nu_{11}$", "v12": r"$\nu_{12}$", "v8+v9": r"$\nu_8{+}\nu_9$"}.get(t.label, t.label)
            ax.annotate(lab, (t.energy, t.intensity / (3.1416 * 4)), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=7)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "formaldehyde_gvpt2.pdf")
    fig.savefig(HERE / "formaldehyde_gvpt2.png", dpi=300)


if __name__ == "__main__":
    water()
    formaldehyde()
    print("written water_spectrum.pdf and formaldehyde_gvpt2.pdf")
