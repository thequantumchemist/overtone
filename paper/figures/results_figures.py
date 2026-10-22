"""
Result figures of the paper from saved OVERTONE results (data/*.json).

    python results_figures.py

water_spectrum.pdf     water, mid- and near-IR with experimental band positions.
                       PRELIMINARY: B3LYP/def2-TZVP test data. Replace the
                       input by paper/01_water/results (MP2/aug-cc-pVTZ).
formaldehyde_gvpt2.pdf Fermi resonance in formaldehyde (GFN2-xTB, tutorial 4),
                       standard VPT2 compared with GVPT2.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from overtone import VPT2Result

HERE = Path(__file__).parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "mathtext.fontset": "dejavusans"})

# experimental band centres, J. Chem. Theory Comput. 10, 5426 (2014)
WATER_EXP = {r"$\nu_2$": 1595, r"$2\nu_2$": 3152, r"$\nu_1$": 3657, r"$\nu_3$": 3756,
             r"$\nu_1{+}\nu_2$": 5235, r"$\nu_2{+}\nu_3$": 5331,
             r"$2\nu_1$": 7202, r"$\nu_1{+}\nu_3$": 7250, r"$2\nu_3$": 7445}


def water(preliminary=True):
    res = VPT2Result.load(HERE / "data" / "water_b3lyp_gvpt2.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    res.plot_spectrum(ax=a1, xmin=1300, xmax=4100, show_harmonic=True, sticks=True,
                      by_kind=False, label="GVPT2")
    res.plot_spectrum(ax=a2, xmin=4700, xmax=7700, log=True, shape="gaussian",
                      kinds=["overtone", "combination"], by_kind=False, label="GVPT2",
                      fwhm={"overtone": 30, "combination": 30})
    a2.set_ylim(1e-5, 1.0)
    for ax, (lo, hi), labels in ((a1, (1300, 4100), False), (a2, (4700, 7700), True)):
        first = True
        for label, nu in WATER_EXP.items():
            if lo <= nu <= hi:
                ax.axvline(nu, color="tab:purple", ls="--", lw=0.7,
                           label="experiment" if first else None)
                first = False
                if labels:
                    # lower-wavenumber partner of a close pair to the right (x is inverted)
                    side = "left" if nu in (5235, 7202) else "right"
                    ax.text(nu, 0.6, label, rotation=90, fontsize=6.5, ha=side,
                            va="top", color="tab:purple")
        ax.legend(frameon=False, fontsize=6.5,
                  loc="lower center" if labels else "upper left")
    a1.set_title("(a) fundamentals", fontsize=8, loc="left")
    a2.set_title("(b) near-IR overtones and combinations", fontsize=8, loc="left")
    a2.set_ylabel("")
    if preliminary:
        fig.text(0.995, 0.01, "preliminary: B3LYP/def2-TZVP test data",
                 color="#c62828", fontsize=7, ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(HERE / "water_spectrum.pdf")
    fig.savefig(HERE / "water_spectrum.png", dpi=300)


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
