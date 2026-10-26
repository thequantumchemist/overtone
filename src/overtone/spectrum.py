"""
Broadened IR / NIR spectra from VPT2 transitions.

Every transition is a stick of integrated intensity A (km/mol) at its VPT2
wavenumber. Broadening uses area-normalised line shapes, so the area under
each band equals its intensity and y is in km mol^-1 per cm^-1.

    from overtone.spectrum import plot_spectrum
    plot_spectrum(result)                                  # Lorentzian, 10 cm-1 FWHM
    plot_spectrum(result, shape="gaussian")                # Gaussian, 20 cm-1 FWHM
    plot_spectrum(result, fwhm={"fundamental": 8, "overtone": 25, "combination": 25})
    plot_spectrum(result, xmin=4000, xmax=9000, log=True)  # NIR region
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from overtone.anharmonics.transitions import VPT2Transition

#: Default full widths at half maximum (cm^-1) for each line shape
DEFAULT_FWHM: Dict[str, float] = {"lorentzian": 10.0, "gaussian": 20.0}

KIND_COLORS = {
    "fundamental": "tab:blue",
    "overtone": "tab:red",
    "combination": "tab:green",
}

FWHMType = Union[None, float, Dict[str, float]]


# ----------------------------------------------------------------------
# line shapes (unit area)
# ----------------------------------------------------------------------
def lorentzian(x: np.ndarray, x0: float, fwhm: float) -> np.ndarray:
    g = 0.5 * fwhm
    return g / np.pi / ((x - x0) ** 2 + g**2)


def gaussian(x: np.ndarray, x0: float, fwhm: float) -> np.ndarray:
    s = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((x - x0) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


LINE_SHAPES = {"lorentzian": lorentzian, "gaussian": gaussian}


def _resolve_fwhm(shape: str, fwhm: FWHMType, kind: str) -> float:
    if fwhm is None:
        return DEFAULT_FWHM[shape]
    if isinstance(fwhm, dict):
        return float(fwhm.get(kind, DEFAULT_FWHM[shape]))
    return float(fwhm)


def broaden(
    positions: Sequence[float],
    intensities: Sequence[float],
    x: Optional[np.ndarray] = None,
    shape: str = "lorentzian",
    fwhm: Union[float, Sequence[float], None] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    resolution: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Broaden sticks into a continuous spectrum.

    Parameters
    ----------
    positions : line positions (cm^-1)
    intensities : integrated intensities (e.g. km/mol)
    x : optional grid (cm^-1); otherwise built from xmin/xmax/resolution
    shape : 'lorentzian' or 'gaussian'
    fwhm : one FWHM for all lines or one per line (cm^-1);
        default DEFAULT_FWHM[shape]
    """
    shape = shape.lower()
    if shape not in LINE_SHAPES:
        raise ValueError(f"shape must be one of {list(LINE_SHAPES)}, got {shape!r}")
    positions = np.asarray(positions, float)
    intensities = np.asarray(intensities, float)
    if fwhm is None:
        fwhm = DEFAULT_FWHM[shape]
    widths = np.broadcast_to(np.asarray(fwhm, float), positions.shape)
    if x is None:
        pad = 5 * (widths.max() if widths.size else 10.0)
        lo = xmin if xmin is not None else (positions.min() - pad if positions.size else 0.0)
        hi = xmax if xmax is not None else (positions.max() + pad if positions.size else 4000.0)
        x = np.arange(lo, hi + resolution, resolution)
    f = LINE_SHAPES[shape]
    y = np.zeros_like(x, dtype=float)
    for x0, a, w in zip(positions, intensities, widths):
        y += a * f(x, x0, w)
    return x, y


def broaden_transitions(
    transitions: Iterable[VPT2Transition],
    x: Optional[np.ndarray] = None,
    shape: str = "lorentzian",
    fwhm: FWHMType = None,
    kinds: Optional[Sequence[str]] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    resolution: float = 0.5,
    use_harmonic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Broadened spectrum from VPT2 transitions (y in km mol^-1 / cm^-1).

    fwhm may be a float or a dict per kind, e.g.
    {'fundamental': 8, 'overtone': 25, 'combination': 25}.
    Transitions without intensity are skipped.
    """
    sel = [
        t for t in transitions
        if t.intensity is not None and (kinds is None or t.kind in kinds)
    ]
    if not sel:
        raise ValueError(
            "No transitions with intensities. Intensities are only computed "
            "on request: run with intensities=True (needs dipole moments)."
        )
    pos = [t.harmonic if use_harmonic else t.energy for t in sel]
    inten = [t.intensity for t in sel]
    widths = [_resolve_fwhm(shape.lower(), fwhm, t.kind) for t in sel]
    return broaden(pos, inten, x=x, shape=shape, fwhm=widths, xmin=xmin,
                   xmax=xmax, resolution=resolution)


def plot_spectrum(
    result,
    shape: str = "lorentzian",
    fwhm: FWHMType = None,
    kinds: Optional[Sequence[str]] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    resolution: float = 0.5,
    sticks: bool = True,
    by_kind: bool = True,
    log: bool = False,
    normalize: bool = False,
    show_harmonic: bool = False,
    x_unit: str = "cm-1",
    invert_x: bool = True,
    ax=None,
    label: Optional[str] = None,
    color: str = "black",
    reference: Optional[dict] = None,
    reference_mode: str = "mirror",
):
    """
    Plot a broadened VPT2 spectrum.

    Parameters
    ----------
    result : VPT2Result or list of VPT2Transition
    shape : 'lorentzian' (default FWHM 10 cm^-1) or 'gaussian' (20 cm^-1)
    fwhm : float or {kind: fwhm}; overrides the defaults
    kinds : restrict to e.g. ['overtone', 'combination']
    xmin, xmax : wavenumber window (cm^-1)
    sticks : draw stick spectrum (scaled to the curve's axis)
    by_kind : additionally draw fundamentals/overtones/combinations as
        separate dashed curves
    log : logarithmic intensity axis (useful in the NIR, where overtones
        are orders of magnitude weaker than fundamentals)
    normalize : scale the total spectrum to a maximum of 1
    show_harmonic : overlay the double-harmonic spectrum (fundamentals at
        harmonic wavenumbers with zeroth-order intensities)
    x_unit : 'cm-1' or 'nm' (NIR convention); broadening is always done in cm^-1
    ax : matplotlib axes to draw into
    reference : dict, optional
        Reference bands (e.g. experiment) to compare with:
        {"positions": [...cm^-1], "intensities": [...km/mol], "label": "exp."}.
        They are broadened with the same line shape and width.
    reference_mode : 'mirror' (drawn downwards, default) or 'overlay'

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "plot_spectrum needs matplotlib: pip install matplotlib "
            "(or pip install overtone-vpt2[plot])"
        ) from err

    transitions = result.transitions if hasattr(result, "transitions") else list(result)
    if transitions is None:
        raise ValueError("This result has no transitions")
    shape = shape.lower()

    x, y = broaden_transitions(transitions, shape=shape, fwhm=fwhm, kinds=kinds,
                               xmin=xmin, xmax=xmax, resolution=resolution)
    scale = 1.0 / y.max() if normalize and y.max() > 0 else 1.0

    if x_unit == "nm":
        xp = 1e7 / x
        to_x = lambda nu: 1e7 / np.asarray(nu)
        xlabel = "Wavelength (nm)"
    elif x_unit == "cm-1":
        xp, to_x = x, (lambda nu: np.asarray(nu))
        xlabel = "Wavenumber (cm$^{-1}$)"
    else:
        raise ValueError("x_unit must be 'cm-1' or 'nm'")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    if show_harmonic:
        fund = [t for t in transitions if t.kind == "fundamental" and t.intensity is not None]
        if fund:
            xh, yh = broaden_transitions(fund, x=x, shape=shape, fwhm=fwhm, use_harmonic=True)
            ax.plot(xp, yh * scale, color="0.6", lw=1, label="harmonic")

    ax.plot(xp, y * scale, color=color, lw=1.5, label=label or f"VPT2 ({shape})")

    if by_kind:
        present = {t.kind for t in transitions if t.intensity is not None}
        for kind in ("fundamental", "overtone", "combination"):
            if kind not in present or (kinds is not None and kind not in kinds):
                continue
            if len(present) == 1:
                break
            _, yk = broaden_transitions(transitions, x=x, shape=shape, fwhm=fwhm, kinds=[kind])
            ax.plot(xp, yk * scale, color=KIND_COLORS[kind], lw=1, ls="--", label=f"{kind}s")

    if sticks:
        sel = [
            t for t in transitions
            if t.intensity is not None
            and (kinds is None or t.kind in kinds)
            and x[0] <= t.energy <= x[-1]
        ]
        if sel:
            top = (y * scale).max()
            imax = max(t.intensity for t in sel)
            for t in sel:
                h = t.intensity / imax * top
                if log:
                    h = max(h, top * 1e-6)
                ax.vlines(to_x(t.energy), top * 1e-6 if log else 0, h,
                          color=KIND_COLORS.get(t.kind, "k"), lw=1,
                          ls=":" if t.resonance_affected else "-")

    if reference is not None:
        if reference_mode not in ("mirror", "overlay"):
            raise ValueError("reference_mode must be 'mirror' or 'overlay'")
        if log and reference_mode == "mirror":
            raise ValueError("a mirrored reference cannot be drawn on a log axis; "
                             "use reference_mode='overlay'")
        ref_pos = np.asarray(reference["positions"], float)
        ref_int = np.asarray(reference["intensities"], float)
        ref_w = _resolve_fwhm(shape, fwhm if not isinstance(fwhm, dict) else None, "fundamental")
        _, yr = broaden(ref_pos, ref_int, x=x, shape=shape, fwhm=ref_w)
        rscale = scale if not normalize else (1.0 / yr.max() if yr.max() > 0 else 1.0)
        sign = -1.0 if reference_mode == "mirror" else 1.0
        ax.plot(xp, sign * yr * rscale, color="tab:purple", lw=1.2,
                label=reference.get("label", "reference"))
        if sticks:
            top = (yr * rscale).max()
            for p0, a0 in zip(ref_pos, ref_int):
                if x[0] <= p0 <= x[-1] and ref_int.max() > 0:
                    ax.vlines(to_x(p0), 0, sign * a0 / ref_int.max() * top,
                              color="tab:purple", lw=1)
        if reference_mode == "mirror":
            ax.axhline(0, color="0.5", lw=0.5)

    if log:
        ax.set_yscale("log")
        top = (y * scale).max()
        ax.set_ylim(top * 1e-5, top * 2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalised intensity" if normalize else "Intensity (km mol$^{-1}$ / cm$^{-1}$)")
    if invert_x and x_unit == "cm-1":
        ax.set_xlim(xp.max(), xp.min())
    else:
        ax.set_xlim(xp.min(), xp.max())
    ax.legend(frameon=False, fontsize=8)
    return ax
