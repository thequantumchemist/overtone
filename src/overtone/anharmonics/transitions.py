"""
Vibrational transitions from the VPT2 anharmonicity constants.

Energies
--------
Term values follow Martin et al., J. Chem. Phys. 103, 2589 (1995), Eq. A4,

    G(v) = sum_r w_r v_r + sum_r X_rr (v_r^2 + v_r)
           + sum_{r<s} X_rs [v_r v_s + (v_r + v_s)/2],

measured from the vibrational ground state (G(0) = 0). Special cases are
Eqs. A5-A7 (fundamentals, overtones, binary combinations). With the
"deperturbed" method, levels involved in a Fermi resonance are deperturbed
values.

Transition moments (optional)
-----------------------------
The dipole surface is expanded in dimensionless normal coordinates,

    mu(q) = mu_0 + sum_a mu_a q_a + 1/2 sum_ab mu_ab q_a q_b,

and the wavefunctions are corrected to first order in the cubic force field
(Rayleigh-Schroedinger, harmonic zeroth order). By parity, the leading
non-vanishing order is

    fundamentals          0th order   <0|mu|1_i>   = mu_i / sqrt(2)
    overtones 2v_i        1st order   electrical (mu_ii) + mechanical (phi*mu)
    combinations v_i+v_j  1st order   electrical (mu_ij) + mechanical (phi*mu)

Second-order corrections to the fundamentals and intensities of states with
three or more quanta need semi-diagonal quartic constants (phi_ijkk) and
third dipole derivatives that are not computed here; those intensities are
left as None.

Terms whose energy denominator belongs to a detected Fermi resonance are
removed (the same deperturbation as in DVPT2), and the affected transitions
are flagged with ``resonance_affected``.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel

SQRT2 = np.sqrt(2.0)


def _intensity_constant() -> float:
    """
    K such that A [km/mol] = K * nu [cm^-1] * |M|^2 [(e*Angstrom)^2].

    A = 8 pi^3 N_A nu |M|^2 / (3 h c 4 pi eps0)   (SI, nu in m^-1),
    i.e. the integrated absorption coefficient of a 0 -> f transition.
    """
    N_A = 6.02214076e23
    h = 6.62607015e-34
    c = 2.99792458e8
    eps0 = 8.8541878128e-12
    e = 1.602176634e-19
    pref = 8 * np.pi**3 * N_A / (3 * h * c * 4 * np.pi * eps0)  # m/mol per (m^-1 (C m)^2)
    return pref * (e * 1e-10) ** 2 * 1e2 * 1e-3  # nu in cm^-1, M in e*A, result in km/mol


INTENSITY_CONSTANT = _intensity_constant()


class VPT2Transition(BaseModel):
    """
    One vibrational transition from the ground state.

    Attributes
    ----------
    quanta : Dict[int, int]
        Mode index -> number of quanta in the final state (non-zero only).
        Mode indices are the same as in X / anharm_freq.
    kind : 'fundamental' | 'overtone' | 'combination'
    label : str
        e.g. "v7", "2v7", "v7+v9" (mode index + 1, as in the summary tables)
    harmonic : float
        Harmonic transition wavenumber (cm^-1)
    energy : float
        VPT2 transition wavenumber (cm^-1)
    intensity : Optional[float]
        Integrated IR intensity (km/mol), only if dipole derivatives were given
    transition_dipole : Optional[List[float]]
        <0|mu|f> in e*Angstrom (x, y, z)
    resonance_affected : bool
        True if the upper state is part of a detected Fermi resonance. With
        method="gvpt2" such levels are treated variationally; with the other
        methods their energies/intensities are only approximate.
    mixing : Optional[Dict[str, float]]
        GVPT2 only: weights (squared coefficients >= 0.01) of the zeroth-order
        states in this level, e.g. {"v11": 0.62, "v2+v12": 0.38}
    """

    quanta: Dict[int, int]
    kind: Literal["fundamental", "overtone", "combination"]
    label: str
    harmonic: float
    energy: float
    intensity: Optional[float] = None
    transition_dipole: Optional[List[float]] = None
    resonance_affected: bool = False
    mixing: Optional[Dict[str, float]] = None

    @property
    def total_quanta(self) -> int:
        return int(sum(self.quanta.values()))


def enumerate_states(
    active_modes: Sequence[int], max_quanta: int
) -> List[Dict[int, int]]:
    """All excited states with 1..max_quanta total quanta in the active modes."""
    states = []
    modes = list(active_modes)
    for n in range(1, max_quanta + 1):
        for combo in itertools.combinations_with_replacement(modes, n):
            q: Dict[int, int] = {}
            for m in combo:
                q[m] = q.get(m, 0) + 1
            states.append(q)
    return states


def term_value(
    freq: np.ndarray,
    X: np.ndarray,
    quanta: Dict[int, int],
    active_modes: Sequence[int],
) -> float:
    """
    VPT2 term value G(v) - G(0) (Martin et al. 1995, Eq. A4).

    Parameters
    ----------
    quanta : {mode: number of quanta}, unexcited modes omitted
    active_modes : all active modes (unexcited modes still enter via X_rs v_r / 2)
    """
    v = {m: quanta.get(m, 0) for m in active_modes}
    e = 0.0
    for a, r in enumerate(active_modes):
        vr = v[r]
        e += freq[r] * vr + X[r, r] * (vr * vr + vr)
        for s in active_modes[a + 1 :]:
            vs = v[s]
            e += X[r, s] * (vr * vs + 0.5 * (vr + vs))
    return float(e)


def _label(quanta: Dict[int, int]) -> str:
    parts = []
    for m in sorted(quanta):
        n = quanta[m]
        parts.append(f"{n if n > 1 else ''}v{m + 1}")
    return "+".join(parts)


def _kind(quanta: Dict[int, int]) -> str:
    if len(quanta) == 1:
        return "fundamental" if next(iter(quanta.values())) == 1 else "overtone"
    return "combination"


class TransitionCalculator:
    """
    Overtone/combination energies and (optionally) IR intensities.

    Parameters
    ----------
    freq : np.ndarray
        Harmonic wavenumbers (cm^-1), indexed like X
    X : np.ndarray
        Anharmonicity constants (cm^-1)
    active_modes : Sequence[int]
    T : Optional[np.ndarray]
        Cubic force constants phi_ijk (cm^-1, dimensionless normal coordinates).
        Required for intensities.
    mu1 : Optional[np.ndarray]
        Dipole first derivatives d mu / d q_i, shape (n_modes, 3), e*Angstrom
    mu2 : Optional[np.ndarray]
        Dipole second derivatives, shape (n_modes, n_modes, 3), e*Angstrom
    fermi_type1, fermi_type2 : lists from VPT2Solver.resonance_check()
    """

    def __init__(
        self,
        freq: np.ndarray,
        X: np.ndarray,
        active_modes: Sequence[int],
        T: Optional[np.ndarray] = None,
        mu1: Optional[np.ndarray] = None,
        mu2: Optional[np.ndarray] = None,
        fermi_type1: Iterable[Tuple[int, int]] = (),
        fermi_type2: Iterable[Tuple[int, int, int]] = (),
    ):
        self.freq = np.asarray(freq, float)
        self.X = np.asarray(X, float)
        self.active = list(active_modes)
        self.T = T
        self.mu1 = mu1
        self.mu2 = mu2
        self.res1 = set(map(tuple, fermi_type1))
        self.res2 = {(h, min(a, b), max(a, b)) for h, a, b in fermi_type2}
        self.with_intensities = mu1 is not None
        if self.with_intensities:
            if T is None:
                raise ValueError("Cubic force constants T are needed for intensities")
            if mu2 is None:
                raise ValueError("Second dipole derivatives mu2 are needed for intensities")
            self._precompute()

    # ------------------------------------------------------------------
    # resonance bookkeeping
    # ------------------------------------------------------------------
    def _resonant_states(self) -> List[Dict[int, int]]:
        states = []
        for hi, lo in self.res1:
            states += [{hi: 1}, {lo: 2}]
        for hi, a, b in self.res2:
            states += [{hi: 1}, {a: 1, b: 1}]
        return states

    def is_resonance_affected(self, quanta: Dict[int, int]) -> bool:
        return any(quanta == s for s in self._resonant_states())

    # ------------------------------------------------------------------
    # transition moments
    # ------------------------------------------------------------------
    def _precompute(self):
        w, T, act = self.freq, self.T, self.active
        # sum_s phi_kss over active modes, for every k
        self._phi_kss = np.zeros(len(w))
        for k in act:
            self._phi_kss[k] = sum(T[k, s, s] for s in act)

    def _c1(self, k):  # <1_k|psi_0^(1)>
        return -self._phi_kss[k] / (4 * SQRT2 * self.freq[k])

    def _c3(self, k):  # <3_k|psi_0^(1)>
        return -self.T[k, k, k] * np.sqrt(3) / (36 * self.freq[k])

    def _c21(self, k, l):  # <2_k 1_l|psi_0^(1)>, k != l
        return -self.T[k, k, l] / (4 * (2 * self.freq[k] + self.freq[l]))

    def _c111(self, k, l, m):  # <1_k 1_l 1_m|psi_0^(1)>, distinct
        return -self.T[k, l, m] / (2 * SQRT2 * (self.freq[k] + self.freq[l] + self.freq[m]))

    def transition_moment(self, quanta: Dict[int, int]) -> Optional[np.ndarray]:
        """<0|mu|f> in e*Angstrom, or None if beyond the implemented order."""
        n = sum(quanta.values())
        mu1, mu2, T, w = self.mu1, self.mu2, self.T, self.freq
        act = self.active

        if n == 1:
            (i,) = quanta
            return mu1[i] / SQRT2

        if n == 2 and len(quanta) == 1:  # overtone 2 v_i
            (i,) = quanta
            M = mu2[i, i] / (2 * SQRT2)
            # ground-state correction
            M = M + self._c1(i) * mu1[i] + self._c3(i) * np.sqrt(1.5) * mu1[i]
            for k in act:
                if k != i:
                    M = M + self._c21(i, k) * mu1[k] / SQRT2
            # final-state correction via |1_k>
            for k in act:
                if k == i:
                    v = T[i, i, i] / 2 + (self._phi_kss[i] - T[i, i, i]) / 4
                    M = M + mu1[i] / SQRT2 * v / w[i]
                else:
                    if (k, i) in self.res1:  # w_k ~ 2 w_i, deperturbed
                        continue
                    M = M + mu1[k] / SQRT2 * (T[i, i, k] / 4) / (2 * w[i] - w[k])
            return M

        if n == 2:  # combination v_i + v_j
            i, j = sorted(quanta)
            M = mu2[i, j] / 2
            # ground-state correction
            M = M + self._c1(i) * mu1[j] / SQRT2 + self._c1(j) * mu1[i] / SQRT2
            M = M + self._c21(i, j) * mu1[i] + self._c21(j, i) * mu1[j]
            for k in act:
                if k not in (i, j):
                    M = M + self._c111(i, j, k) * mu1[k] / SQRT2
            # final-state correction via |1_k>
            v_i = (self._phi_kss[j] / 4 + T[i, i, j] / 2) / SQRT2  # <1_i|V|1_i 1_j>
            v_j = (self._phi_kss[i] / 4 + T[i, j, j] / 2) / SQRT2  # <1_j|V|1_i 1_j>
            M = M + mu1[i] / SQRT2 * v_i / w[j] + mu1[j] / SQRT2 * v_j / w[i]
            for k in act:
                if k in (i, j):
                    continue
                if (k, i, j) in self.res2:  # w_k ~ w_i + w_j, deperturbed
                    continue
                v = T[i, j, k] / (2 * SQRT2)
                M = M + mu1[k] / SQRT2 * v / (w[i] + w[j] - w[k])
            return M

        return None

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def calculate(self, max_quanta: int = 2, variational: bool = False) -> List[VPT2Transition]:
        """
        Parameters
        ----------
        max_quanta : highest total number of quanta in the band list
        variational : GVPT2. Diagonalise every group of Fermi-coupled states
            (deperturbed energies on the diagonal, cubic couplings off the
            diagonal) and mix the transition moments with the eigenvectors.
            Requires X from the deperturbed method.
        """
        out = self._calculate_plain(max_quanta)
        if variational and (self.res1 or self.res2):
            out = self._apply_gvpt2(out, max_quanta)
        return out

    # ------------------------------------------------------------------
    # GVPT2: variational treatment of Fermi-resonance polyads
    # ------------------------------------------------------------------
    @staticmethod
    def _key(quanta):
        return tuple(sorted((m, n) for m, n in quanta.items() if n))

    def _neighbours(self, q):
        """States coupled to q by a detected Fermi resonance, with the
        coupling matrix element <q'|H3|q> (cm^-1)."""
        T = self.T
        out = []

        def shifted(changes):
            new = dict(q)
            for m, d in changes:
                new[m] = new.get(m, 0) + d
                if new[m] < 0:
                    return None
            return {m: n for m, n in new.items() if n}

        for hi, lo in self.res1:  # w_hi ~ 2 w_lo, operator 1/2 phi_lo,lo,hi q_lo^2 q_hi
            # down: remove one quantum of hi, add two of lo
            if q.get(hi, 0) >= 1:
                n_lo, n_hi = q.get(lo, 0), q[hi]
                v = 0.5 * T[lo, lo, hi] * np.sqrt((n_lo + 1) * (n_lo + 2)) / 2 * np.sqrt(n_hi / 2)
                out.append((shifted([(hi, -1), (lo, 2)]), v))
            # up: the reverse
            if q.get(lo, 0) >= 2:
                n_lo, n_hi = q[lo], q.get(hi, 0)
                v = 0.5 * T[lo, lo, hi] * np.sqrt(n_lo * (n_lo - 1)) / 2 * np.sqrt((n_hi + 1) / 2)
                out.append((shifted([(hi, 1), (lo, -2)]), v))
        for hi, a, b in self.res2:  # w_hi ~ w_a + w_b, operator phi_hi,a,b q_hi q_a q_b
            if q.get(hi, 0) >= 1:
                v = T[hi, a, b] * np.sqrt((q.get(a, 0) + 1) / 2) * np.sqrt(
                    (q.get(b, 0) + 1) / 2) * np.sqrt(q[hi] / 2)
                out.append((shifted([(hi, -1), (a, 1), (b, 1)]), v))
            if q.get(a, 0) >= 1 and q.get(b, 0) >= 1:
                v = T[hi, a, b] * np.sqrt(q[a] / 2) * np.sqrt(q[b] / 2) * np.sqrt(
                    (q.get(hi, 0) + 1) / 2)
                out.append((shifted([(hi, 1), (a, -1), (b, -1)]), v))
        return [(n, v) for n, v in out if n]

    def _apply_gvpt2(self, transitions, max_quanta):
        from scipy.optimize import linear_sum_assignment

        by_key = {self._key(t.quanta): t for t in transitions}
        limit = max_quanta + 1  # partners of the highest states are included
        done = set()
        for start in list(by_key):
            if start in done:
                continue
            # collect the polyad (connected component) by breadth-first search
            states, couplings, queue = {start: dict(start)}, {}, [start]
            while queue:
                k = queue.pop()
                for nb, v in self._neighbours(states[k]):
                    if sum(nb.values()) > limit:
                        continue
                    kn = self._key(nb)
                    couplings[frozenset((k, kn))] = v
                    if kn not in states:
                        states[kn] = nb
                        queue.append(kn)
            done.update(states)
            if len(states) == 1:
                continue
            keys = list(states)
            n = len(keys)
            H = np.zeros((n, n))
            for i, k in enumerate(keys):
                H[i, i] = term_value(self.freq, self.X, states[k], self.active)
            for i in range(n):
                for j in range(i + 1, n):
                    v = couplings.get(frozenset((keys[i], keys[j])), 0.0)
                    H[i, j] = H[j, i] = v
            E, C = np.linalg.eigh(H)
            # one-to-one assignment of eigenvectors to zeroth-order states
            rows, cols = linear_sum_assignment(-(C**2))  # rows: basis, cols: eigvec
            eig_of = dict(zip(rows, cols))
            moments = None
            if self.with_intensities:
                moments = []
                for k in keys:
                    M = self.transition_moment(states[k])
                    moments.append(np.zeros(3) if M is None else M)
                moments = np.array(moments)
            for i, k in enumerate(keys):
                if k not in by_key:
                    continue  # partner state above max_quanta, not reported
                t = by_key[k]
                c = C[:, eig_of[i]]
                t.energy = float(E[eig_of[i]])
                t.resonance_affected = True
                t.mixing = {
                    _label(states[keys[j]]): round(float(c[j] ** 2), 3)
                    for j in np.argsort(-(c**2)) if c[j] ** 2 >= 0.01
                }
                if moments is not None:
                    M = c @ moments
                    t.transition_dipole = [float(x) for x in M]
                    t.intensity = float(INTENSITY_CONSTANT * t.energy * np.dot(M, M))
        return list(by_key.values())

    def _calculate_plain(self, max_quanta: int) -> List[VPT2Transition]:
        out = []
        for q in enumerate_states(self.active, max_quanta):
            harm = float(sum(self.freq[m] * n for m, n in q.items()))
            energy = term_value(self.freq, self.X, q, self.active)
            intensity, tdm = None, None
            if self.with_intensities:
                M = self.transition_moment(q)
                if M is not None:
                    tdm = [float(x) for x in M]
                    intensity = float(INTENSITY_CONSTANT * energy * np.dot(M, M))
            out.append(
                VPT2Transition(
                    quanta=q,
                    kind=_kind(q),
                    label=_label(q),
                    harmonic=harm,
                    energy=energy,
                    intensity=intensity,
                    transition_dipole=tdm,
                    resonance_affected=self.is_resonance_affected(q),
                )
            )
        out.sort(key=lambda t: t.energy)
        return out
