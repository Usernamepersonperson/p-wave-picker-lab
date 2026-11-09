"""Quick magnitude from the first seconds after the P pick (Pd and tau_c style)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def velocity_to_displacement(v: np.ndarray, fs: float, hp: float = 0.075) -> np.ndarray:
    """Causal integration of velocity with a high-pass to suppress drift."""
    v = np.asarray(v, dtype=float)
    sos = butter(2, hp, btype="highpass", fs=fs, output="sos")
    return sosfilt(sos, np.cumsum(v) / fs)


def window_params(v: np.ndarray, fs: float, t_pick: float, win_s: float = 3.0):
    """Return (pd, tau_c) for ``win_s`` seconds after ``t_pick``.

    ``v`` is vertical ground velocity (m/s). pd is the peak displacement (m);
    tau_c = 2*pi/sqrt(r) with r = int(v^2) / int(d^2) over the window.
    """
    if win_s <= 0:
        raise ValueError("win_s must be positive")
    i0 = round(t_pick * fs)
    n = round(win_s * fs)
    if i0 < 0 or i0 + n > len(v):
        raise ValueError("window outside the record")
    d = velocity_to_displacement(v, fs)
    seg_v, seg_d = np.asarray(v[i0 : i0 + n], float), d[i0 : i0 + n]
    pd = float(np.max(np.abs(seg_d)))
    num, den = float(np.sum(seg_v**2)), float(np.sum(seg_d**2))
    if den <= 0 or num <= 0:
        return pd, float("nan")
    return pd, float(2 * np.pi / np.sqrt(num / den))


class MagnitudeModel:
    """Two empirical relations fitted by least squares.

    with distance:    M = a + b*log10(pd_m) + c*log10(R_km)
    without distance: M = a + b*log10(tau_c_s)
    """

    def __init__(self):
        self.coef_pd: np.ndarray | None = None
        self.coef_tau: np.ndarray | None = None

    def fit(self, pd, tau, r_km, mag):
        pd, tau, r_km, mag = (np.asarray(a, float) for a in (pd, tau, r_km, mag))
        if not (pd.size == tau.size == r_km.size == mag.size) or pd.size < 4:
            raise ValueError("need >= 4 matching samples")
        if np.any(pd <= 0) or np.any(r_km <= 0):
            raise ValueError("pd and r_km must be positive")
        A = np.column_stack([np.ones_like(pd), np.log10(pd), np.log10(r_km)])
        self.coef_pd = np.linalg.lstsq(A, mag, rcond=None)[0]
        ok = np.isfinite(tau) & (tau > 0)
        if ok.sum() < 3:
            raise ValueError("need >= 3 valid tau_c values")
        B = np.column_stack([np.ones(ok.sum()), np.log10(tau[ok])])
        self.coef_tau = np.linalg.lstsq(B, mag[ok], rcond=None)[0]
        return self

    def predict(self, pd=None, tau=None, r_km=None) -> float:
        """Pd model if distance is known, else the tau_c model."""
        if r_km is not None and pd is not None:
            if self.coef_pd is None:
                raise RuntimeError("model not fitted")
            a, b, c = self.coef_pd
            return float(a + b * np.log10(pd) + c * np.log10(r_km))
        if tau is not None and np.isfinite(tau):
            if self.coef_tau is None:
                raise RuntimeError("model not fitted")
            a, b = self.coef_tau
            return float(a + b * np.log10(tau))
        raise ValueError("need (pd and r_km) or tau")
