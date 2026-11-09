"""Classic P-wave onset detection: STA/LTA trigger and AIC refinement.

All functions work on plain 1-D numpy arrays (ground velocity or counts) so they port
easily to a microcontroller-class node.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def bandpass(x: np.ndarray, fs: float, lo: float = 1.0, hi: float = 20.0, order: int = 4):
    """Causal Butterworth band-pass (what a node can run on a live stream)."""
    if not 0 < lo < hi < fs / 2:
        raise ValueError("need 0 < lo < hi < fs/2")
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfilt(sos, np.asarray(x, dtype=float))


def sta_lta(x: np.ndarray, fs: float, sta_s: float = 0.5, lta_s: float = 10.0) -> np.ndarray:
    """Energy STA/LTA ratio; the LTA window ends where the STA window starts.

    Output has the same length as ``x``; samples before one full STA+LTA span are 0.
    """
    x = np.asarray(x, dtype=float)
    nsta, nlta = round(sta_s * fs), round(lta_s * fs)
    if nsta < 1 or nlta <= nsta:
        raise ValueError("need 1 <= STA < LTA (in samples)")
    out = np.zeros(x.size)
    if x.size < nsta + nlta:
        return out
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    idx = np.arange(nsta + nlta, x.size + 1)  # exclusive end of the STA window
    sta = (c[idx] - c[idx - nsta]) / nsta
    lta = (c[idx - nsta] - c[idx - nsta - nlta]) / nlta
    out[idx - 1] = sta / np.maximum(lta, 1e-30)
    return out


class StreamingStaLta:
    """Recursive STA/LTA with O(1) state, fed one sample at a time."""

    def __init__(self, fs: float, sta_s: float = 0.5, lta_s: float = 10.0):
        if sta_s <= 0 or lta_s <= sta_s:
            raise ValueError("need 0 < sta_s < lta_s")
        self.a_sta = 1.0 / (sta_s * fs)
        self.a_lta = 1.0 / (lta_s * fs)
        self.sta = 0.0
        self.lta = 0.0
        self.n = 0
        self.warm = int(lta_s * fs)

    def push(self, v: float) -> float:
        """Feed one sample, return the current ratio (0 while warming up)."""
        e = v * v
        self.n += 1
        if self.n == 1:
            self.sta = self.lta = e
        self.sta += self.a_sta * (e - self.sta)
        if self.n > self.warm:
            self.lta += self.a_lta * (e - self.lta)
        else:
            self.lta += (e - self.lta) / self.n
        return self.sta / max(self.lta, 1e-30) if self.n > self.warm else 0.0


def first_trigger(ratio: np.ndarray, on: float, start: int = 0) -> int | None:
    """Index of the first sample >= ``start`` where ``ratio`` is at or above ``on``."""
    above = np.asarray(ratio) >= on
    above[:start] = False
    hits = np.flatnonzero(above)
    return int(hits[0]) if hits.size else None


def trigger_times(ratio: np.ndarray, on: float, off: float, fs: float, holdoff_s: float = 30.0):
    """All trigger onsets (s) with hysteresis (on/off) and a hold-off after each trigger."""
    if off > on:
        raise ValueError("off threshold must not exceed on threshold")
    ratio = np.asarray(ratio)
    ons = np.flatnonzero(ratio >= on)
    offs = np.flatnonzero(ratio <= off)
    times, pos = [], 0
    while True:
        k = np.searchsorted(ons, pos)
        if k >= ons.size:
            return times
        i = int(ons[k])
        times.append(i / fs)
        m = np.searchsorted(offs, i)
        j = int(offs[m]) if m < offs.size else ratio.size
        pos = max(j, i + int(holdoff_s * fs))


def aic(x: np.ndarray) -> np.ndarray:
    """Maeda AIC function; the minimum marks the change from noise to signal."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        raise ValueError("need at least 8 samples")
    c1, c2 = np.cumsum(x), np.cumsum(x * x)
    k = np.arange(2, n - 2)
    m1 = c1[k - 1] / k
    v1 = c2[k - 1] / k - m1 * m1
    n2 = n - k
    m2 = (c1[-1] - c1[k - 1]) / n2
    v2 = (c2[-1] - c2[k - 1]) / n2 - m2 * m2
    out = np.full(n, np.nan)
    out[k] = k * np.log(np.maximum(v1, 1e-30)) + (n - k - 1) * np.log(np.maximum(v2, 1e-30))
    return out


def aic_refine(x: np.ndarray, fs: float, t_trigger: float, back_s: float = 2.0, fwd_s: float = 1.0):
    """Refine a trigger time (s) to the AIC minimum in [t-back, t+fwd]."""
    i0 = max(0, int((t_trigger - back_s) * fs))
    i1 = min(len(x), int((t_trigger + fwd_s) * fs))
    if i1 - i0 < 8:
        return t_trigger
    seg = aic(x[i0:i1])
    return (i0 + int(np.nanargmin(seg))) / fs


def pick_sta_lta_aic(x, fs, sta_s=0.5, lta_s=10.0, on=4.0, lo=1.0, hi=20.0, aic_back=2.0):
    """Classic picker: band-pass, STA/LTA trigger, AIC refinement. Returns time (s) or None."""
    xf = bandpass(x, fs, lo, hi)
    r = sta_lta(xf, fs, sta_s, lta_s)
    i = first_trigger(r, on, start=int((lta_s + sta_s) * fs))
    if i is None:
        return None
    return aic_refine(xf, fs, i / fs, back_s=aic_back)
