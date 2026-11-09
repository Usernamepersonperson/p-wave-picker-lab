"""A small learned picker: logistic regression on five STA/LTA-style features.

Cheap enough for a microcontroller-class node (five coefficients plus a scaler).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .pickers import aic_refine, bandpass, sta_lta

STEP_S = 0.1
LTA_S = 10.0
STA_SET = (0.25, 1.0, 2.5)
WARM_S = LTA_S + max(STA_SET)


def features(xf: np.ndarray, fs: float, step_s: float = STEP_S):
    """Feature matrix on a ``step_s`` grid for an already band-passed trace.

    Columns: log10 STA/LTA for each STA length, log10 of 1 s RMS, and the
    0.5 s change of the 1 s log-ratio. Returns (times_s, F); rows before the
    LTA is full are dropped.
    """
    xf = np.asarray(xf, dtype=float)
    step = max(1, round(step_s * fs))
    ratios = [np.log10(np.maximum(sta_lta(xf, fs, s, LTA_S), 1e-3)) for s in STA_SET]
    nrms = int(fs)
    c = np.concatenate([[0.0], np.cumsum(xf * xf)])
    idx = np.arange(xf.size)
    lo = np.maximum(idx + 1 - nrms, 0)
    rms = np.sqrt(np.maximum((c[idx + 1] - c[lo]) / nrms, 1e-40))
    lag = round(0.5 * fs)
    slope = ratios[1] - np.concatenate([np.full(lag, ratios[1][0]), ratios[1][:-lag]])
    first = int(WARM_S * fs)
    sel = np.arange(first, xf.size, step)
    F = np.column_stack([r[sel] for r in ratios] + [np.log10(rms[sel]), slope[sel]])
    return sel / fs, F


class LearnedPicker:
    """Window classifier + 3-in-a-row rule + AIC refinement."""

    def __init__(self, p_on: float = 0.9, consec: int = 3, pos_s: float = 1.5):
        self.p_on, self.consec, self.pos_s = p_on, consec, pos_s
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")

    def rows(self, t: np.ndarray, F: np.ndarray, onset: float | None):
        """Labelled training rows from one record's feature matrix (onset None = pure noise)."""
        if onset is None:
            keep = np.arange(0, len(t), 3)
            return F[keep], np.zeros(keep.size)
        pos = (t >= onset) & (t < onset + self.pos_s)
        m = pos | (t < onset - 1.0)
        return F[m], pos[m].astype(float)

    def fit_rows(self, parts):
        """Fit on a list of (X, y) pieces from :meth:`rows`."""
        X = np.vstack([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        if y.sum() == 0 or y.sum() == y.size:
            raise ValueError("need both positive and negative windows")
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def fit(self, traces):
        """``traces``: iterable of (velocity, fs, onset_s or None for pure noise)."""
        parts = []
        for v, fs, onset in traces:
            t, F = features(bandpass(v, fs), fs)
            parts.append(self.rows(t, F, onset))
        return self.fit_rows(parts)

    def proba_features(self, F: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(F))[:, 1]

    def proba(self, v: np.ndarray, fs: float):
        t, F = features(bandpass(v, fs), fs)
        return t, self.proba_features(F)

    def triggers(self, t: np.ndarray, p: np.ndarray, p_on: float | None = None, after: float = 0.0):
        """First time >= ``after`` where ``consec`` consecutive probabilities reach ``p_on``."""
        thr = self.p_on if p_on is None else p_on
        ok = (p >= thr) & (t >= after)
        run = np.convolve(ok.astype(int), np.ones(self.consec, int), mode="valid")
        hits = np.flatnonzero(run == self.consec)
        return float(t[hits[0]]) if hits.size else None

    def pick(self, v: np.ndarray, fs: float, p_on: float | None = None, after: float = 0.0):
        t, p = self.proba(v, fs)
        trig = self.triggers(t, p, p_on, after)
        if trig is None:
            return None
        return aic_refine(bandpass(v, fs), fs, trig, back_s=2.5, fwd_s=0.5)
