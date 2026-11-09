"""Benchmark the pickers, magnitude model and alert rule against analyst picks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .alert import AlertRule, blind_zone_km, warning_time
from .data import Noise, Trace
from .learned import LearnedPicker, features
from .magnitude import MagnitudeModel, window_params
from .pickers import (
    aic_refine,
    bandpass,
    first_trigger,
    sta_lta,
    trigger_times,
)

EARLY_S, LATE_S = 2.0, 10.0  # a pick counts as "found" in [P - 2 s, P + 10 s]
STA_S, LTA_S, OFF_RATIO = 0.5, 10.0, 1.5
STA_GRID = (0.5, 1.0, 2.0)
ON_GRID = (8.0, 14.0, 20.0, 30.0, 50.0, 80.0)
P_GRID = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999)
MIN_PICK_RATE = 0.97
FOLDS = 5
DIST_BINS = ((0, 25), (25, 50), (50, 75), (75, 125))


def error_stats(errors) -> dict:
    """Median / p90 of absolute error and signed bias; empty input gives NaNs."""
    e = np.asarray(errors, float)
    if e.size == 0:
        return {"n": 0, "median_abs": float("nan"), "p90_abs": float("nan"), "bias": float("nan")}
    a = np.abs(e)
    return {
        "n": int(e.size),
        "median_abs": float(np.median(a)),
        "p90_abs": float(np.percentile(a, 90)),
        "bias": float(np.median(e)),
    }


def pick_window(t_pick, t_true) -> bool:
    return t_pick is not None and t_true - EARLY_S <= t_pick <= t_true + LATE_S


def sta_lta_pick(v, fs, on, start_s):
    """STA/LTA + AIC pick of the first trigger at or after ``start_s``."""
    xf = bandpass(v, fs)
    r = sta_lta(xf, fs, STA_S, LTA_S)
    i = first_trigger(r, on, start=int(max(start_s, LTA_S + STA_S) * fs))
    return None if i is None else aic_refine(xf, fs, i / fs)


def noise_ratios(noise: list[Noise], sta_s: float = STA_S):
    return [sta_lta(bandpass(rec.v, rec.fs), rec.fs, sta_s, LTA_S) for rec in noise]


def noise_false_triggers_sta_lta(noise: list[Noise], on: float, ratios) -> int:
    return sum(
        len(trigger_times(r, on, min(OFF_RATIO, on), rec.fs))
        for r, rec in zip(ratios, noise, strict=True)
    )


def station_days(noise: list[Noise]) -> float:
    return sum(len(r.v) / r.fs for r in noise) / 86400.0


def choose_threshold(cands, ft_rate, pick_rate, min_rate=MIN_PICK_RATE):
    """Fewest false triggers among settings that still find ``min_rate`` of the picks.

    Ties go to the higher pick rate. Falls back to the highest pick rate if none qualify.
    """
    ok = [i for i, r in enumerate(pick_rate) if r >= min_rate]
    if not ok:
        return int(np.argmax(pick_rate))
    return min(ok, key=lambda i: (ft_rate[i], -pick_rate[i]))


def eval_sta_lta(traces: list[Trace], noise: list[Noise]):
    days = station_days(noise)
    cands, ft, rate, picks_by = [], [], [], []
    for sta_s in STA_GRID:
        ratios = noise_ratios(noise, sta_s)
        cache = []
        for t in traces:
            xf = bandpass(t.v, t.fs)
            cache.append((xf, sta_lta(xf, t.fs, sta_s, LTA_S)))
        for on in ON_GRID:
            picks = []
            for t, (xf, r) in zip(traces, cache, strict=True):
                i = first_trigger(r, on, start=int(max(t.t_p - EARLY_S, LTA_S + sta_s) * t.fs))
                picks.append(None if i is None else aic_refine(xf, t.fs, i / t.fs))
            cands.append((sta_s, on))
            ft.append(noise_false_triggers_sta_lta(noise, on, ratios) / days)
            rate.append(sum(pick_window(p, t.t_p) for t, p in zip(traces, picks, strict=True))
                        / len(traces))
            picks_by.append(picks)
    k = choose_threshold(cands, ft, rate)
    sta_s, on = cands[k]
    info = {"sta_s": sta_s, "on": on, "ft_per_station_day": ft[k],
            "grid": [{"sta_s": c[0], "on": c[1], "ft": f, "rate": r}
                     for c, f, r in zip(cands, ft, rate, strict=True)]}
    return info, picks_by[k]


def eval_learned(traces: list[Trace], noise: list[Noise]):
    """Group k-fold by event (noise records by index); everything scored out-of-fold."""
    ev_ids = sorted({t.event for t in traces})
    fold_of = {e: i % FOLDS for i, e in enumerate(ev_ids)}
    feats_tr = [features(bandpass(t.v, t.fs), t.fs) for t in traces]
    feats_no = [features(bandpass(n.v, n.fs), n.fs) for n in noise]
    probe = LearnedPicker()
    parts_tr = [probe.rows(*f, t.t_p) for f, t in zip(feats_tr, traces, strict=True)]
    parts_no = [probe.rows(*f, None) for f in feats_no]
    oof_tr, oof_no = {}, {}
    for f in range(FOLDS):
        train = [p for p, t in zip(parts_tr, traces, strict=True) if fold_of[t.event] != f]
        train += [p for i, p in enumerate(parts_no) if i % FOLDS != f]
        m = LearnedPicker().fit_rows(train)
        for i, t in enumerate(traces):
            if fold_of[t.event] == f:
                oof_tr[i] = (feats_tr[i][0], m.proba_features(feats_tr[i][1]))
        for i in range(len(noise)):
            if i % FOLDS == f:
                oof_no[i] = (feats_no[i][0], m.proba_features(feats_no[i][1]))
    helper = LearnedPicker()
    days = station_days(noise)
    ft, rate = [], []
    for p in P_GRID:
        ft.append(sum(_count_runs(t, pr, helper, p) for t, pr in oof_no.values()) / days)
        hit = 0
        for i, t in enumerate(traces):
            tt, pr = oof_tr[i]
            trig = helper.triggers(tt, pr, p, after=t.t_p - EARLY_S)
            hit += pick_window(trig, t.t_p)
        rate.append(hit / len(traces))
    k = choose_threshold(P_GRID, ft, rate)
    p_on = P_GRID[k]
    picks = []
    for i, t in enumerate(traces):
        tt, pr = oof_tr[i]
        trig = helper.triggers(tt, pr, p_on, after=t.t_p - EARLY_S)
        picks.append(None if trig is None else
                     aic_refine(bandpass(t.v, t.fs), t.fs, trig, back_s=2.5, fwd_s=0.5))
    info = {"p_on": p_on, "ft_per_station_day": ft[k],
            "grid": [{"p_on": p, "ft": f, "rate": r}
                     for p, f, r in zip(P_GRID, ft, rate, strict=True)]}
    return info, picks


def _count_runs(t, p, helper: LearnedPicker, thr: float, holdoff_s: float = 30.0) -> int:
    """Number of learned-picker triggers on a noise record, with the same hold-off."""
    n, after = 0, 0.0
    while True:
        trig = helper.triggers(t, p, thr, after=after)
        if trig is None:
            return n
        n += 1
        after = trig + holdoff_s


def summarise_picks(traces: list[Trace], picks) -> dict:
    found = [(t, p) for t, p in zip(traces, picks, strict=True) if pick_window(p, t.t_p)]
    errs = [p - t.t_p for t, p in found]
    s = error_stats(errs)
    s["pick_rate"] = len(found) / len(traces) if traces else float("nan")
    s["frac_within_0.5s"] = float(np.mean(np.abs(errs) <= 0.5)) if errs else float("nan")
    return s


def eval_magnitude(traces: list[Trace], picks, win_s: float, rule: AlertRule, r_assumed: float):
    """Leave-one-event-out Pd/tau_c models on the STA/LTA picks; returns metrics + rows."""
    rows = []
    for k, (t, p) in enumerate(zip(traces, picks, strict=True)):
        if not pick_window(p, t.t_p):
            continue
        try:
            pd, tau = window_params(t.v, t.fs, p, win_s)
        except ValueError:
            continue
        if pd > 0:
            rows.append((t, p, pd, tau, k))
    ev = np.array([r[0].event for r in rows])
    pd = np.array([r[2] for r in rows])
    tau = np.array([r[3] for r in rows])
    rk = np.array([r[0].dist_km for r in rows])
    mag = np.array([r[0].mag for r in rows])
    est_pd, est_tau = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for e in np.unique(ev):
        tr, te = ev != e, ev == e
        m = MagnitudeModel().fit(pd[tr], tau[tr], rk[tr], mag[tr])
        for i in np.flatnonzero(te):
            est_pd[i] = m.predict(pd=pd[i], r_km=rk[i])
            if np.isfinite(tau[i]):
                est_tau[i] = m.predict(tau=tau[i])
    full = MagnitudeModel().fit(pd, tau, rk, mag)
    fin = np.isfinite(est_tau)
    out = {
        "n_traces": len(rows),
        "pd_model": error_stats(est_pd - mag),
        "tau_model": error_stats((est_tau - mag)[np.isfinite(est_tau)]),
        "pd_coef": [float(c) for c in full.coef_pd],
        "tau_coef": [float(c) for c in full.coef_tau],
        "pd_corr": float(np.corrcoef(est_pd, mag)[0, 1]),
        "tau_corr": float(np.corrcoef(est_tau[fin], mag[fin])[0, 1]),
    }
    # per-event median estimate across its stations
    ev_err = []
    for e in np.unique(ev):
        k = ev == e
        ev_err.append(float(np.median(est_pd[k]) - mag[k][0]))
    out["constant_guess_median_abs"] = float(np.median(np.abs(mag - np.median(mag))))
    out["pd_event_level"] = error_stats(ev_err)
    # alert rule on out-of-fold Pd estimates
    big = mag >= rule.min_mag
    alert = np.array([rule.decide(est_pd[i], pd[i]) for i in range(len(rows))])
    out["alert"] = {
        "min_mag": rule.min_mag,
        "recall_big": float(alert[big].mean()) if big.any() else float("nan"),
        "false_alert_rate_small": float(alert[~big].mean()) if (~big).any() else float("nan"),
        "n_big": int(big.sum()), "n_small": int((~big).sum()),
    }
    out["_model"] = full
    out["_assumed_r"] = r_assumed
    est_trace = np.full(len(traces), np.nan)
    est_trace[[r[4] for r in rows]] = est_pd
    return out, rows, est_trace


def noise_false_alerts(
    noise: list[Noise], on: float, sta_s: float, model: MagnitudeModel, rule: AlertRule,
    r_assumed: float,
) -> int:
    """False alerts on pure noise: trigger, quick magnitude at an assumed distance, rule."""
    n = 0
    for rec in noise:
        xf = bandpass(rec.v, rec.fs)
        r = sta_lta(xf, rec.fs, sta_s, LTA_S)
        for ts in trigger_times(r, on, min(OFF_RATIO, on), rec.fs):
            pk = aic_refine(xf, rec.fs, ts)
            try:
                pd, _ = window_params(rec.v, rec.fs, pk, rule.win_s)
            except ValueError:
                continue
            if pd > 0 and rule.decide(model.predict(pd=pd, r_km=r_assumed), pd):
                n += 1
    return n


def warning_times(traces: list[Trace], picks, rule: AlertRule):
    """Measured: analyst S arrival minus (detected P + window + latency), per distance bin."""
    vals = []
    for t, p in zip(traces, picks, strict=True):
        if t.t_s is not None and pick_window(p, t.t_p):
            vals.append((t.dist_km, t.t_s - (p + rule.win_s + rule.latency_s)))
    table = []
    for lo, hi in DIST_BINS:
        w = [v for d, v in vals if lo <= d < hi]
        table.append({
            "bin_km": [lo, hi], "n": len(w),
            "median_s": float(np.median(w)) if w else float("nan"),
            "frac_positive": float(np.mean(np.array(w) > 0)) if w else float("nan"),
        })
    return table


def run(traces: list[Trace], noise: list[Noise], out_json: Path | None = None) -> dict:
    rule_base = AlertRule()
    sl, picks_sl = eval_sta_lta(traces, noise)
    ln, picks_ln = eval_learned(traces, noise)
    r_assumed = float(np.median([t.dist_km for t in traces]))
    # data-driven amplitude gate: 10th percentile of detected-event Pd
    _, rows0, _ = eval_magnitude(traces, picks_sl, rule_base.win_s, rule_base, r_assumed)
    pd_all = np.array([r[2] for r in rows0])
    rule = AlertRule(min_pd_m=float(np.percentile(pd_all, 10)))
    mag, _rows, est = eval_magnitude(traces, picks_sl, rule.win_s, rule, r_assumed)
    model = mag.pop("_model")
    mag.pop("_assumed_r")
    fa = noise_false_alerts(noise, sl["on"], sl["sta_s"], model, rule, r_assumed)
    days = station_days(noise)
    res = {
        "n_events": len({t.event for t in traces}),
        "n_traces": len(traces),
        "n_stations": len({t.sta for t in traces}),
        "mag_range": [min(t.mag for t in traces), max(t.mag for t in traces)],
        "dist_range_km": [min(t.dist_km for t in traces), max(t.dist_km for t in traces)],
        "noise_station_days": days,
        "sta_lta_aic": {**sl, **summarise_picks(traces, picks_sl)},
        "learned": {**ln, **summarise_picks(traces, picks_ln)},
        "magnitude": mag,
        "alert_rule": {"min_mag": rule.min_mag, "min_pd_m": rule.min_pd_m, "win_s": rule.win_s,
                       "latency_s": rule.latency_s, "assumed_r_km": r_assumed,
                       "false_alerts_per_station_day": fa / days},
        "warning_time_measured": warning_times(traces, picks_sl, rule),
        "warning_time_theory": [
            {"dist_km": d, "s": warning_time(d, rule.win_s, rule.latency_s)}
            for d in (25, 50, 75, 100)
        ],
        "blind_zone_km": blind_zone_km(rule.win_s, rule.latency_s),
    }
    res["_picks"] = {"sta_lta": picks_sl, "learned": picks_ln, "est": est.tolist()}
    if out_json:
        out = {k: v for k, v in res.items() if not k.startswith("_")}
        out_json.write_text(json.dumps(out, indent=2))
    return res
