"""README and LinkedIn figures (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .alert import VP_KM_S, VS_KM_S
from .pickers import bandpass

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = (
    "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA",
)
SOURCE = (
    "Source: Swiss Seismological Service (SED) at ETH Zurich, network CH, "
    "doi:10.12686/sed/networks/ch"
)


def _font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Segoe UI" if "Segoe UI" in names else "DejaVu Sans"


def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)


def _setup(size_px_fs: float):
    plt.rcParams.update({"font.family": _font(), "font.size": size_px_fs, "text.color": TEXT})


def best_example(traces, picks_sl, picks_ln):
    """Clear example: M>=2.8, 35-70 km, both pickers close to the analyst; nearest to 50 km."""
    cands = []
    for i, t in enumerate(traces):
        a, b = picks_sl[i], picks_ln[i]
        if a is None or b is None or t.mag < 2.8 or not 35 <= t.dist_km <= 70:
            continue
        if abs(a - t.t_p) < 0.3 and abs(b - t.t_p) < 0.5:
            cands.append((abs(t.dist_km - 50), i))
    return min(cands)[1] if cands else 0


def make_main(traces, res, out: Path):
    _setup(12)
    p = res["_picks"]
    sl, ln, est = p["sta_lta"], p["learned"], np.array(p["est"])
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.975, top=0.84, bottom=0.095,
                          hspace=0.42, wspace=0.2)
    fig.text(0.065, 0.945, "P-wave picking and quick magnitude on real Swiss broadband data",
             fontsize=21, fontweight="bold", color=TEXT)
    n = res["n_events"]
    fig.text(0.065, 0.905,
             f"{n} events M{res['mag_range'][0]:.1f}-{res['mag_range'][1]:.1f}, "
             f"{res['n_traces']} station records vs analyst picks. "
             "STA/LTA+AIC and a learned picker (5-fold, by event).", fontsize=13, color=MUTED)

    # (a) example waveform
    i = best_example(traces, sl, ln)
    t = traces[i]
    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    xf = bandpass(t.v, t.fs)
    tt = np.arange(xf.size) / t.fs - t.t_p
    ax.plot(tt, xf / np.max(np.abs(xf)), color=ACCENT, lw=1.0)
    ax.axvline(0, color=GREEN, lw=2, label="analyst pick")
    ax.axvline(sl[i] - t.t_p, color=AMBER, lw=1.6, ls="--", label="STA/LTA+AIC")
    ax.axvline(ln[i] - t.t_p, color=VIOLET, lw=1.6, ls=":", label="learned")
    ax.set_xlim(-6, 14)
    ax.set_xlabel("time from analyst P pick (s)")
    ax.set_ylabel("norm. velocity")
    ax.set_title(f"Example: M{t.mag:.1f}, {t.dist_km:.0f} km, station {t.sta}", fontsize=13,
                 loc="left")
    ax.legend(loc="upper left", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)

    # (b) pick error histogram
    ax = fig.add_subplot(gs[0, 1])
    _style(ax)
    bins = np.arange(-1, 3.01, 0.2)
    for name, picks, col in (("STA/LTA+AIC", sl, AMBER), ("learned", ln, VIOLET)):
        e = [q - tr.t_p for tr, q in zip(traces, picks, strict=True)
             if q is not None and -2 <= q - tr.t_p <= 10]
        ax.hist(np.clip(e, -1, 2.99), bins=bins, histtype="step", lw=2, color=col,
                label=f"{name} (n={len(e)})")
    ax.axvline(0, color=GREEN, lw=1.5)
    ax.set_xlabel("pick error vs analyst (s), errors above 3 s in the last bin")
    ax.set_ylabel("records")
    s1, s2 = res["sta_lta_aic"], res["learned"]
    ax.set_title(f"Median |error| {s1['median_abs']:.2f} s (STA/LTA), {s2['median_abs']:.2f} s "
                 "(learned)", fontsize=13, loc="left")
    ax.legend(loc="upper right", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)

    # (c) magnitude scatter
    ax = fig.add_subplot(gs[1, 0])
    _style(ax)
    ok = [(tr.mag, e) for tr, q, e in zip(traces, sl, est, strict=True)
          if q is not None and np.isfinite(e)]
    m = np.array(ok)
    ax.scatter(m[:, 0], m[:, 1], s=16, color=ACCENT, alpha=0.8)
    lim = (1.5, 5.0)
    ax.plot(lim, lim, color=GREEN, lw=1.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("catalogue magnitude")
    ax.set_ylabel("quick estimate (3 s window)")
    mg = res["magnitude"]["pd_model"]
    ax.set_title(f"Pd magnitude: median |error| {mg['median_abs']:.2f}, p90 {mg['p90_abs']:.2f}",
                 fontsize=13, loc="left")

    # (d) warning time
    ax = fig.add_subplot(gs[1, 1])
    _style(ax)
    lat = res["alert_rule"]
    d = np.linspace(0, 125, 200)
    w = d / VS_KM_S - (d / VP_KM_S + lat["win_s"] + lat["latency_s"])
    ax.plot(d, w, color=GREEN, lw=2.2, label="Vp 6.0, Vs 3.5 km/s")
    pts = [(tr.dist_km, tr.t_s - (q + lat["win_s"] + lat["latency_s"]))
           for tr, q in zip(traces, sl, strict=True)
           if q is not None and tr.t_s is not None and -2 <= q - tr.t_p <= 10]
    if pts:
        pa = np.array(pts)
        ax.scatter(pa[:, 0], pa[:, 1], s=14, color=AMBER, alpha=0.8, label="measured (S pick)")
    ax.axhline(0, color=RED, lw=1.5)
    ax.set_xlabel("hypocentral distance (km)")
    ax.set_ylabel("warning time (s)")
    ax.set_title(f"Blind zone ~{res['blind_zone_km']:.0f} km "
                 f"({lat['win_s']:.0f} s window + {lat['latency_s']:.0f} s latency)",
                 fontsize=13, loc="left")
    ax.legend(loc="upper left", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
    fig.text(0.065, 0.02, SOURCE, fontsize=10, color=MUTED)
    fig.savefig(out / "figure.png", dpi=100, facecolor=BG)
    plt.close(fig)


def make_post(traces, res, out: Path):
    _setup(11.5)
    p = res["_picks"]
    sl, ln = p["sta_lta"], p["learned"]
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    fig.text(0.05, 0.94, "Picking P waves on real earthquakes, 3 s of data",
             fontsize=14.5, fontweight="bold", color=TEXT)
    fig.text(0.05, 0.895, f"Open data: SED / ETH Zurich, {res['n_events']} Swiss events, "
             f"M{res['mag_range'][0]:.1f}-{res['mag_range'][1]:.1f}", fontsize=11.5, color=MUTED)
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.97, top=0.8, bottom=0.14, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    bins = np.arange(-1, 3.01, 0.2)
    for name, picks, col in (("STA/LTA+AIC", sl, AMBER), ("learned", ln, VIOLET)):
        e = [q - tr.t_p for tr, q in zip(traces, picks, strict=True)
             if q is not None and -2 <= q - tr.t_p <= 10]
        ax.hist(np.clip(e, -1, 2.99), bins=bins, histtype="step", lw=3, color=col, label=name)
    ax.axvline(0, color=GREEN, lw=2.5)
    ax.set_xlabel("pick error vs analyst (s)", fontsize=12)
    ax.set_ylabel("records", fontsize=12)
    ax.set_title("Pick error", fontsize=13, loc="left")
    ax.legend(loc="upper right", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=11.5)

    lat = res["alert_rule"]
    ax = fig.add_subplot(gs[0, 1])
    _style(ax)
    d = np.linspace(0, 125, 200)
    ax.plot(d, d / VS_KM_S - (d / VP_KM_S + lat["win_s"] + lat["latency_s"]), color=GREEN, lw=3.5)
    pts = [(tr.dist_km, tr.t_s - (q + lat["win_s"] + lat["latency_s"]))
           for tr, q in zip(traces, sl, strict=True)
           if q is not None and tr.t_s is not None and -2 <= q - tr.t_p <= 10]
    pa = np.array(pts)
    ax.scatter(pa[:, 0], pa[:, 1], s=22, color=AMBER, alpha=0.85, zorder=3)
    ax.axhline(0, color=RED, lw=2.5)
    ax.axvspan(0, res["blind_zone_km"], color=RED, alpha=0.15)
    ax.set_xlabel("distance (km)", fontsize=12)
    ax.set_ylabel("warning time (s)", fontsize=12)
    ax.set_title("Warning time (line: theory, dots: measured)", fontsize=13, loc="left")
    ax.text(res["blind_zone_km"] + 2, -3.2, "blind zone", color=RED, fontsize=11.5)
    fig.text(0.05, 0.035, "Source: SED at ETH Zurich, network CH, doi:10.12686/sed/networks/ch",
             fontsize=11.5, color=MUTED)
    fig.savefig(out / "post-figure.png", dpi=100, facecolor=BG)
    plt.close(fig)


def make_all(traces, noise, res, out: Path):
    out.mkdir(exist_ok=True)
    make_main(traces, res, out)
    make_post(traces, res, out)
