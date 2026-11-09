"""Fetch events, manual P/S picks and waveforms from the Swiss national FDSN service.

Everything lands in ``cache/`` (gitignored). Needs the optional ``obspy`` dependency.
Run with ``python -m pwave fetch``.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

SERVER = "ETH"
NETWORK = "CH"
START, END = "2021-01-01", "2026-10-31"
BBOX = dict(minlatitude=45.8, maxlatitude=47.8, minlongitude=5.9, maxlongitude=10.5)
MAG_MIN, MAG_MAX = 2.0, 4.5
MAX_DIST_KM = 120.0
N_STATIONS = 10
N_EVENTS = 48
PRE_S, POST_S = 40.0, 60.0
NOISE_HOURS_PER_STATION = 8
SEED = 7
KM_PER_DEG = 111.19


def event_key(event_id: str) -> str:
    """File-system safe key (Windows rejects ':' in names)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", event_id).strip("_")[-60:]


def qualifying_picks(event):
    """Manual vertical P picks on broadband channels, plus the S pick if any, per station."""
    origin = event.preferred_origin() or event.origins[0]
    picks = {p.resource_id: p for p in event.picks}
    by_sta: dict[str, dict] = {}
    for arr in origin.arrivals:
        pk = picks.get(arr.pick_id)
        if pk is None or pk.waveform_id.network_code != NETWORK:
            continue
        sta = pk.waveform_id.station_code
        dist = (arr.distance or 0.0) * KM_PER_DEG
        rec = by_sta.setdefault(sta, {"dist_km": dist})
        ph = (arr.phase or "").strip()
        if (
            ph in ("P", "Pg", "Pn")
            and pk.evaluation_mode == "manual"
            and pk.waveform_id.channel_code == "HHZ"
        ):
            rec["p"] = str(pk.time)
        elif ph in ("S", "Sg", "Sn") and pk.evaluation_mode == "manual":
            rec["s"] = str(pk.time)
    return {s: r for s, r in by_sta.items() if "p" in r and r["dist_km"] <= MAX_DIST_KM}


def fetch_catalogue(client, cache: Path):
    from obspy import UTCDateTime

    t0, t1 = UTCDateTime(START), UTCDateTime(END)
    rows = []
    t = t0
    while t < t1:
        t_next = min(t + 90 * 86400, t1)
        cat = client.get_events(
            starttime=t, endtime=t_next, minmagnitude=MAG_MIN, maxmagnitude=MAG_MAX,
            includearrivals=True, **BBOX,
        )
        for ev in cat:
            q = qualifying_picks(ev)
            mag = ev.preferred_magnitude() or ev.magnitudes[0]
            org = ev.preferred_origin() or ev.origins[0]
            if q:
                rows.append({
                    "id": str(ev.resource_id), "time": str(org.time), "mag": float(mag.mag),
                    "mag_type": mag.magnitude_type, "lat": org.latitude, "lon": org.longitude,
                    "depth_km": (org.depth or 0.0) / 1000.0, "picks": q,
                })
        t = t_next
    (cache / "catalogue_all.json").write_text(json.dumps(rows))
    return rows


def select_events(rows):
    """Choose the stations with most picks, then magnitude-stratified events (seeded)."""
    cnt = Counter(s for r in rows for s in r["picks"])
    stations = [s for s, _ in cnt.most_common(N_STATIONS)]
    cand = []
    for r in rows:
        pk = {s: v for s, v in r["picks"].items() if s in stations}
        if pk:
            cand.append({**r, "picks": pk})
    rng = random.Random(SEED)
    bins = [(2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.6)]
    chosen = []
    for lo, hi in bins:
        pool = [c for c in cand if lo <= c["mag"] < hi]
        rng.shuffle(pool)
        chosen += pool[: N_EVENTS // len(bins)]
    return stations, sorted(chosen, key=lambda c: c["time"])


def fetch_waveforms(client, stations, events, cache: Path):
    from obspy import UTCDateTime

    wf = cache / "waveforms"
    wf.mkdir(exist_ok=True)
    inv = client.get_stations(
        network=NETWORK, station=",".join(stations), channel="HHZ", level="response",
        starttime=UTCDateTime(START), endtime=UTCDateTime(END),
    )
    for ev in events:
        for sta, pk in ev["picks"].items():
            out = wf / f"{event_key(ev['id'])}_{sta}.npz"
            if out.exists():
                continue
            tp = UTCDateTime(pk["p"])
            try:
                st = client.get_waveforms(NETWORK, sta, "*", "HHZ", tp - PRE_S, tp + POST_S)
                st.merge(fill_value=0)
                tr = st[0]
                tr.remove_response(inventory=inv, output="VEL", pre_filt=(0.05, 0.1, 40, 45))
            except Exception as exc:
                print("skip", sta, ev["time"], exc)
                continue
            np.savez_compressed(
                out, data=tr.data.astype("float32"), fs=tr.stats.sampling_rate,
                t0=str(tr.stats.starttime), meta=json.dumps({**pk, "event": ev["id"], "sta": sta}),
            )


def fetch_noise(client, stations, events, cache: Path):
    """One-hour velocity records at random times with no catalogue event (M>=1) +-15 min."""
    from obspy import UTCDateTime

    out_dir = cache / "noise"
    out_dir.mkdir(exist_ok=True)
    rng = random.Random(SEED + 1)
    inv = client.get_stations(
        network=NETWORK, station=",".join(stations[:6]), channel="HHZ", level="response",
        starttime=UTCDateTime(START), endtime=UTCDateTime(END),
    )
    t_lo, t_hi = UTCDateTime("2024-01-01").timestamp, UTCDateTime("2026-09-30").timestamp
    for sta in stations[:6]:
        done = 0
        while done < NOISE_HOURS_PER_STATION:
            t = UTCDateTime(rng.uniform(t_lo, t_hi)).replace(minute=0, second=0, microsecond=0)
            out = out_dir / f"{sta}_{int(t.timestamp)}.npz"
            if out.exists():
                done += 1
                continue
            if _has_events(client, t, BBOX):
                continue
            try:
                st = client.get_waveforms(NETWORK, sta, "*", "HHZ", t, t + 3600)
                st.merge(fill_value=0)
                tr = st[0]
                tr.remove_response(inventory=inv, output="VEL", pre_filt=(0.05, 0.1, 40, 45))
            except Exception as exc:
                print("skip noise", sta, t, exc)
                continue
            np.savez_compressed(out, data=tr.data.astype("float32"), fs=tr.stats.sampling_rate,
                                t0=str(tr.stats.starttime))
            done += 1


def _has_events(client, t, bbox) -> bool:
    from obspy.clients.fdsn.header import FDSNNoDataException

    try:
        client.get_events(starttime=t - 900, endtime=t + 4500, minmagnitude=1.0, **bbox)
    except FDSNNoDataException:
        return False
    return True


def run(cache: Path) -> None:
    from obspy.clients.fdsn import Client

    cache.mkdir(exist_ok=True)
    client = Client(SERVER)
    rows = fetch_catalogue(client, cache)
    stations, events = select_events(rows)
    (cache / "selected.json").write_text(json.dumps({"stations": stations, "events": events}))
    print(f"{len(rows)} usable events, {len(events)} selected, stations {stations}")
    fetch_waveforms(client, stations, events, cache)
    fetch_noise(client, stations, events, cache)
