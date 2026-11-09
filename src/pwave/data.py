"""Load the cached waveforms written by ``pwave.fetch``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fetch import PRE_S, event_key


@dataclass
class Trace:
    event: str
    sta: str
    mag: float
    dist_km: float  # hypocentral
    fs: float
    v: np.ndarray = field(repr=False)
    t_p: float = PRE_S  # analyst P pick, seconds from trace start
    t_s: float | None = None  # analyst S pick, seconds from trace start
    ev_time: str = ""


@dataclass
class Noise:
    sta: str
    fs: float
    v: np.ndarray = field(repr=False)


def _event_key(event_id: str) -> str:
    return event_key(event_id)


def load(cache: Path):
    """Return (traces, noise_records). Raises FileNotFoundError if nothing was fetched."""
    from obspy import UTCDateTime

    sel = json.loads((cache / "selected.json").read_text())
    traces: list[Trace] = []
    for ev in sel["events"]:
        for sta, pk in ev["picks"].items():
            f = cache / "waveforms" / f"{_event_key(ev['id'])}_{sta}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            t0 = UTCDateTime(str(z["t0"]))
            t_s = (UTCDateTime(pk["s"]) - t0) if "s" in pk else None
            d = float(np.hypot(pk["dist_km"], ev["depth_km"]))
            traces.append(Trace(
                ev["id"], sta, ev["mag"], d, float(z["fs"]), z["data"].astype(float),
                UTCDateTime(pk["p"]) - t0, t_s, ev["time"],
            ))
    noise = []
    for f in sorted((cache / "noise").glob("*.npz")):
        z = np.load(f)
        noise.append(Noise(f.name.split("_")[0], float(z["fs"]), z["data"].astype(float)))
    if not traces:
        raise FileNotFoundError(f"no waveforms in {cache}; run `python -m pwave fetch`")
    return traces, noise
