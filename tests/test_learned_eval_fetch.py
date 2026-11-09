import json
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FS, synth

from pwave import evaluate as ev
from pwave import fetch
from pwave.__main__ import build_parser
from pwave.data import Noise, Trace
from pwave.learned import LearnedPicker, features


def test_features_shape_and_finite():
    t, F = features(synth(dur_s=60.0), FS)
    assert F.shape == (len(t), 5) and np.all(np.isfinite(F))
    assert t[0] >= 12.0


def test_features_step_spacing():
    t, _ = features(synth(), FS, step_s=0.2)
    assert np.allclose(np.diff(t), 0.2)


def make_training(n=8):
    return [(synth(onset_s=30.0, seed=s, snr=8.0), FS, 30.0) for s in range(n)] + [
        (synth(onset_s=1e6, seed=100 + s), FS, None) for s in range(3)
    ]


def test_learned_fit_needs_both_classes():
    with pytest.raises(ValueError):
        LearnedPicker().fit([(synth(onset_s=1e6), FS, None)])


def test_learned_picks_synthetic_onset():
    m = LearnedPicker(p_on=0.9).fit(make_training())
    t = m.pick(synth(onset_s=32.0, seed=50, snr=8.0), FS)
    assert t is not None and abs(t - 32.0) < 1.0


def test_learned_quiet_on_noise():
    m = LearnedPicker(p_on=0.99).fit(make_training())
    assert m.pick(synth(onset_s=1e6, seed=77), FS) is None


def test_learned_triggers_needs_consecutive():
    m = LearnedPicker(consec=3)
    t = np.arange(10) * 0.1
    p = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0], float)
    assert m.triggers(t, p, 0.9) == pytest.approx(0.4)


def test_learned_triggers_after_filter():
    m = LearnedPicker(consec=1)
    t = np.arange(10) * 0.1
    p = np.ones(10)
    assert m.triggers(t, p, 0.5, after=0.45) == pytest.approx(0.5)


def test_error_stats_values():
    s = ev.error_stats([-1, 1, -2, 2, 0])
    assert s["n"] == 5 and s["median_abs"] == 1 and s["bias"] == 0
    assert s["p90_abs"] == pytest.approx(2.0)


def test_error_stats_empty():
    s = ev.error_stats([])
    assert s["n"] == 0 and np.isnan(s["median_abs"])


def test_pick_window_bounds():
    assert ev.pick_window(40.0, 40.0)
    assert ev.pick_window(38.0, 40.0)
    assert not ev.pick_window(37.9, 40.0)
    assert not ev.pick_window(50.1, 40.0)
    assert not ev.pick_window(None, 40.0)


def test_choose_threshold_min_false_triggers_among_good_rates():
    k = ev.choose_threshold(None, [5.0, 0.5, 0.1], [0.99, 0.98, 0.90], min_rate=0.97)
    assert k == 1


def test_choose_threshold_fallback_best_rate():
    assert ev.choose_threshold(None, [9.0, 8.0], [0.5, 0.6], min_rate=0.97) == 1


def test_choose_threshold_tie_prefers_higher_rate():
    assert ev.choose_threshold(None, [1.0, 1.0], [0.98, 0.99], min_rate=0.97) == 1


def test_station_days():
    n = [Noise("A", 100.0, np.zeros(3600 * 100)) for _ in range(24)]
    assert ev.station_days(n) == pytest.approx(1.0)


def test_sta_lta_pick_on_synthetic_trace():
    t = ev.sta_lta_pick(synth(), FS, 4.0, 28.0)
    assert t is not None and abs(t - 30.0) < 0.4


def test_summarise_picks_counts_misses():
    tr = [Trace("e", "S", 3.0, 30.0, FS, np.zeros(10), t_p=40.0) for _ in range(4)]
    s = ev.summarise_picks(tr, [40.1, 39.5, None, 70.0])
    assert s["pick_rate"] == 0.5 and s["n"] == 2


def test_warning_times_bins():
    tr = [Trace("e", "S", 3.0, 30.0, FS, np.zeros(10), t_p=40.0, t_s=50.0)]
    out = ev.warning_times(tr, [40.0], ev.AlertRule(win_s=3.0, latency_s=1.0))
    row = next(r for r in out if r["bin_km"] == [25, 50])
    assert row["n"] == 1 and row["median_s"] == pytest.approx(6.0)


def fake_event(phase="Pg", mode="manual", chan="HHZ", net="CH", dist_deg=0.5):
    pid = "smi:pick/1"
    pick = SimpleNamespace(
        resource_id=pid, evaluation_mode=mode, time="2025-01-01T00:00:10Z",
        waveform_id=SimpleNamespace(network_code=net, station_code="STA1", channel_code=chan),
    )
    arr = SimpleNamespace(pick_id=pid, phase=phase, distance=dist_deg)
    origin = SimpleNamespace(arrivals=[arr])
    return SimpleNamespace(picks=[pick], preferred_origin=lambda: origin, origins=[origin])


def test_qualifying_picks_accepts_manual_p():
    q = fetch.qualifying_picks(fake_event())
    assert "STA1" in q and q["STA1"]["dist_km"] == pytest.approx(0.5 * fetch.KM_PER_DEG)


@pytest.mark.parametrize("kw", [
    {"mode": "automatic"}, {"chan": "HNZ"}, {"net": "XX"}, {"phase": "Sg"}, {"dist_deg": 2.0},
])
def test_qualifying_picks_rejects(kw):
    assert fetch.qualifying_picks(fake_event(**kw)) == {}


def test_select_events_is_deterministic_and_stratified():
    rows = [
        {"id": f"e{i}", "time": f"2024-01-{i % 28 + 1:02d}T00:00:00",
         "mag": 2.0 + (i % 25) / 10, "picks": {"A": {}, "B": {}}}
        for i in range(200)
    ]
    s1, e1 = fetch.select_events(rows)
    s2, e2 = fetch.select_events(rows)
    assert s1 == s2 and [x["id"] for x in e1] == [x["id"] for x in e2]
    assert len(e1) == fetch.N_EVENTS
    assert min(x["mag"] for x in e1) < 2.5 and max(x["mag"] for x in e1) >= 3.5


def test_cli_parser_commands():
    p = build_parser()
    assert p.parse_args(["demo"]).cmd == "demo"
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_load_missing_cache_raises(tmp_path):
    from pwave import data

    (tmp_path / "selected.json").write_text(json.dumps({"stations": [], "events": []}))
    (tmp_path / "noise").mkdir()
    with pytest.raises(FileNotFoundError):
        data.load(tmp_path)
