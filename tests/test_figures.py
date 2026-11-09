import numpy as np
from conftest import FS, synth
from matplotlib.image import imread

from pwave import figures
from pwave.data import Trace


def fake(n=6):
    traces = [Trace(f"e{i}", "S", 3.0, 40.0 + i, FS, synth(seed=i), t_p=30.0, t_s=40.0)
              for i in range(n)]
    picks = [30.05] * n
    res = {
        "n_events": n, "mag_range": [2.0, 4.0], "blind_zone_km": 33.6,
        "alert_rule": {"win_s": 3.0, "latency_s": 1.0},
        "_picks": {"sta_lta": picks, "learned": picks, "est": [3.0] * n},
    }
    return traces, res


def test_post_figure_is_exactly_960x700(tmp_path):
    traces, res = fake()
    figures.make_post(traces, res, tmp_path)
    img = imread(tmp_path / "post-figure.png")
    assert img.shape[:2] == (700, 960)


def test_palette_constants():
    assert figures.BG == "#16213A" and figures.ACCENT == "#38BDF8"


def test_best_example_falls_back_to_zero():
    traces, _ = fake(2)
    assert figures.best_example(traces, [None, None], [None, None]) == 0


def test_best_example_prefers_close_to_50km():
    traces, _ = fake(3)
    traces[0].dist_km, traces[1].dist_km, traces[2].dist_km = 36.0, 49.0, 69.0
    p = [30.0, 30.0, 30.0]
    assert figures.best_example(traces, p, p) == 1
    assert np.isfinite(traces[1].dist_km)
