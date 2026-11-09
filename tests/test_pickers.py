import numpy as np
import pytest
from conftest import FS, synth

from pwave import pickers as pk


def test_sta_lta_same_length(burst):
    assert pk.sta_lta(burst, FS).shape == burst.shape


def test_sta_lta_zero_before_warmup(burst):
    r = pk.sta_lta(burst, FS, 0.5, 10.0)
    assert np.all(r[: int(10.5 * FS) - 1] == 0)


def test_sta_lta_short_input_all_zero():
    assert np.all(pk.sta_lta(np.ones(100), FS) == 0)


def test_sta_lta_white_noise_ratio_near_one():
    x = np.random.default_rng(1).normal(size=60 * int(FS))
    r = pk.sta_lta(x, FS)[int(11 * FS):]
    assert 0.9 < r.mean() < 1.2


def test_sta_lta_step_gives_large_ratio():
    x = np.concatenate([np.ones(2000), 10 * np.ones(500)])
    assert pk.sta_lta(x, FS).max() > 50


@pytest.mark.parametrize("sta,lta", [(0, 10), (10, 5), (0.001, 10)])
def test_sta_lta_bad_windows(sta, lta):
    with pytest.raises(ValueError):
        pk.sta_lta(np.ones(5000), FS, sta, lta)


def test_streaming_matches_batch_trigger_time(burst):
    s = pk.StreamingStaLta(FS)
    r = np.array([s.push(v) for v in burst])
    i = pk.first_trigger(r, 4.0)
    j = pk.first_trigger(pk.sta_lta(burst, FS), 4.0)
    assert abs(i - j) < 1.0 * FS


def test_streaming_zero_during_warmup():
    s = pk.StreamingStaLta(FS, 0.5, 10.0)
    assert all(s.push(1.0) == 0 for _ in range(int(10 * FS)))


def test_streaming_bad_args():
    with pytest.raises(ValueError):
        pk.StreamingStaLta(FS, 5.0, 1.0)


def test_first_trigger_none_below_threshold():
    assert pk.first_trigger(np.ones(10), 2.0) is None


def test_first_trigger_respects_start():
    r = np.array([5.0, 0, 0, 6.0])
    assert pk.first_trigger(r, 4.0, start=1) == 3


def test_trigger_times_hysteresis_and_holdoff():
    r = np.zeros(5000)
    r[1000:1100] = 6.0
    r[1200:1300] = 6.0  # inside the 30 s hold-off, must not retrigger
    r[4500:4600] = 6.0
    t = pk.trigger_times(r, 4.0, 1.5, FS, holdoff_s=30.0)
    assert t == [10.0, 45.0]


def test_trigger_times_off_above_on_raises():
    with pytest.raises(ValueError):
        pk.trigger_times(np.zeros(10), 2.0, 3.0, FS)


def test_aic_minimum_at_change():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(0, 1, 500), rng.normal(0, 8, 500)])
    assert abs(int(np.nanargmin(pk.aic(x))) - 500) < 15


def test_aic_too_short():
    with pytest.raises(ValueError):
        pk.aic(np.ones(4))


def test_aic_constant_signal_is_finite():
    a = pk.aic(np.ones(50))
    assert np.all(np.isfinite(a[2:-2]))


def test_aic_refine_improves_late_trigger(burst):
    xf = pk.bandpass(burst, FS)
    refined = pk.aic_refine(xf, FS, 31.5)
    assert abs(refined - 30.0) < 0.3


def test_aic_refine_tiny_window_returns_input():
    assert pk.aic_refine(np.ones(100), FS, 0.0, back_s=0.0, fwd_s=0.01) == 0.0


def test_pick_accuracy_synthetic(burst):
    t = pk.pick_sta_lta_aic(burst, FS)
    assert t is not None and abs(t - 30.0) < 0.3


@pytest.mark.parametrize("seed", range(5))
def test_pick_accuracy_many_seeds(seed):
    t = pk.pick_sta_lta_aic(synth(seed=seed, onset_s=25.0), FS)
    assert t is not None and abs(t - 25.0) < 0.4


def test_pick_none_on_pure_noise():
    x = np.random.default_rng(3).normal(size=60 * int(FS))
    assert pk.pick_sta_lta_aic(x, FS, on=6.0) is None


def test_bandpass_attenuates_out_of_band():
    t = np.arange(0, 60, 1 / FS)
    slow = np.sin(2 * np.pi * 0.05 * t)
    assert np.std(pk.bandpass(slow, FS)[int(20 * FS):]) < 0.02 * np.std(slow)


def test_bandpass_passes_in_band():
    t = np.arange(0, 30, 1 / FS)
    x = np.sin(2 * np.pi * 5 * t)
    y = pk.bandpass(x, FS)[int(10 * FS):]
    assert 0.8 < np.std(y) / np.std(x) < 1.2


def test_bandpass_invalid_band():
    with pytest.raises(ValueError):
        pk.bandpass(np.ones(100), FS, 10, 60)
