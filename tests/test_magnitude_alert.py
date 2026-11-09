import math

import numpy as np
import pytest

from pwave.alert import AlertRule, blind_zone_km, warning_time
from pwave.magnitude import MagnitudeModel, velocity_to_displacement, window_params

FS = 100.0


def sine_velocity(f, amp_v=1e-6, dur=20.0):
    t = np.arange(0, dur, 1 / FS)
    return amp_v * np.cos(2 * np.pi * f * t)


def test_displacement_of_sine_velocity():
    f, a = 2.0, 1e-6
    d = velocity_to_displacement(sine_velocity(f, a), FS)[int(10 * FS):]
    expected = a / (2 * math.pi * f)
    assert np.max(np.abs(d)) == pytest.approx(expected, rel=0.08)


def test_pd_matches_known_amplitude():
    f, a = 2.0, 1e-6
    pd, _ = window_params(sine_velocity(f, a), FS, 10.0, 3.0)
    assert pd == pytest.approx(a / (2 * math.pi * f), rel=0.1)


@pytest.mark.parametrize("f", [0.5, 1.0, 2.0, 5.0])
def test_tau_c_of_sine_is_period(f):
    _, tau = window_params(sine_velocity(f), FS, 8.0, 4.0)
    assert tau == pytest.approx(1.0 / f, rel=0.15)


def test_window_outside_record_raises():
    with pytest.raises(ValueError):
        window_params(np.ones(100), FS, 0.5, 3.0)


def test_negative_pick_raises():
    with pytest.raises(ValueError):
        window_params(np.ones(1000), FS, -1.0, 3.0)


def test_nonpositive_window_raises():
    with pytest.raises(ValueError):
        window_params(np.ones(1000), FS, 1.0, 0.0)


def test_zero_signal_gives_nan_tau():
    pd, tau = window_params(np.zeros(2000), FS, 1.0, 3.0)
    assert pd == 0.0 and math.isnan(tau)


def make_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    mag = rng.uniform(2, 4.5, n)
    r = rng.uniform(10, 120, n)
    pd = 10 ** ((mag - 5.0 - 1.2 * np.log10(r)) / 1.1 - 2)
    tau = 10 ** ((mag - 5.0) / 2.0)
    return pd, tau, r, mag


def test_model_recovers_exact_relation():
    pd, tau, r, mag = make_data()
    m = MagnitudeModel().fit(pd, tau, r, mag)
    assert m.predict(pd=pd[0], r_km=r[0]) == pytest.approx(mag[0], abs=1e-6)
    assert m.predict(tau=tau[1]) == pytest.approx(mag[1], abs=1e-6)


def test_model_pd_preferred_when_distance_known():
    pd, tau, r, mag = make_data()
    m = MagnitudeModel().fit(pd, tau, r, mag)
    assert m.predict(pd=pd[2], tau=tau[2] * 5, r_km=r[2]) == pytest.approx(mag[2], abs=1e-6)


def test_model_needs_inputs():
    pd, tau, r, mag = make_data()
    m = MagnitudeModel().fit(pd, tau, r, mag)
    with pytest.raises(ValueError):
        m.predict(pd=1e-6)


def test_model_unfitted_raises():
    with pytest.raises(RuntimeError):
        MagnitudeModel().predict(pd=1e-6, r_km=10)
    with pytest.raises(RuntimeError):
        MagnitudeModel().predict(tau=1.0)


def test_model_fit_too_few_samples():
    with pytest.raises(ValueError):
        MagnitudeModel().fit([1e-6] * 3, [1.0] * 3, [10.0] * 3, [3.0] * 3)


def test_model_fit_mismatched_lengths():
    with pytest.raises(ValueError):
        MagnitudeModel().fit([1e-6] * 5, [1.0] * 4, [10.0] * 5, [3.0] * 5)


def test_model_fit_rejects_nonpositive_pd():
    pd, tau, r, mag = make_data()
    pd[0] = 0.0
    with pytest.raises(ValueError):
        MagnitudeModel().fit(pd, tau, r, mag)


def test_warning_time_known_value():
    assert warning_time(50.0, 3.0, 1.0, 6.0, 3.5) == pytest.approx(50 / 3.5 - 50 / 6 - 4, rel=1e-9)


def test_warning_time_negative_inside_blind_zone():
    assert warning_time(10.0) < 0


def test_blind_zone_is_where_warning_is_zero():
    assert warning_time(blind_zone_km()) == pytest.approx(0.0, abs=1e-9)


def test_warning_time_grows_with_distance():
    assert warning_time(100) > warning_time(50) > warning_time(25)


def test_warning_time_rejects_bad_speeds():
    with pytest.raises(ValueError):
        warning_time(10, vp=3.0, vs=3.5)
    with pytest.raises(ValueError):
        warning_time(-1)


def test_blind_zone_rejects_bad_speeds():
    with pytest.raises(ValueError):
        blind_zone_km(vp=3.0, vs=3.5)


def test_alert_rule_threshold_edges():
    rule = AlertRule(min_mag=3.0, min_pd_m=1e-6)
    assert rule.decide(3.0, 1e-6)
    assert not rule.decide(2.99, 1e-6)
    assert not rule.decide(3.5, 9e-7)


def test_alert_rule_nan_is_no_alert():
    assert not AlertRule().decide(float("nan"), 1.0)
