import numpy as np
import pytest

FS = 100.0


def synth(onset_s=30.0, dur_s=60.0, snr=15.0, f0=5.0, seed=0, fs=FS):
    """Gaussian noise (sigma 1) plus a decaying sine burst starting at ``onset_s``."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * fs)
    t = np.arange(n) / fs
    x = rng.normal(0, 1.0, n)
    m = t >= onset_s
    tt = t[m] - onset_s
    x[m] += snr * np.sin(2 * np.pi * f0 * tt) * (1 - np.exp(-tt / 0.2)) * np.exp(-tt / 8.0)
    return x


@pytest.fixture
def burst():
    return synth()
