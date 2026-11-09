"""Decision rule and warning-time arithmetic."""

from __future__ import annotations

from dataclasses import dataclass

VP_KM_S = 6.0  # crustal P speed used for the example table
VS_KM_S = 3.5


@dataclass(frozen=True)
class AlertRule:
    """Raise an alert when a P trigger is followed by a large enough quick magnitude.

    ``min_mag``: alert if the estimate after ``win_s`` is at least this.
    ``min_pd_m``: amplitude gate on peak displacement, rejects weak/noisy triggers.
    """

    min_mag: float = 3.0
    min_pd_m: float = 2e-7
    win_s: float = 3.0
    latency_s: float = 1.0

    def decide(self, mag_est: float, pd_m: float) -> bool:
        if mag_est != mag_est:  # NaN
            return False
        return mag_est >= self.min_mag and pd_m >= self.min_pd_m


def warning_time(dist_km: float, win_s: float = 3.0, latency_s: float = 1.0,
                 vp: float = VP_KM_S, vs: float = VS_KM_S) -> float:
    """Seconds between the alert and S-wave arrival at ``dist_km`` (negative = too late)."""
    if dist_km < 0 or vp <= vs or vs <= 0:
        raise ValueError("need dist >= 0 and vp > vs > 0")
    return dist_km / vs - (dist_km / vp + win_s + latency_s)


def blind_zone_km(win_s: float = 3.0, latency_s: float = 1.0,
                  vp: float = VP_KM_S, vs: float = VS_KM_S) -> float:
    """Distance inside which the alert arrives after the S wave."""
    if vp <= vs or vs <= 0:
        raise ValueError("need vp > vs > 0")
    return (win_s + latency_s) / (1 / vs - 1 / vp)
