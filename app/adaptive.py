"""Bandwidth-adaptive control: EWMA throughput estimate -> ABR profile pick.

Mirrors the source platform's ladder (ultra/high/medium/low). In the real
platform throughput arrives from WebRTC receiver stats; here measurements are
reported explicitly, so tests and experiments can simulate any network."""

from app import config
from app.codecs.lossy import jpeg_encode
from app.media_source import SyntheticXRSource
from app.viewport import foveated_jpeg, gaze_roi


class BandwidthEstimator:
    """Exponentially-weighted moving average of observed throughput."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._kbps: float | None = None

    def update(self, bytes_sent: int, seconds: float) -> None:
        if seconds <= 0:
            return
        kbps = bytes_sent * 8 / 1000 / seconds
        self._kbps = kbps if self._kbps is None else self.alpha * kbps + (1 - self.alpha) * self._kbps

    @property
    def kbps(self) -> float:
        return self._kbps or 0.0


def pick_profile(kbps: float) -> str:
    return next(name for floor, name in config.BANDWIDTH_LADDER if kbps >= floor)


class AdaptiveStreamSession:
    """One viewer's session: throughput measurements in, profile knobs out.

    Constrained profiles additionally switch on viewport-aware (foveated)
    encoding, combining the bandwidth and viewport dimensions."""

    def __init__(self):
        self.estimator = BandwidthEstimator()
        self.profile = "high"

    def report_throughput(self, bytes_sent: int, seconds: float) -> None:
        self.estimator.update(bytes_sent, seconds)
        self.profile = pick_profile(self.estimator.kbps)

    @property
    def knobs(self) -> dict:
        return config.PROFILES[self.profile]

    def encode_frame(
        self, source: SyntheticXRSource, t: float, yaw_deg: float = 0.0, pitch_deg: float = 0.0
    ) -> bytes:
        img = source.frame(t)
        k = self.knobs
        if k["foveated"]:
            roi = gaze_roi(img.width, img.height, yaw_deg, pitch_deg)
            return foveated_jpeg(img, roi, quality=k["jpeg_quality"])
        return jpeg_encode(img, quality=k["jpeg_quality"])
