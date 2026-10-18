"""Bandwidth-adaptive control: EWMA throughput estimate -> ABR profile pick.

Mirrors the source platform's ladder (ultra/high/medium/low). In the real
platform throughput arrives from WebRTC receiver stats; here measurements are
reported explicitly, so tests and experiments can simulate any network."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

from app import config
from app.codecs.lossy import jpeg_encode
from app.media_source import SyntheticXRSource
from app.viewport import foveated_jpeg, gaze_roi
from app.viewport_tiles import TiledFrameResult, encode_and_reconstruct_tiles


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

    Constrained profiles additionally switch on viewport-aware (foveated or tiled)
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

    def frame_budget_bytes(self) -> int:
        """Compute the byte budget allocated for one frame under current profile."""
        k = self.knobs
        target_kbps = k.get("target_kbps", 1800)
        fps = max(k.get("fps", 30), 1)
        return int((target_kbps * 1000 / 8) / fps)

    def encode_frame(
        self,
        source: SyntheticXRSource,
        t: float,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        mode: Literal["blur", "tiled"] = "blur",
    ) -> bytes:
        img = source.frame(t)
        k = self.knobs
        if k["foveated"]:
            if mode == "tiled":
                res = encode_and_reconstruct_tiles(
                    img,
                    yaw_deg=yaw_deg,
                    pitch_deg=pitch_deg,
                    quality_ladder=k.get("tile_ladder"),
                    codec="jpeg",
                )
                return jpeg_encode(res.reconstructed, quality=k["jpeg_quality"])
            roi = gaze_roi(img.width, img.height, yaw_deg, pitch_deg)
            return foveated_jpeg(img, roi, quality=k["jpeg_quality"])
        return jpeg_encode(img, quality=k["jpeg_quality"])

    def encode_frame_tiled(
        self,
        source: SyntheticXRSource,
        t: float,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        codec: Literal["jpeg", "h264"] = "jpeg",
    ) -> Tuple[TiledFrameResult, Dict[str, Any]]:
        """Encode frame using genuine 4x4 spatial tiling and evaluate budget compliance."""
        img = source.frame(t)
        k = self.knobs
        ladder = k.get("tile_ladder")
        res = encode_and_reconstruct_tiles(
            img,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            quality_ladder=ladder,
            codec=codec,
            base_bitrate_kbps=k.get("target_kbps", 1000),
        )

        budget = self.frame_budget_bytes()
        actual = res.total_bytes
        overshoot = max(0, actual - budget)
        overshoot_pct = round(100.0 * (actual - budget) / budget, 1) if budget > 0 else 0.0

        budget_report = {
            "profile": self.profile,
            "target_kbps": k.get("target_kbps"),
            "fps": k.get("fps"),
            "budget_bytes": budget,
            "actual_bytes": actual,
            "overshoot_bytes": overshoot,
            "overshoot_pct": overshoot_pct,
            "within_budget": actual <= budget,
        }

        return res, budget_report
