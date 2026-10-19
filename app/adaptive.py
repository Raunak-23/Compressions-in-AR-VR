"""Bandwidth-adaptive control: EWMA throughput estimate, fluctuation tracking,
hysteresis-based profile selection, and verified bitrate budget enforcement.

Supports:
- Classical EWMA throughput estimation with variance and volatility tracking.
- IntelligentBudgetController with fast congestion downswitching and dwell-time upswitching.
- Latency-budget and dropped-frame awareness.
- Strict budget auditing: never claims target was met if measurements contradict it.
- Backward compatibility with course baseline (pick_profile, AdaptiveStreamSession).
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple

from app import config
from app.codecs.lossy import jpeg_encode
from app.media_source import SyntheticXRSource
from app.viewport import foveated_jpeg, gaze_roi
from app.viewport_tiles import TiledFrameResult, encode_and_reconstruct_tiles


class BandwidthEstimator:
    """Exponentially-weighted moving average of observed throughput with variance tracking."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._kbps: float | None = None
        self._variance: float = 0.0
        self._history: Deque[Tuple[float, float]] = collections.deque(maxlen=30)  # (t, kbps)

    def update(self, bytes_sent: int, seconds: float) -> None:
        if seconds <= 0:
            return
        inst_kbps = bytes_sent * 8 / 1000 / seconds
        now = time.perf_counter()
        self._history.append((now, inst_kbps))

        if self._kbps is None:
            self._kbps = inst_kbps
            self._variance = 0.0
        else:
            diff = inst_kbps - self._kbps
            # EWMA update of variance
            self._variance = self.alpha * (diff ** 2) + (1 - self.alpha) * self._variance
            self._kbps = self.alpha * inst_kbps + (1 - self.alpha) * self._kbps

    @property
    def kbps(self) -> float:
        return self._kbps or 0.0

    @property
    def std_kbps(self) -> float:
        return math.sqrt(max(0.0, self._variance))

    @property
    def volatility_ratio(self) -> float:
        """Normalized dispersion metric (coefficient of variation)."""
        if self._kbps is None or self._kbps <= 0:
            return 0.0
        return round(self.std_kbps / self._kbps, 4)

    def safe_kbps(self, volatility_weight: float = 0.5) -> float:
        """Conservative bandwidth target discounting for recent fluctuations."""
        if self._kbps is None:
            return 0.0
        # Reduce effective headroom when link is volatile
        safe = self._kbps - (volatility_weight * self.std_kbps)
        return max(50.0, round(safe, 1))

    @property
    def trend(self) -> float:
        """Linear slope over recent samples: positive for rising, negative for falling."""
        if len(self._history) < 3:
            return 0.0
        times = [t for t, _ in self._history]
        vals = [v for _, v in self._history]
        t_mean = sum(times) / len(times)
        v_mean = sum(vals) / len(vals)
        denom = sum((t - t_mean) ** 2 for t in times)
        if denom == 0:
            return 0.0
        numer = sum((t - t_mean) * (v - v_mean) for t, v in zip(times, vals))
        return round(numer / denom, 2)


def pick_profile(kbps: float) -> str:
    """Standard profile ladder rung selection (first floor met)."""
    return next(name for floor, name in config.BANDWIDTH_LADDER if kbps >= floor)


@dataclass
class ControllerDecision:
    profile: str
    target_kbps: int
    frame_budget_bytes: int
    estimated_kbps: float
    safe_kbps: float
    volatility_ratio: float
    action: Literal["initial", "maintain", "upswitch", "downswitch", "throttle_for_latency"]
    reason: str


@dataclass
class BudgetAuditReport:
    profile: str
    target_kbps: int
    fps: int
    budget_bytes: int
    actual_bytes: int
    overshoot_bytes: int
    overshoot_pct: float
    within_budget: bool
    budget_retries: int
    nominal_codec_bitrate_kbps: int
    measured_frame_bitrate_kbps: float
    processing_delay_s: float
    latency_budget_s: float
    delay_exceeded: bool
    adaptation_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IntelligentBudgetController:
    """Bitrate-budget controller incorporating throughput, volatility, delay, and hysteresis.

    Key Features:
    - Rapid downswitching when bandwidth collapses or latency blows out.
    - Dwell-time / confirmation hysteresis before stepping up to avoid profile flapping.
    - Conservative headroom backoff under high volatility.
    - Verified per-frame budget enforcement: records exact overshoot and never claims
      compliance when budget was exceeded.
    """

    def __init__(
        self,
        estimator: Optional[BandwidthEstimator] = None,
        initial_profile: str = "high",
        upswitch_dwell_frames: int = 4,
        headroom_factor: float = 0.15,
        max_delay_ratio: float = 0.90,
    ):
        self.estimator = estimator or BandwidthEstimator()
        self.current_profile = initial_profile
        self.upswitch_dwell_frames = upswitch_dwell_frames
        self.headroom_factor = headroom_factor
        self.max_delay_ratio = max_delay_ratio

        self._dwell_counter: int = 0
        self._last_action: str = "initial"
        self._last_reason: str = "initial_profile"
        self.decisions_log: Deque[Dict[str, Any]] = collections.deque(maxlen=100)
        self.audit_log: Deque[Dict[str, Any]] = collections.deque(maxlen=100)

    @property
    def knobs(self) -> Dict[str, Any]:
        return config.PROFILES[self.current_profile]

    def frame_budget_bytes(self, profile: Optional[str] = None) -> int:
        p = profile or self.current_profile
        k = config.PROFILES[p]
        target_kbps = k.get("target_kbps", 1800)
        fps = max(k.get("fps", 30), 1)
        return int((target_kbps * 1000 / 8) / fps)

    def _get_profile_floor(self, prof_name: str) -> float:
        for floor, name in config.BANDWIDTH_LADDER:
            if name == prof_name:
                return float(floor)
        return 0.0

    def _find_higher_profile(self, current: str) -> Optional[str]:
        ladder = [name for _, name in reversed(config.BANDWIDTH_LADDER)]
        idx = ladder.index(current)
        if idx + 1 < len(ladder):
            return ladder[idx + 1]
        return None

    def _find_lower_profile(self, current: str) -> Optional[str]:
        ladder = [name for _, name in reversed(config.BANDWIDTH_LADDER)]
        idx = ladder.index(current)
        if idx > 0:
            return ladder[idx - 1]
        return None

    def update(
        self,
        bytes_sent: int,
        seconds: float,
        processing_delay_s: float = 0.0,
        dropped_frames: int = 0,
    ) -> ControllerDecision:
        """Update network observations and perform hysteresis-bounded adaptation."""
        self.estimator.update(bytes_sent, seconds)
        raw_kbps = self.estimator.kbps
        safe_kbps = self.estimator.safe_kbps(volatility_weight=0.6)
        volatility = self.estimator.volatility_ratio

        fps = max(self.knobs.get("fps", 30), 1)
        latency_budget = 1.0 / fps

        action: Literal["initial", "maintain", "upswitch", "downswitch", "throttle_for_latency"] = "maintain"
        reason = "stable_headroom"

        # 1. Urgent downswitch check: processing delay exceeded or dropped frames
        if dropped_frames > 0 or (processing_delay_s > latency_budget * self.max_delay_ratio):
            lower = self._find_lower_profile(self.current_profile)
            if lower is not None:
                self.current_profile = lower
                self._dwell_counter = 0
                action = "throttle_for_latency"
                reason = f"processing_delay_{processing_delay_s*1000:.1f}ms_exceeded_budget"
            else:
                action = "throttle_for_latency"
                reason = "lowest_profile_reached_under_latency_pressure"

        # 2. Bandwidth-based adaptation
        else:
            current_floor = self._get_profile_floor(self.current_profile)

            # Check if current safe bandwidth fell below the floor of the active profile
            if safe_kbps < current_floor:
                # Urgent downswitch to appropriate rung
                candidate = pick_profile(safe_kbps)
                if candidate != self.current_profile:
                    self.current_profile = candidate
                    self._dwell_counter = 0
                    action = "downswitch"
                    reason = f"safe_bw_{safe_kbps:.1f}k_dropped_below_floor_{current_floor:.1f}k"
                else:
                    reason = "floor_deficit_at_lowest_rung"
            else:
                # Check candidate higher profile
                higher = self._find_higher_profile(self.current_profile)
                if higher is not None:
                    higher_floor = self._get_profile_floor(higher)
                    required_kbps = higher_floor * (1.0 + self.headroom_factor)
                    if safe_kbps >= required_kbps and volatility < 0.40:
                        self._dwell_counter += 1
                        if self._dwell_counter >= self.upswitch_dwell_frames:
                            self.current_profile = higher
                            self._dwell_counter = 0
                            action = "upswitch"
                            reason = f"sustained_bw_{safe_kbps:.1f}k_cleared_{higher}_target_{required_kbps:.1f}k"
                        else:
                            reason = f"upswitch_dwelling_step_{self._dwell_counter}_of_{self.upswitch_dwell_frames}"
                    else:
                        self._dwell_counter = max(0, self._dwell_counter - 1)
                        reason = "headroom_insufficient_or_volatile"
                else:
                    self._dwell_counter = 0
                    reason = "maximum_rung_reached"

        self._last_action = action
        self._last_reason = reason

        decision = ControllerDecision(
            profile=self.current_profile,
            target_kbps=self.knobs.get("target_kbps", 1800),
            frame_budget_bytes=self.frame_budget_bytes(),
            estimated_kbps=round(raw_kbps, 1),
            safe_kbps=round(safe_kbps, 1),
            volatility_ratio=volatility,
            action=action,
            reason=reason,
        )
        self.decisions_log.append(asdict(decision))
        return decision

    def audit_frame(
        self,
        actual_bytes: int,
        encode_time_s: float = 0.0,
        decode_time_s: float = 0.0,
        retries: int = 0,
    ) -> BudgetAuditReport:
        """Audit frame output against budget and delay limits with zero falsification."""
        frame_budget = self.frame_budget_bytes()
        overshoot = max(0, actual_bytes - frame_budget)
        overshoot_pct = round(100.0 * overshoot / frame_budget, 1) if frame_budget > 0 else 0.0
        fps = max(self.knobs.get("fps", 30), 1)
        target_kbps = self.knobs.get("target_kbps", 1800)
        measured_kbps = round(actual_bytes * 8 * fps / 1000, 1)

        total_proc = encode_time_s + decode_time_s
        latency_budget = 1.0 / fps
        delay_exceeded = total_proc > latency_budget

        report = BudgetAuditReport(
            profile=self.current_profile,
            target_kbps=target_kbps,
            fps=fps,
            budget_bytes=frame_budget,
            actual_bytes=actual_bytes,
            overshoot_bytes=overshoot,
            overshoot_pct=overshoot_pct,
            within_budget=(actual_bytes <= frame_budget),
            budget_retries=retries,
            nominal_codec_bitrate_kbps=target_kbps,
            measured_frame_bitrate_kbps=measured_kbps,
            processing_delay_s=round(total_proc, 4),
            latency_budget_s=round(latency_budget, 4),
            delay_exceeded=delay_exceeded,
            adaptation_reason=self._last_reason,
        )
        self.audit_log.append(report.to_dict())
        return report


class AdaptiveStreamSession:
    """One viewer's session: throughput measurements in, profile knobs out.

    Constrained profiles additionally switch on viewport-aware (foveated or tiled)
    encoding, combining the bandwidth and viewport dimensions.
    Integrates the IntelligentBudgetController with verified budget enforcement.
    """

    def __init__(self, initial_profile: str = "high"):
        self.controller = IntelligentBudgetController(initial_profile=initial_profile)
        self.estimator = self.controller.estimator

    @property
    def profile(self) -> str:
        return self.controller.current_profile

    @profile.setter
    def profile(self, val: str) -> None:
        self.controller.current_profile = val

    def report_throughput(
        self,
        bytes_sent: int,
        seconds: float,
        processing_delay_s: float = 0.0,
        dropped_frames: int = 0,
    ) -> None:
        """Update observed throughput through the controller."""
        self.controller.update(
            bytes_sent=bytes_sent,
            seconds=seconds,
            processing_delay_s=processing_delay_s,
            dropped_frames=dropped_frames,
        )
        self.profile = pick_profile(self.estimator.kbps)

    @property
    def knobs(self) -> dict:
        return self.controller.knobs

    def frame_budget_bytes(self) -> int:
        return self.controller.frame_budget_bytes()

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
        enforce_budget: bool = True,
        max_retries: int = 3,
        allocation_strategy: Literal["fixed_tiers", "intelligent_allocator"] = "fixed_tiers",
    ) -> Tuple[TiledFrameResult, Dict[str, Any]]:
        """Encode frame using 4x4 spatial tiling with verified budget enforcement.

        When enforce_budget is True, the encoder reduces quality and re-encodes
        if the total size exceeds the per-frame budget, up to max_retries attempts.
        """
        img = source.frame(t)
        k = self.knobs
        ladder = k.get("tile_ladder")
        budget = self.frame_budget_bytes() if enforce_budget else None

        res = encode_and_reconstruct_tiles(
            img,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            quality_ladder=ladder,
            codec=codec,
            base_bitrate_kbps=k.get("target_kbps", 1000),
            budget_bytes=budget,
            max_retries=max_retries,
            allocation_strategy=allocation_strategy,
        )

        audit = self.controller.audit_frame(
            actual_bytes=res.total_bytes,
            encode_time_s=res.encode_time_s,
            decode_time_s=res.decode_time_s,
            retries=res.budget_retries,
        )

        return res, audit.to_dict()
