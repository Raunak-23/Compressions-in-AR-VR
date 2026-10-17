"""Live bridge state: one phone publisher, N independently-adapted viewers.

Holds the latest published video/audio frames, the global gaze, and per-viewer
adaptation state (WebRTC receiver stats -> EWMA -> profile ladder, plus demo
overrides). Kept free of aiortc imports so the control logic is unit-testable
headlessly; the media plumbing lives in app.rtc.tracks / app.rtc.signaling.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app import config
from app.adaptive import BandwidthEstimator, pick_profile

logger = logging.getLogger(__name__)


@dataclass
class ViewerState:
    """One viewer's adaptation state: throughput measurements in, knobs out."""

    id: str
    estimator: BandwidthEstimator = field(default_factory=BandwidthEstimator)
    force_profile: Optional[str] = None  # demo override: pick a ladder rung
    cap_kbps: Optional[float] = None  # demo override: simulated link cap
    profile: str = "high"
    has_measurement: bool = False
    shadow: dict[str, Any] = field(default_factory=dict)
    frames_sent: int = 0
    audio_frames: int = 0

    @property
    def knobs(self) -> dict:
        return config.PROFILES[self.profile]

    @property
    def profile_source(self) -> str:
        if self.force_profile:
            return "forced"
        if self.cap_kbps is not None:
            return "capped"
        if self.has_measurement:
            return "measured"
        return "default"

    def report_throughput(self, nbytes: int, seconds: float) -> None:
        """Feed WebRTC receiver stats (bytes received over dt) into the EWMA."""
        if seconds <= 0:
            return
        self.estimator.update(nbytes, seconds)
        self.has_measurement = True
        self.repick_profile()

    def repick_profile(self) -> None:
        if self.force_profile:
            self.profile = self.force_profile
        elif self.cap_kbps is not None:
            self.profile = pick_profile(self.cap_kbps)
        elif self.has_measurement:
            self.profile = pick_profile(self.estimator.kbps)
        # else: keep the startup rung until the first measurement arrives


class VideoSlot:
    """Latest published video frame, guarded by a condition for fan-out.

    Slow consumers simply skip ahead to the newest frame, which is exactly
    the semantics a live stream wants.
    """

    def __init__(self) -> None:
        self.cond = asyncio.Condition()
        self.image = None  # PIL.Image, already normalized to the canvas
        self.seq = 0

    async def publish(self, image) -> int:
        async with self.cond:
            self.image = image
            self.seq += 1
            self.cond.notify_all()
            return self.seq

    async def wait_for(self, last_seen: int) -> tuple[Any, int]:
        async with self.cond:
            await self.cond.wait_for(lambda: self.image is not None and self.seq > last_seen)
            return self.image, self.seq


class AudioSlot:
    """Latest published audio frame, guarded by a condition for fan-out."""

    def __init__(self) -> None:
        self.cond = asyncio.Condition()
        self.frame = None  # av.AudioFrame
        self.seq = 0

    async def publish(self, frame) -> int:
        async with self.cond:
            self.frame = frame
            self.seq += 1
            self.cond.notify_all()
            return self.seq

    async def wait_for(self, last_seen: int) -> tuple[Any, int]:
        async with self.cond:
            await self.cond.wait_for(lambda: self.frame is not None and self.seq > last_seen)
            return self.frame, self.seq


class Bridge:
    """Wires the phone publisher to every connected viewer session."""

    def __init__(self) -> None:
        self.video = VideoSlot()
        self.audio = AudioSlot()
        self.gaze_norm = (0.5, 0.5)  # (nx, ny) in [0, 1] of the frustum
        self.audio_mode = "mic"  # "mic" (relay) | "foa" (synthesized B-format)
        self.phone_connected = False
        self.phone_pc = None  # aiortc RTCPeerConnection, set by signaling
        self.phone_stats: dict[str, Any] = {
            "inbound_kbps": None,
            "packets_lost": None,
            "frames_decoded": None,
        }
        self.viewers: dict[str, ViewerState] = {}
        self._ids = itertools.count(1)

    def new_viewer(self) -> ViewerState:
        state = ViewerState(id=f"v{next(self._ids)}")
        self.viewers[state.id] = state
        return state

    def drop_viewer(self, state: ViewerState) -> None:
        self.viewers.pop(state.id, None)

    def set_gaze(self, nx: float, ny: float) -> None:
        self.gaze_norm = (min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0))
