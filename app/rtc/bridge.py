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
from app.adaptive import BandwidthEstimator, IntelligentBudgetController, pick_profile
from app.prediction import ViewportPredictor

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
    shadow_sample: dict[str, Any] = field(default_factory=dict)  # last JPEG/PSNR sample (sticky between samples)
    frames_sent: int = 0
    audio_frames: int = 0
    frames_dropped: int = 0  # adapter skips: consumer arrived late, newest frame won
    last_report_kbps: float | None = None  # instantaneous wire rate from the last stats report
    controller: IntelligentBudgetController = field(init=False)
    allocation_strategy: str = "intelligent_allocator"
    prediction_enabled: bool = False
    predictor: ViewportPredictor = field(default_factory=ViewportPredictor)

    def __post_init__(self):
        self.controller = IntelligentBudgetController(estimator=self.estimator, initial_profile=self.profile)

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

    def report_throughput(self, nbytes: int, seconds: float, processing_delay_s: float = 0.0) -> None:
        """Feed WebRTC receiver stats (bytes received over dt) into the controller."""
        if seconds <= 0:
            return
        self.last_report_kbps = round(nbytes * 8 / 1000 / seconds, 1)
        if self.cap_kbps is not None:
            effective_bytes = min(nbytes, int(self.cap_kbps * 1000 * seconds / 8))
            self.controller.update(effective_bytes, seconds, processing_delay_s=processing_delay_s)
        else:
            self.controller.update(nbytes, seconds, processing_delay_s=processing_delay_s)
        self.has_measurement = True
        self.repick_profile()

    def repick_profile(self) -> None:
        if self.force_profile:
            self.profile = self.force_profile
            self.controller.current_profile = self.force_profile
        elif self.cap_kbps is not None:
            candidate = pick_profile(self.cap_kbps)
            self.profile = candidate
            self.controller.current_profile = candidate
        elif self.has_measurement:
            self.profile = pick_profile(self.estimator.kbps)
            self.controller.current_profile = self.profile


class VideoSlot:
    """Latest published video frame, guarded by a condition for fan-out.

    Slow consumers simply skip ahead to the newest frame, which is exactly
    the semantics a live stream wants.
    """

    def __init__(self) -> None:
        self._cond: asyncio.Condition | None = None
        self.image = None  # PIL.Image, already normalized to the canvas
        self.seq = 0

    @property
    def cond(self) -> asyncio.Condition:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._cond is None or (loop is not None and getattr(self._cond, "_loop", None) not in (None, loop)):
            self._cond = asyncio.Condition()
        return self._cond

    @cond.setter
    def cond(self, val: asyncio.Condition) -> None:
        self._cond = val

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
        self._cond: asyncio.Condition | None = None
        self.frame = None  # av.AudioFrame
        self.seq = 0

    @property
    def cond(self) -> asyncio.Condition:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._cond is None or (loop is not None and getattr(self._cond, "_loop", None) not in (None, loop)):
            self._cond = asyncio.Condition()
        return self._cond

    @cond.setter
    def cond(self, val: asyncio.Condition) -> None:
        self._cond = val

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
        self.phone_ws = None  # signaling websocket of the current owner
        self.phone_frames = 0  # total decoded phone frames published
        self.phone_capture: list[int] | None = None  # [w, h] of the incoming stream
        self.phone_stats: dict[str, Any] = {
            "inbound_kbps": None,
            "packets_lost": None,
            "frames_decoded": None,
        }
        # what the phone's own getStats sampler reports (video-only wire rate)
        self.phone_reported: dict[str, Any] = {"sent_kbps": None, "sent_fps": None}
        self.viewers: dict[str, ViewerState] = {}
        self._ids = itertools.count(1)
        self._synthetic_active = False  # True when synthetic pump is feeding frames

    def new_viewer(self) -> ViewerState:
        state = ViewerState(id=f"v{next(self._ids)}")
        self.viewers[state.id] = state
        return state

    def drop_viewer(self, state: ViewerState) -> None:
        self.viewers.pop(state.id, None)

    def set_gaze(self, nx: float, ny: float) -> None:
        self.gaze_norm = (min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0))

    async def claim_phone(self, ws, pc) -> None:
        """Take over as the active phone; the previous publisher is kicked so
        two sources can never interleave into the same frame slot."""
        old_ws, old_pc = self.phone_ws, self.phone_pc
        self.phone_ws, self.phone_pc = ws, pc
        self.phone_connected = True
        if old_pc is not None and old_pc is not pc:
            try:
                await old_pc.close()
            except Exception:
                pass
        if old_ws is not None and old_ws is not ws:
            try:
                await old_ws.close()
            except Exception:
                pass

    def release_phone(self, ws) -> None:
        """Clear publisher state only if `ws` still owns it (a newer phone may
        have taken over while this session was cleaning up)."""
        if self.phone_ws is ws:
            self.phone_ws = None
            self.phone_pc = None
            self.phone_connected = False
            self.phone_capture = None
            self.phone_reported = {"sent_kbps": None, "sent_fps": None}
