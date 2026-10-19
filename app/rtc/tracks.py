"""aiortc media tracks for the live bridge.

ProcessedVideoTrack: pulls the newest phone frame and runs the lab's adaptive
stage over it -- profile encode resolution, viewport foveation from the gaze,
and shadow metrics (JPEG size / PSNR via the course codecs) sampled every Nth
frame. Output is always the fixed canvas size, so the VP8 encoder never needs
renegotiation; downshifted profiles ship fewer real pixels upscaled.

RelayAudioTrack: passes the phone's decoded mic audio through.
FoaAudioTrack: synthesizes the 4-channel FOA B-format field and folds it to
the viewer's current profile channel count -- the spatial-audio study, live.
"""

from __future__ import annotations

import asyncio
import fractions
import logging

import numpy as np
from av import AudioFrame, AudioResampler, VideoFrame
from PIL import Image
from PIL.Image import Resampling

from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from app import config
from app.codecs import lossy, spatial_audio, video_codec
from app.media_source import SyntheticXRSource
from app.rtc.bridge import Bridge, ViewerState
from app import viewport_tiles
from app.viewport import foveate_image, gaze_roi

logger = logging.getLogger(__name__)

VIDEO_TIME_BASE = fractions.Fraction(1, config.VIDEO_CLOCK_RATE)
AUDIO_TIME_BASE = fractions.Fraction(1, config.AUDIO_CLOCK_RATE)
CANVAS = (config.VIDEO_CANVAS_WIDTH, config.VIDEO_CANVAS_HEIGHT)


def process_frame(
    img: Image.Image,
    knobs: dict,
    gaze_norm: tuple[float, float],
    do_sample: bool,
    experimental_codec: str | None = None,
    allocation_strategy: str = "intelligent_allocator",
) -> tuple[Image.Image, dict]:
    """Adaptive stage for one frame: profile resolution + foveation + shadow
    metrics. Returns the canvas-sized image to transmit and a metrics dict."""
    eff_w = max(8, int(CANVAS[0] * knobs["scale"]))
    eff_h = max(8, int(CANVAS[1] * knobs["scale"]))
    work = img.resize((eff_w, eff_h), Resampling.BILINEAR)
    roi = None
    yaw = (gaze_norm[0] - 0.5) * config.HFOV_DEG
    pitch = (0.5 - gaze_norm[1]) * config.VFOV_DEG
    if knobs["foveated"]:
        roi = gaze_roi(eff_w, eff_h, yaw_deg=yaw, pitch_deg=pitch)
        work = foveate_image(work, roi, blur_radius=knobs["blur"])

    h264_shadow_bytes = None
    if experimental_codec == "h264" and do_sample:
        try:
            target_kbps = knobs.get("target_kbps", 1000)
            h264_blob = video_codec.encode_frame(work, bitrate_kbps=target_kbps, fps=knobs.get("fps", 30))
            h264_shadow_bytes = len(h264_blob)
            work = video_codec.decode_frame(h264_blob, target_size=(eff_w, eff_h))
        except Exception as e:
            logger.debug("experimental H.264 stage error: %s", e)

    out = work.resize(CANVAS, Resampling.BILINEAR)

    shadow: dict = {
        "jpeg_bytes": None,
        "psnr_db": None,
        "raw_bytes": eff_w * eff_h * 3,
        "effective": [eff_w, eff_h],
        "roi": list(roi) if roi else None,
        "tile_bytes": None,
        "viewport_psnr_db": None,
        "guard_band_psnr_db": None,
        "peripheral_psnr_db": None,
        "budget_bytes": None,
        "overshoot_bytes": None,
        "budget_retries": 0,
        "within_budget": True,
        "boundary_artifact_mse": None,
        "h264_bytes": h264_shadow_bytes,
        "allocation_strategy": allocation_strategy,
        "note": "Shadow metrics are experimental calculations, not actual VP8 transport payload",
    }
    if do_sample:
        blob = lossy.jpeg_encode(work, quality=knobs["jpeg_quality"])
        ref = img.resize((eff_w, eff_h), Resampling.BILINEAR)
        shadow["jpeg_bytes"] = len(blob)
        psnr = lossy.psnr(ref, work)
        shadow["psnr_db"] = round(psnr, 2) if psnr != float("inf") else None

        if knobs.get("enable_tile_shadow"):
            target_kbps = knobs.get("target_kbps", 1800)
            fps = max(knobs.get("fps", 30), 1)
            budget_bytes = int((target_kbps * 1000 / 8) / fps)
            shadow["budget_bytes"] = budget_bytes
            try:
                tile_res = viewport_tiles.encode_and_reconstruct_tiles(
                    work,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    quality_ladder=knobs.get("tile_ladder"),
                    codec="jpeg",
                    budget_bytes=budget_bytes,
                    allocation_strategy=allocation_strategy,
                )
                shadow["tile_bytes"] = tile_res.total_bytes
                shadow["viewport_psnr_db"] = tile_res.viewport_psnr_db
                shadow["guard_band_psnr_db"] = tile_res.guard_band_psnr_db
                shadow["peripheral_psnr_db"] = tile_res.peripheral_psnr_db
                shadow["overshoot_bytes"] = tile_res.overshoot_bytes
                shadow["budget_retries"] = tile_res.budget_retries
                shadow["within_budget"] = tile_res.within_budget
                shadow["boundary_artifact_mse"] = tile_res.boundary_artifact_mse
            except Exception as e:
                logger.debug("tile shadow metric error: %s", e)

    return out, shadow


class ProcessedVideoTrack(MediaStreamTrack):
    """One viewer's video: newest phone frame through the adaptive stage."""

    kind = "video"

    def __init__(self, bridge: Bridge, state: ViewerState):
        super().__init__()
        self._bridge = bridge
        self._state = state
        self._last_seen = 0
        self._frame_count = 0
        self._start: float | None = None
        self._next_send: float | None = None
        self._pts = 0

    async def recv(self) -> VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError
        img, seq = await self._bridge.video.wait_for(self._last_seen)
        self._last_seen = seq

        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._start is None:
            self._start = now
            self._next_send = now
        if self._next_send > now:
            await asyncio.sleep(self._next_send - now)

        self._frame_count += 1
        knobs = self._state.knobs
        do_sample = self._frame_count % config.SHADOW_SAMPLE_EVERY_N == 0
        out, shadow = await asyncio.to_thread(
            process_frame,
            img,
            knobs,
            self._bridge.gaze_norm,
            do_sample,
            None,
            getattr(self._state, "allocation_strategy", "intelligent_allocator"),
        )
        self._state.shadow = shadow
        self._state.frames_sent += 1
        self._next_send = max(loop.time(), self._next_send) + 1.0 / max(knobs["fps"], 1)

        frame = VideoFrame.from_ndarray(np.asarray(out, dtype=np.uint8), format="rgb24")
        pts = int((loop.time() - self._start) * config.VIDEO_CLOCK_RATE)
        if pts <= self._pts:  # keep RTP timestamps strictly increasing
            pts = self._pts + 1
        self._pts = pts
        frame.pts = pts
        frame.time_base = VIDEO_TIME_BASE
        return frame


class RelayAudioTrack(MediaStreamTrack):
    """Passes the phone's decoded audio through, normalized to 48k stereo s16.
    If no phone is sending audio (e.g. synthetic feed or muted), outputs silence
    at 50 pps (20 ms) to keep WebRTC A/V synchronization alive."""

    kind = "audio"

    def __init__(self, bridge: Bridge):
        super().__init__()
        self._bridge = bridge
        self._last_seen = 0
        self._resampler = AudioResampler(
            format="s16", layout="stereo", rate=config.AUDIO_CLOCK_RATE
        )
        self._samples = int(config.AUDIO_CLOCK_RATE * config.AUDIO_PTIME_MS / 1000)
        self._pts = 0
        self._next_send: float | None = None

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._next_send is None:
            self._next_send = now
        if self._next_send > now:
            await asyncio.sleep(self._next_send - now)

        try:
            frame, seq = await asyncio.wait_for(
                self._bridge.audio.wait_for(self._last_seen), timeout=0.03
            )
            self._last_seen = seq
            outs = self._resampler.resample(frame)
            out_frame = outs[0] if outs else frame
        except (asyncio.TimeoutError, Exception):
            silence = np.zeros((1, self._samples * 2), dtype=np.int16)
            out_frame = AudioFrame.from_ndarray(silence, format="s16", layout="stereo")
            out_frame.sample_rate = config.AUDIO_CLOCK_RATE

        out_frame.pts = self._pts
        out_frame.time_base = AUDIO_TIME_BASE
        self._pts += self._samples
        self._next_send = max(loop.time(), self._next_send) + self._samples / config.AUDIO_CLOCK_RATE
        return out_frame


class FoaAudioTrack(MediaStreamTrack):
    """Synthetic FOA field folded to the viewer's profile channel count.

    The 4 s loop wraps at integer multiples of both the 440 Hz tone and the
    0.25 Hz circling azimuth, so the seam is click-free."""

    kind = "audio"

    def __init__(self, bridge: Bridge, state: ViewerState):
        super().__init__()
        self._bridge = bridge
        self._state = state
        self._samples = int(config.AUDIO_CLOCK_RATE * config.AUDIO_PTIME_MS / 1000)  # 960
        self._buffer = SyntheticXRSource().foa_audio(4.0, config.AUDIO_CLOCK_RATE)
        self._pos = 0
        self._pts = 0
        self._next_send: float | None = None

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._next_send is None:
            self._next_send = now
        if self._next_send > now:
            await asyncio.sleep(self._next_send - now)

        n = self._samples
        if self._pos + n > self._buffer.shape[1]:
            self._pos = 0
        chunk = self._buffer[:, self._pos : self._pos + n]
        self._pos += n

        channels = max(1, min(self._state.knobs["audio_channels"], 2))
        folded = spatial_audio.fold_foa(chunk, channels)
        pcm = spatial_audio.quantize_i16(folded)
        if pcm.shape[0] == 1:  # transport is stereo; mono content duplicated
            pcm = np.repeat(pcm, 2, axis=0)
        elif pcm.shape[0] > 2:
            pcm = pcm[:2]
        interleaved = np.ascontiguousarray(pcm.T).reshape(1, -1)

        frame = AudioFrame.from_ndarray(interleaved, format="s16", layout="stereo")
        frame.sample_rate = config.AUDIO_CLOCK_RATE
        frame.pts = self._pts
        frame.time_base = AUDIO_TIME_BASE
        self._pts += n
        self._state.audio_frames += 1
        self._next_send = max(loop.time(), self._next_send) + n / config.AUDIO_CLOCK_RATE
        return frame
