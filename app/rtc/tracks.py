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
import time

import numpy as np
from av import AudioFrame, AudioResampler, VideoFrame
from PIL import Image, ImageFilter
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
    experimental_codec: str | None = None,
    allocation_strategy: str = "intelligent_allocator",
) -> tuple[Image.Image, Image.Image, dict]:
    """Adaptive stage for one frame (hot path): profile resolution + foveation.
    Returns (canvas frame, effective-res work image, per-frame shadow). JPEG/PSNR
    shadow sampling lives in ProcessedVideoTrack._sample_shadow so its ~40 ms
    never sits on the frame path (it ran inline and spiked latency every Nth
    frame)."""
    eff_w = max(8, int(CANVAS[0] * knobs["scale"]))
    eff_h = max(8, int(CANVAS[1] * knobs["scale"]))
    work = img if img.size == (eff_w, eff_h) else img.resize((eff_w, eff_h), Resampling.BILINEAR)
    roi = None
    yaw = (gaze_norm[0] - 0.5) * config.HFOV_DEG
    pitch = (0.5 - gaze_norm[1]) * config.VFOV_DEG
    if knobs["foveated"]:
        roi = gaze_roi(eff_w, eff_h, yaw_deg=yaw, pitch_deg=pitch)
        # shortcut: periphery blur approximated at quarter size (Gaussian r/4
        # upscaled ~= Gaussian r) -- ~3 ms vs ~28 ms inline. The study plane
        # keeps the exact foveate_image(); upgrade if the blur ever looks soft.
        small = work.resize((max(8, eff_w // 4), max(8, eff_h // 4)), Resampling.BILINEAR)
        blurred = small.filter(ImageFilter.GaussianBlur(max(knobs["blur"] / 4, 0.5)))
        blurred = blurred.resize((eff_w, eff_h), Resampling.BILINEAR)
        blurred.paste(work.crop(roi), roi[:2])
        work = blurred

    h264_shadow_bytes = None
    if experimental_codec == "h264":
        try:
            target_kbps = knobs.get("target_kbps", 1000)
            h264_blob = video_codec.encode_frame(work, bitrate_kbps=target_kbps, fps=knobs.get("fps", 30))
            h264_shadow_bytes = len(h264_blob)
            work = video_codec.decode_frame(h264_blob, target_size=(eff_w, eff_h))
        except Exception as e:
            logger.debug("experimental H.264 stage error: %s", e)

    out = work if work.size == CANVAS else work.resize(CANVAS, Resampling.BILINEAR)

    shadow: dict = {
        "raw_bytes": eff_w * eff_h * 3,
        "effective": [eff_w, eff_h],
        "roi": list(roi) if roi else None,
        "h264_bytes": h264_shadow_bytes,
        "allocation_strategy": allocation_strategy,
        "note": "Shadow metrics are experimental calculations, not actual VP8 transport payload",
    }
    return out, work, shadow


def _stage_and_frame(
    img: Image.Image,
    knobs: dict,
    gaze_norm: tuple[float, float],
    experimental_codec: str | None,
    allocation_strategy: str,
    pts: int,
) -> tuple[Image.Image, Image.Image, dict, VideoFrame]:
    """Stage one frame and build its yuv420p VideoFrame. Measured slower off
    the loop on this machine (thread ping-pong), so recv runs it in-loop."""
    out, work, shadow = process_frame(img, knobs, gaze_norm, experimental_codec, allocation_strategy)
    frame = VideoFrame.from_ndarray(np.asarray(out, dtype=np.uint8), format="rgb24")
    frame = frame.reformat(format="yuv420p")  # saves the encoder its own reformat
    frame.pts = pts
    frame.time_base = VIDEO_TIME_BASE
    return out, work, shadow, frame


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
        self._shadow_task: asyncio.Task | None = None

    async def _sample_shadow(self, img: Image.Image, work: Image.Image, knobs: dict) -> None:
        """JPEG size / PSNR shadow metrics, off the frame path (they cost
        ~40 ms; inline they added a latency spike every Nth frame). Skipped,
        never queued, if the previous sample is still running."""
        def run() -> dict:
            eff = work.size
            ref = img if img.size == eff else img.resize(eff, Resampling.BILINEAR)
            psnr = lossy.psnr(ref, work)
            sample: dict = {
                "jpeg_bytes": len(lossy.jpeg_encode(work, quality=knobs["jpeg_quality"])),
                "psnr_db": round(psnr, 2) if psnr != float("inf") else None,
            }
            if knobs.get("enable_tile_shadow", True):
                yaw = (self._bridge.gaze_norm[0] - 0.5) * config.HFOV_DEG
                pitch = (0.5 - self._bridge.gaze_norm[1]) * config.VFOV_DEG
                fps = max(knobs.get("fps", 30), 1)
                budget_bytes = int((knobs.get("target_kbps", 1800) * 1000 / 8) / fps)
                sample["budget_bytes"] = budget_bytes

                # Track trajectory in predictor
                t_now = time.monotonic()
                if hasattr(self._state, "predictor"):
                    self._state.predictor.observe(t_now, yaw, pitch)

                target_yaw = yaw
                target_pitch = pitch
                predicted_roi = None
                pred_conf = 1.0

                if getattr(self._state, "prediction_enabled", False) and hasattr(self._state, "predictor"):
                    try:
                        pred = self._state.predictor.predict(
                            width=work.width,
                            height=work.height,
                            horizon_s=0.100,
                            apply_fallback=True,
                        )
                        target_yaw = pred.predicted_yaw_deg
                        target_pitch = pred.predicted_pitch_deg
                        predicted_roi = list(pred.roi_box)
                        pred_conf = pred.confidence
                    except Exception as pe:
                        logger.debug("prediction error: %s", pe)

                try:
                    tile_res = viewport_tiles.encode_and_reconstruct_tiles(
                        work,
                        yaw_deg=target_yaw,
                        pitch_deg=target_pitch,
                        quality_ladder=knobs.get("tile_ladder"),
                        codec="jpeg",
                        budget_bytes=budget_bytes,
                        allocation_strategy=self._state.allocation_strategy,
                    )
                    sample.update(
                        tile_bytes=tile_res.total_bytes,
                        viewport_psnr_db=tile_res.viewport_psnr_db,
                        guard_band_psnr_db=tile_res.guard_band_psnr_db,
                        peripheral_psnr_db=tile_res.peripheral_psnr_db,
                        overshoot_bytes=tile_res.overshoot_bytes,
                        budget_retries=tile_res.budget_retries,
                        within_budget=tile_res.within_budget,
                        boundary_artifact_mse=tile_res.boundary_artifact_mse,
                        tile_qualities=[t.quality_val for t in tile_res.tiles],
                        high_count=sum(1 for t in tile_res.tiles if t.tier == "high"),
                        medium_count=sum(1 for t in tile_res.tiles if t.tier == "medium"),
                        low_count=sum(1 for t in tile_res.tiles if t.tier == "low"),
                        encode_time_ms=round(tile_res.encode_time_s * 1000, 2),
                        predicted_roi=predicted_roi,
                        prediction_confidence=round(pred_conf, 2),
                    )
                except Exception as e:
                    logger.debug("tile shadow metric error: %s", e)
            return sample

        try:
            self._state.shadow_sample = await asyncio.to_thread(run)
        except Exception:
            logger.debug("shadow sampling failed", exc_info=True)

    async def recv(self) -> VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError
        img, seq = await self._bridge.video.wait_for(self._last_seen)
        if seq > self._last_seen + 1:  # skipped frames: this viewer fell behind
            self._state.frames_dropped += seq - self._last_seen - 1
        self._last_seen = seq

        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._start is None:
            self._start = now
            self._next_send = now
        cycle_start = now
        if self._next_send > now:
            await asyncio.sleep(self._next_send - now)

        self._frame_count += 1
        knobs = self._state.knobs
        pts = int((loop.time() - self._start) * config.VIDEO_CLOCK_RATE)
        if pts <= self._pts:  # keep RTP timestamps strictly increasing
            pts = self._pts + 1
        out, work, shadow, frame = _stage_and_frame(
            img,
            knobs,
            self._bridge.gaze_norm,
            None,
            getattr(self._state, "allocation_strategy", "intelligent_allocator"),
            pts,
        )
        self._pts = pts
        self._state.shadow = shadow
        self._state.frames_sent += 1
        if (self._frame_count == 1 or self._frame_count % config.SHADOW_SAMPLE_EVERY_N == 0) and (
            self._shadow_task is None or self._shadow_task.done()
        ):
            self._shadow_task = asyncio.create_task(self._sample_shadow(img, work, knobs))
        # pace at the profile fps when ahead; when behind, never add a sleep --
        # the next recv runs immediately (a full period of extra sleep here
        # capped the pipeline below its real capacity whenever it fell behind)
        self._next_send = max(cycle_start + 1.0 / max(knobs["fps"], 1), loop.time())
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
