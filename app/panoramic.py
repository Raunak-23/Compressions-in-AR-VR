"""Equirectangular 360° Content Workflow and Spherical Viewport Mapping.

Supports:
1. Equirectangular metadata validation (aspect ratio, mono vs SBS stereo, dimensions).
2. Spherical projection: mapping yaw (-180° to +180°) and pitch (-90° to +90°)
   to equirectangular (u, v) image coordinates.
3. Repeatable viewport trajectories:
   - Central fixation (yaw=0, pitch=0)
   - Smooth horizontal panning (smooth pursuit across horizon)
   - Sudden saccadic turn (90° jump within 50ms)
4. Procedural 360° panoramic source (equator markings, celestial poles, textured ground,
   and moving celestial target) as well as video/image file loading.
5. Panoramic tile encoding, reconstruction, and seam boundary MSE analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import av
import numpy as np
from PIL import Image, ImageDraw

from app import config
from app.codecs import lossy
from app.codecs.video_codec import compute_ssim
from app.viewport_tiles import (
    METRIC_UNAVAILABLE,
    _measure_boundary_artifacts,
    compute_grid_boundaries,
    encode_and_reconstruct_tiles,
)


@dataclass
class PanoramicMetadata:
    width: int
    height: int
    aspect_ratio: float
    layout: Literal["mono", "stereo_sbs", "stereo_tb", "unknown"]
    is_valid_equirectangular: bool
    fps: float = 30.0
    duration_s: float = 1.0
    frame_count: int = 30


def validate_panoramic_dimensions(
    width: int,
    height: int,
) -> PanoramicMetadata:
    """Validate panoramic aspect ratio and determine layout."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions: {width}x{height}")

    ratio = round(width / float(height), 3)

    # Standard equirectangular projections:
    # 2:1 is standard mono 360°
    # 4:1 is side-by-side stereo 360°
    # 1:1 is top-bottom stereo 360°
    if abs(ratio - 2.0) <= 0.05:
        layout = "mono"
        is_valid = True
    elif abs(ratio - 4.0) <= 0.10:
        layout = "stereo_sbs"
        is_valid = True
    elif abs(ratio - 1.0) <= 0.05:
        layout = "stereo_tb"
        is_valid = True
    else:
        layout = "unknown"
        is_valid = False

    return PanoramicMetadata(
        width=width,
        height=height,
        aspect_ratio=ratio,
        layout=layout,
        is_valid_equirectangular=is_valid,
    )


def sphere_to_equirect(
    yaw_deg: float,
    pitch_deg: float,
    width: int,
    height: int,
) -> Tuple[int, int]:
    """Map yaw (-180 to 180) and pitch (-90 to 90) to equirectangular pixel coordinates."""
    # Wrap yaw into [-180, 180)
    yaw_wrapped = ((yaw_deg + 180.0) % 360.0) - 180.0
    pitch_clamped = max(-90.0, min(90.0, pitch_deg))

    u = (yaw_wrapped + 180.0) / 360.0
    v = (90.0 - pitch_clamped) / 180.0

    x = int(round(u * (width - 1)))
    y = int(round(v * (height - 1)))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def equirect_roi(
    width: int,
    height: int,
    yaw_deg: float,
    pitch_deg: float,
    hfov_deg: float = 100.0,
    vfov_deg: float = 62.0,
) -> Tuple[int, int, int, int]:
    """Calculate the equirectangular bounding box covering the FOV frustum."""
    cx, cy = sphere_to_equirect(yaw_deg, pitch_deg, width, height)
    rw = int((hfov_deg / 360.0) * width)
    rh = int((vfov_deg / 180.0) * height)

    x0 = max(0, cx - rw // 2)
    y0 = max(0, cy - rh // 2)
    x1 = min(width, x0 + rw)
    y1 = min(height, y0 + rh)

    # Re-adjust if clamped against boundary
    if x1 == width:
        x0 = max(0, width - rw)
    if y1 == height:
        y0 = max(0, height - rh)

    return x0, y0, x1, y1


class ProceduralPanoramicSource:
    """Generates authentic 2:1 equirectangular test scenes with spatial markers and trajectories."""

    def __init__(self, width: int = 1920, height: int = 960):
        # 1920x960 is an exact 2:1 equirectangular mono canvas
        self.metadata = validate_panoramic_dimensions(width, height)
        self.width = width
        self.height = height

    def frame(self, t: float = 0.0) -> Image.Image:
        """Render a synthetic equirectangular 360° frame with latitude/longitude markers."""
        img = Image.new("RGB", (self.width, self.height), (12, 14, 24))
        draw = ImageDraw.Draw(img)

        # 1. Sky and ground gradient
        horizon_y = self.height // 2
        for y in range(self.height):
            if y < horizon_y:
                # Zenith down to horizon
                c = int(25 + 50 * (y / horizon_y))
                draw.line([(0, y), (self.width, y)], fill=(15, 20, c))
            else:
                # Horizon down to nadir
                c = int(50 * ((self.height - y) / horizon_y))
                draw.line([(0, y), (self.width, y)], fill=(c, int(c * 0.7), int(c * 0.4)))

        # 2. Latitude (parallels) and longitude (meridians) grid lines
        for lat in range(-60, 61, 30):
            _, y = sphere_to_equirect(0, lat, self.width, self.height)
            color = (200, 200, 200) if lat == 0 else (70, 80, 110)
            draw.line([(0, y), (self.width, y)], fill=color, width=2 if lat == 0 else 1)

        for lon in range(-180, 180, 45):
            x, _ = sphere_to_equirect(lon, 0, self.width, self.height)
            color = (220, 150, 40) if lon == 0 else (60, 70, 95)
            draw.line([(x, 0), (x, self.height)], fill=color, width=2 if lon == 0 else 1)

        # 3. Dynamic celestial / moving target circling across azimuth
        target_yaw = ((t * 45.0) % 360.0) - 180.0
        target_pitch = 15.0 * math.sin(t * 1.5)
        tx, ty = sphere_to_equirect(target_yaw, target_pitch, self.width, self.height)
        r = int(self.height * 0.06)
        draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=(255, 215, 80), outline=(255, 255, 240))

        # 4. Stationary textured landmark at yaw=45, pitch=0
        lx, ly = sphere_to_equirect(45.0, 0.0, self.width, self.height)
        draw.rectangle([lx - 40, ly - 80, lx + 40, ly + 40], fill=(180, 60, 50), outline=(230, 120, 100))

        return img


class PanoramicVideoSource:
    """Loads and streams real equirectangular video files using PyAV.

    Validates format, aspect ratio, frame rate, and supports frame random access
    and sequential iteration.
    """

    def __init__(self, file_path: Union[str, Path]):
        self.path = Path(file_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._container = av.open(str(self.path))
        video_streams = [s for s in self._container.streams if s.type == "video"]
        if not video_streams:
            raise ValueError(f"No video streams found in {self.path}")

        self._stream = video_streams[0]
        self.width = self._stream.width
        self.height = self._stream.height
        self.fps = float(self._stream.average_rate or 30.0)

        # Approximate duration and frame count
        if self._stream.duration and self._stream.time_base:
            self.duration_s = float(self._stream.duration * self._stream.time_base)
        else:
            self.duration_s = 0.0
        self.frame_count = int(self._stream.frames or 0)

        self.metadata = validate_panoramic_dimensions(self.width, self.height)
        self.metadata.fps = self.fps
        self.metadata.duration_s = self.duration_s
        self.metadata.frame_count = self.frame_count

        # Pre-cache frames for fast random access if video is reasonably short
        self._cached_frames: List[Image.Image] = []
        self._load_cache()

    def _load_cache(self, max_cache: int = 120) -> None:
        """Cache first max_cache frames for fast repeatable trajectory evaluation."""
        self._container.seek(0)
        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                self._cached_frames.append(frame.to_image().convert("RGB"))
                if len(self._cached_frames) >= max_cache:
                    return

    def frame(self, t: float = 0.0) -> Image.Image:
        """Fetch frame at time t (in seconds)."""
        if not self._cached_frames:
            raise RuntimeError("No frames available in video source")
        idx = int(round(t * self.fps)) % len(self._cached_frames)
        return self._cached_frames[idx]

    def frame_at(self, index: int) -> Image.Image:
        """Fetch frame at discrete index."""
        if not self._cached_frames:
            raise RuntimeError("No frames available in video source")
        return self._cached_frames[index % len(self._cached_frames)]

    def close(self) -> None:
        """Close video container."""
        if self._container:
            self._container.close()

    def __enter__(self) -> PanoramicVideoSource:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def create_test_panoramic_video(
    output_path: Union[str, Path],
    width: int = 1920,
    height: int = 960,
    fps: int = 30,
    duration_s: float = 1.0,
) -> Path:
    """Create an authentic 2:1 equirectangular MP4 video file on disk for end-to-end testing."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = ProceduralPanoramicSource(width=width, height=height)
    num_frames = max(1, int(fps * duration_s))

    container = av.open(str(path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i in range(num_frames):
        t = i / float(fps)
        img = source.frame(t)
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


# Repeatable Viewport Trajectories
def trajectory_central(t: float) -> Tuple[float, float]:
    """Central fixation: constant centre gaze (0°, 0°)."""
    return 0.0, 0.0


def trajectory_smooth_panning(t: float, speed_dps: float = 30.0) -> Tuple[float, float]:
    """Smooth pursuit horizontal panning along the equator."""
    yaw = ((speed_dps * t) % 360.0) - 180.0
    pitch = 5.0 * math.sin(2 * math.pi * 0.1 * t)
    return round(yaw, 2), round(pitch, 2)


def trajectory_sudden_turn(t: float, saccade_time_s: float = 1.0) -> Tuple[float, float]:
    """Sudden saccade turn: stationary at 0°, then abrupt 90° azimuth turn within 50ms."""
    if t < saccade_time_s:
        return 0.0, 0.0
    elif t < saccade_time_s + 0.05:
        # Rapid transition over 50ms
        frac = (t - saccade_time_s) / 0.05
        return round(frac * 90.0, 2), 0.0
    else:
        return 90.0, 0.0


def evaluate_panoramic_tiles(
    panoramic_source: Union[ProceduralPanoramicSource, PanoramicVideoSource],
    trajectory_fn,
    duration_s: float = 2.0,
    fps: int = 15,
    codec: Literal["jpeg", "h264"] = "jpeg",
    allocation_strategy: Literal["fixed_tiers", "intelligent_allocator", "uniform"] = "intelligent_allocator",
    budget_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate panoramic compression over a repeatable viewport trajectory."""
    frames = max(1, int(duration_s * fps))
    viewport_psnrs = []
    viewport_ssims = []
    guard_psnrs = []
    guard_ssims = []
    periph_psnrs = []
    periph_ssims = []
    full_psnrs = []
    full_ssims = []
    boundary_mses = []
    total_bytes_list = []
    enc_times = []
    dec_times = []

    for i in range(frames):
        t = i / float(fps)
        yaw, pitch = trajectory_fn(t)
        frame_img = panoramic_source.frame(t)

        res = encode_and_reconstruct_tiles(
            img=frame_img,
            yaw_deg=yaw,
            pitch_deg=pitch,
            codec=codec,
            budget_bytes=budget_bytes,
            allocation_strategy=allocation_strategy,
        )

        viewport_psnrs.append(res.viewport_psnr_db)
        viewport_ssims.append(res.viewport_ssim)
        if res.guard_band_psnr_db >= 0:
            guard_psnrs.append(res.guard_band_psnr_db)
            guard_ssims.append(res.guard_band_ssim)
        if res.peripheral_psnr_db >= 0:
            periph_psnrs.append(res.peripheral_psnr_db)
            periph_ssims.append(res.peripheral_ssim)
        full_psnrs.append(res.full_psnr_db)
        full_ssims.append(res.full_ssim)
        boundary_mses.append(res.boundary_artifact_mse)
        total_bytes_list.append(res.total_bytes)
        enc_times.append(res.encode_time_s)
        dec_times.append(res.decode_time_s)

    return {
        "frames_evaluated": frames,
        "mean_viewport_psnr_db": round(float(np.mean(viewport_psnrs)), 2) if viewport_psnrs else 0.0,
        "mean_viewport_ssim": round(float(np.mean(viewport_ssims)), 4) if viewport_ssims else 0.0,
        "mean_guard_band_psnr_db": round(float(np.mean(guard_psnrs)), 2) if guard_psnrs else -1.0,
        "mean_guard_band_ssim": round(float(np.mean(guard_ssims)), 4) if guard_ssims else -1.0,
        "mean_peripheral_psnr_db": round(float(np.mean(periph_psnrs)), 2) if periph_psnrs else -1.0,
        "mean_peripheral_ssim": round(float(np.mean(periph_ssims)), 4) if periph_ssims else -1.0,
        "mean_full_psnr_db": round(float(np.mean(full_psnrs)), 2) if full_psnrs else 0.0,
        "mean_full_ssim": round(float(np.mean(full_ssims)), 4) if full_ssims else 0.0,
        "mean_boundary_mse": round(float(np.mean(boundary_mses)), 2) if boundary_mses else 0.0,
        "mean_bytes_per_frame": int(np.mean(total_bytes_list)) if total_bytes_list else 0,
        "mean_bitrate_kbps": round(float(np.mean(total_bytes_list)) * 8 * fps / 1000, 1) if total_bytes_list else 0.0,
        "mean_encode_time_ms": round(float(np.mean(enc_times)) * 1000, 2) if enc_times else 0.0,
        "mean_decode_time_ms": round(float(np.mean(dec_times)) * 1000, 2) if dec_times else 0.0,
        "codec": codec,
        "allocation_strategy": allocation_strategy,
    }
