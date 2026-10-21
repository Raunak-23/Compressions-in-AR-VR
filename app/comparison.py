"""Before/After Quality Comparison and Video Compression Playground backend.

Provides:
1. Single-image compression using JPEG, H.264, and 4×4 viewport tiles.
2. Video file compression using H.264 with configurable quality.
3. Frame extraction for before/after visual comparison.
4. Measured quality metrics: PSNR, SSIM, file sizes, compression ratio, timing.

All compression uses the existing codec implementations in app/codecs/.
"""

from __future__ import annotations

import io
import math
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

from app.codecs import lossless, lossy
from app.codecs.video_codec import (
    compute_ssim,
    decode_frame,
    encode_frame as h264_encode_frame,
    h264_roundtrip,
)
from app.viewport import foveated_jpeg, gaze_roi
from app.viewport_tiles import encode_and_reconstruct_tiles

# Maximum upload size: 100 MB
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
# Supported image formats
SUPPORTED_IMAGE_FORMATS = {"jpeg", "jpg", "png", "bmp", "webp", "tiff"}
# Supported video formats
SUPPORTED_VIDEO_FORMATS = {"mp4", "webm", "mkv", "avi", "mov"}

# Temp directory for uploads and compressed outputs
_TEMP_DIR = Path(tempfile.gettempdir()) / "xr_compression_lab"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CompressionResult:
    """Result of a single compression operation."""
    method: str
    original_bytes: int
    compressed_bytes: int
    size_reduction_bytes: int
    size_reduction_pct: float
    compression_ratio: float
    original_file_bytes: Optional[int] = None
    raw_bytes: Optional[int] = None
    psnr_db: Optional[float] = None
    ssim: Optional[float] = None
    processing_time_s: float = 0.0
    width: int = 0
    height: int = 0
    output_width: int = 0
    output_height: int = 0
    # Only populated for methods that produce image data
    compressed_image_b64: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Remove None values
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class VideoCompressionResult:
    """Result of a video compression operation."""
    job_id: str
    status: str  # "processing", "completed", "failed"
    method: str = ""
    original_bytes: int = 0
    compressed_bytes: int = 0
    size_reduction_pct: float = 0.0
    compression_ratio: float = 0.0
    original_width: int = 0
    original_height: int = 0
    output_width: int = 0
    output_height: int = 0
    original_duration_s: float = 0.0
    output_duration_s: float = 0.0
    original_fps: float = 0.0
    output_fps: float = 0.0
    processing_time_s: float = 0.0
    avg_psnr_db: Optional[float] = None
    avg_ssim: Optional[float] = None
    avg_bitrate_kbps: Optional[float] = None
    target_bitrate_kbps: Optional[int] = None
    target_met: Optional[bool] = None
    target_diff_pct: Optional[float] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    progress_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


def _compute_metrics(
    original: Image.Image, compressed: Image.Image
) -> Tuple[float, float]:
    """Compute PSNR and SSIM between two images of the same size."""
    # Resize compressed to match original if needed
    if compressed.size != original.size:
        compressed = compressed.resize(original.size, Image.BILINEAR)
    psnr_val = lossy.psnr(original, compressed)
    ssim_val = compute_ssim(original, compressed)
    return (
        round(psnr_val, 2) if psnr_val != float("inf") else 99.0,
        round(ssim_val, 4),
    )


def compress_image_jpeg(
    img: Image.Image,
    quality: int = 75,
    chroma_subsample: bool = True,
    original_file_bytes: Optional[int] = None,
) -> CompressionResult:
    """Compress image using JPEG and measure quality metrics."""
    t0 = time.perf_counter()
    raw_bytes = len(img.tobytes())
    base_bytes = original_file_bytes if original_file_bytes is not None and original_file_bytes > 0 else raw_bytes

    buf = io.BytesIO()
    subsampling = 2 if chroma_subsample else 0  # 2 = 4:2:0, 0 = 4:4:4
    img.save(buf, "JPEG", quality=quality, subsampling=subsampling)
    jpeg_blob = buf.getvalue()

    # Decode back for comparison
    rec = Image.open(io.BytesIO(jpeg_blob)).convert("RGB")
    t1 = time.perf_counter()

    psnr_val, ssim_val = _compute_metrics(img, rec)
    compressed_bytes = len(jpeg_blob)

    import base64
    compressed_b64 = base64.b64encode(jpeg_blob).decode("ascii")

    return CompressionResult(
        method=f"JPEG (q={quality}, {'4:2:0' if chroma_subsample else '4:4:4'})",
        original_bytes=base_bytes,
        compressed_bytes=compressed_bytes,
        size_reduction_bytes=base_bytes - compressed_bytes,
        size_reduction_pct=round(100.0 * (base_bytes - compressed_bytes) / base_bytes, 2),
        compression_ratio=round(base_bytes / compressed_bytes, 2) if compressed_bytes > 0 else float("inf"),
        original_file_bytes=original_file_bytes,
        raw_bytes=raw_bytes,
        psnr_db=psnr_val,
        ssim=ssim_val,
        processing_time_s=round(t1 - t0, 4),
        width=img.width,
        height=img.height,
        output_width=rec.width,
        output_height=rec.height,
        compressed_image_b64=compressed_b64,
        extra={"quality": quality, "chroma_subsampling": "4:2:0" if chroma_subsample else "4:4:4"},
    )


def compress_image_h264(
    img: Image.Image,
    bitrate_kbps: int = 1800,
    original_file_bytes: Optional[int] = None,
) -> CompressionResult:
    """Compress image using H.264 (single-frame) and measure quality metrics."""
    t0 = time.perf_counter()
    raw_bytes = len(img.tobytes())
    base_bytes = original_file_bytes if original_file_bytes is not None and original_file_bytes > 0 else raw_bytes

    blob, rec, metrics = h264_roundtrip(img, bitrate_kbps=bitrate_kbps, fps=30)
    t1 = time.perf_counter()

    # Encode reconstructed as JPEG for browser display
    import base64
    display_buf = io.BytesIO()
    rec.save(display_buf, "JPEG", quality=95)
    compressed_b64 = base64.b64encode(display_buf.getvalue()).decode("ascii")

    compressed_bytes = len(blob)
    return CompressionResult(
        method=f"H.264 (bitrate={bitrate_kbps} kbps)",
        original_bytes=base_bytes,
        compressed_bytes=compressed_bytes,
        size_reduction_bytes=base_bytes - compressed_bytes,
        size_reduction_pct=round(100.0 * (base_bytes - compressed_bytes) / base_bytes, 2),
        compression_ratio=round(base_bytes / compressed_bytes, 2) if compressed_bytes > 0 else float("inf"),
        original_file_bytes=original_file_bytes,
        raw_bytes=raw_bytes,
        psnr_db=metrics["psnr_db"],
        ssim=metrics["ssim"],
        processing_time_s=round(t1 - t0, 4),
        width=img.width,
        height=img.height,
        output_width=rec.width,
        output_height=rec.height,
        compressed_image_b64=compressed_b64,
        extra={"bitrate_kbps": bitrate_kbps, "encode_time_s": metrics["encode_time_s"]},
    )


def compress_image_tiles(
    img: Image.Image,
    quality_high: int = 85,
    quality_medium: int = 55,
    quality_low: int = 25,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    original_file_bytes: Optional[int] = None,
) -> CompressionResult:
    """Compress image using 4×4 viewport-aware tiles."""
    t0 = time.perf_counter()
    raw_bytes = len(img.tobytes())
    base_bytes = original_file_bytes if original_file_bytes is not None and original_file_bytes > 0 else raw_bytes

    quality_ladder = {"high": quality_high, "medium": quality_medium, "low": quality_low}
    result = encode_and_reconstruct_tiles(
        img,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        quality_ladder=quality_ladder,
        codec="jpeg",
    )
    t1 = time.perf_counter()

    psnr_val, ssim_val = _compute_metrics(img, result.reconstructed)

    # Encode reconstructed as JPEG for browser display
    import base64
    display_buf = io.BytesIO()
    result.reconstructed.save(display_buf, "JPEG", quality=95)
    compressed_b64 = base64.b64encode(display_buf.getvalue()).decode("ascii")

    return CompressionResult(
        method=f"Viewport Tiles (high={quality_high}, med={quality_medium}, low={quality_low})",
        original_bytes=base_bytes,
        compressed_bytes=result.total_bytes,
        size_reduction_bytes=base_bytes - result.total_bytes,
        size_reduction_pct=round(100.0 * (base_bytes - result.total_bytes) / base_bytes, 2),
        compression_ratio=round(base_bytes / result.total_bytes, 2) if result.total_bytes > 0 else float("inf"),
        original_file_bytes=original_file_bytes,
        raw_bytes=raw_bytes,
        psnr_db=result.full_psnr_db,
        ssim=result.full_ssim,
        processing_time_s=round(t1 - t0, 4),
        width=img.width,
        height=img.height,
        output_width=result.reconstructed.width,
        output_height=result.reconstructed.height,
        compressed_image_b64=compressed_b64,
        extra={
            "viewport_psnr_db": result.viewport_psnr_db,
            "viewport_ssim": result.viewport_ssim,
            "peripheral_psnr_db": result.peripheral_psnr_db,
            "peripheral_ssim": result.peripheral_ssim,
            "boundary_artifact_mse": result.boundary_artifact_mse,
        },
    )


def get_supported_methods() -> List[Dict[str, Any]]:
    """Return the list of supported compression methods with their controls."""
    return [
        {
            "id": "jpeg",
            "name": "JPEG",
            "description": "Lossy transform coding with DCT, Huffman entropy coding, and optional chroma subsampling.",
            "type": "image",
            "controls": [
                {"id": "quality", "label": "Quality", "type": "range", "min": 1, "max": 100, "default": 75, "step": 1},
                {"id": "chroma_subsample", "label": "Chroma Subsampling (4:2:0)", "type": "checkbox", "default": True},
            ],
        },
        {
            "id": "h264",
            "name": "H.264 / AVC",
            "description": "Video codec using spatial prediction, transform coding, and in-loop deblocking. Single-frame mode (IDR only).",
            "type": "image",
            "controls": [
                {"id": "bitrate_kbps", "label": "Target Bitrate (kbps)", "type": "range", "min": 100, "max": 5000, "default": 1800, "step": 50},
            ],
        },
        {
            "id": "tiles",
            "name": "Viewport Tiles (4×4)",
            "description": "Spatial tiling with gaze-directed quality allocation: high in viewport, medium in guard band, low in periphery.",
            "type": "image",
            "controls": [
                {"id": "quality_high", "label": "Viewport Quality", "type": "range", "min": 10, "max": 100, "default": 85, "step": 5},
                {"id": "quality_medium", "label": "Guard Band Quality", "type": "range", "min": 10, "max": 100, "default": 55, "step": 5},
                {"id": "quality_low", "label": "Periphery Quality", "type": "range", "min": 5, "max": 100, "default": 25, "step": 5},
                {"id": "yaw_deg", "label": "Gaze Yaw (°)", "type": "range", "min": -50, "max": 50, "default": 0, "step": 1},
                {"id": "pitch_deg", "label": "Gaze Pitch (°)", "type": "range", "min": -30, "max": 30, "default": 0, "step": 1},
            ],
        },
    ]


def get_video_compression_methods() -> List[Dict[str, Any]]:
    """Return the list of supported video compression methods."""
    methods = [
        {
            "id": "h264",
            "name": "H.264 / AVC",
            "description": "Inter-frame video compression using motion estimation, spatial and temporal prediction. Uses libx264 via PyAV.",
            "controls": [
                {"id": "bitrate_kbps", "label": "Target Bitrate (kbps)", "type": "range", "min": 100, "max": 10000, "default": 2000, "step": 100},
                {"id": "fps", "label": "Output FPS", "type": "range", "min": 5, "max": 60, "default": 30, "step": 1},
                {"id": "resolution", "label": "Resolution", "type": "select", "options": [
                    {"value": "original", "label": "Original"},
                    {"value": "720p", "label": "1280×720"},
                    {"value": "480p", "label": "854×480"},
                    {"value": "360p", "label": "640×360"},
                ], "default": "original"},
            ],
        },
        {
            "id": "jpeg_frames",
            "name": "JPEG Frame Sequence (M-JPEG)",
            "description": "Frame-by-frame JPEG compression without temporal redundancy. Each frame compressed independently.",
            "controls": [
                {"id": "quality", "label": "JPEG Quality", "type": "range", "min": 1, "max": 100, "default": 75, "step": 1},
                {"id": "fps", "label": "Output FPS", "type": "range", "min": 5, "max": 60, "default": 30, "step": 1},
                {"id": "resolution", "label": "Resolution", "type": "select", "options": [
                    {"value": "original", "label": "Original"},
                    {"value": "720p", "label": "1280×720"},
                    {"value": "480p", "label": "854×480"},
                    {"value": "360p", "label": "640×360"},
                ], "default": "original"},
            ],
        },
    ]
    return methods


# --- Video compression jobs ---

_active_jobs: Dict[str, VideoCompressionResult] = {}


def _resolve_resolution(target: str, orig_w: int, orig_h: int) -> Tuple[int, int]:
    """Resolve target resolution, maintaining aspect ratio."""
    if target == "original":
        # Ensure even dimensions for H.264
        w = orig_w if orig_w % 2 == 0 else orig_w + 1
        h = orig_h if orig_h % 2 == 0 else orig_h + 1
        return w, h
    presets = {"720p": (1280, 720), "480p": (854, 480), "360p": (640, 360)}
    tw, th = presets.get(target, (orig_w, orig_h))
    # Maintain aspect ratio
    ar = orig_w / orig_h if orig_h > 0 else 16 / 9
    if ar > tw / th:
        out_w = tw
        out_h = int(tw / ar)
    else:
        out_h = th
        out_w = int(th * ar)
    # Ensure even dimensions
    out_w = out_w if out_w % 2 == 0 else out_w + 1
    out_h = out_h if out_h % 2 == 0 else out_h + 1
    return out_w, out_h


def validate_upload(filename: str, file_size: int) -> Tuple[bool, str]:
    """Validate an uploaded file before processing."""
    if not filename:
        return False, "Filename cannot be empty"
    if file_size <= 0:
        return False, "File is empty"
    if file_size > MAX_UPLOAD_BYTES:
        return False, f"File too large: {file_size / 1024 / 1024:.1f} MB exceeds limit of {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if not ext:
        return False, "File has no extension. Please provide a supported file format."
    if ext not in SUPPORTED_VIDEO_FORMATS and ext not in SUPPORTED_IMAGE_FORMATS:
        return False, f"Unsupported format: .{ext}. Supported: {', '.join(sorted(SUPPORTED_VIDEO_FORMATS | SUPPORTED_IMAGE_FORMATS))}"
    return True, ""


def get_job_status(job_id: str) -> Optional[VideoCompressionResult]:
    """Get the status of a compression job."""
    return _active_jobs.get(job_id)


def compress_video_h264(
    input_path: str,
    job_id: str,
    bitrate_kbps: int = 2000,
    target_fps: int = 30,
    resolution: str = "original",
) -> VideoCompressionResult:
    """Compress a video file using H.264 via PyAV. Runs synchronously."""
    import av

    result = _active_jobs.get(job_id)
    if result is None:
        result = VideoCompressionResult(job_id=job_id, status="processing", method="H.264")
        _active_jobs[job_id] = result

    result.method = f"H.264 ({bitrate_kbps} kbps)"
    result.status = "processing"

    output_path = str(_TEMP_DIR / f"{job_id}_compressed.mp4")
    t_start = time.perf_counter()

    try:
        # Open input
        in_container = av.open(input_path, mode="r")
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"

        orig_w = in_stream.width
        orig_h = in_stream.height
        orig_fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0
        orig_duration = float(in_stream.duration * in_stream.time_base) if in_stream.duration else 0.0
        if orig_duration <= 0 and in_container.duration:
            orig_duration = float(in_container.duration) / av.time_base

        result.original_bytes = os.path.getsize(input_path)
        result.original_width = orig_w
        result.original_height = orig_h
        result.original_fps = round(orig_fps, 2)
        result.original_duration_s = round(orig_duration, 2)

        out_w, out_h = _resolve_resolution(resolution, orig_w, orig_h)
        result.output_width = out_w
        result.output_height = out_h
        result.output_fps = target_fps

        # Open output
        out_container = av.open(output_path, mode="w")
        out_stream = out_container.add_stream("libx264", rate=target_fps)
        out_stream.width = out_w
        out_stream.height = out_h
        out_stream.pix_fmt = "yuv420p"
        out_stream.bit_rate = int(bitrate_kbps * 1000)
        out_stream.options = {"preset": "medium", "tune": "zerolatency"}

        # Quality metrics sampling
        psnr_samples = []
        ssim_samples = []
        frame_count = 0
        total_frames_est = int(orig_duration * orig_fps) if orig_duration > 0 else 0

        for frame in in_container.decode(video=0):
            pil_img = frame.to_image().convert("RGB")

            if (out_w, out_h) != (orig_w, orig_h):
                pil_img_resized = pil_img.resize((out_w, out_h), Image.LANCZOS)
            else:
                pil_img_resized = pil_img

            video_frame = av.VideoFrame.from_image(pil_img_resized)
            video_frame.pts = frame_count
            video_frame.time_base = __import__("fractions").Fraction(1, target_fps)

            for pkt in out_stream.encode(video_frame):
                out_container.mux(pkt)

            # Sample quality metrics every 10th frame (or first 5 frames)
            if frame_count < 5 or frame_count % 10 == 0:
                # We'll compute metrics after decoding the output
                pass

            frame_count += 1
            if total_frames_est > 0:
                result.progress_pct = min(95.0, round(100.0 * frame_count / total_frames_est, 1))

        # Flush encoder
        for pkt in out_stream.encode(None):
            out_container.mux(pkt)

        out_container.close()
        in_container.close()

        t_end = time.perf_counter()

        # Measure output
        compressed_size = os.path.getsize(output_path)
        result.compressed_bytes = compressed_size
        result.size_reduction_pct = round(
            100.0 * (result.original_bytes - compressed_size) / result.original_bytes, 2
        ) if result.original_bytes > 0 else 0.0
        result.compression_ratio = round(
            result.original_bytes / compressed_size, 2
        ) if compressed_size > 0 else 0.0
        result.processing_time_s = round(t_end - t_start, 2)
        result.output_path = output_path

        # Compute average bitrate
        output_duration = frame_count / target_fps if target_fps > 0 else 0
        result.output_duration_s = round(output_duration, 2)
        if output_duration > 0:
            result.avg_bitrate_kbps = round(
                (compressed_size * 8 / 1000) / output_duration, 1
            )

        # Track target bitrate evaluation
        result.target_bitrate_kbps = bitrate_kbps
        if result.avg_bitrate_kbps and bitrate_kbps > 0:
            diff_pct = round(100.0 * (result.avg_bitrate_kbps - bitrate_kbps) / bitrate_kbps, 1)
            result.target_diff_pct = diff_pct
            result.target_met = abs(diff_pct) <= 25.0

        # Compute PSNR/SSIM by sampling frames from both input and output
        try:
            _compute_video_quality_metrics(input_path, output_path, result, out_w, out_h)
        except Exception:
            # Metrics are best-effort
            result.avg_psnr_db = None
            result.avg_ssim = None

        result.status = "completed"
        result.progress_pct = 100.0

    except Exception as e:
        result.status = "failed"
        result.error = str(e)

    return result


def compress_video_mjpeg(
    input_path: str,
    job_id: str,
    quality: int = 75,
    target_fps: int = 30,
    resolution: str = "original",
) -> VideoCompressionResult:
    """Compress a video as M-JPEG (frame-by-frame JPEG) via PyAV. Runs synchronously."""
    import av
    import fractions

    result = _active_jobs.get(job_id)
    if result is None:
        result = VideoCompressionResult(job_id=job_id, status="processing", method="M-JPEG")
        _active_jobs[job_id] = result

    result.method = f"M-JPEG (q={quality})"
    result.status = "processing"

    output_path = str(_TEMP_DIR / f"{job_id}_compressed.avi")
    t_start = time.perf_counter()

    try:
        in_container = av.open(input_path, mode="r")
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"

        orig_w = in_stream.width
        orig_h = in_stream.height
        orig_fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0
        orig_duration = float(in_stream.duration * in_stream.time_base) if in_stream.duration else 0.0
        if orig_duration <= 0 and in_container.duration:
            orig_duration = float(in_container.duration) / av.time_base

        result.original_bytes = os.path.getsize(input_path)
        result.original_width = orig_w
        result.original_height = orig_h
        result.original_fps = round(orig_fps, 2)
        result.original_duration_s = round(orig_duration, 2)

        out_w, out_h = _resolve_resolution(resolution, orig_w, orig_h)
        result.output_width = out_w
        result.output_height = out_h
        result.output_fps = target_fps

        out_container = av.open(output_path, mode="w")
        out_stream = out_container.add_stream("mjpeg", rate=target_fps)
        out_stream.width = out_w
        out_stream.height = out_h
        out_stream.pix_fmt = "yuvj420p"
        out_stream.options = {"q:v": str(max(1, min(31, int(32 - quality * 31 / 100))))}

        frame_count = 0
        total_frames_est = int(orig_duration * orig_fps) if orig_duration > 0 else 0
        psnr_vals = []
        ssim_vals = []

        for frame in in_container.decode(video=0):
            pil_img = frame.to_image().convert("RGB")

            if (out_w, out_h) != (orig_w, orig_h):
                pil_img_resized = pil_img.resize((out_w, out_h), Image.LANCZOS)
            else:
                pil_img_resized = pil_img

            video_frame = av.VideoFrame.from_image(pil_img_resized)
            video_frame.pts = frame_count
            video_frame.time_base = fractions.Fraction(1, target_fps)

            for pkt in out_stream.encode(video_frame):
                out_container.mux(pkt)

            frame_count += 1
            if total_frames_est > 0:
                result.progress_pct = min(95.0, round(100.0 * frame_count / total_frames_est, 1))

        for pkt in out_stream.encode(None):
            out_container.mux(pkt)

        out_container.close()
        in_container.close()

        t_end = time.perf_counter()

        compressed_size = os.path.getsize(output_path)
        result.compressed_bytes = compressed_size
        result.size_reduction_pct = round(
            100.0 * (result.original_bytes - compressed_size) / result.original_bytes, 2
        ) if result.original_bytes > 0 else 0.0
        result.compression_ratio = round(
            result.original_bytes / compressed_size, 2
        ) if compressed_size > 0 else 0.0
        result.processing_time_s = round(t_end - t_start, 2)
        result.output_path = output_path

        output_duration = frame_count / target_fps if target_fps > 0 else 0
        result.output_duration_s = round(output_duration, 2)
        if output_duration > 0:
            result.avg_bitrate_kbps = round(
                (compressed_size * 8 / 1000) / output_duration, 1
            )

        try:
            _compute_video_quality_metrics(input_path, output_path, result, out_w, out_h)
        except Exception:
            result.avg_psnr_db = None
            result.avg_ssim = None

        result.status = "completed"
        result.progress_pct = 100.0

    except Exception as e:
        result.status = "failed"
        result.error = str(e)

    return result


def _compute_video_quality_metrics(
    input_path: str,
    output_path: str,
    result: VideoCompressionResult,
    out_w: int,
    out_h: int,
    max_sample_frames: int = 8,
) -> None:
    """Sample frames from original and compressed at matching timestamps to compute avg PSNR/SSIM."""
    try:
        duration = result.output_duration_s or result.original_duration_s
        if duration <= 0:
            duration = 1.0

        psnr_vals = []
        ssim_vals = []

        for i in range(max_sample_frames):
            ts = round(i * (duration / max(1, max_sample_frames)), 2)
            orig_img = extract_frame_from_video(input_path, ts)
            comp_img = extract_frame_from_video(output_path, ts)
            if orig_img is not None and comp_img is not None:
                if orig_img.size != comp_img.size:
                    comp_img = comp_img.resize(orig_img.size, Image.BILINEAR)
                p, s = _compute_metrics(orig_img, comp_img)
                psnr_vals.append(p)
                ssim_vals.append(s)

        if psnr_vals:
            result.avg_psnr_db = round(sum(psnr_vals) / len(psnr_vals), 2)
            result.avg_ssim = round(sum(ssim_vals) / len(ssim_vals), 4)
        else:
            result.avg_psnr_db = None
            result.avg_ssim = None
    except Exception:
        result.avg_psnr_db = None
        result.avg_ssim = None


def extract_frame_from_video(video_path: str, time_s: float = 0.0) -> Optional[Image.Image]:
    """Extract a single frame from a video file at the given timestamp."""
    import av
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]

        # Seek to timestamp
        if time_s > 0:
            ts = int(time_s / stream.time_base) if stream.time_base else int(time_s * 1000000)
            container.seek(ts, stream=stream)

        for frame in container.decode(video=0):
            img = frame.to_image().convert("RGB")
            container.close()
            return img

        container.close()
    except Exception:
        pass
    return None


def cleanup_job(job_id: str) -> None:
    """Clean up temporary files for a compression job."""
    result = _active_jobs.pop(job_id, None)
    if result and result.output_path:
        try:
            os.unlink(result.output_path)
        except OSError:
            pass
    # Clean up input file too
    input_path = _TEMP_DIR / f"{job_id}_input.*"
    import glob
    for f in glob.glob(str(_TEMP_DIR / f"{job_id}_input*")):
        try:
            os.unlink(f)
        except OSError:
            pass


def get_temp_dir() -> Path:
    """Return the temp directory path for uploads."""
    return _TEMP_DIR
