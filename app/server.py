"""Demo server: adaptive + viewport-aware MJPEG streaming of the synthetic XR feed,
plus the live WebRTC plane (phone -> bridge -> viewers) under /phone and /viewer.

In the source platform the client reports WebRTC receiver stats; in the MJPEG
baseline the client passes its measured bandwidth as a query parameter and the
server walks the same profile ladder. In the live plane the stats are real.

When no phone is connected, a built-in synthetic pump feeds XR frames into the
bridge automatically so the viewer works immediately."""

import asyncio
import base64
import io
import json
import logging
import os
import shutil
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, Query, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.adaptive import AdaptiveStreamSession
from app.codecs import audio_codec, lossless, lossy, spatial_audio, video_codec
from app.comparison import (
    MAX_UPLOAD_BYTES,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    VideoCompressionResult,
    cleanup_job,
    compress_image_h264,
    compress_image_jpeg,
    compress_image_tiles,
    compress_video_h264,
    compress_video_mjpeg,
    extract_frame_from_video,
    get_job_status,
    get_supported_methods,
    get_temp_dir,
    get_video_compression_methods,
    validate_upload,
    _active_jobs,
    _compute_metrics,
)
from app.media_source import SyntheticXRSource
from app.rtc.bridge import Bridge
from app.rtc.signaling import phone_endpoint, phone_stats_loop, viewer_endpoint
from app.viewport import foveated_jpeg, gaze_roi
from app.viewport_tiles import encode_and_reconstruct_tiles

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
bridge = Bridge()


async def _synthetic_pump(bridge: Bridge) -> None:
    """Feed synthetic XR frames into the bridge when no phone is connected.

    Starts automatically; pauses while a real phone is publishing and resumes
    when it disconnects. This ensures the viewer always has video to display."""
    src = SyntheticXRSource()
    t = 0.0
    while True:
        # Pause while a real phone peer is actively connected
        if bridge.phone_pc is not None or (bridge.phone_connected and not bridge._synthetic_active):
            bridge._synthetic_active = False
            await asyncio.sleep(0.5)
            continue
        bridge._synthetic_active = True
        img = await asyncio.to_thread(src.frame, t)
        await bridge.video.publish(img)
        t += 1.0 / 15  # 15 fps synthetic feed
        await asyncio.sleep(1.0 / 15)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    stats_task = asyncio.create_task(phone_stats_loop(bridge))
    pump_task = asyncio.create_task(_synthetic_pump(bridge))
    logger.info("Synthetic video pump started — viewer will show XR frames immediately")
    yield
    pump_task.cancel()
    stats_task.cancel()


app = FastAPI(title="XR Compression Lab", description="Data Compression course baseline", lifespan=_lifespan)
source = SyntheticXRSource()


def _entry(raw_len: int, blob: bytes) -> dict:
    return {"bytes": len(blob), "ratio": round(lossless.compression_ratio(raw_len, len(blob)), 3)}


@app.get("/metrics")
def metrics(t: float = 1.0, quality: int = 75):
    """One frame through every codec: sizes, ratios and distortion."""
    img = source.frame(t)
    raw = img.tobytes()
    rle = lossless.rle_encode(raw)
    huff = lossless.huffman_encode(raw)
    zlb = lossless.zlib_compress(raw)
    lzm = lossless.lzma_compress(raw)
    jpeg_blob, jpeg_img = lossy.jpeg_roundtrip(img, quality)
    roi = (img.width // 4, img.height // 4, img.width * 3 // 4, img.height * 3 // 4)
    fov = foveated_jpeg(img, roi, quality=quality)
    h264_blob, h264_rec, h264_m = video_codec.h264_roundtrip(img, bitrate_kbps=1800, fps=30)
    tiled_res = encode_and_reconstruct_tiles(img, yaw_deg=0.0, pitch_deg=0.0, codec="jpeg")
    return {
        "frame": {"width": img.width, "height": img.height, "raw_bytes": len(raw), "quality": quality},
        "lossless": {
            "rle": _entry(len(raw), rle),
            "huffman": _entry(len(raw), huff),
            "zlib_deflate": _entry(len(raw), zlb),
            "lzma": _entry(len(raw), lzm),
        },
        "lossy": {
            "jpeg": _entry(len(raw), jpeg_blob) | {"psnr_db": round(lossy.psnr(img, jpeg_img), 2)},
            "foveated_jpeg": _entry(len(raw), fov) | {"roi_box": roi},
            "h264": _entry(len(raw), h264_blob) | {"psnr_db": h264_m["psnr_db"], "ssim": h264_m["ssim"]},
            "viewport_tiles_4x4": {
                "bytes": tiled_res.total_bytes,
                "ratio": round(lossless.compression_ratio(len(raw), tiled_res.total_bytes), 3),
                "viewport_psnr_db": tiled_res.viewport_psnr_db,
                "peripheral_psnr_db": tiled_res.peripheral_psnr_db,
                "full_psnr_db": tiled_res.full_psnr_db,
                "viewport_ssim": tiled_res.viewport_ssim,
            },
        },
    }


@app.get("/stream")
def stream(yaw: float = 0.0, pitch: float = 0.0, bw_kbps: float | None = None, fps: int = 15, frames: int = 0):
    """MJPEG stream. `bw_kbps` simulates client-reported throughput: each frame's
    transmission time feeds the EWMA, which walks the profile ladder. `frames=0`
    streams until the client disconnects."""
    session = AdaptiveStreamSession()

    def generate():
        n = 0
        while frames <= 0 or n < frames:
            jpg = session.encode_frame(source, n / fps, yaw, pitch)
            if bw_kbps:
                session.report_throughput(len(jpg), len(jpg) * 8 / (bw_kbps * 1000))
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            n += 1

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# --- Live WebRTC demo plane ---------------------------------------------------

@app.get("/", include_in_schema=False)
@app.get("/viewer", include_in_schema=False)
def viewer_page():
    """Laptop dashboard: adapted video feed + live compression metrics."""
    return FileResponse(WEB_DIR / "viewer.html", media_type="text/html")


@app.get("/phone", include_in_schema=False)
def phone_page():
    """Phone source page: publish camera/mic, touch to steer the gaze."""
    return FileResponse(WEB_DIR / "phone.html", media_type="text/html")


@app.get("/api/ice", include_in_schema=False)
def ice_servers():
    """ICE configuration for the browser peers (signaling-only relays)."""
    return {"iceServers": [{"urls": config.ICE_SERVERS}]}


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {
        "ok": True,
        "phone_connected": bridge.phone_connected,
        "synthetic_active": getattr(bridge, "_synthetic_active", False),
        "viewers": len(bridge.viewers),
        "audio_mode": bridge.audio_mode,
    }


@app.websocket("/ws/phone")
async def ws_phone(websocket: WebSocket):
    await phone_endpoint(bridge, websocket)


@app.websocket("/ws/viewer")
async def ws_viewer(websocket: WebSocket):
    await viewer_endpoint(bridge, websocket)


@app.get("/audio")
def audio(seconds: float = 1.0, profile: str = "high"):
    """FOA spatial audio folded to the profile's channel count, as WAV."""
    channels = config.PROFILES[profile]["audio_channels"]
    foa = source.foa_audio(seconds)
    pcm = spatial_audio.quantize_i16(spatial_audio.fold_foa(foa, channels))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(pcm.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(np.ascontiguousarray(pcm.T).tobytes())
    return Response(content=buf.getvalue(), media_type="audio/wav")


# --- Quality Comparison & Video Compression Playground ---

@app.get("/compare", include_in_schema=False)
def compare_page():
    """Before/After Quality Comparison slider page."""
    return FileResponse(WEB_DIR / "compare.html", media_type="text/html")


@app.get("/playground", include_in_schema=False)
def playground_page():
    """Video Compression Playground page."""
    return FileResponse(WEB_DIR / "playground.html", media_type="text/html")


@app.get("/api/compression-methods")
def api_compression_methods():
    """Return available compression methods for images."""
    return {"methods": get_supported_methods()}


@app.get("/api/video-compression-methods")
def api_video_compression_methods():
    """Return available compression methods for video."""
    return {"methods": get_video_compression_methods()}


@app.get("/api/sample-frame")
def api_sample_frame(t: float = 1.0):
    """Return a synthetic XR frame as a JPEG for the comparison tool."""
    img = source.frame(t)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=98)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "image_b64": b64,
        "width": img.width,
        "height": img.height,
        "raw_bytes": len(img.tobytes()),
        "source": "synthetic_xr",
    }


@app.post("/api/compress-image")
async def api_compress_image(
    method: str = Form("jpeg"),
    quality: int = Form(75),
    bitrate_kbps: int = Form(1800),
    chroma_subsample: bool = Form(True),
    quality_high: int = Form(85),
    quality_medium: int = Form(55),
    quality_low: int = Form(25),
    yaw_deg: float = Form(0.0),
    pitch_deg: float = Form(0.0),
    source_type: str = Form("synthetic"),
    t: float = Form(1.0),
    file: UploadFile | None = File(None),
):
    """Compress an image using the specified method and return metrics + result."""
    try:
        orig_file_bytes = None
        # Get the source image
        if source_type == "upload" and file:
            filename = file.filename or "upload.png"
            contents = await file.read()
            valid, err = validate_upload(filename, len(contents))
            if not valid:
                return JSONResponse(status_code=400, content={"error": err})

            orig_file_bytes = len(contents)
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

            if ext in SUPPORTED_VIDEO_FORMATS:
                # Video file uploaded: extract frame at timestamp t
                temp_video = get_temp_dir() / f"temp_{uuid.uuid4().hex[:8]}.{ext}"
                try:
                    with open(temp_video, "wb") as f_out:
                        f_out.write(contents)
                    img = extract_frame_from_video(str(temp_video), time_s=t)
                    if img is None:
                        return JSONResponse(
                            status_code=400,
                            content={"error": f"Could not extract video frame at t={t}s. File may be unreadable or empty."},
                        )
                finally:
                    if temp_video.exists():
                        try:
                            temp_video.unlink()
                        except OSError:
                            pass
            else:
                try:
                    img = Image.open(io.BytesIO(contents)).convert("RGB")
                except Exception:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Could not decode image. Ensure it is a valid, uncorrupted image file."},
                    )
        else:
            # Use synthetic XR source
            img = source.frame(t)

        # Compress using the requested method
        if method == "jpeg":
            result = compress_image_jpeg(
                img,
                quality=quality,
                chroma_subsample=chroma_subsample,
                original_file_bytes=orig_file_bytes,
            )
        elif method == "h264":
            result = compress_image_h264(
                img,
                bitrate_kbps=bitrate_kbps,
                original_file_bytes=orig_file_bytes,
            )
        elif method == "tiles":
            result = compress_image_tiles(
                img,
                quality_high=quality_high,
                quality_medium=quality_medium,
                quality_low=quality_low,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                original_file_bytes=orig_file_bytes,
            )
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unknown compression method: {method}"},
            )

        # Also return the original as base64 for display
        orig_buf = io.BytesIO()
        img.save(orig_buf, "JPEG", quality=98)
        orig_b64 = base64.b64encode(orig_buf.getvalue()).decode("ascii")

        return {
            "original_image_b64": orig_b64,
            "result": result.to_dict(),
        }

    except Exception as e:
        logger.exception("Image compression failed")
        return JSONResponse(
            status_code=500,
            content={"error": f"Compression failed: {str(e)}"},
        )


@app.post("/api/upload-video")
async def api_upload_video(file: UploadFile = File(...)):
    """Upload a video file for compression. Returns a job_id and metadata."""
    filename = file.filename or "video.mp4"
    # Read content length from headers if available
    contents = await file.read()
    file_size = len(contents)

    valid, error_msg = validate_upload(filename, file_size)
    if not valid:
        return JSONResponse(status_code=400, content={"error": error_msg})

    # Save to temp
    job_id = str(uuid.uuid4())[:12]
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "mp4"
    input_path = get_temp_dir() / f"{job_id}_input.{ext}"

    try:
        with open(input_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to save upload: {str(e)}"})

    # Extract metadata
    metadata = {"job_id": job_id, "filename": filename, "file_size": file_size}
    try:
        import av
        container = av.open(str(input_path))
        vstream = container.streams.video[0]
        metadata["width"] = vstream.width
        metadata["height"] = vstream.height
        metadata["fps"] = float(vstream.average_rate) if vstream.average_rate else None
        if vstream.duration and vstream.time_base:
            metadata["duration_s"] = round(float(vstream.duration * vstream.time_base), 2)
        elif container.duration:
            metadata["duration_s"] = round(float(container.duration) / av.time_base, 2)
        metadata["codec"] = vstream.codec_context.name if vstream.codec_context else None
        container.close()
    except Exception as e:
        # Video metadata extraction failed, but file might still be processable
        metadata["metadata_error"] = str(e)

    # Initialize job
    _active_jobs[job_id] = VideoCompressionResult(job_id=job_id, status="uploaded")

    return metadata


@app.post("/api/compress-video")
async def api_compress_video(
    job_id: str = Form(...),
    method: str = Form("h264"),
    bitrate_kbps: int = Form(2000),
    quality: int = Form(75),
    fps: int = Form(30),
    resolution: str = Form("original"),
):
    """Start video compression for an uploaded file."""
    import glob as glob_mod

    # Find the input file
    matches = list(get_temp_dir().glob(f"{job_id}_input.*"))
    if not matches:
        return JSONResponse(status_code=404, content={"error": "Upload not found. Please upload a video first."})

    input_path = str(matches[0])

    # Validate it's a real file
    if not os.path.isfile(input_path):
        return JSONResponse(status_code=404, content={"error": "Upload file missing."})

    try:
        # Validate video can be opened
        import av
        container = av.open(input_path)
        if not container.streams.video:
            container.close()
            return JSONResponse(status_code=400, content={"error": "File contains no video stream."})
        container.close()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Cannot read video: {str(e)}. The file may be corrupted or in an unsupported format."},
        )

    # Run compression in a thread to avoid blocking
    if method == "h264":
        result = await asyncio.to_thread(
            compress_video_h264,
            input_path, job_id, bitrate_kbps, fps, resolution,
        )
    elif method == "jpeg_frames":
        result = await asyncio.to_thread(
            compress_video_mjpeg,
            input_path, job_id, quality, fps, resolution,
        )
    else:
        return JSONResponse(status_code=400, content={"error": f"Unknown video compression method: {method}"})

    return result.to_dict()


@app.get("/api/job-status/{job_id}")
def api_job_status(job_id: str):
    """Check the status of a compression job."""
    result = get_job_status(job_id)
    if result is None:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return result.to_dict()


@app.get("/api/download/{job_id}")
def api_download(job_id: str):
    """Download the compressed video file."""
    result = get_job_status(job_id)
    if result is None or result.output_path is None:
        return JSONResponse(status_code=404, content={"error": "Compressed file not found"})
    if not os.path.isfile(result.output_path):
        return JSONResponse(status_code=404, content={"error": "Compressed file no longer available"})

    ext = result.output_path.rsplit(".", 1)[-1]
    media_types = {"mp4": "video/mp4", "avi": "video/x-msvideo", "webm": "video/webm", "mkv": "video/x-matroska"}
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        result.output_path,
        media_type=media_type,
        filename=f"compressed_{job_id}.{ext}",
    )


@app.get("/api/video-frame/{job_id}")
def api_video_frame(job_id: str, which: str = "original", time_s: float = 0.0):
    """Extract a frame from the original or compressed video for comparison."""
    if which == "original":
        matches = list(get_temp_dir().glob(f"{job_id}_input.*"))
        if not matches:
            return JSONResponse(status_code=404, content={"error": "Original video not found"})
        video_path = str(matches[0])
    elif which == "compressed":
        result = get_job_status(job_id)
        if result is None or result.output_path is None:
            return JSONResponse(status_code=404, content={"error": "Compressed video not found"})
        video_path = result.output_path
    else:
        return JSONResponse(status_code=400, content={"error": "which must be 'original' or 'compressed'"})

    img = extract_frame_from_video(video_path, time_s)
    if img is None:
        return JSONResponse(status_code=500, content={"error": "Could not extract frame"})

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "image_b64": b64,
        "width": img.width,
        "height": img.height,
    }


@app.get("/api/compare-video-frames/{job_id}")
def api_compare_video_frames(job_id: str, time_s: float = 0.0):
    """Extract and compare frames from both original and compressed videos of a job."""
    matches = list(get_temp_dir().glob(f"{job_id}_input.*"))
    if not matches:
        return JSONResponse(status_code=404, content={"error": "Original video not found"})
    orig_path = str(matches[0])

    result = get_job_status(job_id)
    if result is None or result.output_path is None:
        return JSONResponse(status_code=404, content={"error": "Compressed video not found for this job"})
    comp_path = result.output_path

    orig_img = extract_frame_from_video(orig_path, time_s)
    comp_img = extract_frame_from_video(comp_path, time_s)

    if orig_img is None or comp_img is None:
        return JSONResponse(status_code=500, content={"error": f"Could not extract frames at timestamp {time_s}s"})

    if comp_img.size != orig_img.size:
        comp_img_matched = comp_img.resize(orig_img.size, Image.BILINEAR)
    else:
        comp_img_matched = comp_img

    psnr_val, ssim_val = _compute_metrics(orig_img, comp_img_matched)

    orig_buf = io.BytesIO()
    orig_img.save(orig_buf, "JPEG", quality=98)
    orig_b64 = base64.b64encode(orig_buf.getvalue()).decode("ascii")

    comp_buf = io.BytesIO()
    comp_img_matched.save(comp_buf, "JPEG", quality=98)
    comp_b64 = base64.b64encode(comp_buf.getvalue()).decode("ascii")

    return {
        "job_id": job_id,
        "time_s": time_s,
        "original_image_b64": orig_b64,
        "compressed_image_b64": comp_b64,
        "width": orig_img.width,
        "height": orig_img.height,
        "frame_psnr_db": psnr_val,
        "frame_ssim": ssim_val,
        "job_metrics": result.to_dict(),
    }


@app.get("/api/serve-video/{job_id}")
def api_serve_video(job_id: str, which: str = "original"):
    """Serve a video for the HTML5 player (original or compressed)."""
    if which == "original":
        matches = list(get_temp_dir().glob(f"{job_id}_input.*"))
        if not matches:
            return JSONResponse(status_code=404, content={"error": "Original video not found"})
        video_path = str(matches[0])
    elif which == "compressed":
        result = get_job_status(job_id)
        if result is None or result.output_path is None:
            return JSONResponse(status_code=404, content={"error": "Compressed video not found"})
        video_path = result.output_path
    else:
        return JSONResponse(status_code=400, content={"error": "which must be 'original' or 'compressed'"})

    if not os.path.isfile(video_path):
        return JSONResponse(status_code=404, content={"error": "Video file not found"})

    ext = video_path.rsplit(".", 1)[-1].lower()
    media_types = {"mp4": "video/mp4", "avi": "video/x-msvideo", "webm": "video/webm", "mkv": "video/x-matroska", "mov": "video/quicktime"}
    return FileResponse(video_path, media_type=media_types.get(ext, "video/mp4"))


@app.delete("/api/cleanup/{job_id}")
def api_cleanup(job_id: str):
    """Clean up temporary files for a compression job."""
    cleanup_job(job_id)
    return {"ok": True}
