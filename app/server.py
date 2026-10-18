"""Demo server: adaptive + viewport-aware MJPEG streaming of the synthetic XR feed,
plus the live WebRTC plane (phone -> bridge -> viewers) under /phone and /viewer.

In the source platform the client reports WebRTC receiver stats; in the MJPEG
baseline the client passes its measured bandwidth as a query parameter and the
server walks the same profile ladder. In the live plane the stats are real.

When no phone is connected, a built-in synthetic pump feeds XR frames into the
bridge automatically so the viewer works immediately."""

import asyncio
import io
import logging
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, Response, StreamingResponse

from app import config
from app.adaptive import AdaptiveStreamSession
from app.codecs import audio_codec, lossless, lossy, spatial_audio, video_codec
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
