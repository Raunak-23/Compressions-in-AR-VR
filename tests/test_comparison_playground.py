"""Automated test suite for Before/After Quality Comparison and Video Compression Playground.

Covers:
1. File upload validation (supported formats, size bounds, empty/corrupt files).
2. Image compression methods (JPEG, H.264, 4×4 Viewport Tiles).
3. Video compression pipeline (H.264 libx264, M-JPEG) via PyAV.
4. Metric correctness (file size reduction %, compression ratio, PSNR, SSIM, bitrate).
5. Frame extraction and temporal alignment.
6. Target bitrate achievement logic and metric unavailability handling.
7. End-to-end FastAPI endpoint integration via TestClient.
"""

import io
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.comparison import (
    MAX_UPLOAD_BYTES,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    CompressionResult,
    VideoCompressionResult,
    _compute_metrics,
    _compute_video_quality_metrics,
    _resolve_resolution,
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
)
from app.media_source import SyntheticXRSource
from app.server import app

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_image():
    """Create a synthetic 320x240 RGB image for testing."""
    arr = np.zeros((240, 320, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 320, dtype=np.uint8)
    arr[:, :, 1] = np.linspace(0, 255, 240, dtype=np.uint8)[:, None]
    arr[:, :, 2] = 128
    return Image.fromarray(arr, "RGB")


@pytest.fixture(scope="module")
def sample_video_path(tmp_path_factory):
    """Create a small valid MP4 video (320x240, 15 frames, 15 fps = 1.0s)."""
    tmp_dir = tmp_path_factory.mktemp("video_test")
    file_path = str(tmp_dir / "sample_test.mp4")

    out_container = av.open(file_path, mode="w")
    stream = out_container.add_stream("libx264", rate=15)
    stream.width = 320
    stream.height = 240
    stream.pix_fmt = "yuv420p"

    for i in range(15):
        arr = np.full((240, 320, 3), (i * 15, 120, 255 - i * 15), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = i
        for pkt in stream.encode(frame):
            out_container.mux(pkt)

    for pkt in stream.encode(None):
        out_container.mux(pkt)

    out_container.close()
    return file_path


# --- 1. Validation Logic ---

def test_validate_upload_valid_formats():
    assert validate_upload("video.mp4", 1024 * 1024)[0] is True
    assert validate_upload("clip.webm", 500 * 1024)[0] is True
    assert validate_upload("movie.mkv", 2 * 1024 * 1024)[0] is True
    assert validate_upload("stream.avi", 1024 * 1024)[0] is True
    assert validate_upload("record.mov", 1024 * 1024)[0] is True
    assert validate_upload("frame.jpg", 100 * 1024)[0] is True
    assert validate_upload("picture.png", 200 * 1024)[0] is True
    assert validate_upload("image.webp", 80 * 1024)[0] is True


def test_validate_upload_invalid_formats():
    valid, err = validate_upload("script.py", 1000)
    assert valid is False
    assert "Unsupported format: .py" in err

    valid, err = validate_upload("binary.exe", 1000)
    assert valid is False

    valid, err = validate_upload("data.xyz", 1000)
    assert valid is False


def test_validate_upload_size_bounds():
    valid, err = validate_upload("empty.mp4", 0)
    assert valid is False
    assert "empty" in err.lower()

    valid, err = validate_upload("huge.mp4", MAX_UPLOAD_BYTES + 1)
    assert valid is False
    assert "exceeds limit" in err

    valid, err = validate_upload("", 1000)
    assert valid is False
    assert "empty" in err.lower()

    valid, err = validate_upload("noext", 1000)
    assert valid is False
    assert "no extension" in err.lower()


# --- 2. Image Compression Pipelines & Metrics ---

def test_compress_image_jpeg(sample_image):
    res_high = compress_image_jpeg(sample_image, quality=90, chroma_subsample=True)
    res_low = compress_image_jpeg(sample_image, quality=30, chroma_subsample=True)

    assert res_high.compressed_bytes > res_low.compressed_bytes
    assert res_high.psnr_db is not None and res_low.psnr_db is not None
    assert res_high.psnr_db > res_low.psnr_db
    assert res_high.ssim is not None and res_high.ssim > 0.8
    assert res_high.compression_ratio > 1.0
    assert 0.0 < res_high.size_reduction_pct < 100.0

    # Formula verification
    expected_pct = round(100.0 * (res_high.original_bytes - res_high.compressed_bytes) / res_high.original_bytes, 2)
    assert res_high.size_reduction_pct == expected_pct
    expected_ratio = round(res_high.original_bytes / res_high.compressed_bytes, 2)
    assert res_high.compression_ratio == expected_ratio


def test_compress_image_jpeg_chroma_subsampling(sample_image):
    res_420 = compress_image_jpeg(sample_image, quality=80, chroma_subsample=True)
    res_444 = compress_image_jpeg(sample_image, quality=80, chroma_subsample=False)

    assert "4:2:0" in res_420.method
    assert "4:4:4" in res_444.method
    # 4:4:4 retains more chroma data and is generally larger
    assert res_444.compressed_bytes >= res_420.compressed_bytes


def test_compress_image_h264(sample_image):
    res = compress_image_h264(sample_image, bitrate_kbps=1500)
    assert res.compressed_bytes > 0
    assert res.compressed_image_b64 is not None
    assert res.psnr_db is not None and res.psnr_db > 20.0
    assert res.ssim is not None and res.ssim > 0.8
    assert res.output_width == sample_image.width
    assert res.output_height == sample_image.height


def test_compress_image_tiles(sample_image):
    res = compress_image_tiles(
        sample_image,
        quality_high=85,
        quality_medium=55,
        quality_low=25,
        yaw_deg=5.0,
        pitch_deg=0.0,
    )
    assert res.compressed_bytes > 0
    assert res.psnr_db is not None
    assert res.ssim is not None
    assert "viewport_psnr_db" in res.extra
    assert "peripheral_psnr_db" in res.extra
    assert "boundary_artifact_mse" in res.extra


def test_custom_original_file_bytes(sample_image):
    uploaded_size = 50000
    res = compress_image_jpeg(sample_image, quality=75, original_file_bytes=uploaded_size)
    assert res.original_bytes == uploaded_size
    assert res.original_file_bytes == uploaded_size
    assert res.raw_bytes == len(sample_image.tobytes())
    expected_pct = round(100.0 * (uploaded_size - res.compressed_bytes) / uploaded_size, 2)
    assert res.size_reduction_pct == expected_pct


# --- 3. Video Compression Pipeline & Metrics ---

def test_resolve_resolution():
    # Original keeps aspect ratio and enforces even dimensions
    w, h = _resolve_resolution("original", 1280, 720)
    assert (w, h) == (1280, 720)

    w, h = _resolve_resolution("720p", 1920, 1080)
    assert (w, h) == (1280, 720)

    w, h = _resolve_resolution("360p", 1280, 720)
    assert (w, h) == (640, 360)


def test_compress_video_h264_end_to_end(sample_video_path):
    job_id = f"test_h264_{uuid.uuid4().hex[:6]}"
    res = compress_video_h264(
        input_path=sample_video_path,
        job_id=job_id,
        bitrate_kbps=800,
        target_fps=15,
        resolution="360p",
    )

    assert res.status == "completed"
    assert res.compressed_bytes > 0
    assert res.output_path is not None and os.path.isfile(res.output_path)
    assert res.original_bytes == os.path.getsize(sample_video_path)
    assert res.output_width == 480
    assert res.output_height == 360
    assert res.output_fps == 15
    assert res.output_duration_s > 0
    assert res.avg_bitrate_kbps is not None
    assert res.avg_psnr_db is not None and res.avg_psnr_db > 25.0
    assert res.avg_ssim is not None and res.avg_ssim > 0.8
    assert res.target_bitrate_kbps == 800
    assert res.target_met is not None

    # Cleanup
    cleanup_job(job_id)
    assert get_job_status(job_id) is None


def test_compress_video_mjpeg_end_to_end(sample_video_path):
    job_id = f"test_mjpeg_{uuid.uuid4().hex[:6]}"
    res = compress_video_mjpeg(
        input_path=sample_video_path,
        job_id=job_id,
        quality=70,
        target_fps=15,
        resolution="original",
    )

    assert res.status == "completed"
    assert res.compressed_bytes > 0
    assert res.output_path is not None and os.path.isfile(res.output_path)
    assert res.original_bytes == os.path.getsize(sample_video_path)
    assert res.output_width == 320
    assert res.output_height == 240
    assert res.avg_bitrate_kbps is not None

    cleanup_job(job_id)


def test_extract_frame_from_video(sample_video_path):
    frame_0 = extract_frame_from_video(sample_video_path, time_s=0.0)
    assert frame_0 is not None
    assert frame_0.size == (320, 240)

    frame_mid = extract_frame_from_video(sample_video_path, time_s=0.5)
    assert frame_mid is not None
    assert frame_mid.size == (320, 240)

    # Invalid file returns None
    assert extract_frame_from_video("non_existent_file.mp4") is None


def test_corrupted_video_handling():
    corrupted_path = str(get_temp_dir() / "corrupted.mp4")
    with open(corrupted_path, "wb") as f:
        f.write(b"NOT A REAL VIDEO FILE CONTENT 1234567890")

    job_id = f"test_corrupt_{uuid.uuid4().hex[:6]}"
    res = compress_video_h264(
        input_path=corrupted_path,
        job_id=job_id,
        bitrate_kbps=1000,
    )
    assert res.status == "failed"
    assert res.error is not None

    if os.path.exists(corrupted_path):
        os.unlink(corrupted_path)


# --- 4. API Endpoints Integration (FastAPI TestClient) ---

def test_page_endpoints():
    r1 = client.get("/compare")
    assert r1.status_code == 200
    assert "Quality Comparison" in r1.text
    assert "sliderContainer" in r1.text

    r2 = client.get("/playground")
    assert r2.status_code == 200
    assert "Video Compression Playground" in r2.text
    assert "uploadZone" in r2.text

    r3 = client.get("/viewer")
    assert r3.status_code == 200
    assert "Quality Comparison" in r3.text  # In top-nav


def test_api_compression_methods():
    resp = client.get("/api/compression-methods")
    assert resp.status_code == 200
    data = resp.json()
    assert "methods" in data
    ids = [m["id"] for m in data["methods"]]
    assert "jpeg" in ids
    assert "h264" in ids
    assert "tiles" in ids


def test_api_video_compression_methods():
    resp = client.get("/api/video-compression-methods")
    assert resp.status_code == 200
    data = resp.json()
    assert "methods" in data
    ids = [m["id"] for m in data["methods"]]
    assert "h264" in ids
    assert "jpeg_frames" in ids


def test_api_sample_frame():
    resp = client.get("/api/sample-frame?t=0.5")
    assert resp.status_code == 200
    data = resp.json()
    assert "image_b64" in data
    assert data["width"] > 0
    assert data["height"] > 0


def test_api_compress_image_synthetic():
    resp = client.post(
        "/api/compress-image",
        data={"method": "jpeg", "quality": 80, "source_type": "synthetic", "t": 1.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "original_image_b64" in data
    assert "result" in data
    assert data["result"]["psnr_db"] > 25.0
    assert data["result"]["compression_ratio"] > 1.0


def test_api_compress_image_uploaded_image(sample_image):
    buf = io.BytesIO()
    sample_image.save(buf, "PNG")
    png_bytes = buf.getvalue()

    resp = client.post(
        "/api/compress-image",
        data={"method": "jpeg", "quality": 85, "source_type": "upload"},
        files={"file": ("test.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["original_bytes"] == len(png_bytes)
    assert data["result"]["psnr_db"] > 25.0


def test_api_compress_image_uploaded_video(sample_video_path):
    with open(sample_video_path, "rb") as f:
        vid_bytes = f.read()

    resp = client.post(
        "/api/compress-image",
        data={"method": "jpeg", "quality": 75, "source_type": "upload", "t": 0.5},
        files={"file": ("sample.mp4", vid_bytes, "video/mp4")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "original_image_b64" in data
    assert data["result"]["psnr_db"] > 20.0


def test_api_upload_video_and_workflow(sample_video_path):
    with open(sample_video_path, "rb") as f:
        vid_bytes = f.read()

    # 1. Upload
    up_resp = client.post(
        "/api/upload-video",
        files={"file": ("sample_clip.mp4", vid_bytes, "video/mp4")},
    )
    assert up_resp.status_code == 200
    meta = up_resp.json()
    assert "job_id" in meta
    job_id = meta["job_id"]
    assert meta["width"] == 320
    assert meta["height"] == 240

    # 2. Status initially uploaded
    stat_resp = client.get(f"/api/job-status/{job_id}")
    assert stat_resp.status_code == 200
    assert stat_resp.json()["status"] == "uploaded"

    # 3. Compress video
    comp_resp = client.post(
        "/api/compress-video",
        data={"job_id": job_id, "method": "h264", "bitrate_kbps": 600, "fps": 15, "resolution": "original"},
    )
    assert comp_resp.status_code == 200
    res_data = comp_resp.json()
    assert res_data["status"] == "completed"
    assert res_data["compressed_bytes"] > 0

    # 4. Extract video frames
    frame_orig = client.get(f"/api/video-frame/{job_id}?which=original&time_s=0.2")
    assert frame_orig.status_code == 200
    assert "image_b64" in frame_orig.json()

    frame_comp = client.get(f"/api/video-frame/{job_id}?which=compressed&time_s=0.2")
    assert frame_comp.status_code == 200
    assert "image_b64" in frame_comp.json()

    # 5. Dual compare-video-frames endpoint
    comp_frames = client.get(f"/api/compare-video-frames/{job_id}?time_s=0.2")
    assert comp_frames.status_code == 200
    cf_data = comp_frames.json()
    assert "original_image_b64" in cf_data
    assert "compressed_image_b64" in cf_data
    assert cf_data["frame_psnr_db"] > 20.0
    assert cf_data["frame_ssim"] > 0.8

    # 6. Serve video
    serve_resp = client.get(f"/api/serve-video/{job_id}?which=compressed")
    assert serve_resp.status_code == 200
    assert serve_resp.headers["content-type"].startswith("video/")

    # 7. Download
    dl_resp = client.get(f"/api/download/{job_id}")
    assert dl_resp.status_code == 200

    # 8. Cleanup
    del_resp = client.delete(f"/api/cleanup/{job_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["ok"] is True
