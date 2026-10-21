"""Unit tests for Equirectangular 360° content workflow, projections, and trajectories."""

import pytest
from PIL import Image

from app.panoramic import (
    ProceduralPanoramicSource,
    equirect_roi,
    evaluate_panoramic_tiles,
    sphere_to_equirect,
    trajectory_central,
    trajectory_smooth_panning,
    trajectory_sudden_turn,
    validate_panoramic_dimensions,
)


def test_validate_panoramic_dimensions():
    # 2:1 is standard mono 360°
    meta_mono = validate_panoramic_dimensions(1920, 960)
    assert meta_mono.is_valid_equirectangular is True
    assert meta_mono.layout == "mono"
    assert meta_mono.aspect_ratio == 2.0

    # 4:1 is SBS stereo 360°
    meta_sbs = validate_panoramic_dimensions(3840, 960)
    assert meta_sbs.is_valid_equirectangular is True
    assert meta_sbs.layout == "stereo_sbs"

    # Arbitrary non-panoramic aspect ratio
    meta_bad = validate_panoramic_dimensions(1280, 720)
    assert meta_bad.is_valid_equirectangular is False
    assert meta_bad.layout == "unknown"


def test_sphere_to_equirect_mapping():
    w, h = 3600, 1800
    # Center (0, 0) should map to center of image
    cx, cy = sphere_to_equirect(0.0, 0.0, w, h)
    assert abs(cx - w // 2) <= 1
    assert abs(cy - h // 2) <= 1

    # North Pole (+90° pitch) should map to top row (y=0)
    _, py_north = sphere_to_equirect(0.0, 90.0, w, h)
    assert py_north == 0

    # South Pole (-90° pitch) should map to bottom row (y=h-1)
    _, py_south = sphere_to_equirect(0.0, -90.0, w, h)
    assert py_south == h - 1


def test_equirect_roi_bounds():
    w, h = 1920, 960
    roi = equirect_roi(w, h, yaw_deg=0.0, pitch_deg=0.0, hfov_deg=100.0, vfov_deg=60.0)
    x0, y0, x1, y1 = roi
    assert 0 <= x0 < x1 <= w
    assert 0 <= y0 < y1 <= h
    # Approximate width based on 100/360 fraction
    expected_w = int((100.0 / 360.0) * w)
    assert abs((x1 - x0) - expected_w) <= 2


def test_procedural_panoramic_source():
    src = ProceduralPanoramicSource(1920, 960)
    img = src.frame(t=0.5)
    assert isinstance(img, Image.Image)
    assert img.size == (1920, 960)
    assert img.mode == "RGB"


def test_repeatable_trajectories():
    # Central
    y0, p0 = trajectory_central(1.5)
    assert y0 == 0.0 and p0 == 0.0

    # Smooth panning
    y1, p1 = trajectory_smooth_panning(2.0, speed_dps=30.0)
    assert abs(y1 - (-120.0)) <= 0.1

    # Sudden saccade
    y_before, _ = trajectory_sudden_turn(0.5, saccade_time_s=1.0)
    y_after, _ = trajectory_sudden_turn(1.5, saccade_time_s=1.0)
    assert y_before == 0.0
    assert y_after == 90.0


def test_evaluate_panoramic_tiles_run():
    src = ProceduralPanoramicSource(960, 480)
    metrics = evaluate_panoramic_tiles(
        panoramic_source=src,
        trajectory_fn=trajectory_central,
        duration_s=0.2,  # 3 frames at 15 fps
        fps=15,
        codec="jpeg",
        allocation_strategy="intelligent_allocator",
    )
    assert metrics["frames_evaluated"] == 3
    assert metrics["mean_viewport_psnr_db"] > 0.0
    assert metrics["mean_viewport_ssim"] > 0.0
    assert metrics["mean_full_psnr_db"] > 0.0
    assert metrics["mean_boundary_mse"] >= 0.0
    assert metrics["mean_encode_time_ms"] > 0.0
    assert metrics["mean_decode_time_ms"] > 0.0


def test_panoramic_video_source_roundtrip(tmp_path):
    """Test generating a real 2:1 equirectangular MP4 video and streaming it."""
    from app.panoramic import PanoramicVideoSource, create_test_panoramic_video

    video_file = tmp_path / "test_360_stream.mp4"
    create_test_panoramic_video(video_file, width=640, height=320, fps=15, duration_s=0.5)
    assert video_file.exists()
    assert video_file.stat().st_size > 0

    with PanoramicVideoSource(video_file) as vsrc:
        assert vsrc.metadata.is_valid_equirectangular is True
        assert vsrc.metadata.layout == "mono"
        assert vsrc.metadata.aspect_ratio == 2.0
        assert vsrc.width == 640
        assert vsrc.height == 320

        # Frame extraction
        f0 = vsrc.frame(0.0)
        assert isinstance(f0, Image.Image)
        assert f0.size == (640, 320)

        # Run evaluation on real video source
        m = evaluate_panoramic_tiles(
            panoramic_source=vsrc,
            trajectory_fn=trajectory_smooth_panning,
            duration_s=0.3,
            fps=15,
            codec="jpeg",
            allocation_strategy="intelligent_allocator",
        )
        assert m["frames_evaluated"] > 0
        assert m["mean_viewport_psnr_db"] > 0.0
        assert m["mean_full_psnr_db"] > 0.0

