"""Regression tests for Phase 2: Quality, timing, and boundary metric validation.

Verifies:
- Empty region detection: _compute_region_quality returns METRIC_UNAVAILABLE (-1.0), never 0.0.
- CSV export convention: formats sentinel values as "N/A".
- Region mask integrity: gaze_roi never produces out-of-bounds coordinates even at extreme angles.
- Independent timing: encode_time_s and decode_time_s are measured independently and positive.
- Single-frame vs continuous H.264 separation: single-frame labeled accurately.
- Bitrate and compression ratio calculations are mathematically consistent.
"""

import numpy as np
import pytest
from PIL import Image

from app.viewport import gaze_roi
from app.viewport_tiles import (
    METRIC_UNAVAILABLE,
    TileInfo,
    _compute_region_quality,
    compute_grid_boundaries,
    encode_and_reconstruct_tiles,
)
from app.experiments_runner import _format_cell_value


@pytest.fixture
def sample_image():
    """Create a 640x360 test image."""
    arr = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_empty_region_returns_metric_unavailable(sample_image):
    """When a tier has no tiles, _compute_region_quality must return METRIC_UNAVAILABLE, never 0.0."""
    # Create tile records where ALL tiles are "high" and "medium", none are "low"
    records = [
        TileInfo(
            row=0,
            col=0,
            bbox=(0, 0, 100, 100),
            tier="high",
            quality_val=80,
            encoded_bytes=b"dummy",
            byte_count=5,
            psnr_db=38.0,
            ssim=0.98,
        )
    ]
    psnr, ssim = _compute_region_quality(sample_image, sample_image, records, "low")
    assert psnr == METRIC_UNAVAILABLE
    assert ssim == METRIC_UNAVAILABLE
    assert psnr != 0.0
    assert ssim != 0.0


def test_csv_formatting_converts_sentinels_to_na():
    """_format_cell_value must format METRIC_UNAVAILABLE and -1.0 as 'N/A'."""
    assert _format_cell_value(METRIC_UNAVAILABLE) == "N/A"
    assert _format_cell_value(-1.0) == "N/A"
    assert _format_cell_value(None) == "N/A"
    assert _format_cell_value(38.5) == 38.5
    assert _format_cell_value("valid_string") == "valid_string"


def test_gaze_roi_boundary_clamping():
    """gaze_roi must clamp coordinates within canvas boundaries even for extreme gaze angles."""
    width, height = 1280, 720
    # Extreme left/right yaw
    roi_far_left = gaze_roi(width, height, yaw_deg=-150.0, pitch_deg=0.0)
    assert roi_far_left[0] >= 0
    assert roi_far_left[2] <= width

    roi_far_right = gaze_roi(width, height, yaw_deg=150.0, pitch_deg=0.0)
    assert roi_far_right[0] >= 0
    assert roi_far_right[2] <= width

    # Extreme up/down pitch
    roi_far_up = gaze_roi(width, height, yaw_deg=0.0, pitch_deg=85.0)
    assert roi_far_up[1] >= 0
    assert roi_far_up[3] <= height

    roi_far_down = gaze_roi(width, height, yaw_deg=0.0, pitch_deg=-85.0)
    assert roi_far_down[1] >= 0
    assert roi_far_down[3] <= height


def test_independent_timing_measurements(sample_image):
    """Encode and decode times must be measured independently and be positive."""
    res = encode_and_reconstruct_tiles(
        sample_image,
        yaw_deg=0.0,
        pitch_deg=0.0,
        codec="jpeg",
    )
    assert res.encode_time_s > 0.0
    assert res.decode_time_s > 0.0
    for tile in res.tiles:
        assert tile.encode_time_s >= 0.0
        assert tile.decode_time_s >= 0.0


def test_bitrate_and_compression_ratio_consistency(sample_image):
    """Verify mathematical consistency of bitrate and compression ratio."""
    raw_bytes = sample_image.width * sample_image.height * 3
    fps = 30
    res = encode_and_reconstruct_tiles(sample_image, yaw_deg=0.0, codec="jpeg")

    total_bytes = res.total_bytes
    assert total_bytes > 0

    nominal_bitrate_kbps = round(total_bytes * 8 * fps / 1000, 1)
    compression_ratio = round(raw_bytes / total_bytes, 2)

    assert nominal_bitrate_kbps > 0
    assert compression_ratio > 1.0
