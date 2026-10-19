"""Tests for 4x4 viewport-aware tile compression and reconstruction."""

import numpy as np
import pytest
from PIL import Image

from app.media_source import SyntheticXRSource
from app.viewport_tiles import (
    assign_tile_tiers,
    compute_grid_boundaries,
    encode_and_reconstruct_tiles,
)


@pytest.fixture
def frame():
    source = SyntheticXRSource(1280, 720)
    return source.frame(1.0)


def test_grid_boundaries_gapless():
    # 1280x720 divisible by 4
    col_bounds, row_bounds = compute_grid_boundaries(1280, 720, 4, 4)
    assert len(col_bounds) == 5
    assert len(row_bounds) == 5
    assert col_bounds == [0, 320, 640, 960, 1280]
    assert row_bounds == [0, 180, 360, 540, 720]

    # Non-divisible dimensions (e.g. 1283 x 719)
    cols_odd, rows_odd = compute_grid_boundaries(1283, 719, 4, 4)
    assert cols_odd[0] == 0 and cols_odd[-1] == 1283
    assert rows_odd[0] == 0 and rows_odd[-1] == 719
    # Gapless check
    for i in range(4):
        assert cols_odd[i] < cols_odd[i + 1]
        assert rows_odd[i] < rows_odd[i + 1]


def test_invalid_grid_parameters():
    with pytest.raises(ValueError, match="Dimensions"):
        compute_grid_boundaries(-10, 720, 4, 4)
    with pytest.raises(ValueError, match="rows and cols"):
        compute_grid_boundaries(1280, 720, 0, 4)


def test_centre_and_offcentre_gaze_tiers():
    # Central gaze: roi in center
    roi_centre = (416, 234, 864, 486)
    tiers, _, _ = assign_tile_tiers(1280, 720, roi_centre, 4, 4)
    assert len(tiers) == 16
    # Center tiles (1,1), (1,2), (2,1), (2,2) should be high
    for r in (1, 2):
        for c in (1, 2):
            assert tiers[(r, c)] == "high"
    # Corners are guard band or low
    high_count = sum(1 for v in tiers.values() if v == "high")
    med_count = sum(1 for v in tiers.values() if v == "medium")
    low_count = sum(1 for v in tiers.values() if v == "low")
    assert high_count > 0
    assert med_count > 0
    assert high_count + med_count + low_count == 16

    # Far corner gaze: top-left (0, 0, 448, 252)
    roi_corner = (0, 0, 448, 252)
    tiers_corner, _, _ = assign_tile_tiers(1280, 720, roi_corner, 4, 4)
    assert tiers_corner[(0, 0)] == "high"
    # Opposite corner (3, 3) must be peripheral "low"
    assert tiers_corner[(3, 3)] == "low"


def test_tile_encoding_and_reconstruction(frame):
    res = encode_and_reconstruct_tiles(
        frame,
        yaw_deg=0.0,
        pitch_deg=0.0,
        quality_ladder={"high": 90, "medium": 60, "low": 20},
        codec="jpeg",
    )

    # 1. Correct tile count and dimensions
    assert len(res.tiles) == 16
    assert res.grid == (4, 4)
    assert res.reconstructed.size == (1280, 720)
    assert res.reconstructed.mode == "RGB"

    # 2. No missing pixels (valid image data)
    rec_arr = np.asarray(res.reconstructed)
    assert rec_arr.shape == (720, 1280, 3)
    assert np.any(rec_arr > 0)

    # 3. Timing and bytes recorded
    assert res.total_bytes > 0
    assert res.encode_time_s > 0
    assert res.decode_time_s > 0

    # 4. Viewport quality exceeds peripheral quality
    assert res.viewport_psnr_db > res.peripheral_psnr_db
    assert res.viewport_ssim >= res.peripheral_ssim


def test_tile_encoding_offcentre_yaw_pitch(frame):
    res = encode_and_reconstruct_tiles(
        frame,
        yaw_deg=25.0,
        pitch_deg=-15.0,
        codec="jpeg",
    )
    assert res.reconstructed.size == (1280, 720)
    assert res.viewport_psnr_db > res.peripheral_psnr_db


def test_tile_encoding_non_standard_size():
    # Odd non-divisible size
    small_img = Image.new("RGB", (643, 361), (120, 60, 200))
    res = encode_and_reconstruct_tiles(small_img, yaw_deg=0.0, codec="jpeg")
    assert res.reconstructed.size == (643, 361)


def test_tile_encoding_h264_codec(frame):
    res = encode_and_reconstruct_tiles(
        frame,
        yaw_deg=0.0,
        pitch_deg=0.0,
        codec="h264",
        base_bitrate_kbps=1500,
    )
    assert len(res.tiles) == 16
    assert res.reconstructed.size == (1280, 720)
    assert res.total_bytes > 0


def test_tile_budget_enforcement_and_retries(frame):
    # Unconstrained encoding baseline
    unconstrained = encode_and_reconstruct_tiles(
        frame,
        quality_ladder={"high": 90, "medium": 70, "low": 50},
        codec="jpeg",
        budget_bytes=None,
    )

    # Set budget to 60% of unconstrained size
    target_budget = int(unconstrained.total_bytes * 0.6)
    budgeted = encode_and_reconstruct_tiles(
        frame,
        quality_ladder={"high": 90, "medium": 70, "low": 50},
        codec="jpeg",
        budget_bytes=target_budget,
        max_retries=3,
        quality_step=10,
    )

    # Must attempt retries
    assert budgeted.budget_retries > 0
    # Encoded bytes must be strictly lower than unconstrained
    assert budgeted.total_bytes < unconstrained.total_bytes
    assert budgeted.budget_bytes == target_budget


def test_guard_band_and_boundary_metrics(frame):
    res = encode_and_reconstruct_tiles(
        frame,
        yaw_deg=0.0,
        pitch_deg=0.0,
        quality_ladder={"high": 85, "medium": 60, "low": 30},
        codec="jpeg",
    )
    # Guard band quality is measured separately from viewport and peripheral
    assert res.guard_band_psnr_db > 0.0
    assert res.guard_band_ssim > 0.0
    assert res.viewport_psnr_db > 0.0
    assert res.peripheral_psnr_db > 0.0
    # Verify all tiers are present in the resulting tile breakdown
    assert any(t.tier == "high" for t in res.tiles)
    assert any(t.tier == "medium" for t in res.tiles)
    assert any(t.tier == "low" for t in res.tiles)
    # Boundary artifact metric is computed (non-negative MSE)
    assert res.boundary_artifact_mse >= 0.0


def test_adaptive_stream_session_budget_control():
    from app.adaptive import AdaptiveStreamSession
    source = SyntheticXRSource(1280, 720)
    session = AdaptiveStreamSession("low")  # low profile: tight budget

    res, report = session.encode_frame_tiled(
        source,
        t=1.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
        enforce_budget=True,
        max_retries=3,
    )
    assert "budget_bytes" in report
    assert "nominal_codec_bitrate_kbps" in report
    assert "measured_frame_bitrate_kbps" in report
    assert "budget_retries" in report
    assert report["fps"] > 0
    assert report["budget_bytes"] > 0

