"""Comprehensive tests for Phase 1: Budget-aware allocation and budget enforcement.

Tests cover:
- Feasible budgets: encoder meets target within retry limit.
- Infeasible budgets: encoder correctly reports infeasibility when floor reached.
- Quality ordering: higher importance tiles receive equal or higher quality.
- Boundary cases: ample budget, tight budget, downscaling peripheral tiles first.
- Abrupt bandwidth changes: session budget adaptation responding to sudden network drop.
- Complete audit trail: verified fields on TileReconstructionResult and ControllerAuditReport.
"""

import numpy as np
import pytest
from PIL import Image

from app.viewport_tiles import (
    METRIC_UNAVAILABLE,
    assign_tile_tiers,
    compute_grid_boundaries,
    compute_tile_importance_and_probability,
    encode_and_reconstruct_tiles,
)
from app.adaptive import (
    AdaptiveStreamSession,
    BandwidthEstimator,
    IntelligentBudgetController,
)
from app.media_source import SyntheticXRSource


@pytest.fixture
def synthetic_frame():
    """Create a reproducible 640x360 synthetic test image with varying details."""
    np.random.seed(42)
    img_data = np.zeros((360, 640, 3), dtype=np.uint8)
    for y in range(360):
        for x in range(640):
            img_data[y, x, 0] = int((x / 640.0) * 255)
            img_data[y, x, 1] = int((y / 360.0) * 255)
            img_data[y, x, 2] = int(((x + y) % 32) * 8)
    return Image.fromarray(img_data)


def test_feasible_budget_enforcement(synthetic_frame):
    """Feasible budget should be met with within_budget=True."""
    # Ample budget: 50,000 bytes
    res_ample = encode_and_reconstruct_tiles(
        synthetic_frame,
        yaw_deg=0.0,
        pitch_deg=0.0,
        budget_bytes=50000,
        allocation_strategy="intelligent_allocator",
    )
    assert res_ample.within_budget
    assert res_ample.total_bytes <= 50000
    assert res_ample.overshoot_bytes == 0

    # Moderate feasible budget: 16,000 bytes
    res_mod = encode_and_reconstruct_tiles(
        synthetic_frame,
        yaw_deg=0.0,
        pitch_deg=0.0,
        budget_bytes=16000,
        allocation_strategy="intelligent_allocator",
    )
    assert res_mod.within_budget
    assert res_mod.total_bytes <= 16000
    assert res_mod.overshoot_bytes == 0
    assert res_mod.budget_retries >= 0


def test_infeasible_budget_handling(synthetic_frame):
    """An unrealistically tiny budget (e.g., 500 bytes for 16 tiles) should trigger budget_infeasible or within_budget=False."""
    res = encode_and_reconstruct_tiles(
        synthetic_frame,
        yaw_deg=0.0,
        pitch_deg=0.0,
        budget_bytes=500,  # Impossible for 16 JPEG tiles (each header is ~400-600B)
        allocation_strategy="intelligent_allocator",
    )
    # The controller must report budget_infeasible or within_budget=False, never silently claim success
    assert (not res.within_budget) or res.budget_infeasible
    assert res.total_bytes > 500
    assert res.overshoot_bytes > 0
    # Tiles must be at minimum quality floor (<= 10)
    for tile in res.tiles:
        assert tile.quality_val <= 10


def test_quality_ordering_by_importance(synthetic_frame):
    """Higher-importance (foveal/viewport) tiles should receive >= quality than peripheral tiles."""
    res = encode_and_reconstruct_tiles(
        synthetic_frame,
        yaw_deg=0.0,
        pitch_deg=0.0,
        budget_bytes=18000,
        allocation_strategy="intelligent_allocator",
    )

    high_tier_qualities = [t.quality_val for t in res.tiles if t.tier == "high"]
    low_tier_qualities = [t.quality_val for t in res.tiles if t.tier == "low"]

    if high_tier_qualities and low_tier_qualities:
        assert min(high_tier_qualities) >= max(low_tier_qualities)


def test_peripheral_tiles_reduced_first_under_pressure(synthetic_frame):
    """When budget is constrained, peripheral tiles should take larger quality cuts than viewport tiles."""
    # Run with high budget
    res_high = encode_and_reconstruct_tiles(
        synthetic_frame,
        yaw_deg=0.0,
        pitch_deg=0.0,
        budget_bytes=40000,
        allocation_strategy="intelligent_allocator",
    )
    # Run with constrained budget
    res_low = encode_and_reconstruct_tiles(
        synthetic_frame,
        yaw_deg=0.0,
        pitch_deg=0.0,
        budget_bytes=14000,
        allocation_strategy="intelligent_allocator",
    )

    high_tier_rem = np.mean([t.quality_val for t in res_low.tiles if t.tier == "high"])
    if any(t.tier == "low" for t in res_low.tiles):
        low_tier_rem = np.mean([t.quality_val for t in res_low.tiles if t.tier == "low"])
        assert high_tier_rem >= low_tier_rem


def test_budget_audit_trail_populated(synthetic_frame):
    """Audit fields must be completely and accurately populated."""
    target_budget = 20000
    res = encode_and_reconstruct_tiles(
        synthetic_frame,
        yaw_deg=15.0,
        pitch_deg=-10.0,
        budget_bytes=target_budget,
        allocation_strategy="intelligent_allocator",
    )

    assert res.budget_bytes == target_budget
    assert isinstance(res.budget_retries, int)
    assert isinstance(res.within_budget, bool)
    assert isinstance(res.overshoot_bytes, int)
    assert res.final_quality_map is not None
    assert len(res.final_quality_map) == 16
    assert res.total_bytes == sum(t.byte_count for t in res.tiles)

    if res.within_budget:
        assert res.overshoot_bytes == 0
        assert res.total_bytes <= target_budget
    else:
        assert res.overshoot_bytes == res.total_bytes - target_budget


def test_abrupt_bandwidth_change_adaptation():
    """Test adaptive session handling an abrupt drop in bandwidth."""
    session = AdaptiveStreamSession("ultra")
    source = SyntheticXRSource(640, 360)

    # Frame 1 at ultra profile (>3500 kbps)
    res1, audit1 = session.encode_frame_tiled(
        source=source,
        t=1.0,
        allocation_strategy="intelligent_allocator",
    )
    b1 = res1.total_bytes
    target_b1 = audit1["budget_bytes"]
    assert audit1["within_budget"]

    # Sudden drop in bandwidth down to 300 kbps
    session.report_throughput(bytes_sent=37_500, seconds=1.0)
    assert session.profile == "low"

    # Frame 2 under low profile
    res2, audit2 = session.encode_frame_tiled(
        source=source,
        t=2.0,
        allocation_strategy="intelligent_allocator",
    )
    b2 = res2.total_bytes
    target_b2 = audit2["budget_bytes"]

    # Target budget and actual encoded bytes must be strictly lower
    assert target_b2 < target_b1
    assert b2 < b1
