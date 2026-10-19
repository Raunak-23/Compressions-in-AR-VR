"""Unit tests comparing the intelligent tile allocator against classical baselines."""

import pytest
from PIL import Image

from app.media_source import SyntheticXRSource
from app.viewport_tiles import (
    compute_tile_importance_and_probability,
    encode_and_reconstruct_tiles,
)


@pytest.fixture
def test_frame():
    source = SyntheticXRSource(1280, 720)
    return source.frame(1.0)


def test_importance_weights_and_view_probabilities():
    roi = (400, 200, 880, 520)  # Central ROI
    weights, probs, cols, rows = compute_tile_importance_and_probability(
        1280, 720, roi, rows=4, cols=4
    )
    assert len(weights) == 16
    assert len(probs) == 16

    # Probabilities must sum to 1.0
    total_prob = sum(probs.values())
    assert abs(total_prob - 1.0) < 0.01

    # Central tiles (1, 1), (1, 2), (2, 1), (2, 2) must have higher importance than corners (0, 0), (3, 3)
    center_w = weights[(1, 1)]
    corner_w = weights[(0, 0)]
    assert center_w > corner_w * 2.0


def test_five_allocation_strategies(test_frame):
    # Condition 1: uniform
    res_uniform = encode_and_reconstruct_tiles(
        test_frame,
        allocation_strategy="uniform",
    )
    assert res_uniform.allocation_strategy == "uniform"
    quals = {t.quality_val for t in res_uniform.tiles}
    assert len(quals) == 1  # exactly uniform quality

    # Condition 2: fixed_tiers
    res_fixed = encode_and_reconstruct_tiles(
        test_frame,
        allocation_strategy="fixed_tiers",
    )
    assert res_fixed.allocation_strategy == "fixed_tiers"
    assert any(t.tier == "high" for t in res_fixed.tiles)
    assert any(t.tier == "low" for t in res_fixed.tiles)

    # Condition 3: viewport_only (unconstrained budget)
    res_vonly = encode_and_reconstruct_tiles(
        test_frame,
        allocation_strategy="viewport_only",
        budget_bytes=10_000,  # Should ignore budget in viewport_only
    )
    assert res_vonly.allocation_strategy == "viewport_only"

    # Condition 4: bandwidth_only
    res_bonly = encode_and_reconstruct_tiles(
        test_frame,
        allocation_strategy="bandwidth_only",
        budget_bytes=25_000,
    )
    assert res_bonly.allocation_strategy == "bandwidth_only"

    # Condition 5: intelligent_allocator (greedy knapsack RDO)
    target_budget = 35_000
    res_intel = encode_and_reconstruct_tiles(
        test_frame,
        allocation_strategy="intelligent_allocator",
        budget_bytes=target_budget,
    )
    assert res_intel.allocation_strategy == "intelligent_allocator"
    assert res_intel.total_bytes <= target_budget * 1.05  # within or tightly bounded
    assert len(res_intel.tile_weights) == 16
    assert len(res_intel.view_probabilities) == 16

    # Viewport tiles should receive higher quality than peripheral corner tiles
    high_tiles = [t for t in res_intel.tiles if t.importance_weight > 0.6]
    low_tiles = [t for t in res_intel.tiles if t.importance_weight < 0.2]
    if high_tiles and low_tiles:
        mean_high_q = sum(t.quality_val for t in high_tiles) / len(high_tiles)
        mean_low_q = sum(t.quality_val for t in low_tiles) / len(low_tiles)
        assert mean_high_q >= mean_low_q
