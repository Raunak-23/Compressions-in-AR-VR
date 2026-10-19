"""Unit and regression tests for the Intelligent Bitrate-Budget Controller."""

import pytest

from app.adaptive import (
    AdaptiveStreamSession,
    BandwidthEstimator,
    IntelligentBudgetController,
    pick_profile,
)
from app.media_source import SyntheticXRSource


def test_bandwidth_estimator_volatility_and_safe_kbps():
    est = BandwidthEstimator(alpha=0.3)
    assert est.kbps == 0.0
    assert est.std_kbps == 0.0

    # Stable feed: 2000 kbps repeated
    for _ in range(5):
        est.update(250_000, 1.0)  # 2000 kbps
    assert 1900 < est.kbps < 2100
    assert est.volatility_ratio < 0.15

    # High jitter feed: alternating 500 kbps and 4000 kbps
    for i in range(10):
        bytes_val = 500_000 if i % 2 == 0 else 62_500
        est.update(bytes_val, 1.0)
    assert est.std_kbps > 200.0
    assert est.volatility_ratio > 0.10
    # Safe bandwidth must be strictly lower than raw EWMA when volatile
    assert est.safe_kbps() < est.kbps


def test_controller_fast_downswitch_on_sudden_congestion():
    controller = IntelligentBudgetController(initial_profile="ultra")
    assert controller.current_profile == "ultra"

    # Sudden drop from ultra (> 3500 kbps) down to 300 kbps
    decision = controller.update(bytes_sent=37_500, seconds=1.0)  # 300 kbps
    assert decision.action == "downswitch"
    assert controller.current_profile == "low"
    assert "dropped_below_floor" in decision.reason


def test_controller_hysteresis_prevents_premature_upswitch():
    # Start at low profile with a 4-frame dwell requirement
    controller = IntelligentBudgetController(initial_profile="low", upswitch_dwell_frames=4)

    # 1. First high bandwidth sample arrives: 2000 kbps (enough for high profile)
    d1 = controller.update(bytes_sent=250_000, seconds=1.0)
    # Must NOT immediately jump to high; should stay dwelling or incrementing
    assert d1.action in ("maintain", "initial")
    assert controller.current_profile == "low"
    assert "dwelling" in d1.reason or "headroom" in d1.reason

    # 2. Feed sustained high bandwidth for remaining dwell frames
    for _ in range(6):
        controller.update(bytes_sent=250_000, seconds=1.0)

    # Sustained stability triggers confirmed upswitch
    assert controller.current_profile in ("medium", "high")


def test_controller_latency_budget_throttles():
    controller = IntelligentBudgetController(initial_profile="high")
    fps = controller.knobs["fps"]  # 30 fps -> latency budget is ~33.3ms

    # Simulate excessive processing delay (e.g. 50ms > 33ms)
    decision = controller.update(
        bytes_sent=250_000,
        seconds=1.0,
        processing_delay_s=0.050,  # 50 ms
        dropped_frames=0,
    )
    assert decision.action == "throttle_for_latency"
    assert controller.current_profile != "ultra"


def test_budget_auditing_never_falsifies_overshoot():
    controller = IntelligentBudgetController(initial_profile="low")
    frame_budget = controller.frame_budget_bytes()

    # Case A: Within budget
    rep_under = controller.audit_frame(actual_bytes=frame_budget - 100)
    assert rep_under.within_budget is True
    assert rep_under.overshoot_bytes == 0
    assert rep_under.overshoot_pct == 0.0

    # Case B: Exceeds budget
    overshoot_amt = 500
    rep_over = controller.audit_frame(actual_bytes=frame_budget + overshoot_amt)
    assert rep_over.within_budget is False
    assert rep_over.overshoot_bytes == overshoot_amt
    assert rep_over.overshoot_pct > 0.0
    # Measured bitrate correctly scales from actual bytes
    assert rep_over.measured_frame_bitrate_kbps > rep_under.measured_frame_bitrate_kbps


def test_adaptive_stream_session_end_to_end():
    session = AdaptiveStreamSession("medium")
    source = SyntheticXRSource(1280, 720)

    res, audit = session.encode_frame_tiled(
        source=source,
        t=1.0,
        allocation_strategy="intelligent_allocator",
    )
    assert res.total_bytes > 0
    assert len(res.tiles) == 16
    assert "within_budget" in audit
    assert "measured_frame_bitrate_kbps" in audit
    assert res.allocation_strategy == "intelligent_allocator"
