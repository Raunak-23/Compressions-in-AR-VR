"""Unit tests for ViewportPredictor: smooth pursuit, saccades, and confidence fallback."""

import pytest

from app.prediction import (
    ViewportPredictor,
    compute_viewport_iou,
    evaluate_prediction,
)
from app.viewport import gaze_roi


def test_predictor_smooth_pursuit_extrapolation():
    pred = ViewportPredictor(default_horizon_s=0.100)

    # Constant horizontal movement at 20 deg/s
    t = 0.0
    for i in range(10):
        pred.observe(t=t, yaw_deg=i * 2.0, pitch_deg=0.0)
        t += 0.100

    # Current yaw is 18.0 deg at t=0.9
    # Predict 100ms into the future at 20 deg/s -> expected ~20.0 deg
    res = pred.predict(horizon_s=0.100)
    assert not res.is_saccade
    assert res.confidence >= 0.50
    assert 18.5 <= res.predicted_yaw_deg <= 21.5
    assert res.fallback_used is False


def test_predictor_saccade_triggers_confidence_fallback():
    pred = ViewportPredictor(default_horizon_s=0.100, saccade_velocity_threshold=100.0)

    # Steady fixation at (0, 0)
    pred.observe(t=0.0, yaw_deg=0.0, pitch_deg=0.0)
    pred.observe(t=0.1, yaw_deg=0.0, pitch_deg=0.0)
    pred.observe(t=0.2, yaw_deg=0.0, pitch_deg=0.0)

    # Sudden high-speed saccadic jump: 30 degrees in 50ms = 600 deg/s!
    pred.observe(t=0.25, yaw_deg=30.0, pitch_deg=0.0)

    res = pred.predict(horizon_s=0.100)
    assert res.is_saccade is True
    # Extreme speed drops confidence and triggers fallback
    assert res.confidence < 0.50
    assert res.fallback_used is True
    # Predicted yaw falls back safely closer to current observation (30°) instead of wild overshoot
    assert abs(res.predicted_yaw_deg - 30.0) < 15.0


def test_compute_viewport_iou_identical_and_disjoint():
    box_a = (100, 100, 300, 300)
    assert compute_viewport_iou(box_a, box_a) == 1.0

    box_b = (500, 500, 700, 700)
    assert compute_viewport_iou(box_a, box_b) == 0.0

    box_c = (200, 100, 400, 300)  # half overlap
    iou = compute_viewport_iou(box_a, box_c)
    assert 0.30 < iou < 0.40


def test_evaluate_prediction_metrics():
    pred = ViewportPredictor(default_horizon_s=0.050)
    pred.observe(0.0, 0.0, 0.0)
    pred.observe(0.05, 5.0, 0.0)
    p_res = pred.predict(horizon_s=0.050)

    eval_res = evaluate_prediction(actual_yaw=10.0, actual_pitch=0.0, pred=p_res)
    assert "angular_error_deg" in eval_res
    assert "viewport_iou" in eval_res
    assert "processing_overhead_us" in eval_res
    assert eval_res["processing_overhead_us"] >= 0.0


def test_viewport_coverage_metric():
    """Verify compute_viewport_coverage correctly measures RoI pixel coverage by high/medium tiles."""
    from app.prediction import compute_viewport_coverage
    from app.viewport_tiles import TileInfo

    actual_roi = (100, 100, 300, 300)  # 200x200 = 40,000 px
    # Tile 1: covers (100, 100, 200, 300) = 100x200 = 20,000 px (50%)
    t1 = TileInfo(row=0, col=0, bbox=(100, 100, 200, 300), tier="high", quality_val=80,
                  encoded_bytes=b"", byte_count=10)
    # Tile 2: covers (200, 100, 300, 300) = 100x200 = 20,000 px (50%)
    t2 = TileInfo(row=0, col=1, bbox=(200, 100, 300, 300), tier="medium", quality_val=60,
                  encoded_bytes=b"", byte_count=10)
    # Tile 3: outside RoI, tier "low"
    t3 = TileInfo(row=1, col=0, bbox=(400, 400, 500, 500), tier="low", quality_val=20,
                  encoded_bytes=b"", byte_count=10)

    # Both high and medium cover RoI completely (1.0)
    cov_full = compute_viewport_coverage(actual_roi, [t1, t2, t3])
    assert cov_full == 1.0

    # Only high tier (50% coverage)
    cov_half = compute_viewport_coverage(actual_roi, [t1, t3])
    assert 0.49 <= cov_half <= 0.51

    # Only low tier (0% coverage)
    cov_zero = compute_viewport_coverage(actual_roi, [t3])
    assert cov_zero == 0.0


def test_compare_prediction_strategies_on_smooth_and_saccade():
    """Test comparing Strategy A (Current), B (Predicted unconstrained), and C (Fallback)."""
    from PIL import Image
    import numpy as np
    from app.prediction import compare_prediction_strategies

    frame = Image.fromarray(np.zeros((360, 640, 3), dtype=np.uint8))
    predictor = ViewportPredictor(default_horizon_s=0.100)

    # 1. Smooth movement: yaw moving at 20 deg/s
    t = 0.0
    for i in range(10):
        predictor.observe(t, yaw_deg=i * 2.0, pitch_deg=0.0)
        t += 0.100

    # Ground truth future at t+0.1 is 20.0 deg
    smooth_comp = compare_prediction_strategies(frame, predictor, actual_yaw=20.0, actual_pitch=0.0, horizon_s=0.100)
    assert "A_current_gaze" in smooth_comp
    assert "B_predicted_unconstrained" in smooth_comp
    assert "C_predicted_fallback" in smooth_comp
    # For smooth movement, prediction should have smaller angular error than current gaze
    assert smooth_comp["C_predicted_fallback"]["angular_error_deg"] < smooth_comp["A_current_gaze"]["angular_error_deg"]
    assert smooth_comp["C_predicted_fallback"]["worse_than_current"] is False

    # 2. Sudden saccade: abrupt 60° jump
    saccade_pred = ViewportPredictor(default_horizon_s=0.100)
    saccade_pred.observe(0.0, 0.0, 0.0)
    saccade_pred.observe(0.1, 0.0, 0.0)
    saccade_pred.observe(0.15, 45.0, 0.0)  # Huge sudden jump

    # Actual gaze at future target settles at 45.0 deg
    saccade_comp = compare_prediction_strategies(frame, saccade_pred, actual_yaw=45.0, actual_pitch=0.0, horizon_s=0.100)
    # Unconstrained extrapolation overshoots significantly, while fallback bounds error
    assert saccade_comp["C_predicted_fallback"]["angular_error_deg"] <= saccade_comp["B_predicted_unconstrained"]["angular_error_deg"]


def test_multi_horizon_prediction_scaling():
    """Verify predictions across 50ms, 100ms, 150ms, 200ms horizons."""
    pred = ViewportPredictor()
    pred.observe(0.0, 0.0, 0.0)
    pred.observe(0.1, 10.0, 0.0)  # 100 deg/s velocity

    p50 = pred.predict(horizon_s=0.050, apply_fallback=False)
    p100 = pred.predict(horizon_s=0.100, apply_fallback=False)
    p150 = pred.predict(horizon_s=0.150, apply_fallback=False)
    p200 = pred.predict(horizon_s=0.200, apply_fallback=False)

    # Monotonically increasing extrapolated offset
    assert p50.predicted_yaw_deg < p100.predicted_yaw_deg < p150.predicted_yaw_deg < p200.predicted_yaw_deg

