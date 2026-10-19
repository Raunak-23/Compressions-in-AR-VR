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
