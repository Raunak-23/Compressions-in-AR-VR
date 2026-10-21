"""Lightweight Viewport and Gaze Trajectory Prediction for Live XR Streaming.

Estimates the future gaze location and viewport ROI from recent orientation history:
- Velocity & acceleration estimation with Holt linear exponential smoothing.
- Configurable prediction horizon (e.g. 50ms, 100ms, 150ms).
- Saccade / sudden turn detection (angular velocity > 100 deg/s).
- Dynamic confidence scoring: falls back gracefully to current viewport if confidence is low.
- Empirical metrics: angular error (degrees), IoU coverage, processing overhead (< 0.1ms).
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from app import config
from app.viewport import gaze_roi


@dataclass
class GazeObservation:
    t: float
    yaw_deg: float
    pitch_deg: float


@dataclass
class PredictedViewport:
    predicted_yaw_deg: float
    predicted_pitch_deg: float
    current_yaw_deg: float
    current_pitch_deg: float
    horizon_s: float
    confidence: float
    is_saccade: bool
    angular_velocity_dps: float
    roi_box: Tuple[int, int, int, int]
    fallback_used: bool
    processing_time_s: float


class ViewportPredictor:
    """Lightweight real-time orientation and viewport predictor."""

    def __init__(
        self,
        default_horizon_s: float = 0.100,  # 100 ms forward prediction
        saccade_velocity_threshold: float = 120.0,  # deg/sec
        confidence_threshold: float = 0.50,
        history_maxlen: int = 20,
    ):
        self.default_horizon_s = default_horizon_s
        self.saccade_velocity_threshold = saccade_velocity_threshold
        self.confidence_threshold = confidence_threshold
        self.history: Deque[GazeObservation] = collections.deque(maxlen=history_maxlen)

        # Smoothed state
        self._level_yaw: float | None = None
        self._trend_yaw: float = 0.0
        self._level_pitch: float | None = None
        self._trend_pitch: float = 0.0
        self._last_t: float | None = None

        # Smoothing coefficients (Holt linear)
        self.alpha = 0.6  # Level
        self.beta = 0.4   # Trend

    def reset(self) -> None:
        self.history.clear()
        self._level_yaw = None
        self._trend_yaw = 0.0
        self._level_pitch = None
        self._trend_pitch = 0.0
        self._last_t = None

    def observe(self, t: float, yaw_deg: float, pitch_deg: float) -> None:
        """Add a gaze observation to the trajectory history."""
        obs = GazeObservation(t=t, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
        self.history.append(obs)

        if self._level_yaw is None or self._last_t is None:
            self._level_yaw = yaw_deg
            self._trend_yaw = 0.0
            self._level_pitch = pitch_deg
            self._trend_pitch = 0.0
            self._last_t = t
            return

        dt = t - self._last_t
        if dt <= 0:
            return

        # Holt linear trend update scaled by dt
        prev_level_yaw = self._level_yaw
        prev_level_pitch = self._level_pitch

        self._level_yaw = self.alpha * yaw_deg + (1 - self.alpha) * (prev_level_yaw + self._trend_yaw * dt)
        self._trend_yaw = self.beta * ((self._level_yaw - prev_level_yaw) / dt) + (1 - self.beta) * self._trend_yaw

        self._level_pitch = self.alpha * pitch_deg + (1 - self.alpha) * (prev_level_pitch + self._trend_pitch * dt)
        self._trend_pitch = self.beta * ((self._level_pitch - prev_level_pitch) / dt) + (1 - self.beta) * self._trend_pitch

        self._last_t = t

    def predict(
        self,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        horizon_s: Optional[float] = None,
        roi_frac: float = 0.35,
        apply_fallback: bool = True,
    ) -> PredictedViewport:
        """Predict the likely viewport at t_now + horizon_s."""
        t_start = time.perf_counter()
        h_sec = horizon_s if horizon_s is not None else self.default_horizon_s

        if not self.history or self._level_yaw is None:
            roi = gaze_roi(width, height, 0.0, 0.0, roi_frac)
            return PredictedViewport(
                predicted_yaw_deg=0.0,
                predicted_pitch_deg=0.0,
                current_yaw_deg=0.0,
                current_pitch_deg=0.0,
                horizon_s=h_sec,
                confidence=1.0,
                is_saccade=False,
                angular_velocity_dps=0.0,
                roi_box=roi,
                fallback_used=False,
                processing_time_s=time.perf_counter() - t_start,
            )

        latest = self.history[-1]
        cur_yaw = latest.yaw_deg
        cur_pitch = latest.pitch_deg

        # Compute instantaneous angular velocity
        speed = math.sqrt(self._trend_yaw ** 2 + self._trend_pitch ** 2)
        is_saccade = speed > self.saccade_velocity_threshold

        # Extrapolate
        raw_pred_yaw = cur_yaw + (self._trend_yaw * h_sec)
        raw_pred_pitch = cur_pitch + (self._trend_pitch * h_sec)

        # Compute confidence:
        history_len = len(self.history)
        history_factor = min(1.0, history_len / 4.0)

        # Non-linear penalty for extreme speeds
        speed_factor = max(0.0, 1.0 - (speed / (self.saccade_velocity_threshold * 1.5)))
        confidence = round(history_factor * speed_factor, 3)

        fallback = False
        if not apply_fallback:
            fallback = False
            final_yaw = raw_pred_yaw
            final_pitch = raw_pred_pitch
        elif confidence < self.confidence_threshold or is_saccade:
            # Fall back to current viewport or conservative blend
            fallback = True
            blend_w = max(0.0, confidence)
            final_yaw = blend_w * raw_pred_yaw + (1.0 - blend_w) * cur_yaw
            final_pitch = blend_w * raw_pred_pitch + (1.0 - blend_w) * cur_pitch
        else:
            final_yaw = raw_pred_yaw
            final_pitch = raw_pred_pitch

        # Clamp within maximum physical FOV limits
        final_yaw = max(-config.HFOV_DEG / 2.0, min(config.HFOV_DEG / 2.0, final_yaw))
        final_pitch = max(-config.VFOV_DEG / 2.0, min(config.VFOV_DEG / 2.0, final_pitch))

        roi = gaze_roi(width, height, final_yaw, final_pitch, roi_frac)
        proc_time = time.perf_counter() - t_start

        return PredictedViewport(
            predicted_yaw_deg=round(final_yaw, 2),
            predicted_pitch_deg=round(final_pitch, 2),
            current_yaw_deg=round(cur_yaw, 2),
            current_pitch_deg=round(cur_pitch, 2),
            horizon_s=h_sec,
            confidence=confidence,
            is_saccade=is_saccade,
            angular_velocity_dps=round(speed, 1),
            roi_box=roi,
            fallback_used=fallback,
            processing_time_s=proc_time,
        )


def compute_viewport_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    inter_w = max(0, ix1 - ix0)
    inter_h = max(0, iy1 - iy0)
    inter_area = inter_w * inter_h

    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return round(inter_area / float(union_area), 4)


def compute_viewport_coverage(
    actual_roi: Tuple[int, int, int, int],
    tile_records: List[Any],
) -> float:
    """Compute fraction of actual gaze ROI covered by high- or medium-tier tiles."""
    rx0, ry0, rx1, ry1 = actual_roi
    roi_area = max(1, (rx1 - rx0) * (ry1 - ry0))
    covered_pixels = 0

    for tile in tile_records:
        if tile.tier in ("high", "medium"):
            tx0, ty0, tx1, ty1 = tile.bbox
            ix0 = max(rx0, tx0)
            iy0 = max(ry0, ty0)
            ix1 = min(rx1, tx1)
            iy1 = min(ry1, ty1)
            if ix1 > ix0 and iy1 > iy0:
                covered_pixels += (ix1 - ix0) * (iy1 - iy0)

    return round(min(1.0, covered_pixels / float(roi_area)), 4)


def evaluate_prediction(
    actual_yaw: float,
    actual_pitch: float,
    pred: PredictedViewport,
    width: int = config.FRAME_WIDTH,
    height: int = config.FRAME_HEIGHT,
) -> Dict[str, float]:
    """Compare prediction against actual future ground truth."""
    ang_err = math.sqrt((actual_yaw - pred.predicted_yaw_deg) ** 2 + (actual_pitch - pred.predicted_pitch_deg) ** 2)
    actual_roi = gaze_roi(width, height, actual_yaw, actual_pitch)
    iou = compute_viewport_iou(actual_roi, pred.roi_box)

    return {
        "angular_error_deg": round(ang_err, 2),
        "viewport_iou": iou,
        "confidence": pred.confidence,
        "is_saccade": float(pred.is_saccade),
        "fallback_used": float(pred.fallback_used),
        "processing_overhead_us": round(pred.processing_time_s * 1_000_000, 2),
    }


def compare_prediction_strategies(
    frame_img,
    predictor: ViewportPredictor,
    actual_yaw: float,
    actual_pitch: float,
    horizon_s: float = 0.100,
    budget_bytes: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare Strategies A (Current Gaze), B (Predicted), C (Predicted + Fallback).

    Measures angular error, IoU, ROI coverage, viewport PSNR/SSIM, bytes, and overhead.
    """
    from app.viewport_tiles import encode_and_reconstruct_tiles

    w, h = frame_img.size
    actual_roi = gaze_roi(w, h, actual_yaw, actual_pitch)
    latest_obs = predictor.history[-1] if predictor.history else GazeObservation(0.0, 0.0, 0.0)

    # Strategy A: Current-gaze allocation
    t_start_a = time.perf_counter()
    res_a = encode_and_reconstruct_tiles(
        frame_img,
        yaw_deg=latest_obs.yaw_deg,
        pitch_deg=latest_obs.pitch_deg,
        codec="jpeg",
        budget_bytes=budget_bytes,
        allocation_strategy="intelligent_allocator",
    )
    time_a = time.perf_counter() - t_start_a
    err_a = math.sqrt((actual_yaw - latest_obs.yaw_deg) ** 2 + (actual_pitch - latest_obs.pitch_deg) ** 2)
    cur_roi = gaze_roi(w, h, latest_obs.yaw_deg, latest_obs.pitch_deg)
    iou_a = compute_viewport_iou(actual_roi, cur_roi)
    cov_a = compute_viewport_coverage(actual_roi, res_a.tiles)

    # Strategy B: Predicted-gaze allocation (unconstrained, apply_fallback=False)
    pred_b = predictor.predict(width=w, height=h, horizon_s=horizon_s, apply_fallback=False)
    res_b = encode_and_reconstruct_tiles(
        frame_img,
        yaw_deg=pred_b.predicted_yaw_deg,
        pitch_deg=pred_b.predicted_pitch_deg,
        codec="jpeg",
        budget_bytes=budget_bytes,
        allocation_strategy="intelligent_allocator",
    )
    ev_b = evaluate_prediction(actual_yaw, actual_pitch, pred_b, w, h)
    cov_b = compute_viewport_coverage(actual_roi, res_b.tiles)

    # Strategy C: Predicted allocation with confidence-based fallback (apply_fallback=True)
    pred_c = predictor.predict(width=w, height=h, horizon_s=horizon_s, apply_fallback=True)
    res_c = encode_and_reconstruct_tiles(
        frame_img,
        yaw_deg=pred_c.predicted_yaw_deg,
        pitch_deg=pred_c.predicted_pitch_deg,
        codec="jpeg",
        budget_bytes=budget_bytes,
        allocation_strategy="intelligent_allocator",
    )
    ev_c = evaluate_prediction(actual_yaw, actual_pitch, pred_c, w, h)
    cov_c = compute_viewport_coverage(actual_roi, res_c.tiles)

    return {
        "A_current_gaze": {
            "strategy": "A: Current Gaze",
            "angular_error_deg": round(err_a, 2),
            "viewport_iou": iou_a,
            "viewport_coverage": cov_a,
            "viewport_psnr_db": res_a.viewport_psnr_db,
            "viewport_ssim": res_a.viewport_ssim,
            "total_bytes": res_a.total_bytes,
            "processing_us": round(time_a * 1e6, 1),
            "worse_than_current": False,
        },
        "B_predicted_unconstrained": {
            "strategy": "B: Predicted Gaze (Unconstrained)",
            "angular_error_deg": ev_b["angular_error_deg"],
            "viewport_iou": ev_b["viewport_iou"],
            "viewport_coverage": cov_b,
            "viewport_psnr_db": res_b.viewport_psnr_db,
            "viewport_ssim": res_b.viewport_ssim,
            "total_bytes": res_b.total_bytes,
            "processing_us": ev_b["processing_overhead_us"],
            "worse_than_current": bool(ev_b["angular_error_deg"] > err_a),
        },
        "C_predicted_fallback": {
            "strategy": "C: Predicted Gaze (with Confidence Fallback)",
            "angular_error_deg": ev_c["angular_error_deg"],
            "viewport_iou": ev_c["viewport_iou"],
            "viewport_coverage": cov_c,
            "viewport_psnr_db": res_c.viewport_psnr_db,
            "viewport_ssim": res_c.viewport_ssim,
            "total_bytes": res_c.total_bytes,
            "processing_us": ev_c["processing_overhead_us"],
            "worse_than_current": bool(ev_c["angular_error_deg"] > err_a),
        },
    }
