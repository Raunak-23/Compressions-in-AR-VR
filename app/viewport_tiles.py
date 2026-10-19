"""4x4 Viewport-Aware Tile Compression, Reconstruction, and Intelligent Allocation.

Supports:
1. Spatial 4x4 partitioning (16 gapless tiles).
2. Viewer gaze mapping (yaw/pitch or normalized coordinates) to primary RoI.
3. Multiple allocation policies:
   - "fixed_tiers": Classical 3-tier policy (high for gaze, medium for guard band, low for periphery).
   - "uniform": Uniform quality across all 16 tiles.
   - "viewport_only": Viewport-prioritized tiling without bandwidth-driven budget reduction.
   - "bandwidth_only": Uniform quality across all tiles scaled to current byte budget.
   - "intelligent_allocator": Measurement-driven greedy knapsack Rate-Distortion optimization
     maximizing viewport-weighted quality under the available byte budget.
4. Independent tile encoding (JPEG or H.264) with per-tile timing and bytes.
5. Seamless, gapless full-frame canvas reconstruction.
6. Region-specific quality metrics (viewport vs guard-band vs peripheral) and boundary artifact MSE.
7. Bounded iterative re-encoding and verified budget compliance reporting.
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

from app.codecs import lossy
from app.codecs.video_codec import compute_ssim, decode_frame as h264_decode, encode_frame as h264_encode
from app.viewport import gaze_roi


@dataclass
class TileInfo:
    row: int
    col: int
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    tier: Literal["high", "medium", "low"]
    quality_val: int
    encoded_bytes: bytes = b""
    byte_count: int = 0
    psnr_db: float = 0.0
    ssim: float = 0.0
    encode_time_s: float = 0.0
    decode_time_s: float = 0.0
    importance_weight: float = 0.0
    view_probability: float = 0.0


@dataclass
class TiledFrameResult:
    reconstructed: Image.Image
    grid: Tuple[int, int]
    tiles: List[TileInfo]
    total_bytes: int
    encode_time_s: float
    decode_time_s: float
    viewport_psnr_db: float
    viewport_ssim: float
    guard_band_psnr_db: float
    guard_band_ssim: float
    peripheral_psnr_db: float
    peripheral_ssim: float
    full_psnr_db: float
    full_ssim: float
    roi_box: Tuple[int, int, int, int]
    codec: str
    boundary_artifact_mse: float = 0.0
    budget_bytes: Optional[int] = None
    budget_retries: int = 0
    within_budget: bool = True
    overshoot_bytes: int = 0
    allocation_strategy: str = "fixed_tiers"
    tile_weights: Dict[Tuple[int, int], float] = field(default_factory=dict)
    view_probabilities: Dict[Tuple[int, int], float] = field(default_factory=dict)


def compute_grid_boundaries(width: int, height: int, rows: int = 4, cols: int = 4) -> Tuple[List[int], List[int]]:
    """Compute non-overlapping, gapless tile boundary coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensions must be positive, got {width}x{height}")
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Grid rows and cols must be positive, got {rows}x{cols}")

    col_bounds = [round(i * width / cols) for i in range(cols + 1)]
    row_bounds = [round(j * height / rows) for j in range(rows + 1)]
    return col_bounds, row_bounds


def assign_tile_tiers(
    width: int,
    height: int,
    roi: Tuple[int, int, int, int],
    rows: int = 4,
    cols: int = 4,
) -> Tuple[Dict[Tuple[int, int], Literal["high", "medium", "low"]], List[int], List[int]]:
    """Map gaze RoI to 4x4 tiles, assigning high, medium (guard band), and low (periphery)."""
    col_bounds, row_bounds = compute_grid_boundaries(width, height, rows, cols)
    gx0, gy0, gx1, gy1 = roi

    high_tiles: set[Tuple[int, int]] = set()

    for r in range(rows):
        ty0, ty1 = row_bounds[r], row_bounds[r + 1]
        for c in range(cols):
            tx0, tx1 = col_bounds[c], col_bounds[c + 1]
            if not (tx1 <= gx0 or tx0 >= gx1 or ty1 <= gy0 or ty0 >= gy1):
                high_tiles.add((r, c))

    if not high_tiles:
        cx = (gx0 + gx1) // 2
        cy = (gy0 + gy1) // 2
        for r in range(rows):
            if row_bounds[r] <= cy < row_bounds[r + 1]:
                for c in range(cols):
                    if col_bounds[c] <= cx < col_bounds[c + 1]:
                        high_tiles.add((r, c))

    medium_tiles: set[Tuple[int, int]] = set()
    for hr, hc in high_tiles:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = hr + dr, hc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) not in high_tiles:
                    medium_tiles.add((nr, nc))

    tiers: Dict[Tuple[int, int], Literal["high", "medium", "low"]] = {}
    for r in range(rows):
        for c in range(cols):
            if (r, c) in high_tiles:
                tiers[(r, c)] = "high"
            elif (r, c) in medium_tiles:
                tiers[(r, c)] = "medium"
            else:
                tiers[(r, c)] = "low"

    return tiers, col_bounds, row_bounds


def compute_tile_importance_and_probability(
    width: int,
    height: int,
    roi: Tuple[int, int, int, int],
    rows: int = 4,
    cols: int = 4,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float], List[int], List[int]]:
    """Compute continuous importance weights and viewing probability for each tile.

    Combines Gaussian radial proximity to gaze center and intersection overlap area.
    """
    col_bounds, row_bounds = compute_grid_boundaries(width, height, rows, cols)
    gx0, gy0, gx1, gy1 = roi
    gcx = (gx0 + gx1) / 2.0
    gcy = (gy0 + gy1) / 2.0
    diag = math.sqrt(width ** 2 + height ** 2)
    sigma = diag * 0.22

    raw_weights: Dict[Tuple[int, int], float] = {}
    for r in range(rows):
        ty0, ty1 = row_bounds[r], row_bounds[r + 1]
        tcy = (ty0 + ty1) / 2.0
        tile_h = max(1, ty1 - ty0)
        for c in range(cols):
            tx0, tx1 = col_bounds[c], col_bounds[c + 1]
            tcx = (tx0 + tx1) / 2.0
            tile_w = max(1, tx1 - tx0)
            tile_area = tile_w * tile_h

            dist = math.sqrt((tcx - gcx) ** 2 + (tcy - gcy) ** 2)
            gaussian = math.exp(-0.5 * (dist / sigma) ** 2)

            ix0 = max(tx0, gx0)
            iy0 = max(ty0, gy0)
            ix1 = min(tx1, gx1)
            iy1 = min(ty1, gy1)
            overlap_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            overlap_frac = overlap_area / float(tile_area)

            # Weight incorporates both gaze proximity and direct ROI overlap
            w = 0.5 * gaussian + 0.5 * overlap_frac
            raw_weights[(r, c)] = max(0.02, w)

    # Compute viewing probabilities (Softmax-like distribution over tiles)
    sum_w = sum(raw_weights.values())
    view_probs = {k: round(v / sum_w, 4) for k, v in raw_weights.items()}

    # Normalize importance weights so peak tile is 1.0
    max_w = max(raw_weights.values()) if raw_weights else 1.0
    importance_weights = {k: round(v / max_w, 4) for k, v in raw_weights.items()}

    return importance_weights, view_probs, col_bounds, row_bounds


def allocate_intelligent_tile_qualities(
    pil_img: Image.Image,
    importance_weights: Dict[Tuple[int, int], float],
    col_bounds: List[int],
    row_bounds: List[int],
    rows: int = 4,
    cols: int = 4,
    codec: Literal["jpeg", "h264"] = "jpeg",
    budget_bytes: Optional[int] = None,
    base_bitrate_kbps: int = 1000,
) -> Dict[Tuple[int, int], int]:
    """Greedy Knapsack Rate-Distortion optimization for tile qualities under budget.

    Maximizes: Sum_i W_i * Q(q_i)
    Subject to: Sum_i S_i(q_i) <= Budget
    """
    if codec == "jpeg":
        candidates = [15, 25, 40, 55, 70, 85, 95]
    else:
        candidates = [
            max(16, int(base_bitrate_kbps * 0.05)),
            max(24, int(base_bitrate_kbps * 0.10)),
            max(48, int(base_bitrate_kbps * 0.20)),
            max(96, int(base_bitrate_kbps * 0.35)),
            max(160, int(base_bitrate_kbps * 0.55)),
            max(250, int(base_bitrate_kbps * 0.80)),
        ]

    # Pre-estimate tile rate curves based on spatial variance / complexity
    arr = np.asarray(pil_img.convert("L"), dtype=np.float32)
    tile_rates: Dict[Tuple[int, int], List[int]] = {}
    tile_quals: List[float] = []

    for q in candidates:
        if codec == "jpeg":
            q_score = 100.0 * (1.0 - math.exp(-q / 22.0))
        else:
            q_score = 100.0 * (1.0 - math.exp(-q / 80.0))
        tile_quals.append(q_score)

    for r in range(rows):
        ty0, ty1 = row_bounds[r], row_bounds[r + 1]
        for c in range(cols):
            tx0, tx1 = col_bounds[c], col_bounds[c + 1]
            tile_w = tx1 - tx0
            tile_h = ty1 - ty0
            patch = arr[ty0:ty1, tx0:tx1]
            std_dev = float(np.std(patch)) if patch.size > 0 else 10.0
            complexity_factor = 0.5 + 0.5 * min(2.0, std_dev / 40.0)

            rates = []
            for q in candidates:
                if codec == "jpeg":
                    est_bytes = (tile_w * tile_h) * (0.008 + 0.00032 * (q ** 1.35)) * complexity_factor
                else:
                    # kbps converted to approximate bytes per 1/30s frame
                    est_bytes = (q * 1000 / 8 / 30.0) * complexity_factor
                rates.append(max(350, int(est_bytes)))
            tile_rates[(r, c)] = rates

    # If no budget, assign qualities purely by importance weight
    if budget_bytes is None:
        alloc: Dict[Tuple[int, int], int] = {}
        for (r, c), w in importance_weights.items():
            if w >= 0.70:
                alloc[(r, c)] = candidates[-1]  # Highest
            elif w >= 0.35:
                alloc[(r, c)] = candidates[-2]
            elif w >= 0.15:
                alloc[(r, c)] = candidates[2]
            else:
                alloc[(r, c)] = candidates[1]
        return alloc

    # Greedy knapsack optimization
    # Start all tiles at lowest rung
    current_idx: Dict[Tuple[int, int], int] = {k: 0 for k in importance_weights}
    current_total_bytes = sum(tile_rates[k][0] for k in importance_weights)

    num_candidates = len(candidates)
    while True:
        best_tile = None
        best_efficiency = -1.0
        best_delta_bytes = 0

        for k, w in importance_weights.items():
            idx = current_idx[k]
            if idx + 1 < num_candidates:
                delta_b = tile_rates[k][idx + 1] - tile_rates[k][idx]
                delta_q = tile_quals[idx + 1] - tile_quals[idx]
                if current_total_bytes + delta_b <= budget_bytes:
                    efficiency = (w * delta_q) / max(1.0, float(delta_b))
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_tile = k
                        best_delta_bytes = delta_b

        if best_tile is None:
            break

        current_idx[best_tile] += 1
        current_total_bytes += best_delta_bytes

    return {k: candidates[idx] for k, idx in current_idx.items()}


def _encode_single_tile(
    tile_img: Image.Image,
    codec: Literal["jpeg", "h264"],
    quality_val: int,
) -> Tuple[bytes, float, float]:
    """Encode and decode a single tile, returning (encoded_bytes, encode_time, decode_time)."""
    t_enc = time.perf_counter()
    if codec == "jpeg":
        blob = lossy.jpeg_encode(tile_img, quality=quality_val)
    elif codec == "h264":
        effective_kbps = max(32, quality_val)
        blob = h264_encode(tile_img, bitrate_kbps=effective_kbps, fps=30)
    else:
        raise ValueError(f"Unsupported tile codec: {codec}")
    encode_time = time.perf_counter() - t_enc

    t_dec = time.perf_counter()
    if codec == "jpeg":
        _ = Image.open(io.BytesIO(blob)).convert("RGB")
    elif codec == "h264":
        tw = tile_img.width
        th = tile_img.height
        _ = h264_decode(blob, target_size=(tw, th))
    decode_time = time.perf_counter() - t_dec

    return blob, encode_time, decode_time


def _measure_boundary_artifacts(
    reconstructed: Image.Image,
    col_bounds: List[int],
    row_bounds: List[int],
) -> float:
    """Measure tile-boundary artifacts across tile boundaries."""
    arr = np.asarray(reconstructed, dtype=np.float64)
    h, w = arr.shape[:2]
    diffs = []

    for x in col_bounds[1:-1]:
        if 0 < x < w:
            left = arr[:, x - 1, :]
            right = arr[:, x, :]
            diffs.append(float(np.mean((left - right) ** 2)))

    for y in row_bounds[1:-1]:
        if 0 < y < h:
            top = arr[y - 1, :, :]
            bottom = arr[y, :, :]
            diffs.append(float(np.mean((top - bottom) ** 2)))

    return float(np.mean(diffs)) if diffs else 0.0


def _compute_region_quality(
    pil_img: Image.Image,
    reconstructed: Image.Image,
    tile_records: List[TileInfo],
    tier_filter: str,
) -> Tuple[float, float]:
    """Compute combined PSNR and SSIM for tiles matching a given tier."""
    refs = []
    recs = []
    ssim_vals = []

    for tile in tile_records:
        if tile.tier == tier_filter:
            ref_tile = pil_img.crop(tile.bbox)
            rec_tile = reconstructed.crop(tile.bbox)
            refs.append(np.asarray(ref_tile))
            recs.append(np.asarray(rec_tile))
            ssim_vals.append(tile.ssim)

    if not refs:
        if tier_filter == "low":
            return _compute_region_quality(pil_img, reconstructed, tile_records, "medium")
        return 0.0, 0.0

    v_ref = np.concatenate([v.flatten() for v in refs])
    v_rec = np.concatenate([v.flatten() for v in recs])
    mse = float(np.mean((v_ref.astype(np.float64) - v_rec.astype(np.float64)) ** 2))
    psnr_val = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0
    ssim_val = float(np.mean(ssim_vals))

    return round(float(psnr_val), 2), round(ssim_val, 4)


def encode_and_reconstruct_tiles(
    img: Union[Image.Image, np.ndarray],
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    quality_ladder: Optional[Dict[str, int]] = None,
    codec: Literal["jpeg", "h264"] = "jpeg",
    rows: int = 4,
    cols: int = 4,
    roi_frac: float = 0.35,
    base_bitrate_kbps: int = 1000,
    budget_bytes: Optional[int] = None,
    max_retries: int = 3,
    quality_step: int = 5,
    allocation_strategy: Literal[
        "fixed_tiers", "uniform", "viewport_only", "bandwidth_only", "intelligent_allocator"
    ] = "fixed_tiers",
) -> TiledFrameResult:
    """Execute 4x4 tile encoding, quality assignment, and full frame reconstruction.

    Parameters:
        img: Input image (PIL Image or NumPy array).
        yaw_deg: Horizontal gaze angle in degrees from center.
        pitch_deg: Vertical gaze angle in degrees from center.
        quality_ladder: Quality values for {"high", "medium", "low"}.
        codec: 'jpeg' (baseline) or 'h264'.
        rows: Number of grid rows (default: 4).
        cols: Number of grid cols (default: 4).
        roi_frac: Size of the gaze RoI relative to canvas.
        base_bitrate_kbps: Base bitrate used when codec='h264'.
        budget_bytes: Byte budget for total encoded frame.
        max_retries: Maximum number of retry attempts for budget enforcement.
        quality_step: Amount to reduce quality per retry.
        allocation_strategy: Strategy choice for assigning per-tile qualities:
            - 'fixed_tiers': Classical 3-tier high/medium/low.
            - 'uniform': Identical quality across all tiles.
            - 'viewport_only': Viewport high/medium/low without bandwidth budget limits.
            - 'bandwidth_only': Uniform quality scaled to budget.
            - 'intelligent_allocator': Greedy knapsack Rate-Distortion optimization.
    """
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img).convert("RGB")
    elif isinstance(img, Image.Image):
        pil_img = img.convert("RGB")
    else:
        raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(img)}")

    w, h = pil_img.size
    roi = gaze_roi(w, h, yaw_deg=yaw_deg, pitch_deg=pitch_deg, roi_frac=roi_frac)

    # 1. Compute geometry, tiers, importance weights, and view probabilities
    tiers, col_bounds, row_bounds = assign_tile_tiers(w, h, roi, rows, cols)
    importance_weights, view_probs, _, _ = compute_tile_importance_and_probability(
        w, h, roi, rows, cols
    )

    if quality_ladder is None:
        if codec == "jpeg":
            quality_ladder = {"high": 85, "medium": 55, "low": 25}
        else:
            quality_ladder = {
                "high": int(base_bitrate_kbps * 0.45),
                "medium": int(base_bitrate_kbps * 0.20),
                "low": int(base_bitrate_kbps * 0.05),
            }

    # 2. Determine per-tile quality assignments according to requested strategy
    current_ladder = dict(quality_ladder)
    active_budget = budget_bytes if allocation_strategy != "viewport_only" else None

    if allocation_strategy == "uniform":
        med_q = current_ladder["medium"]
        tile_qualities: Dict[Tuple[int, int], int] = {
            (r, c): med_q for r in range(rows) for c in range(cols)
        }
    elif allocation_strategy == "bandwidth_only":
        # Uniform quality scaled to available budget
        scale = 1.0
        if active_budget is not None:
            baseline_est = (w * h * 0.04)
            scale = min(1.0, max(0.2, active_budget / max(1.0, baseline_est)))
        q_target = max(10, int(current_ladder["high"] * scale))
        tile_qualities = {(r, c): q_target for r in range(rows) for c in range(cols)}
    elif allocation_strategy == "intelligent_allocator":
        tile_qualities = allocate_intelligent_tile_qualities(
            pil_img=pil_img,
            importance_weights=importance_weights,
            col_bounds=col_bounds,
            row_bounds=row_bounds,
            rows=rows,
            cols=cols,
            codec=codec,
            budget_bytes=active_budget,
            base_bitrate_kbps=base_bitrate_kbps,
        )
    else:
        # 'fixed_tiers' or 'viewport_only'
        tile_qualities = {
            (r, c): current_ladder[tiers[(r, c)]] for r in range(rows) for c in range(cols)
        }

    # 3. Encoding loop with retry budget control
    retries = 0
    best_result = None

    while True:
        t_enc_start = time.perf_counter()
        tile_records: List[TileInfo] = []

        for r in range(rows):
            ty0, ty1 = row_bounds[r], row_bounds[r + 1]
            for c in range(cols):
                tx0, tx1 = col_bounds[c], col_bounds[c + 1]
                bbox = (tx0, ty0, tx1, ty1)
                tier = tiers[(r, c)]
                q_val = tile_qualities[(r, c)]

                tile_img = pil_img.crop(bbox)
                blob, tile_enc_time, tile_dec_time = _encode_single_tile(
                    tile_img, codec, q_val
                )

                tile_records.append(
                    TileInfo(
                        row=r,
                        col=c,
                        bbox=bbox,
                        tier=tier,
                        quality_val=q_val,
                        encoded_bytes=blob,
                        byte_count=len(blob),
                        encode_time_s=tile_enc_time,
                        decode_time_s=tile_dec_time,
                        importance_weight=importance_weights[(r, c)],
                        view_probability=view_probs[(r, c)],
                    )
                )

        total_encode_time = time.perf_counter() - t_enc_start
        total_bytes = sum(t.byte_count for t in tile_records)

        # Check budget compliance
        if active_budget is not None and total_bytes > active_budget and retries < max_retries:
            # Step down lower-importance tiles first
            sorted_tiles = sorted(importance_weights.keys(), key=lambda k: importance_weights[k])
            for k in sorted_tiles:
                min_q = 10 if codec == "jpeg" else 16
                if tile_qualities[k] > min_q:
                    step = quality_step if codec == "jpeg" else max(16, int(tile_qualities[k] * 0.25))
                    tile_qualities[k] = max(min_q, tile_qualities[k] - step)
            retries += 1
            continue

        best_result = (tile_records, total_encode_time, total_bytes)
        break

    tile_records, encode_time_s, total_bytes = best_result

    # 4. Seamless full canvas reconstruction
    t_dec_start = time.perf_counter()
    reconstructed = Image.new("RGB", (w, h))

    for tile in tile_records:
        if codec == "jpeg":
            rec_tile = Image.open(io.BytesIO(tile.encoded_bytes)).convert("RGB")
        elif codec == "h264":
            orig_tile_w = tile.bbox[2] - tile.bbox[0]
            orig_tile_h = tile.bbox[3] - tile.bbox[1]
            rec_tile = h264_decode(tile.encoded_bytes, target_size=(orig_tile_w, orig_tile_h))
        else:
            raise ValueError(f"Unsupported tile codec: {codec}")

        reconstructed.paste(rec_tile, (tile.bbox[0], tile.bbox[1]))

    decode_time_s = time.perf_counter() - t_dec_start

    # 5. Measure quality per tile & region
    for tile in tile_records:
        ref_tile = pil_img.crop(tile.bbox)
        rec_tile = reconstructed.crop(tile.bbox)
        tile.psnr_db = round(lossy.psnr(ref_tile, rec_tile), 2)
        tile.ssim = round(compute_ssim(ref_tile, rec_tile), 4)

    viewport_psnr, viewport_ssim = _compute_region_quality(pil_img, reconstructed, tile_records, "high")
    guard_band_psnr, guard_band_ssim = _compute_region_quality(pil_img, reconstructed, tile_records, "medium")
    peripheral_psnr, peripheral_ssim = _compute_region_quality(pil_img, reconstructed, tile_records, "low")

    full_psnr = round(lossy.psnr(pil_img, reconstructed), 2)
    full_ssim = round(compute_ssim(pil_img, reconstructed), 4)

    # 6. Measure seam / boundary artifacts
    boundary_mse = _measure_boundary_artifacts(reconstructed, col_bounds, row_bounds)

    # 7. Budget accounting
    overshoot = max(0, total_bytes - active_budget) if active_budget is not None else 0
    within_budget = (active_budget is None) or (total_bytes <= active_budget)

    return TiledFrameResult(
        reconstructed=reconstructed,
        grid=(rows, cols),
        tiles=tile_records,
        total_bytes=total_bytes,
        encode_time_s=encode_time_s,
        decode_time_s=decode_time_s,
        viewport_psnr_db=viewport_psnr,
        viewport_ssim=viewport_ssim,
        guard_band_psnr_db=guard_band_psnr,
        guard_band_ssim=guard_band_ssim,
        peripheral_psnr_db=peripheral_psnr,
        peripheral_ssim=peripheral_ssim,
        full_psnr_db=full_psnr,
        full_ssim=full_ssim,
        roi_box=roi,
        codec=codec,
        boundary_artifact_mse=round(boundary_mse, 2),
        budget_bytes=active_budget,
        budget_retries=retries,
        within_budget=within_budget,
        overshoot_bytes=overshoot,
        allocation_strategy=allocation_strategy,
        tile_weights=importance_weights,
        view_probabilities=view_probs,
    )
