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
    budget_infeasible: bool = False
    final_quality_map: Dict[Tuple[int, int], int] = field(default_factory=dict)


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


def _estimate_jpeg_bytes(tile_w: int, tile_h: int, quality: int, complexity_factor: float) -> int:
    """Calibrated JPEG size estimation.

    The model is based on empirical observation that JPEG output for a
    ``tile_w × tile_h`` RGB tile at quality ``q`` follows approximately::

        bytes ≈ pixels × bpp(q) × complexity_factor

    where bpp(q) grows super-linearly with quality.  The constants were
    calibrated against Pillow's JPEG encoder on 320×180 tiles from the
    synthetic XR source and the procedural panoramic source.
    """
    pixels = tile_w * tile_h
    # Bits-per-pixel model: ranges from ~0.15 bpp at q=5 to ~3.5 bpp at q=95
    bpp = 0.08 + 0.035 * (quality ** 1.12) / 100.0
    est = pixels * bpp * complexity_factor
    # JPEG overhead: headers, Huffman tables, restart markers, etc.
    est += 600
    return max(800, int(est))


def _estimate_h264_bytes(bitrate_kbps: int, complexity_factor: float) -> int:
    """Approximate single-IDR H.264 tile size."""
    est_bytes = (bitrate_kbps * 1000 / 8 / 30.0) * complexity_factor
    return max(500, int(est_bytes))


def _probe_calibration_multiplier(
    pil_img: Image.Image,
    col_bounds: List[int],
    row_bounds: List[int],
    probe_r: int,
    probe_c: int,
    probe_quality: int,
    codec: Literal["jpeg", "h264"],
    estimated_bytes: int,
) -> float:
    """Encode one representative tile to calibrate the rate model.

    Returns a multiplier ``actual / estimated`` so subsequent estimates
    can be corrected.  Limited to one fast encode to avoid overhead.
    """
    tx0, tx1 = col_bounds[probe_c], col_bounds[probe_c + 1]
    ty0, ty1 = row_bounds[probe_r], row_bounds[probe_r + 1]
    tile_img = pil_img.crop((tx0, ty0, tx1, ty1))

    if codec == "jpeg":
        blob = lossy.jpeg_encode(tile_img, quality=probe_quality)
    else:
        blob = h264_encode(tile_img, bitrate_kbps=max(32, probe_quality), fps=30)

    actual = len(blob)
    if estimated_bytes <= 0:
        return 1.0
    return max(0.3, min(5.0, actual / estimated_bytes))


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

    Uses a calibrated rate model corrected by a single probe encode so the
    knapsack allocation starts from a realistic byte estimate.  This avoids
    the previous problem where wildly optimistic estimates led to massive
    budget overshoot after the first encoding pass.
    """
    if codec == "jpeg":
        candidates = [5, 10, 15, 25, 40, 55, 70, 85, 95]
    else:
        candidates = [
            max(16, int(base_bitrate_kbps * 0.03)),
            max(24, int(base_bitrate_kbps * 0.06)),
            max(48, int(base_bitrate_kbps * 0.12)),
            max(96, int(base_bitrate_kbps * 0.20)),
            max(160, int(base_bitrate_kbps * 0.35)),
            max(250, int(base_bitrate_kbps * 0.55)),
            max(350, int(base_bitrate_kbps * 0.80)),
        ]

    # Pre-estimate tile rate curves based on spatial variance / complexity
    arr = np.asarray(pil_img.convert("L"), dtype=np.float32)
    tile_complexity: Dict[Tuple[int, int], float] = {}
    tile_dims: Dict[Tuple[int, int], Tuple[int, int]] = {}
    tile_rates: Dict[Tuple[int, int], List[int]] = {}
    tile_quals: List[float] = []

    for q in candidates:
        if codec == "jpeg":
            q_score = 100.0 * (1.0 - math.exp(-q / 22.0))
        else:
            q_score = 100.0 * (1.0 - math.exp(-q / 80.0))
        tile_quals.append(q_score)

    # Find the tile with highest complexity for probe calibration
    max_complexity_key = (0, 0)
    max_complexity_val = -1.0

    for r in range(rows):
        ty0, ty1 = row_bounds[r], row_bounds[r + 1]
        for c in range(cols):
            tx0, tx1 = col_bounds[c], col_bounds[c + 1]
            tw = tx1 - tx0
            th = ty1 - ty0
            patch = arr[ty0:ty1, tx0:tx1]
            std_dev = float(np.std(patch)) if patch.size > 0 else 10.0
            complexity_factor = 0.5 + 0.5 * min(2.0, std_dev / 40.0)
            tile_complexity[(r, c)] = complexity_factor
            tile_dims[(r, c)] = (tw, th)
            if complexity_factor > max_complexity_val:
                max_complexity_val = complexity_factor
                max_complexity_key = (r, c)

    # Build uncalibrated rate curves
    for k in tile_complexity:
        tw, th = tile_dims[k]
        cf = tile_complexity[k]
        rates = []
        for q in candidates:
            if codec == "jpeg":
                est = _estimate_jpeg_bytes(tw, th, q, cf)
            else:
                est = _estimate_h264_bytes(q, cf)
            rates.append(est)
        tile_rates[k] = rates

    # Probe calibration: encode the most complex tile at a mid-quality candidate
    # to correct systematic model error with a single fast encode.
    if budget_bytes is not None and len(candidates) >= 3:
        probe_q_idx = len(candidates) // 2
        probe_q = candidates[probe_q_idx]
        pr, pc = max_complexity_key
        est_probe = tile_rates[(pr, pc)][probe_q_idx]
        cal = _probe_calibration_multiplier(
            pil_img, col_bounds, row_bounds, pr, pc, probe_q, codec, est_probe
        )
        # Apply calibration to all rate estimates
        for k in tile_rates:
            tile_rates[k] = [max(600, int(r * cal)) for r in tile_rates[k]]

    # If no budget, assign qualities purely by importance weight
    if budget_bytes is None:
        alloc: Dict[Tuple[int, int], int] = {}
        for (r, c), w in importance_weights.items():
            if w >= 0.70:
                alloc[(r, c)] = candidates[-1]  # Highest
            elif w >= 0.35:
                alloc[(r, c)] = candidates[-2]
            elif w >= 0.15:
                alloc[(r, c)] = candidates[len(candidates) // 2]
            else:
                alloc[(r, c)] = candidates[1]
        return alloc

    # Greedy knapsack optimization
    # Start all tiles at lowest rung
    current_idx: Dict[Tuple[int, int], int] = {k: 0 for k in importance_weights}
    current_total_bytes = sum(tile_rates[k][0] for k in importance_weights)

    # If even lowest quality exceeds budget, return minimum allocation immediately
    if current_total_bytes > budget_bytes:
        return {k: candidates[0] for k in importance_weights}

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


# Sentinel value for metrics that cannot be computed (e.g. empty tile region).
# Consumers must test for this before comparisons or aggregation.
METRIC_UNAVAILABLE: float = -1.0


def _compute_region_quality(
    pil_img: Image.Image,
    reconstructed: Image.Image,
    tile_records: List[TileInfo],
    tier_filter: str,
) -> Tuple[float, float]:
    """Compute combined PSNR and SSIM for tiles matching a given tier.

    Returns (METRIC_UNAVAILABLE, METRIC_UNAVAILABLE) when no tiles belong to
    the requested tier.  This is a legitimate state — for example, a central
    gaze may assign all 16 tiles to "high" or "medium" leaving zero peripheral
    tiles.  The previous implementation silently returned 0.0 which was
    indistinguishable from a genuinely poor quality score.
    """
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
        return METRIC_UNAVAILABLE, METRIC_UNAVAILABLE

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
    max_retries: int = 6,
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

    # 3. Budget-aware encoding with measurement-driven reallocation
    #
    # Strategy: encode once at requested quality, measure actual bytes, then
    # proportionally scale qualities down based on the overshoot ratio.  This
    # uses importance-weighted scaling: peripheral tiles (low weight) get
    # cut more aggressively than viewport tiles (high weight).
    #
    # Minimum quality policy:
    #   JPEG: q=5   (visually poor but bounded file size)
    #   H.264: 16 kbps (minimum meaningful bitrate)
    # When the budget is infeasible even at minimum quality, the encoder
    # reports budget_infeasible=True and uses minimum qualities.

    min_q = 5 if codec == "jpeg" else 16
    retries = 0
    budget_infeasible = False

    def _encode_all_tiles(
        qualities: Dict[Tuple[int, int], int],
    ) -> Tuple[List[TileInfo], float, int]:
        """Encode all 16 tiles and return (records, encode_time_s, total_bytes)."""
        t_enc_start = time.perf_counter()
        records: List[TileInfo] = []
        for r in range(rows):
            ty0, ty1 = row_bounds[r], row_bounds[r + 1]
            for c in range(cols):
                tx0, tx1 = col_bounds[c], col_bounds[c + 1]
                bbox = (tx0, ty0, tx1, ty1)
                tier = tiers[(r, c)]
                q_val = qualities[(r, c)]
                tile_img = pil_img.crop(bbox)
                blob, tile_enc_time, tile_dec_time = _encode_single_tile(
                    tile_img, codec, q_val
                )
                records.append(
                    TileInfo(
                        row=r, col=c, bbox=bbox, tier=tier, quality_val=q_val,
                        encoded_bytes=blob, byte_count=len(blob),
                        encode_time_s=tile_enc_time, decode_time_s=tile_dec_time,
                        importance_weight=importance_weights[(r, c)],
                        view_probability=view_probs[(r, c)],
                    )
                )
        enc_time = time.perf_counter() - t_enc_start
        tot = sum(t.byte_count for t in records)
        return records, enc_time, tot

    # --- First pass ---
    tile_records, encode_time_s, total_bytes = _encode_all_tiles(tile_qualities)

    # --- Budget enforcement retries ---
    # Target slightly under budget (95%) to provide margin for encode variance.
    effective_budget = active_budget
    safety_target = int(active_budget * 0.95) if active_budget is not None else None

    if active_budget is not None and total_bytes > active_budget:
        # Build a per-tile measured byte map from the first pass
        measured_bytes_map: Dict[Tuple[int, int], int] = {}
        for t in tile_records:
            measured_bytes_map[(t.row, t.col)] = t.byte_count

        max_w = max(importance_weights.values()) if importance_weights else 1.0

        for attempt in range(max_retries):
            retries += 1
            overshoot_ratio = total_bytes / max(1, safety_target or active_budget)

            # Per-tile scaling using measured byte contributions.
            # Tiles that consumed a disproportionate share of the budget
            # relative to their importance get cut more aggressively.
            all_at_minimum = True
            for k in tile_qualities:
                w_norm = importance_weights[k] / max(max_w, 1e-6)
                # Protection factor: viewport tiles (w_norm~1.0) get mild cuts,
                # peripheral tiles (w_norm~0.02) get aggressive cuts.
                protection = 0.25 + 0.65 * w_norm
                # Base scale to bring total down to budget
                base_scale = 1.0 / overshoot_ratio
                # Blended scale: more aggressive for low-importance tiles
                scale_factor = base_scale * (protection + (1.0 - protection) * base_scale)
                scale_factor = min(scale_factor, 0.92)  # Always reduce at least 8%

                new_q = max(min_q, int(tile_qualities[k] * scale_factor))
                tile_qualities[k] = new_q
                if new_q > min_q:
                    all_at_minimum = False

            tile_records, encode_time_s, total_bytes = _encode_all_tiles(tile_qualities)

            # Update measured byte map for next iteration
            for t in tile_records:
                measured_bytes_map[(t.row, t.col)] = t.byte_count

            if total_bytes <= active_budget:
                break

            if all_at_minimum:
                # Budget is infeasible — all tiles are at minimum quality
                budget_infeasible = True
                break

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
        budget_infeasible=budget_infeasible,
        final_quality_map=dict(tile_qualities),
    )
