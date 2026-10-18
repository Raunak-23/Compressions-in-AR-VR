"""4x4 Viewport-Aware Tile Compression and Reconstruction.

Unlike blur-based foveated JPEG (which blurs peripheral pixels and encodes one
single full-frame image), this module implements true spatial tiling:
1. Partitions the video frame into a 4x4 spatial grid (16 tiles).
2. Maps viewer gaze (yaw/pitch or normalized coordinates) to a primary Region of Interest (RoI).
3. Assigns quality tiers across tiles:
   - "high": Tiles intersecting the primary gaze viewport.
   - "medium": Guard band tiles surrounding the viewport (1-hop 8-neighbors).
   - "low": Peripheral tiles outside the gaze region.
4. Independently encodes each tile at its assigned quality.
5. Reconstructs the complete seamless canvas with zero gaps or overlaps.
6. Measures viewport quality vs. peripheral quality separately.
"""

from __future__ import annotations

import io
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
    peripheral_psnr_db: float
    peripheral_ssim: float
    full_psnr_db: float
    full_ssim: float
    roi_box: Tuple[int, int, int, int]
    codec: str


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
            # Check intersection
            if not (tx1 <= gx0 or tx0 >= gx1 or ty1 <= gy0 or ty0 >= gy1):
                high_tiles.add((r, c))

    # Fallback if RoI was somehow 0-sized
    if not high_tiles:
        cx = (gx0 + gx1) // 2
        cy = (gy0 + gy1) // 2
        for r in range(rows):
            if row_bounds[r] <= cy < row_bounds[r + 1]:
                for c in range(cols):
                    if col_bounds[c] <= cx < col_bounds[c + 1]:
                        high_tiles.add((r, c))

    # Guard band: 8-neighbors of high tiles
    medium_tiles: set[Tuple[int, int]] = set()
    for hr, hc in high_tiles:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
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
) -> TiledFrameResult:
    """Execute 4x4 tile encoding, quality assignment, and full frame reconstruction.
    
    Parameters:
        img: Input image (PIL Image or NumPy array).
        yaw_deg: Horizontal gaze angle in degrees from center.
        pitch_deg: Vertical gaze angle in degrees from center.
        quality_ladder: Quality values for {"high", "medium", "low"}.
                        For JPEG: JPEG quality 1-100.
                        For H.264: Bitrate multipliers (e.g. kbps per tier).
        codec: 'jpeg' (baseline) or 'h264'.
        rows: Number of grid rows (default: 4).
        cols: Number of grid cols (default: 4).
        roi_frac: Size of the gaze RoI relative to canvas.
        base_bitrate_kbps: Base bitrate used when codec='h264'.
    """
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img).convert("RGB")
    elif isinstance(img, Image.Image):
        pil_img = img.convert("RGB")
    else:
        raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(img)}")

    w, h = pil_img.size
    roi = gaze_roi(w, h, yaw_deg=yaw_deg, pitch_deg=pitch_deg, roi_frac=roi_frac)

    if quality_ladder is None:
        if codec == "jpeg":
            quality_ladder = {"high": 85, "medium": 55, "low": 25}
        else:
            quality_ladder = {"high": int(base_bitrate_kbps * 0.45), "medium": int(base_bitrate_kbps * 0.20), "low": int(base_bitrate_kbps * 0.05)}

    tiers, col_bounds, row_bounds = assign_tile_tiers(w, h, roi, rows, cols)

    t_enc_start = time.perf_counter()
    tile_records: List[TileInfo] = []

    # 1. Independent Tile Encoding
    for r in range(rows):
        ty0, ty1 = row_bounds[r], row_bounds[r + 1]
        for c in range(cols):
            tx0, tx1 = col_bounds[c], col_bounds[c + 1]
            bbox = (tx0, ty0, tx1, ty1)
            tier = tiers[(r, c)]
            q_val = quality_ladder[tier]

            tile_img = pil_img.crop(bbox)

            if codec == "jpeg":
                blob = lossy.jpeg_encode(tile_img, quality=q_val)
            elif codec == "h264":
                # Ensure minimum 32 kbps for tile
                effective_kbps = max(32, q_val)
                blob = h264_encode(tile_img, bitrate_kbps=effective_kbps, fps=30)
            else:
                raise ValueError(f"Unsupported tile codec: {codec}")

            tile_records.append(
                TileInfo(
                    row=r,
                    col=c,
                    bbox=bbox,
                    tier=tier,
                    quality_val=q_val,
                    encoded_bytes=blob,
                    byte_count=len(blob),
                )
            )

    encode_time_s = time.perf_counter() - t_enc_start

    # 2. Frame Reconstruction
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

    # 3. Quality Metrics Computation
    total_bytes = sum(t.byte_count for t in tile_records)

    viewport_refs, viewport_recs = [], []
    periph_refs, periph_recs = [], []

    for tile in tile_records:
        ref_tile = pil_img.crop(tile.bbox)
        rec_tile = reconstructed.crop(tile.bbox)

        tile.psnr_db = round(lossy.psnr(ref_tile, rec_tile), 2)
        tile.ssim = round(compute_ssim(ref_tile, rec_tile), 4)

        if tile.tier == "high":
            viewport_refs.append(np.asarray(ref_tile))
            viewport_recs.append(np.asarray(rec_tile))
        else:
            periph_refs.append(np.asarray(ref_tile))
            periph_recs.append(np.asarray(rec_tile))

    # Evaluate combined viewport region
    if viewport_refs:
        v_ref = np.concatenate([v.flatten() for v in viewport_refs])
        v_rec = np.concatenate([v.flatten() for v in viewport_recs])
        mse_v = float(np.mean((v_ref.astype(np.float64) - v_rec.astype(np.float64)) ** 2))
        viewport_psnr = 10 * np.log10(255.0**2 / mse_v) if mse_v > 0 else 99.0
        viewport_ssim = float(np.mean([t.ssim for t in tile_records if t.tier == "high"]))
    else:
        viewport_psnr, viewport_ssim = 0.0, 0.0

    # Evaluate peripheral region (all tiles outside high-quality viewport)
    if periph_refs:
        p_ref = np.concatenate([p.flatten() for p in periph_refs])
        p_rec = np.concatenate([p.flatten() for p in periph_recs])
        mse_p = float(np.mean((p_ref.astype(np.float64) - p_rec.astype(np.float64)) ** 2))
        peripheral_psnr = 10 * np.log10(255.0**2 / mse_p) if mse_p > 0 else 99.0
        peripheral_ssim = float(np.mean([t.ssim for t in tile_records if t.tier != "high"]))
    else:
        peripheral_psnr, peripheral_ssim = 0.0, 0.0

    full_psnr = round(lossy.psnr(pil_img, reconstructed), 2)
    full_ssim = round(compute_ssim(pil_img, reconstructed), 4)

    return TiledFrameResult(
        reconstructed=reconstructed,
        grid=(rows, cols),
        tiles=tile_records,
        total_bytes=total_bytes,
        encode_time_s=encode_time_s,
        decode_time_s=decode_time_s,
        viewport_psnr_db=round(viewport_psnr, 2),
        viewport_ssim=round(viewport_ssim, 4),
        peripheral_psnr_db=round(peripheral_psnr, 2),
        peripheral_ssim=round(peripheral_ssim, 4),
        full_psnr_db=full_psnr,
        full_ssim=full_ssim,
        roi_box=roi,
        codec=codec,
    )
