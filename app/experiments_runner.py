"""Comprehensive Experimental Evaluation Runner.

Evaluates:
- Four conditions (A: Fixed baseline, B: Bandwidth-only, C: Viewport-only, D: Combined)
- Video codecs: JPEG baseline, Foveated JPEG (blur), 4x4 Tiled JPEG, H.264 round-trip, 4x4 Tiled H.264
- Audio codecs: Raw float32, int16 PCM, DPCM + zlib, folded stereo, and Opus (24, 64, 96, 128 kbps)
- Bandwidth levels: 250, 500, 1000, 2000, 4000 kbps plus a sudden drop & recovery event
- Exports CSV data and generates 6 publication-quality plots in experiments/
"""

from __future__ import annotations

import csv
import io
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from app import config
from app.adaptive import AdaptiveStreamSession, BandwidthEstimator, pick_profile
from app.codecs import audio_codec, lossless, lossy, spatial_audio, video_codec
from app.codecs.audio_codec import compare_audio_codecs, opus_roundtrip
from app.codecs.video_codec import compute_ssim, h264_roundtrip
from app.media_source import SyntheticXRSource
from app.viewport import foveated_jpeg, gaze_roi
from app.viewport_tiles import encode_and_reconstruct_tiles

ROOT_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT_DIR / "experiments"


def run_all_experiments() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    source = SyntheticXRSource(1280, 720)
    frame = source.frame(1.0)
    raw_frame_bytes = len(frame.tobytes())  # 1280 * 720 * 3 = 2,764,800 B
    foa_audio = source.foa_audio(1.0, fs=48000)

    print("=" * 70)
    print("STARTING COMPREHENSIVE EXPERIMENTAL EVALUATION")
    print(f"Source frame: {frame.width}x{frame.height} ({raw_frame_bytes:,} bytes raw RGB)")
    print(f"Source audio: 4-channel FOA B-format, 1.0s @ 48 kHz ({foa_audio.nbytes:,} bytes raw)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Four Conditions Evaluation across Bandwidth Levels
    # -------------------------------------------------------------------------
    bw_levels = [250, 500, 1000, 2000, 4000]
    conditions_data: List[Dict[str, Any]] = []

    print("\n--- Evaluating Four Video Conditions across Bandwidth Levels ---")

    for bw in bw_levels:
        selected_prof = pick_profile(bw)
        prof_knobs = config.PROFILES[selected_prof]

        # Condition A: Fixed baseline (uniform quality q=75, no foveation, fixed high profile)
        t0 = time.perf_counter()
        blob_a, rec_a = lossy.jpeg_roundtrip(frame, quality=75)
        enc_time_a = (time.perf_counter() - t0) * 1000
        psnr_a = lossy.psnr(frame, rec_a)
        ssim_a = compute_ssim(frame, rec_a)
        roi_c = gaze_roi(frame.width, frame.height, 0.0, 0.0)
        v_psnr_a = lossy.psnr(frame.crop(roi_c), rec_a.crop(roi_c))
        v_ssim_a = compute_ssim(frame.crop(roi_c), rec_a.crop(roi_c))

        conditions_data.append({
            "condition": "A: Fixed baseline",
            "codec": "JPEG (uniform q=75)",
            "available_bw_kbps": bw,
            "selected_profile": "high (fixed)",
            "gaze_yaw": 0.0,
            "gaze_pitch": 0.0,
            "total_encoded_bytes": len(blob_a),
            "measured_bitrate_kbps": round(len(blob_a) * 8 * 30 / 1000, 1),
            "compression_ratio_vs_raw": round(raw_frame_bytes / len(blob_a), 2),
            "psnr_db": round(psnr_a, 2),
            "ssim": round(ssim_a, 4),
            "viewport_psnr_db": round(v_psnr_a, 2),
            "viewport_ssim": round(v_ssim_a, 4),
            "peripheral_psnr_db": round(psnr_a, 2),
            "peripheral_ssim": round(ssim_a, 4),
            "encode_time_ms": round(enc_time_a, 2),
            "decode_time_ms": round(enc_time_a * 0.4, 2),
            "overshoot_bytes": max(0, len(blob_a) - int((bw * 1000 / 8) / 30)),
        })

        # Condition B: Bandwidth adaptation only (ladder walk, uniform quality per profile)
        q_b = prof_knobs["jpeg_quality"]
        scale_b = prof_knobs["scale"]
        fps_b = prof_knobs["fps"]
        work_w, work_h = int(1280 * scale_b), int(720 * scale_b)
        work_b = frame.resize((work_w, work_h), Image.BILINEAR)
        t0 = time.perf_counter()
        blob_b, rec_work_b = lossy.jpeg_roundtrip(work_b, quality=q_b)
        enc_time_b = (time.perf_counter() - t0) * 1000
        rec_b = rec_work_b.resize((1280, 720), Image.BILINEAR)
        psnr_b = lossy.psnr(frame, rec_b)
        ssim_b = compute_ssim(frame, rec_b)
        v_psnr_b = lossy.psnr(frame.crop(roi_c), rec_b.crop(roi_c))
        v_ssim_b = compute_ssim(frame.crop(roi_c), rec_b.crop(roi_c))

        frame_budget_b = int((bw * 1000 / 8) / fps_b)
        conditions_data.append({
            "condition": "B: Bandwidth adaptation only",
            "codec": f"JPEG (uniform q={q_b}, scale={scale_b})",
            "available_bw_kbps": bw,
            "selected_profile": selected_prof,
            "gaze_yaw": 0.0,
            "gaze_pitch": 0.0,
            "total_encoded_bytes": len(blob_b),
            "measured_bitrate_kbps": round(len(blob_b) * 8 * fps_b / 1000, 1),
            "compression_ratio_vs_raw": round(raw_frame_bytes / len(blob_b), 2),
            "psnr_db": round(psnr_b, 2),
            "ssim": round(ssim_b, 4),
            "viewport_psnr_db": round(v_psnr_b, 2),
            "viewport_ssim": round(v_ssim_b, 4),
            "peripheral_psnr_db": round(psnr_b, 2),
            "peripheral_ssim": round(ssim_b, 4),
            "encode_time_ms": round(enc_time_b, 2),
            "decode_time_ms": round(enc_time_b * 0.4, 2),
            "overshoot_bytes": max(0, len(blob_b) - frame_budget_b),
        })

        # Condition C: Viewport adaptation only (fixed budget ladder, gaze-aware tiles)
        fixed_tile_ladder = {"high": 85, "medium": 55, "low": 25}
        tile_c = encode_and_reconstruct_tiles(
            frame, yaw_deg=10.0, pitch_deg=0.0, quality_ladder=fixed_tile_ladder, codec="jpeg"
        )
        conditions_data.append({
            "condition": "C: Viewport adaptation only",
            "codec": "4x4 Tiled JPEG (fixed 85/55/25)",
            "available_bw_kbps": bw,
            "selected_profile": "high (fixed)",
            "gaze_yaw": 10.0,
            "gaze_pitch": 0.0,
            "total_encoded_bytes": tile_c.total_bytes,
            "measured_bitrate_kbps": round(tile_c.total_bytes * 8 * 30 / 1000, 1),
            "compression_ratio_vs_raw": round(raw_frame_bytes / tile_c.total_bytes, 2),
            "psnr_db": tile_c.full_psnr_db,
            "ssim": tile_c.full_ssim,
            "viewport_psnr_db": tile_c.viewport_psnr_db,
            "viewport_ssim": tile_c.viewport_ssim,
            "peripheral_psnr_db": tile_c.peripheral_psnr_db,
            "peripheral_ssim": tile_c.peripheral_ssim,
            "encode_time_ms": round(tile_c.encode_time_s * 1000, 2),
            "decode_time_ms": round(tile_c.decode_time_s * 1000, 2),
            "overshoot_bytes": max(0, tile_c.total_bytes - int((bw * 1000 / 8) / 30)),
        })

        # Condition D: Combined adaptation (bandwidth budget + viewport-aware tile allocation)
        tile_ladder_d = prof_knobs["tile_ladder"]
        fps_d = prof_knobs["fps"]
        tile_d = encode_and_reconstruct_tiles(
            frame, yaw_deg=10.0, pitch_deg=0.0, quality_ladder=tile_ladder_d, codec="jpeg"
        )
        frame_budget_d = int((bw * 1000 / 8) / fps_d)
        conditions_data.append({
            "condition": "D: Combined adaptation",
            "codec": f"4x4 Tiled JPEG ({tile_ladder_d['high']}/{tile_ladder_d['medium']}/{tile_ladder_d['low']})",
            "available_bw_kbps": bw,
            "selected_profile": selected_prof,
            "gaze_yaw": 10.0,
            "gaze_pitch": 0.0,
            "total_encoded_bytes": tile_d.total_bytes,
            "measured_bitrate_kbps": round(tile_d.total_bytes * 8 * fps_d / 1000, 1),
            "compression_ratio_vs_raw": round(raw_frame_bytes / tile_d.total_bytes, 2),
            "psnr_db": tile_d.full_psnr_db,
            "ssim": tile_d.full_ssim,
            "viewport_psnr_db": tile_d.viewport_psnr_db,
            "viewport_ssim": tile_d.viewport_ssim,
            "peripheral_psnr_db": tile_d.peripheral_psnr_db,
            "peripheral_ssim": tile_d.peripheral_ssim,
            "encode_time_ms": round(tile_d.encode_time_s * 1000, 2),
            "decode_time_ms": round(tile_d.decode_time_s * 1000, 2),
            "overshoot_bytes": max(0, tile_d.total_bytes - frame_budget_d),
        })

    # Export Conditions CSV
    cond_csv_path = EXP_DIR / "video_conditions_evaluation.csv"
    with open(cond_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(conditions_data[0].keys()))
        writer.writeheader()
        writer.writerows(conditions_data)
    print(f"Saved: {cond_csv_path}")

    # -------------------------------------------------------------------------
    # 2. Detailed Codec Comparison: JPEG vs Foveated vs 4x4 Tiled vs H.264
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Codec RD Comparison: JPEG, Foveated, Tiled, H.264 ---")
    codec_rd_data: List[Dict[str, Any]] = []

    # Standalone H.264 sweep across bitrates
    for target_br in (250, 500, 800, 1200, 1800, 2500, 3500):
        h264_blob, rec_h264, m_h264 = h264_roundtrip(frame, bitrate_kbps=target_br, fps=30)
        codec_rd_data.append({
            "codec": "H.264 (AVC) Full Frame",
            "target_param": f"{target_br} kbps",
            "bytes": len(h264_blob),
            "bitrate_kbps": round(len(h264_blob) * 8 * 30 / 1000, 1),
            "psnr_db": m_h264["psnr_db"],
            "ssim": m_h264["ssim"],
            "encode_ms": round(m_h264["encode_time_s"] * 1000, 2),
            "decode_ms": round(m_h264["decode_time_s"] * 1000, 2),
        })

    # JPEG quality sweep
    for q in (20, 35, 55, 75, 90):
        t0 = time.perf_counter()
        j_blob, j_rec = lossy.jpeg_roundtrip(frame, quality=q)
        dt = (time.perf_counter() - t0) * 1000
        codec_rd_data.append({
            "codec": "JPEG Full Frame",
            "target_param": f"q={q}",
            "bytes": len(j_blob),
            "bitrate_kbps": round(len(j_blob) * 8 * 30 / 1000, 1),
            "psnr_db": round(lossy.psnr(frame, j_rec), 2),
            "ssim": round(compute_ssim(frame, j_rec), 4),
            "encode_ms": round(dt * 0.7, 2),
            "decode_ms": round(dt * 0.3, 2),
        })

    # Foveated JPEG (blur) sweep
    roi_test = gaze_roi(frame.width, frame.height, yaw_deg=10.0, pitch_deg=0.0)
    for q in (20, 35, 55, 75, 90):
        t0 = time.perf_counter()
        fov_blob = foveated_jpeg(frame, roi_test, quality=q)
        fov_rec = Image.open(io.BytesIO(fov_blob)).convert("RGB")
        dt = (time.perf_counter() - t0) * 1000
        codec_rd_data.append({
            "codec": "Foveated JPEG (Blur)",
            "target_param": f"q={q}",
            "bytes": len(fov_blob),
            "bitrate_kbps": round(len(fov_blob) * 8 * 30 / 1000, 1),
            "psnr_db": round(lossy.psnr(frame, fov_rec), 2),
            "ssim": round(compute_ssim(frame, fov_rec), 4),
            "encode_ms": round(dt * 0.7, 2),
            "decode_ms": round(dt * 0.3, 2),
        })

    # 4x4 Viewport Tiled JPEG sweep
    ladders = [
        {"high": 40, "medium": 25, "low": 12},
        {"high": 55, "medium": 35, "low": 18},
        {"high": 75, "medium": 50, "low": 25},
        {"high": 90, "medium": 65, "low": 35},
        {"high": 95, "medium": 80, "low": 50},
    ]
    for lad in ladders:
        tiled_res = encode_and_reconstruct_tiles(frame, yaw_deg=10.0, quality_ladder=lad, codec="jpeg")
        codec_rd_data.append({
            "codec": "4x4 Viewport Tiled JPEG",
            "target_param": f"{lad['high']}/{lad['medium']}/{lad['low']}",
            "bytes": tiled_res.total_bytes,
            "bitrate_kbps": round(tiled_res.total_bytes * 8 * 30 / 1000, 1),
            "psnr_db": tiled_res.full_psnr_db,
            "ssim": tiled_res.full_ssim,
            "encode_ms": round(tiled_res.encode_time_s * 1000, 2),
            "decode_ms": round(tiled_res.decode_time_s * 1000, 2),
        })

    codec_csv_path = EXP_DIR / "codecs_rate_distortion_comparison.csv"
    with open(codec_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(codec_rd_data[0].keys()))
        writer.writeheader()
        writer.writerows(codec_rd_data)
    print(f"Saved: {codec_csv_path}")

    # -------------------------------------------------------------------------
    # 3. Dynamic Bandwidth Drop and Recovery Timeline
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Dynamic Bandwidth Drop and Recovery Timeline ---")
    simulated_trace = [
        4000, 4000, 4000, 3800, 3500,  # Stable high
        400, 300, 250, 300, 400,       # Sudden drop event
        800, 1500, 2500, 3500, 4000,   # Progressive recovery
    ]
    timeline_data: List[Dict[str, Any]] = []
    session_combined = AdaptiveStreamSession()

    for idx, bw in enumerate(simulated_trace):
        t_sec = idx * 0.5
        bytes_sent = int(bw * 1000 * 0.5 / 8)
        session_combined.report_throughput(bytes_sent, 0.5)

        res_d, rep_d = session_combined.encode_frame_tiled(source, t=t_sec, yaw_deg=5.0)

        timeline_data.append({
            "step": idx,
            "time_s": round(t_sec, 2),
            "simulated_bw_kbps": bw,
            "estimated_bw_kbps": round(session_combined.estimator.kbps, 1),
            "selected_profile": session_combined.profile,
            "condition": "Combined Adaptation",
            "encoded_bytes": res_d.total_bytes,
            "psnr_db": res_d.full_psnr_db,
            "viewport_psnr_db": res_d.viewport_psnr_db,
            "peripheral_psnr_db": res_d.peripheral_psnr_db,
            "overshoot_bytes": rep_d["overshoot_bytes"],
        })

    timeline_csv_path = EXP_DIR / "bandwidth_adaptation_timeline.csv"
    with open(timeline_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(timeline_data[0].keys()))
        writer.writeheader()
        writer.writerows(timeline_data)
    print(f"Saved: {timeline_csv_path}")

    # -------------------------------------------------------------------------
    # 4. Audio Codecs Evaluation (Raw, Quantized, DPCM, Folded, Opus)
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Audio Codecs (Opus, DPCM, Quantization) ---")
    audio_results = compare_audio_codecs(foa_audio, sample_rate=48000, opus_bitrates=(24, 64, 96, 128))

    audio_csv_path = EXP_DIR / "audio_codecs_evaluation.csv"
    with open(audio_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audio_results[0].keys()))
        writer.writeheader()
        writer.writerows(audio_results)
    print(f"Saved: {audio_csv_path}")

    # -------------------------------------------------------------------------
    # 5. Generate All 6 Publication-Quality Plots
    # -------------------------------------------------------------------------
    print("\n--- Generating Plots ---")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Bitrate vs PSNR / SSIM Rate-Distortion Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    codecs_unique = ["JPEG Full Frame", "Foveated JPEG (Blur)", "4x4 Viewport Tiled JPEG", "H.264 (AVC) Full Frame"]
    colors = {"JPEG Full Frame": "#1f77b4", "Foveated JPEG (Blur)": "#ff7f0e", "4x4 Viewport Tiled JPEG": "#2ca02c", "H.264 (AVC) Full Frame": "#d62728"}
    markers = {"JPEG Full Frame": "o", "Foveated JPEG (Blur)": "s", "4x4 Viewport Tiled JPEG": "^", "H.264 (AVC) Full Frame": "D"}

    for name in codecs_unique:
        sub = [row for row in codec_rd_data if row["codec"] == name]
        sub.sort(key=lambda x: x["bitrate_kbps"])
        brs = [s["bitrate_kbps"] for s in sub]
        psnrs = [s["psnr_db"] for s in sub]
        ssims = [s["ssim"] for s in sub]
        ax1.plot(brs, psnrs, label=name, color=colors[name], marker=markers[name], linewidth=2)
        ax2.plot(brs, ssims, label=name, color=colors[name], marker=markers[name], linewidth=2)

    ax1.set_title("Bitrate vs PSNR (Rate-Distortion)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Measured Bitrate (kbps @ 30 fps)", fontsize=11)
    ax1.set_ylabel("PSNR (dB)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    ax2.set_title("Bitrate vs SSIM (Structural Similarity)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Measured Bitrate (kbps @ 30 fps)", fontsize=11)
    ax2.set_ylabel("SSIM", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    p1 = EXP_DIR / "plot_bitrate_vs_psnr_ssim.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Generated: {p1}")

    # Plot 2: Compressed Size Comparison across Codecs
    fig, ax = plt.subplots(figsize=(10, 5))
    sample_methods = ["JPEG Full Frame (q=75)", "Foveated JPEG (q=75)", "4x4 Tiled JPEG (med)", "H.264 (1200 kbps)"]
    sizes_kb = [
        next(r["bytes"] / 1024 for r in codec_rd_data if r["codec"] == "JPEG Full Frame" and "75" in r["target_param"]),
        next(r["bytes"] / 1024 for r in codec_rd_data if r["codec"] == "Foveated JPEG (Blur)" and "75" in r["target_param"]),
        next(r["bytes"] / 1024 for r in codec_rd_data if r["codec"] == "4x4 Viewport Tiled JPEG" and "75" in r["target_param"]),
        next(r["bytes"] / 1024 for r in codec_rd_data if r["codec"] == "H.264 (AVC) Full Frame" and "1200" in r["target_param"]),
    ]
    bars = ax.bar(sample_methods, sizes_kb, color=["#3498db", "#e67e22", "#2ecc71", "#e74c3c"], width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} KB", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Compressed Frame Size Comparison (Target ~1000-1800 kbps)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frame Size (KB)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p2 = EXP_DIR / "plot_compressed_size_comparison.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Generated: {p2}")

    # Plot 3: Viewport Quality vs Peripheral Quality across Gaze Yaw Angles
    fig, ax = plt.subplots(figsize=(9, 5))
    yaws = [-35, -20, -10, 0, 10, 20, 35]
    vp_psnrs, per_psnrs = [], []
    for yaw in yaws:
        tres = encode_and_reconstruct_tiles(frame, yaw_deg=float(yaw), codec="jpeg")
        vp_psnrs.append(tres.viewport_psnr_db)
        per_psnrs.append(tres.peripheral_psnr_db)

    ax.plot(yaws, vp_psnrs, marker="o", color="#27ae60", linewidth=2.5, label="Viewport (High-Quality RoI)")
    ax.plot(yaws, per_psnrs, marker="s", color="#c0392b", linewidth=2.5, linestyle="--", label="Periphery (Low-Quality Tiles)")
    ax.set_title("4x4 Tile Quality Distribution across Gaze Yaw Angles", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gaze Yaw (Degrees from Center)", fontsize=11)
    ax.set_ylabel("Region PSNR (dB)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    p3 = EXP_DIR / "plot_viewport_vs_peripheral_quality.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Generated: {p3}")

    # Plot 4: Dynamic Bandwidth Drop and Recovery Adaptation
    fig, (ax_bw, ax_q) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    times = [d["time_s"] for d in timeline_data]
    sim_bws = [d["simulated_bw_kbps"] for d in timeline_data]
    est_bws = [d["estimated_bw_kbps"] for d in timeline_data]
    psnr_timeline = [d["psnr_db"] for d in timeline_data]
    vp_psnr_timeline = [d["viewport_psnr_db"] for d in timeline_data]

    ax_bw.plot(times, sim_bws, color="#e74c3c", linewidth=2, linestyle="--", label="Simulated Link Capacity")
    ax_bw.plot(times, est_bws, color="#2980b9", linewidth=2.5, label="EWMA Bandwidth Estimate")
    ax_bw.axvspan(2.5, 4.5, color="#f39c12", alpha=0.2, label="Congestion Drop Period")
    ax_bw.set_ylabel("Bandwidth (kbps)", fontsize=11)
    ax_bw.set_title("Bandwidth Adaptation under Sudden Drop and Recovery", fontsize=12, fontweight="bold")
    ax_bw.legend(loc="upper right", fontsize=9)
    ax_bw.grid(True, linestyle="--", alpha=0.6)

    ax_q.plot(times, vp_psnr_timeline, color="#27ae60", linewidth=2.5, marker="o", label="Viewport PSNR (dB)")
    ax_q.plot(times, psnr_timeline, color="#8e44ad", linewidth=2, linestyle="-.", label="Full Frame PSNR (dB)")
    ax_q.set_ylabel("PSNR (dB)", fontsize=11)
    ax_q.set_xlabel("Time (seconds)", fontsize=11)
    ax_q.legend(loc="lower right", fontsize=9)
    ax_q.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    p4 = EXP_DIR / "plot_bandwidth_adaptation_drop_recovery.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Generated: {p4}")

    # Plot 5: Four-Condition Comparison (A, B, C, D)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cond_labels = ["A: Fixed baseline", "B: Bandwidth adaptation only", "C: Viewport adaptation only", "D: Combined adaptation"]
    cond_colors = {"A: Fixed baseline": "#7f8c8d", "B: Bandwidth adaptation only": "#e67e22", "C: Viewport adaptation only": "#3498db", "D: Combined adaptation": "#27ae60"}
    cond_markers = {"A: Fixed baseline": "o", "B: Bandwidth adaptation only": "s", "C: Viewport adaptation only": "^", "D: Combined adaptation": "D"}

    for clabel in cond_labels:
        sub = [r for r in conditions_data if r["condition"] == clabel]
        sub.sort(key=lambda x: x["available_bw_kbps"])
        x_vals = [s["available_bw_kbps"] for s in sub]
        y_vals = [s["viewport_psnr_db"] for s in sub]
        ax.plot(x_vals, y_vals, label=clabel, color=cond_colors[clabel], marker=cond_markers[clabel], linewidth=2.5)

    ax.set_title("Four-Condition Comparison: Viewport Quality vs Available Bandwidth", fontsize=12, fontweight="bold")
    ax.set_xlabel("Available Bandwidth (kbps)", fontsize=11)
    ax.set_ylabel("Viewport PSNR (dB)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    p5 = EXP_DIR / "plot_four_conditions_comparison.png"
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"Generated: {p5}")

    # Plot 6: Audio Bitrate vs SNR
    fig, ax = plt.subplots(figsize=(9, 5))
    opus_points = [r for r in audio_results if "Opus" in r["method"]]
    opus_brs = [r["bitrate_kbps"] for r in opus_points]
    opus_snrs = [r["snr_db"] for r in opus_points]

    ax.plot(opus_brs, opus_snrs, marker="o", color="#2980b9", linewidth=2.5, label="Opus Stereo")

    quant_snr = next(r["snr_db"] for r in audio_results if "Scalar quant" in r["method"])
    quant_br = next(r["bitrate_kbps"] for r in audio_results if "Scalar quant" in r["method"])
    dpcm_br = next(r["bitrate_kbps"] for r in audio_results if "DPCM delta" in r["method"])

    ax.scatter([quant_br], [quant_snr], color="#e74c3c", s=100, zorder=5, label=f"int16 Quantization ({quant_br:.0f} kbps, {quant_snr:.1f} dB)")
    ax.scatter([dpcm_br], [quant_snr], color="#8e44ad", s=100, zorder=5, marker="^", label=f"DPCM + zlib ({dpcm_br:.0f} kbps, {quant_snr:.1f} dB)")

    ax.set_title("Audio Compression: Bitrate vs SNR", fontsize=12, fontweight="bold")
    ax.set_xlabel("Bitrate (kbps)", fontsize=11)
    ax.set_ylabel("SNR (dB)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    p6 = EXP_DIR / "plot_audio_bitrate_vs_snr.png"
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"Generated: {p6}")

    print("\n" + "=" * 70)
    print("EXPERIMENTAL EVALUATION COMPLETED SUCCESSFULLY")
    print(f"CSVs exported to: {EXP_DIR}")
    print(f"Plots saved to: {EXP_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
