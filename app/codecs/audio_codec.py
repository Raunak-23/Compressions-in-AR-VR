"""Real-time Opus audio codec module using PyAV (FFmpeg / libopus).

Provides:
1. Real Opus encode/decode round-trips for mono and stereo streams.
2. Evaluation across standard bitrates (24, 64, 96, 128 kbps).
3. SNR, bitstream byte counts, and timing measurements.
4. Comparison against int16 scalar quantization and DPCM + zlib.
5. Spatial audio handling notes regarding FOA down-folding.

Spatial Audio Representation Note:
----------------------------------
Standard Opus supports mono and stereo (and up to 8 surround channels with mapping).
When 4-channel First-Order Ambisonics (B-format: W, X, Y, Z) is folded into stereo
(W+X, W+Y) or mono (W + 0.5(X+Y)), the 3D spherical soundfield is collapsed:
- Height information (Z channel) is completely discarded.
- Side/front distinction is merged, losing full 360-degree acoustic localization.
- This folding is an aggressive bandwidth-reduction trade-off, not a lossless spatial compression.
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Literal, Tuple, Union

import av
import numpy as np

from app.codecs import spatial_audio


def _normalize_pcm(pcm: np.ndarray) -> Tuple[np.ndarray, int]:
    """Ensure PCM array has shape (channels, samples) and int16 dtype.
    
    Returns (pcm_i16, channel_count).
    """
    arr = np.asarray(pcm)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim != 2:
        raise ValueError(f"PCM array must be 1D or 2D (channels, samples), got shape {arr.shape}")

    channels, samples = arr.shape
    if channels not in (1, 2):
        raise ValueError(f"Opus module supports mono (1) or stereo (2) channels, got {channels} channels")

    if np.issubdtype(arr.dtype, np.floating):
        arr = spatial_audio.quantize_i16(arr)
    elif arr.dtype != np.int16:
        arr = np.clip(arr, -32768, 32767).astype(np.int16)

    return arr, channels


def encode_opus(
    pcm: np.ndarray,
    sample_rate: int = 48000,
    bitrate_kbps: int = 64,
    frame_size_ms: int = 20,
) -> bytes:
    """Encode mono or stereo PCM audio into an Ogg/Opus bitstream.
    
    Parameters:
        pcm: Audio samples, shape (channels, samples) or (samples,).
        sample_rate: Audio sampling rate (default: 48000).
        bitrate_kbps: Target Opus bitrate in kbps (e.g. 24, 64, 96, 128).
        frame_size_ms: Opus frame duration in milliseconds (default: 20 ms).
        
    Returns:
        Ogg/Opus bitstream as bytes.
    """
    if bitrate_kbps <= 0:
        raise ValueError(f"Bitrate must be positive, got {bitrate_kbps}")
    if sample_rate not in (8000, 12000, 16000, 24000, 48000):
        raise ValueError(f"Unsupported Opus sample rate: {sample_rate}")

    pcm_i16, channels = _normalize_pcm(pcm)
    layout = "mono" if channels == 1 else "stereo"

    # Opus frame size in samples (e.g. 48000 * 20 / 1000 = 960)
    frame_samples = int(sample_rate * frame_size_ms / 1000)
    total_samples = pcm_i16.shape[1]

    buf = io.BytesIO()
    try:
        container = av.open(buf, mode="w", format="ogg")
        stream = container.add_stream("libopus", rate=sample_rate, layout=layout)
        stream.bit_rate = int(bitrate_kbps * 1000)

        for i in range(0, total_samples, frame_samples):
            chunk = pcm_i16[:, i : i + frame_samples]
            if chunk.shape[1] < frame_samples:
                # Pad final frame to exact frame_samples
                pad_len = frame_samples - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad_len)))

            if channels == 1:
                packed = np.ascontiguousarray(chunk.reshape(1, -1))
            else:
                packed = np.ascontiguousarray(chunk.T).reshape(1, -1)

            af = av.AudioFrame.from_ndarray(packed, format="s16", layout=layout)
            af.sample_rate = sample_rate
            af.pts = i

            for packet in stream.encode(af):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)

        container.close()
    except Exception as e:
        raise RuntimeError(f"Opus encoding failed: {e}") from e

    data = buf.getvalue()
    if not data:
        raise RuntimeError("Opus encoder produced empty bitstream")
    return data


def decode_opus(
    blob: bytes,
    sample_rate: int = 48000,
    expected_channels: int = 2,
    target_samples: int | None = None,
) -> np.ndarray:
    """Decode an Ogg/Opus bitstream into int16 PCM audio.
    
    Returns:
        PCM array of shape (channels, samples), dtype int16.
    """
    if not blob:
        raise ValueError("Cannot decode empty Opus bitstream")

    buf = io.BytesIO(blob)
    try:
        container = av.open(buf, mode="r", format="ogg")
        decoded_frames = []
        for frame in container.decode(audio=0):
            decoded_frames.append(frame.to_ndarray())
        container.close()
    except Exception as e:
        raise RuntimeError(f"Opus decoding failed: {e}") from e

    if not decoded_frames:
        raise RuntimeError("Opus decoder produced no frames")

    # PyAV decodes audio in planar format: shape (channels, samples)
    raw = np.concatenate(decoded_frames, axis=1)

    if np.issubdtype(raw.dtype, np.floating):
        pcm = (np.clip(raw, -1.0, 1.0) * 32767.0).astype(np.int16)
    else:
        pcm = raw.astype(np.int16)

    if target_samples is not None and pcm.shape[1] > target_samples:
        pcm = pcm[:, :target_samples]

    return pcm


def opus_roundtrip(
    pcm: np.ndarray,
    sample_rate: int = 48000,
    bitrate_kbps: int = 64,
) -> Tuple[bytes, np.ndarray, Dict[str, Any]]:
    """Perform Opus encode/decode round-trip and record performance metrics."""
    pcm_i16, channels = _normalize_pcm(pcm)
    orig_samples = pcm_i16.shape[1]
    duration_s = orig_samples / sample_rate

    t0 = time.perf_counter()
    blob = encode_opus(pcm_i16, sample_rate=sample_rate, bitrate_kbps=bitrate_kbps)
    t1 = time.perf_counter()

    rec_pcm = decode_opus(blob, sample_rate=sample_rate, expected_channels=channels, target_samples=orig_samples)
    t2 = time.perf_counter()

    encode_time_s = t1 - t0
    decode_time_s = t2 - t1

    snr_val = spatial_audio.audio_snr(pcm_i16, rec_pcm)
    measured_kbps = (len(blob) * 8 / 1000) / max(duration_s, 1e-6)

    metrics = {
        "compressed_bytes": len(blob),
        "target_bitrate_kbps": bitrate_kbps,
        "measured_bitrate_kbps": round(measured_kbps, 1),
        "encode_time_s": encode_time_s,
        "decode_time_s": decode_time_s,
        "snr_db": round(snr_val, 2),
        "sample_rate": sample_rate,
        "channels": channels,
        "samples": orig_samples,
    }

    return blob, rec_pcm, metrics


def compare_audio_codecs(
    foa_audio: np.ndarray,
    sample_rate: int = 48000,
    opus_bitrates: Tuple[int, ...] = (24, 64, 96, 128),
) -> List[Dict[str, Any]]:
    """Benchmark raw PCM, int16 quantization, DPCM + zlib, and Opus.
    
    Accepts 4-channel FOA or 2-channel stereo.
    """
    results: List[Dict[str, Any]] = []
    duration_s = foa_audio.shape[1] / sample_rate

    # 1. Raw float32 representation
    raw_f32_bytes = foa_audio.nbytes
    results.append({
        "method": "Raw float32 (4ch FOA)",
        "bytes": raw_f32_bytes,
        "bitrate_kbps": round((raw_f32_bytes * 8 / 1000) / duration_s, 1),
        "snr_db": float("inf"),
        "channels": foa_audio.shape[0],
        "encode_time_s": 0.0,
        "decode_time_s": 0.0,
    })

    # 2. int16 scalar quantization
    t0 = time.perf_counter()
    pcm_i16 = spatial_audio.quantize_i16(foa_audio)
    t1 = time.perf_counter()
    snr_i16 = spatial_audio.audio_snr(foa_audio, spatial_audio.dequantize_f32(pcm_i16))
    results.append({
        "method": "Scalar quant int16 (4ch FOA)",
        "bytes": pcm_i16.nbytes,
        "bitrate_kbps": round((pcm_i16.nbytes * 8 / 1000) / duration_s, 1),
        "snr_db": round(snr_i16, 1),
        "channels": pcm_i16.shape[0],
        "encode_time_s": round(t1 - t0, 5),
        "decode_time_s": 0.0,
    })

    # 3. First-order DPCM + zlib
    import zlib
    t0 = time.perf_counter()
    dpcm_bytes = sum(len(zlib.compress(spatial_audio.delta_encode(ch), 9)) for ch in pcm_i16)
    t1 = time.perf_counter()
    results.append({
        "method": "DPCM delta + zlib (4ch FOA)",
        "bytes": dpcm_bytes,
        "bitrate_kbps": round((dpcm_bytes * 8 / 1000) / duration_s, 1),
        "snr_db": round(snr_i16, 1),  # lossless on int16
        "channels": pcm_i16.shape[0],
        "encode_time_s": round(t1 - t0, 5),
        "decode_time_s": 0.0,
    })

    # 4. Channel folding (FOA -> Stereo) + DPCM + zlib
    t0 = time.perf_counter()
    folded_stereo = spatial_audio.quantize_i16(spatial_audio.fold_foa(foa_audio, 2))
    folded_stereo_bytes = sum(len(zlib.compress(spatial_audio.delta_encode(ch), 9)) for ch in folded_stereo)
    t1 = time.perf_counter()
    results.append({
        "method": "Fold to stereo + DPCM + zlib",
        "bytes": folded_stereo_bytes,
        "bitrate_kbps": round((folded_stereo_bytes * 8 / 1000) / duration_s, 1),
        "snr_db": round(spatial_audio.audio_snr(pcm_i16[:2], folded_stereo), 1),
        "channels": 2,
        "encode_time_s": round(t1 - t0, 5),
        "decode_time_s": 0.0,
    })

    # 5. Opus across target bitrates on stereo folded audio
    for br in opus_bitrates:
        blob, rec_pcm, m = opus_roundtrip(folded_stereo, sample_rate=sample_rate, bitrate_kbps=br)
        results.append({
            "method": f"Opus {br} kbps (folded stereo)",
            "bytes": m["compressed_bytes"],
            "bitrate_kbps": m["measured_bitrate_kbps"],
            "snr_db": m["snr_db"],
            "encode_time_s": m["encode_time_s"],
            "decode_time_s": m["decode_time_s"],
            "channels": 2,
        })

    return results
