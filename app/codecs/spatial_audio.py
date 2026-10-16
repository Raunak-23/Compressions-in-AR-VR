"""Compression for the spatial (FOA) audio plane.

The source livestream carries 4-channel First-Order Ambisonics (B-format).
Course levers implemented here:
- Scalar quantization: float32 -> int16 PCM (2:1 byte ratio, ~90 dB SNR).
- Predictive coding: first-order DPCM deltas before entropy coding.
- Channel folding: 4ch B-format -> stereo -> mono as bandwidth drops."""

import math
import zlib

import numpy as np


def quantize_i16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def dequantize_f32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32767.0


def delta_encode(channel: np.ndarray) -> bytes:
    """First-order DPCM: transmit successive differences (small ints compress better)."""
    d = np.diff(channel.astype(np.int32), prepend=np.int32(0)).astype(np.int16)
    return d.tobytes()


def delta_decode(blob: bytes) -> np.ndarray:
    return np.cumsum(np.frombuffer(blob, dtype=np.int16), dtype=np.int64).astype(np.int16)


def zlib_channels(pcm: np.ndarray) -> int:
    """Compressed byte size when each channel is zlib-deflated separately."""
    return sum(len(zlib.compress(ch.tobytes(), 9)) for ch in pcm)


def fold_foa(foa: np.ndarray, channels: int) -> np.ndarray:
    """Bandwidth-scaled channel folding of B-format audio (input shape (4, n))."""
    if channels >= 4:
        return foa
    w, x, y = foa[0], foa[1], foa[2]
    if channels == 2:
        return np.stack([w + x, w + y])
    if channels == 1:
        return (w + 0.5 * (x + y))[None, :]
    raise ValueError(f"Unsupported channel count: {channels}")


def audio_snr(reference: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-noise ratio in dB between original and reconstructed audio."""
    ref = reference.astype(np.float64)
    err = ref - test.astype(np.float64)
    return 10 * math.log10(float(np.sum(ref**2)) / max(float(np.sum(err**2)), 1e-12))
