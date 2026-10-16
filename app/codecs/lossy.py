"""Lossy codecs for the video plane: the JPEG quality ladder, the classic
8x8 block-DCT + quantization pipeline (JPEG's core transform stage), and
4:2:0 chroma subsampling."""

import io
import math

import numpy as np
from PIL import Image

# Standard JPEG Annex K luminance quantization matrix
_Q_LUMA = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)

_K = np.arange(8)
_C = np.where(_K == 0, np.sqrt(1 / 8), np.sqrt(2 / 8))
# Orthonormal DCT-II kernel: T[k, n] = c(k) * cos((2n + 1) * k * pi / 16)
_DCT = _C[:, None] * np.cos((2 * _K[None, :] + 1) * _K[:, None] * np.pi / 16)


def jpeg_encode(img: Image.Image, quality: int = 75) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def jpeg_roundtrip(img: Image.Image, quality: int = 75) -> tuple[bytes, Image.Image]:
    blob = jpeg_encode(img, quality)
    return blob, Image.open(io.BytesIO(blob)).convert("RGB")


def psnr(reference, test) -> float:
    """Peak signal-to-noise ratio in dB for images or raw arrays."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    return 10 * math.log10(255.0**2 / mse) if mse > 0 else float("inf")


def dct_quantize(img: Image.Image, q_scale: float = 1.0) -> tuple[Image.Image, int]:
    """Textbook JPEG luma pipeline: 8x8 block DCT -> quantize -> dequantize -> IDCT.
    Returns (reconstructed grayscale image, count of surviving coefficients)."""
    gray = np.asarray(img.convert("L"), dtype=np.float64) - 128.0
    h, w = gray.shape
    h8, w8 = h - h % 8, w - w % 8
    blocks = gray[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3)
    coef = _DCT @ blocks @ _DCT.T
    q = np.round(coef / (_Q_LUMA * q_scale))
    rec = _DCT.T @ (q * _Q_LUMA * q_scale) @ _DCT
    rec_img = np.clip(rec.transpose(0, 2, 1, 3).reshape(h8, w8) + 128.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rec_img, "L"), int(np.count_nonzero(q))


def chroma_subsample_420(img: Image.Image):
    """4:2:0: halve Cb/Cr resolution, keep full luma (quarter of the chroma bytes)."""
    y, cb, cr = img.convert("YCbCr").split()
    half = (img.width // 2, img.height // 2)
    return y, cb.resize(half, Image.BILINEAR), cr.resize(half, Image.BILINEAR)


def chroma_restore_420(y, cb_s, cr_s, size) -> Image.Image:
    img = Image.merge(
        "YCbCr",
        (y, cb_s.resize(size, Image.BILINEAR), cr_s.resize(size, Image.BILINEAR)),
    )
    return img.convert("RGB")
