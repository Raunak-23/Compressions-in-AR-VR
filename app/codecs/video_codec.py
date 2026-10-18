"""H.264 / AVC video codec module using PyAV (FFmpeg / libx264).

This module implements:
1. Real H.264 encoding and decoding round-trips via PyAV.
2. Parameter validation for dimensions, bitrates, frame rates, and formats.
3. Explicit timestamps, time bases, and YUV420p pixel format.
4. Robust bitstream packetization and flushing.
5. Quality metrics: compressed bytes, encoding time, decoding time, PSNR, and SSIM.
6. Both standalone frame-level encode/decode and reusable streaming codec classes.

Limitations of Single-Frame H.264 Encoding:
------------------------------------------
H.264 is fundamentally an inter-frame (temporal) video codec relying on Groups
of Pictures (GOPs) with I-frames (Intra), P-frames (Predicted), and B-frames
(Bi-directional). When encoding a single standalone frame:
1. The encoder must emit full SPS (Sequence Parameter Set) and PPS (Picture
   Parameter Set) NAL units alongside an IDR (Instantaneous Decoder Refresh)
   keyframe.
2. Single-frame compression cannot leverage motion estimation or temporal
   redundancy.
3. Consequently, at very low bitrates, standalone single-frame H.264 carries
   header overhead compared to JPEG, whereas continuous video streams achieve
   far higher compression ratios through inter-frame prediction.
"""

from __future__ import annotations

import fractions
import io
import time
from typing import Any, Tuple, Union

import av
import numpy as np
from PIL import Image

from app.codecs.lossy import psnr


def _to_pil_rgb(img: Union[Image.Image, np.ndarray]) -> Image.Image:
    """Normalize input into an RGB PIL Image and validate."""
    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            return img.convert("RGB")
        return img
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img).convert("RGB")
        elif img.ndim == 3 and img.shape[2] in (3, 4):
            if img.shape[2] == 4:
                return Image.fromarray(img[..., :3].astype(np.uint8), "RGB")
            return Image.fromarray(img.astype(np.uint8), "RGB")
        else:
            raise ValueError(f"Unsupported numpy image shape: {img.shape}")
    else:
        raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(img)}")


def validate_codec_params(width: int, height: int, bitrate_kbps: int, fps: int) -> None:
    """Validate encoding dimensions, bitrate, and frame rate."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensions must be positive, got {width}x{height}")
    if bitrate_kbps <= 0:
        raise ValueError(f"Bitrate must be positive, got {bitrate_kbps} kbps")
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")


def compute_ssim(reference: Union[Image.Image, np.ndarray], test: Union[Image.Image, np.ndarray], block_size: int = 8) -> float:
    """Compute mean Structural Similarity Index (SSIM) across color channels.
    
    Uses standard Wang et al. constants:
    C1 = (K1 * L)^2, C2 = (K2 * L)^2 with K1=0.01, K2=0.03, L=255.
    Evaluates statistics over non-overlapping block_size x block_size windows.
    """
    a = np.asarray(_to_pil_rgb(reference) if isinstance(reference, Image.Image) else reference, dtype=np.float64)
    b = np.asarray(_to_pil_rgb(test) if isinstance(test, Image.Image) else test, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(f"Image shapes must match for SSIM calculation: {a.shape} vs {b.shape}")

    if a.ndim == 3:
        channel_ssims = [
            compute_ssim(a[..., c], b[..., c], block_size=block_size)
            for c in range(a.shape[2])
        ]
        return float(np.mean(channel_ssims))

    h, w = a.shape
    h_adj = h - (h % block_size)
    w_adj = w - (w % block_size)

    K1, K2 = 0.01, 0.03
    L = 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    if h_adj == 0 or w_adj == 0:
        mu1, mu2 = float(np.mean(a)), float(np.mean(b))
        s1_sq, s2_sq = float(np.var(a)), float(np.var(b))
        s12 = float(np.mean((a - mu1) * (b - mu2)))
        num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (s1_sq + s2_sq + C2)
        return float(num / den)

    a_blocks = (
        a[:h_adj, :w_adj]
        .reshape(h_adj // block_size, block_size, w_adj // block_size, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block_size * block_size)
    )
    b_blocks = (
        b[:h_adj, :w_adj]
        .reshape(h_adj // block_size, block_size, w_adj // block_size, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block_size * block_size)
    )

    mu1 = np.mean(a_blocks, axis=1)
    mu2 = np.mean(b_blocks, axis=1)
    s1_sq = np.var(a_blocks, axis=1)
    s2_sq = np.var(b_blocks, axis=1)
    s12 = np.mean((a_blocks - mu1[:, None]) * (b_blocks - mu2[:, None]), axis=1)

    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1_sq + s2_sq + C2)
    return float(np.mean(num / den))


def encode_frame(
    img: Union[Image.Image, np.ndarray],
    bitrate_kbps: int = 1000,
    fps: int = 30,
    preset: str = "veryfast",
    tune: str = "zerolatency",
) -> bytes:
    """Encode a single image into an Annex-B H.264 bitstream.
    
    Parameters:
        img: Input PIL Image or NumPy RGB array.
        bitrate_kbps: Target bitrate budget in kbps.
        fps: Target framerate for timebase configuration.
        preset: libx264 speed preset (default: 'veryfast').
        tune: libx264 tuning (default: 'zerolatency').
        
    Returns:
        Raw bytes of the H.264 bitstream (containing SPS, PPS, and IDR slice).
    """
    pil_img = _to_pil_rgb(img)
    w, h = pil_img.size
    validate_codec_params(w, h, bitrate_kbps, fps)

    # libx264 YUV420p requires even width and height
    pad_w = w % 2
    pad_h = h % 2
    if pad_w or pad_h:
        padded = Image.new("RGB", (w + pad_w, h + pad_h))
        padded.paste(pil_img, (0, 0))
        target_img = padded
        enc_w, enc_h = w + pad_w, h + pad_h
    else:
        target_img = pil_img
        enc_w, enc_h = w, h

    buf = io.BytesIO()
    try:
        container = av.open(buf, mode="w", format="h264")
        stream = container.add_stream("libx264", rate=fps)
        stream.width = enc_w
        stream.height = enc_h
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = int(bitrate_kbps * 1000)
        stream.options = {"preset": preset, "tune": tune}

        vf = av.VideoFrame.from_image(target_img)
        vf.pts = 0
        vf.time_base = fractions.Fraction(1, fps)

        for pkt in stream.encode(vf):
            container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)

        container.close()
    except Exception as e:
        raise RuntimeError(f"H.264 encoding failed: {e}") from e

    data = buf.getvalue()
    if not data:
        raise RuntimeError("H.264 encoder produced empty bitstream")
    return data


def decode_frame(encoded_bytes: bytes, target_size: tuple[int, int] | None = None) -> Image.Image:
    """Decode an H.264 Annex-B bitstream into a PIL Image.
    
    Parameters:
        encoded_bytes: Raw H.264 bitstream.
        target_size: Optional (width, height) to crop/rescale to if padded.
        
    Returns:
        Reconstructed RGB PIL Image.
    """
    if not encoded_bytes:
        raise ValueError("Cannot decode empty H.264 bitstream")

    buf = io.BytesIO(encoded_bytes)
    try:
        container = av.open(buf, mode="r", format="h264")
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_image().convert("RGB"))
        container.close()
    except Exception as e:
        raise RuntimeError(f"H.264 decoding failed: {e}") from e

    if not frames:
        raise RuntimeError("H.264 decoder did not yield any frames from bitstream")

    out_img = frames[0]
    if target_size is not None and out_img.size != target_size:
        # If padded to even dimensions, crop back to target size
        if out_img.width >= target_size[0] and out_img.height >= target_size[1]:
            out_img = out_img.crop((0, 0, target_size[0], target_size[1]))
        else:
            out_img = out_img.resize(target_size, Image.BILINEAR)

    return out_img


def h264_roundtrip(
    img: Union[Image.Image, np.ndarray],
    bitrate_kbps: int = 1000,
    fps: int = 30,
) -> Tuple[bytes, Image.Image, dict[str, Any]]:
    """Perform a complete H.264 encode/decode round-trip and measure metrics.
    
    Returns:
        (encoded_bytes, reconstructed_image, metrics_dict)
    """
    pil_img = _to_pil_rgb(img)
    orig_size = pil_img.size

    t0 = time.perf_counter()
    blob = encode_frame(pil_img, bitrate_kbps=bitrate_kbps, fps=fps)
    t1 = time.perf_counter()

    rec = decode_frame(blob, target_size=orig_size)
    t2 = time.perf_counter()

    encode_time_s = t1 - t0
    decode_time_s = t2 - t1

    psnr_val = psnr(pil_img, rec)
    ssim_val = compute_ssim(pil_img, rec)

    metrics = {
        "compressed_bytes": len(blob),
        "encode_time_s": encode_time_s,
        "decode_time_s": decode_time_s,
        "psnr_db": round(psnr_val, 2) if psnr_val != float("inf") else 99.0,
        "ssim": round(ssim_val, 4),
        "bitrate_kbps": bitrate_kbps,
        "fps": fps,
        "width": orig_size[0],
        "height": orig_size[1],
    }

    return blob, rec, metrics


class H264StreamEncoder:
    """Reusable streaming H.264 encoder for continuous sequence encoding.
    
    Maintains a continuous GOP, time base, and avoids per-frame container
    initialization overhead.
    """

    def __init__(self, width: int, height: int, bitrate_kbps: int = 1000, fps: int = 30):
        validate_codec_params(width, height, bitrate_kbps, fps)
        self.width = width
        self.height = height
        self.pad_w = width % 2
        self.pad_h = height % 2
        self.enc_w = width + self.pad_w
        self.enc_h = height + self.pad_h
        self.bitrate_kbps = bitrate_kbps
        self.fps = fps
        self.pts = 0

        self.buf = io.BytesIO()
        self.container = av.open(self.buf, mode="w", format="h264")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = self.enc_w
        self.stream.height = self.enc_h
        self.stream.pix_fmt = "yuv420p"
        self.stream.bit_rate = int(bitrate_kbps * 1000)
        self.stream.options = {"preset": "veryfast", "tune": "zerolatency"}

    def encode_frame(self, img: Union[Image.Image, np.ndarray]) -> bytes:
        pil_img = _to_pil_rgb(img)
        if self.pad_w or self.pad_h:
            padded = Image.new("RGB", (self.enc_w, self.enc_h))
            padded.paste(pil_img, (0, 0))
            work_img = padded
        else:
            work_img = pil_img

        vf = av.VideoFrame.from_image(work_img)
        vf.pts = self.pts
        vf.time_base = fractions.Fraction(1, self.fps)
        self.pts += 1

        pos = self.buf.tell()
        for pkt in self.stream.encode(vf):
            self.container.mux(pkt)
        cur = self.buf.tell()
        self.buf.seek(pos)
        chunk = self.buf.read(cur - pos)
        self.buf.seek(cur)
        return chunk

    def flush(self) -> bytes:
        pos = self.buf.tell()
        for pkt in self.stream.encode(None):
            self.container.mux(pkt)
        cur = self.buf.tell()
        self.buf.seek(pos)
        chunk = self.buf.read(cur - pos)
        self.buf.seek(cur)
        return chunk

    def close(self) -> None:
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
            self.container = None
