"""Run the course experiments and print the measurement tables.

Usage: uv run python experiments.py"""

import io
import zlib

from PIL import Image

from app.adaptive import AdaptiveStreamSession
from app.codecs import lossless, lossy, spatial_audio
from app.media_source import SyntheticXRSource
from app.viewport import foveated_jpeg, gaze_roi

source = SyntheticXRSource()
frame = source.frame(1.0)
raw = frame.tobytes()
RAW = len(raw)


def row(name: str, blob_len: int):
    print(f"  {name:<26} {blob_len:>10,} B   x{lossless.compression_ratio(RAW, blob_len):.2f}")


print(f"== 1. Lossless on one raw SBS frame ({frame.width}x{frame.height}, {RAW:,} B) ==")
row("RLE (count,value)", len(lossless.rle_encode(raw)))
row("Huffman", len(lossless.huffman_encode(raw)))
row("zlib (DEFLATE)", len(lossless.zlib_compress(raw)))
row("LZMA", len(lossless.lzma_compress(raw)))

print("\n== 2. Lossy JPEG quality ladder (rate-distortion) ==")
for q in (90, 75, 55, 35):
    blob, rec = lossy.jpeg_roundtrip(frame, q)
    print(
        f"  q={q:<3} {len(blob):>9,} B   x{lossless.compression_ratio(RAW, len(blob)):>4.0f}"
        f"   PSNR {lossy.psnr(frame, rec):5.1f} dB"
    )

print("\n== 3. Block-DCT quantization sweep (JPEG core, luma) ==")
ref = frame.convert("L").crop((0, 0, frame.width - frame.width % 8, frame.height - frame.height % 8))
for scale in (0.5, 1.0, 2.0, 4.0):
    rec, nonzero = lossy.dct_quantize(frame, scale)
    print(f"  q_scale={scale:<4} surviving coefficients {nonzero:>8,}   PSNR {lossy.psnr(ref, rec):5.1f} dB")

print("\n== 4. Viewport-aware (foveated) encoding, gaze vs savings ==")
plain = len(lossy.jpeg_encode(frame, 75))
for yaw in (0, 15, 30):
    roi = gaze_roi(frame.width, frame.height, yaw_deg=float(yaw))
    fov = foveated_jpeg(frame, roi, quality=75)
    rec = Image.open(io.BytesIO(fov)).convert("RGB")
    roi_psnr = lossy.psnr(frame.crop(roi), rec.crop(roi))
    print(
        f"  gaze yaw={yaw:>2} deg  plain {plain:,} B -> foveated {len(fov):,} B"
        f" ({100 * (1 - len(fov) / plain):>2.0f}% saved)   RoI PSNR {roi_psnr:5.1f} dB"
    )

print("\n== 5. Spatial (FOA) audio compression (1 s, 4 ch) ==")
foa = source.foa_audio(1.0)
pcm = spatial_audio.quantize_i16(foa)
print(f"  raw float32 B-format        {foa.nbytes:>10,} B")
print(
    f"  int16 scalar quantization   {pcm.nbytes:>10,} B"
    f"   SNR {spatial_audio.audio_snr(foa, spatial_audio.dequantize_f32(pcm)):.1f} dB"
)
print(f"  int16 + zlib per channel    {spatial_audio.zlib_channels(pcm):>10,} B")
delta_z = sum(len(zlib.compress(spatial_audio.delta_encode(ch), 9)) for ch in pcm)
print(f"  DPCM delta + zlib/channel   {delta_z:>10,} B")
for ch in (4, 2, 1):
    folded = spatial_audio.zlib_channels(spatial_audio.quantize_i16(spatial_audio.fold_foa(foa, ch)))
    print(f"  fold to {ch}ch + zlib           {folded:>10,} B")

print("\n== 6. Bandwidth-adaptive ladder walk (EWMA controller, 8 reports/link) ==")
session = AdaptiveStreamSession()
for kbps in (5000, 4800, 1000, 300, 2500, 6000):
    sent = 50_000
    seconds = sent * 8 / (kbps * 1000)
    for _ in range(8):
        session.report_throughput(sent, seconds)
    k = session.knobs
    print(
        f"  link ~{kbps:>5} kbps -> profile {session.profile:<6}"
        f" jpeg q={k['jpeg_quality']:<3} audio {k['audio_channels']}ch  foveated={k['foveated']}"
    )
