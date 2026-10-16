"""Baseline guarantees for every compression lever in the lab."""

import io
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.adaptive import AdaptiveStreamSession, BandwidthEstimator, pick_profile
from app.codecs import lossless, lossy, spatial_audio
from app.media_source import SyntheticXRSource
from app.server import app
from app.viewport import foveated_jpeg, gaze_roi

BIG = SyntheticXRSource()  # full 1280x720 SBS frame
SMALL = SyntheticXRSource(320, 180)  # fast for pure-Python lossless roundtrips


@pytest.fixture(scope="module")
def frame():
    return BIG.frame(1.0)


def test_lossless_roundtrips_and_dictionary_beats_entropy(frame):
    raw = frame.tobytes()
    huff = lossless.huffman_encode(raw)
    zlb = lossless.zlib_compress(raw)

    assert lossless.rle_decode(lossless.rle_encode(raw)) == raw
    assert lossless.huffman_decode(huff) == raw
    assert lossless.compression_ratio(len(raw), len(huff)) > 1.0
    # Dictionary coding (LZ77 + entropy) subsumes run-length, so it wins on media.
    assert len(zlb) < len(huff)


def test_lossless_roundtrip_on_small_frame():
    raw = SMALL.frame(0.5).tobytes()
    assert lossless.huffman_decode(lossless.huffman_encode(raw)) == raw


def test_jpeg_quality_ladder_trades_bytes_for_fidelity(frame):
    sizes, psnrs = [], []
    for q in (90, 75, 55, 35):
        blob, rec = lossy.jpeg_roundtrip(frame, q)
        sizes.append(len(blob))
        psnrs.append(lossy.psnr(frame, rec))
    assert sizes == sorted(sizes, reverse=True)
    assert psnrs == sorted(psnrs, reverse=True)
    assert psnrs[-1] > 25.0  # even the cheapest rung stays watchable


def test_block_dct_quantization_pipeline(frame):
    h8, w8 = frame.height - frame.height % 8, frame.width - frame.width % 8
    rec, nonzero = lossy.dct_quantize(frame, q_scale=1.0)
    ref = frame.convert("L").crop((0, 0, w8, h8))
    assert lossy.psnr(ref, rec) > 25.0
    assert nonzero < (h8 // 8) * (w8 // 8) * 64 * 0.5


def test_chroma_subsampling_roundtrip(frame):
    y, cb_s, cr_s = lossy.chroma_subsample_420(frame)
    rec = lossy.chroma_restore_420(y, cb_s, cr_s, frame.size)
    assert lossy.psnr(frame, rec) > 25.0  # halved chroma is barely visible


def test_audio_quantization_snr_and_predictive_coding():
    foa = BIG.foa_audio(0.5)
    pcm = spatial_audio.quantize_i16(foa)
    snr = spatial_audio.audio_snr(foa, spatial_audio.dequantize_f32(pcm))
    assert snr > 60.0

    assert np.array_equal(
        spatial_audio.delta_decode(spatial_audio.delta_encode(pcm[0])), pcm[0]
    )
    raw_z = spatial_audio.zlib_channels(pcm)
    delta_z = sum(len(zlib.compress(spatial_audio.delta_encode(ch), 9)) for ch in pcm)
    assert delta_z < raw_z  # prediction before entropy coding pays on audio


def test_foa_folding_scales_with_bandwidth():
    foa = BIG.foa_audio(0.25)
    sizes = [
        spatial_audio.zlib_channels(spatial_audio.quantize_i16(spatial_audio.fold_foa(foa, ch)))
        for ch in (4, 2, 1)
    ]
    assert sizes[0] > sizes[1] > sizes[2]


def test_foveated_encoding_spends_bits_in_the_viewport(frame):
    roi = gaze_roi(frame.width, frame.height, yaw_deg=8.0)
    plain = lossy.jpeg_encode(frame, 75)
    foveated = foveated_jpeg(frame, roi, quality=75)

    assert len(foveated) < len(plain)
    rec = Image.open(io.BytesIO(foveated)).convert("RGB")
    roi_psnr = lossy.psnr(frame.crop(roi), rec.crop(roi))
    periph_box = (0, 0, roi[0], frame.height)
    periph_psnr = lossy.psnr(frame.crop(periph_box), rec.crop(periph_box))
    assert roi_psnr > periph_psnr + 3.0  # the gaze region keeps its fidelity


def test_adaptive_ladder_and_foveation_handshake():
    session = AdaptiveStreamSession()
    # A healthy link first...
    session.report_throughput(125_000, 0.25)  # 4000 kbps
    assert session.profile == "ultra" and not session.knobs["foveated"]
    # ...then a throttled link downshifts and enables viewport-aware encoding.
    for _ in range(10):
        session.report_throughput(10_000, 0.5)  # 160 kbps
    assert session.profile == "low" and session.knobs["foveated"]

    jpg = session.encode_frame(BIG, 1.0, yaw_deg=5.0)
    assert jpg[:2] == b"\xff\xd8"  # valid JPEG out the other end


def test_bandwidth_estimator_and_ladder_pick():
    est = BandwidthEstimator()
    est.update(100_000, 0.5)  # 1600 kbps
    assert pick_profile(est.kbps) == "medium"


def test_demo_server_endpoints():
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.json()["lossless"]["huffman"]["ratio"] > 1.0

    r = client.get("/audio?profile=low")
    assert r.status_code == 200 and r.content[:4] == b"RIFF"

    r = client.get("/stream?frames=2&bw_kbps=500")
    assert r.status_code == 200 and r.content.count(b"\xff\xd8") == 2
