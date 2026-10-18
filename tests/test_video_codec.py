"""Tests for the H.264 / AVC video codec module."""

import numpy as np
import pytest
from PIL import Image

from app.codecs.video_codec import (
    H264StreamEncoder,
    compute_ssim,
    decode_frame,
    encode_frame,
    h264_roundtrip,
    validate_codec_params,
)
from app.media_source import SyntheticXRSource


@pytest.fixture
def test_frame():
    source = SyntheticXRSource(320, 180)
    return source.frame(1.0)


def test_validate_codec_params():
    validate_codec_params(320, 240, 1000, 30)
    with pytest.raises(ValueError, match="Dimensions"):
        validate_codec_params(-10, 240, 1000, 30)
    with pytest.raises(ValueError, match="Bitrate"):
        validate_codec_params(320, 240, 0, 30)
    with pytest.raises(ValueError, match="FPS"):
        validate_codec_params(320, 240, 1000, -5)


def test_compute_ssim_identical_and_different():
    img1 = Image.new("RGB", (64, 64), (100, 150, 200))
    img2 = Image.new("RGB", (64, 64), (100, 150, 200))
    assert compute_ssim(img1, img2) == pytest.approx(1.0, abs=1e-4)

    img3 = Image.new("RGB", (64, 64), (250, 20, 30))
    ssim_diff = compute_ssim(img1, img3)
    assert 0.0 <= ssim_diff < 0.5


def test_h264_roundtrip_success(test_frame):
    blob, rec, metrics = h264_roundtrip(test_frame, bitrate_kbps=800, fps=30)
    assert len(blob) > 0
    # First 3-4 bytes of Annex-B H.264 NAL unit start with 0x000001 or 0x00000001
    assert blob[:3] == b"\x00\x00\x01" or blob[:4] == b"\x00\x00\x00\x01"
    assert rec.size == test_frame.size
    assert rec.mode == "RGB"

    assert metrics["compressed_bytes"] == len(blob)
    assert metrics["encode_time_s"] > 0
    assert metrics["decode_time_s"] > 0
    assert metrics["psnr_db"] > 25.0
    assert 0.70 < metrics["ssim"] <= 1.0


def test_h264_odd_dimensions_handling():
    # 321x179 has odd width and height
    odd_img = Image.new("RGB", (321, 179), (80, 120, 160))
    blob = encode_frame(odd_img, bitrate_kbps=500, fps=15)
    rec = decode_frame(blob, target_size=(321, 179))
    assert rec.size == (321, 179)


def test_h264_decoder_empty_or_corrupt_error():
    with pytest.raises(ValueError, match="empty"):
        decode_frame(b"")
    with pytest.raises(RuntimeError):
        decode_frame(b"not a valid h264 bitstream")


def test_h264_numpy_input_support():
    arr = np.zeros((180, 320, 3), dtype=np.uint8)
    arr[:, :, 0] = 200
    blob = encode_frame(arr, bitrate_kbps=600, fps=30)
    rec = decode_frame(blob)
    assert rec.size == (320, 180)


def test_h264_stream_encoder():
    enc = H264StreamEncoder(width=160, height=120, bitrate_kbps=400, fps=30)
    img1 = Image.new("RGB", (160, 120), (200, 50, 50))
    img2 = Image.new("RGB", (160, 120), (50, 200, 50))

    chunk1 = enc.encode_frame(img1)
    chunk2 = enc.encode_frame(img2)
    flushed = enc.flush()
    enc.close()

    assert len(chunk1) > 0
    total_data = chunk1 + chunk2 + flushed
    assert len(total_data) > len(chunk1)
