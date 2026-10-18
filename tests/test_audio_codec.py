"""Tests for the Opus audio codec module."""

import numpy as np
import pytest

from app.codecs.audio_codec import (
    compare_audio_codecs,
    decode_opus,
    encode_opus,
    opus_roundtrip,
)
from app.media_source import SyntheticXRSource


@pytest.fixture
def test_audio():
    # 0.5 s of synthetic 4-channel FOA
    source = SyntheticXRSource()
    return source.foa_audio(0.5, fs=48000)


def test_opus_mono_roundtrip():
    # 48000 Hz, 0.5s mono tone
    fs = 48000
    t = np.arange(int(fs * 0.5)) / fs
    mono = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    blob, rec, metrics = opus_roundtrip(mono, sample_rate=fs, bitrate_kbps=64)
    assert len(blob) > 0
    assert rec.shape[0] == 1  # 1 channel
    assert abs(rec.shape[1] - len(mono)) < 960  # sample count preserved
    assert metrics["snr_db"] > 15.0
    assert metrics["encode_time_s"] > 0
    assert metrics["decode_time_s"] > 0


def test_opus_stereo_roundtrip():
    fs = 48000
    t = np.arange(int(fs * 0.5)) / fs
    ch1 = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    ch2 = (0.5 * np.cos(2 * np.pi * 880 * t) * 32767).astype(np.int16)
    stereo = np.stack([ch1, ch2])

    blob, rec, metrics = opus_roundtrip(stereo, sample_rate=fs, bitrate_kbps=96)
    assert len(blob) > 0
    assert rec.shape[0] == 2  # stereo
    assert metrics["snr_db"] > 15.0
    assert metrics["sample_rate"] == 48000


def test_opus_bitrate_ladder():
    fs = 48000
    t = np.arange(int(fs * 0.5)) / fs
    tone = (0.6 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    stereo = np.stack([tone, tone])

    sizes = []
    for br in (24, 64, 96, 128):
        blob, _, m = opus_roundtrip(stereo, sample_rate=fs, bitrate_kbps=br)
        sizes.append(len(blob))
        assert m["snr_db"] > 12.0

    # Higher target bitrates generally produce larger byte payloads
    assert sizes[-1] > sizes[0]


def test_opus_invalid_parameters():
    tone = np.zeros(1000, dtype=np.int16)
    with pytest.raises(ValueError, match="Bitrate"):
        encode_opus(tone, bitrate_kbps=0)
    with pytest.raises(ValueError, match="sample rate"):
        encode_opus(tone, sample_rate=44100)
    with pytest.raises(ValueError, match="empty"):
        decode_opus(b"")


def test_compare_audio_codecs(test_audio):
    report = compare_audio_codecs(test_audio, sample_rate=48000, opus_bitrates=(24, 64, 128))
    assert len(report) >= 6
    methods = [r["method"] for r in report]
    assert any("Raw float32" in m for m in methods)
    assert any("Scalar quant" in m for m in methods)
    assert any("DPCM" in m for m in methods)
    assert any("Opus 64" in m for m in methods)
