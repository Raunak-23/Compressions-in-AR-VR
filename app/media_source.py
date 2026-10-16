"""Synthetic XR broadcast source.

Stands in for the stereoscopic rig of the source livestream: a moving bright
"idol" region on a dark structured background, rendered once per eye as a
Side-by-Side (SBS) frame, plus a 4-channel First-Order Ambisonics (FOA) audio
field with a source circling the listener. Swap this class for a tap on the
live WebRTC/JPEG pipeline to run the same experiments on the real broadcast.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from app import config


def _draw_eye(width: int, height: int, t: float, parallax_px: int) -> Image.Image:
    img = Image.new("RGB", (width, height), (18, 16, 22))
    draw = ImageDraw.Draw(img)
    # Static pillars: high-frequency edges for the encoder to chew on.
    for x in (int(width * 0.12), int(width * 0.88)):
        draw.rectangle([x, 0, x + 26, height], fill=(88, 62, 30), outline=(140, 105, 55))
    # Floor band: low-frequency structure.
    draw.rectangle([0, int(height * 0.82), width, height], fill=(38, 28, 18))
    # The moving "idol": bright high-contrast region, shifted per eye for stereo.
    cx = int(width * (0.5 + 0.25 * math.sin(2 * math.pi * 0.1 * t))) + parallax_px
    cy = int(height * 0.48)
    r = int(height * 0.22)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(196, 148, 52))
    draw.polygon(
        [(cx, cy - r // 2), (cx - r // 3, cy + r // 3), (cx + r // 3, cy + r // 3)],
        fill=(255, 232, 150),
    )
    return img


class SyntheticXRSource:
    """Generates the SBS stereo video frames and FOA spatial audio to compress."""

    def __init__(self, width: int = config.FRAME_WIDTH, height: int = config.FRAME_HEIGHT):
        self.width = width
        self.height = height

    def frame(self, t: float) -> Image.Image:
        """SBS stereo frame at time t: [left eye | right eye] with 12 px parallax."""
        img = Image.new("RGB", (self.width, self.height))
        img.paste(_draw_eye(self.width // 2, self.height, t, 0), (0, 0))
        img.paste(_draw_eye(self.width // 2, self.height, t, 12), (self.width // 2, 0))
        return img

    def foa_audio(self, duration_s: float, fs: int = 48000) -> np.ndarray:
        """FOA B-format (W, X, Y, Z), float32 in [-1, 1]; source circles at 0.25 Hz."""
        n = int(duration_s * fs)
        t = np.arange(n, dtype=np.float32) / fs
        tone = 0.6 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        azimuth = 2 * np.pi * 0.25 * t
        w = tone / np.sqrt(2.0)
        x = tone * np.cos(azimuth)
        y = tone * np.sin(azimuth)
        return np.stack([w, x, y, np.zeros_like(tone)]).astype(np.float32)
