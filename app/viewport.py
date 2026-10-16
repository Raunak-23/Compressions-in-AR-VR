"""Viewport-aware (foveated) encoding.

The source XR stream is a fixed ~100 deg x 62 deg frustum. The client's gaze
(yaw/pitch in degrees from view centre) picks a sharp Region-of-Interest; the
periphery is pre-blurred before the single JPEG encode, so bits are spent only
where the eye is looking."""

import io

import numpy as np
from PIL import Image, ImageFilter

from app import config


def gaze_roi(
    width: int,
    height: int,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roi_frac: float = 0.35,
) -> tuple[int, int, int, int]:
    """Map gaze offsets (deg from view centre) to a pixel RoI box (x0, y0, x1, y1)."""
    cx = width / 2 + (yaw_deg / config.HFOV_DEG) * width
    cy = height / 2 - (pitch_deg / config.VFOV_DEG) * height
    rw, rh = width * roi_frac, height * roi_frac
    x0 = min(max(cx - rw / 2, 0.0), width - rw)
    y0 = min(max(cy - rh / 2, 0.0), height - rh)
    return int(x0), int(y0), int(x0 + rw), int(y0 + rh)


def foveate_image(img: Image.Image, roi: tuple[int, int, int, int], blur_radius: float = 6.0) -> Image.Image:
    """Blend one frame: everything outside the gaze RoI is pre-blurred."""
    base = img.filter(ImageFilter.GaussianBlur(blur_radius))
    base.paste(img.crop(roi), roi[:2])
    return base


def foveated_jpeg(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    quality: int = 75,
    blur_radius: float = 6.0,
) -> bytes:
    """One JPEG encode of the frame with only the gaze RoI kept sharp."""
    buf = io.BytesIO()
    foveate_image(img, roi, blur_radius).save(buf, "JPEG", quality=quality)
    return buf.getvalue()
