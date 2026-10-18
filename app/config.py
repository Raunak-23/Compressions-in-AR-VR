"""Configuration. Profile names and media geometry mirror the source XR
livestream (Virtual Darshan): fixed-viewpoint 100° SBS stereo video and
4-channel First-Order Ambisonics spatial audio."""

import os

# Fixed-XR media geometry (matches the source livestream)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
HFOV_DEG = 100
VFOV_DEG = 62
FOA_CHANNELS = 4  # B-format: W (omni), X, Y, Z (direction cosines)

# Bandwidth ladder: (minimum kbps, profile). First match wins.
BANDWIDTH_LADDER = [
    (3500, "ultra"),
    (1800, "high"),
    (800, "medium"),
    (0, "low"),
]

# Per-profile encoder knobs. `foveated` = spend bits only inside the gaze RoI.
# The live pipeline adds transport levers: `scale` (encode resolution as a
# fraction of the canvas), `fps` (frame pacing) and `blur` (periphery radius).
PROFILES = {
    "ultra": {"jpeg_quality": 90, "audio_channels": 4, "foveated": False, "scale": 1.0, "fps": 30, "blur": 0.0, "tile_ladder": {"high": 90, "medium": 75, "low": 40}, "target_kbps": 3500},
    "high": {"jpeg_quality": 75, "audio_channels": 4, "foveated": False, "scale": 1.0, "fps": 30, "blur": 0.0, "tile_ladder": {"high": 80, "medium": 60, "low": 30}, "target_kbps": 1800},
    "medium": {"jpeg_quality": 55, "audio_channels": 2, "foveated": True, "scale": 0.75, "fps": 20, "blur": 6.0, "tile_ladder": {"high": 65, "medium": 45, "low": 20}, "target_kbps": 800},
    "low": {"jpeg_quality": 35, "audio_channels": 1, "foveated": True, "scale": 0.5, "fps": 15, "blur": 8.0, "tile_ladder": {"high": 45, "medium": 30, "low": 15}, "target_kbps": 400},
}

# --- Live WebRTC demo ---------------------------------------------------------

# Signaling-only ICE servers; media is peer-to-peer. For cellular demos behind
# symmetric NAT, set XR_ICE_SERVERS to a TURN uri, e.g.
#   turn:user@host:3478?transport=udp  (with username/credential supported by
#   the aiortc RTCIceServer shape in app/rtc/signaling.py).
ICE_SERVERS = [
    s.strip()
    for s in os.environ.get("XR_ICE_SERVERS", "stun:stun.l.google.com:19302").split(",")
    if s.strip()
]

VIDEO_CANVAS_WIDTH = 1280  # every outbound frame is normalized to this canvas
VIDEO_CANVAS_HEIGHT = 720  # so the VP8 encoder never needs renegotiation
VIDEO_CLOCK_RATE = 90000
AUDIO_CLOCK_RATE = 48000
AUDIO_PTIME_MS = 20  # one Opus packet
METRICS_INTERVAL_S = 0.5  # dashboard push cadence
SHADOW_SAMPLE_EVERY_N = 5  # JPEG/PSNR shadow metrics on every Nth frame
