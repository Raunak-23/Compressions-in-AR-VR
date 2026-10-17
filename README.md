# XR Compression Lab

Course baseline for a **Data Compression** project: studying classical and
media compression techniques, **network-bandwidth-adaptive** compression, and
**viewport-aware** compression — measured on the spatial audio + video of an
AR/VR/XR livestream (the synthetic stereoscopic rig from the Virtual Darshan
platform).

## The media plane under test

`app/media_source.py` synthesizes the same media the real platform streams:

- **Video**: fixed-viewpoint ~100°x62° frustum, **Side-by-Side (SBS)** stereo
  frame (left | right eye halves with parallax), 1280×720.
- **Audio**: 4-channel **First-Order Ambisonics (FOA)** B-format field
  (W, X, Y, Z) with a source circling the listener.

Swap `SyntheticXRSource` for a tap on the live WebRTC/JPEG pipeline to run the
same experiments on the real broadcast.

## Topic map

| Course topic | Where | What it shows |
|---|---|---|
| Run-length coding | `app/codecs/lossless.py` | When RLE wins (flat runs) and when it explodes |
| Entropy coding | `app/codecs/lossless.py` | Huffman codes; ratio vs dictionary coding |
| Dictionary coding | `zlib` / `lzma` wrappers | LZ77+Huffman beats pure Huffman on media bytes |
| Transform coding | `app/codecs/lossy.py` | 8×8 block DCT + JPEG quantization matrix |
| Scalar quantization | `app/codecs/lossy.py`, `spatial_audio.py` | JPEG ladder; float32→int16 audio SNR |
| Chroma subsampling | `app/codecs/lossy.py` | 4:2:0 and why it is nearly free |
| Predictive coding | `app/codecs/spatial_audio.py` | DPCM deltas before entropy coding |
| Multichannel spatial audio | `app/codecs/spatial_audio.py` | FOA channel folding vs bandwidth |
| Rate–distortion | `experiments.py` §2–3 | bytes vs PSNR curves |
| Viewport-aware / foveated | `app/viewport.py` | bits spent only inside the gaze RoI |
| Bandwidth-adaptive (ABR) | `app/adaptive.py` | EWMA throughput → profile ladder |
| Adaptive + viewport combined | `app/adaptive.py` | constrained profiles switch foveation on |

## Quickstart

```bash
uv sync
uv run pytest                 # baseline guarantees + live WebRTC loopback test
uv run python experiments.py  # print the measurement tables
uv run uvicorn app.server:app --port 8200
```

Then open `http://localhost:8200/viewer` (laptop) and `http://localhost:8200/phone`
(phone, or the tunnel URL — see [DEMO.md](DEMO.md)).

Demo endpoints (offline study plane):

- `GET /metrics?t=1.0&quality=75` — one frame through every codec: bytes, ratios, PSNR.
- `GET /stream?bw_kbps=400&yaw=10&frames=30` — adaptive + viewport-aware MJPEG;
  the client reports its measured bandwidth (as WebRTC stats would) and the
  server walks the ladder.
- `GET /audio?profile=low&seconds=1` — FOA audio folded to the profile's
  channel count, as WAV.

## Live demo: real WebRTC, phone -> laptop

The adaptive + viewport-aware pipeline also runs **live over real WebRTC**:

1. **Phone** (`/phone`, over an HTTPS tunnel) publishes its camera + mic via
   `RTCPeerConnection` and streams touch position as the gaze.
2. **Bridge** (aiortc, in this server) runs the adaptive stage per viewer:
   profile encode resolution, foveation from the gaze, shadow JPEG/PSNR via the
   course codecs, FOA channel folding for the audio plane.
3. **Viewer** (`/viewer`, on the laptop) receives the adapted VP8/Opus stream
   and reports real WebRTC receiver stats (`bytesReceived` deltas) over the
   signaling WebSocket; the EWMA walks the same profile ladder. The dashboard
   shows the live metrics (profile, EWMA, effective resolution, foveation,
   shadow bytes, stage PSNR, phone-link throughput).

See **[DEMO.md](DEMO.md)** for the step-by-step run (ngrok / localtunnel).

- Live endpoints: `GET /phone`, `GET /viewer`, `WS /ws/phone`, `WS /ws/viewer`,
  `GET /api/ice`, `GET /healthz`.
- ICE: STUN by default (`XR_ICE_SERVERS` env to add TURN for cellular demos);
  media is peer-to-peer, the tunnel carries only signaling.

## Troubleshooting

- **`uv run ...` fails with `error: uv trampoline failed to canonicalize script path`**
  — the `.venv` was created at a different folder location (the project was moved
  or renamed), and uv's console-script launchers embed the old absolute path.
  Fix: recreate the venv, which is fully reproducible from `uv.lock`:
  `Remove-Item -Recurse -Force .venv; uv venv; uv sync`.
  Fallback that always works regardless: `.venv\Scripts\python.exe -m uvicorn app.server:app --port 8200`.
- **Camera blocked on the phone** — allow camera + mic for the site (tap the
  padlock / permission icon in the mobile browser); must be HTTPS except on localhost.

## Baseline scope / next steps

Deliberately baseline: JPEG is Pillow's, DEFLATE/LZMA are stdlib, distortion is
PSNR-only, and the demo's *link cap* control simulates scarcity (real receiver
stats drive the ladder otherwise). Natural extensions for the course report:
SSIM/VMAF distortion, AV1/H.265 encoders, TURN-backed cellular demos, and
rate–distortion plots per technique.
