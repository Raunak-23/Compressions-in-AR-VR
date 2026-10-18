# XR Compression Lab

Course project for **Data Compression**: studying classical and media compression techniques,
**real H.264 video compression**, **4×4 viewport-aware tile compression**, **real Opus audio coding**,
**network-bandwidth-adaptive** streaming, and **combined adaptive policies** — measured on the spatial audio + video of an
AR/VR/XR livestream (the synthetic stereoscopic rig from the Virtual Darshan platform).

## The media plane under test

`app/media_source.py` synthesizes the media under test:

- **Video**: fixed-viewpoint ~100°×62° frustum, **Side-by-Side (SBS)** stereo frame (left | right eye halves with parallax), 1280×720.
- **Audio**: 4-channel **First-Order Ambisonics (FOA)** B-format field (W, X, Y, Z) with a source circling the listener at 440 Hz.

Swap `SyntheticXRSource` for a tap on the live WebRTC/JPEG pipeline to run the same experiments on the real broadcast.

## Topic map

| Course topic | Where | What it shows |
|---|---|---|
| Run-length coding | `app/codecs/lossless.py` | When RLE wins (flat runs) and when it explodes |
| Entropy coding | `app/codecs/lossless.py` | Huffman codes; ratio vs dictionary coding |
| Dictionary coding | `zlib` / `lzma` wrappers | LZ77+Huffman beats pure Huffman on media bytes |
| Transform coding | `app/codecs/lossy.py` | 8×8 block DCT + JPEG Annex K quantization matrix |
| Scalar quantization | `app/codecs/lossy.py`, `spatial_audio.py` | JPEG ladder; float32→int16 audio SNR |
| Chroma subsampling | `app/codecs/lossy.py` | 4:2:0 YUV subsampling and restoration |
| Predictive coding | `app/codecs/spatial_audio.py` | DPCM deltas before entropy coding |
| Multichannel spatial audio | `app/codecs/spatial_audio.py`, `audio_codec.py` | FOA channel folding vs Opus & bandwidth |
| Real H.264 / AVC video | `app/codecs/video_codec.py` | libx264 Annex-B round-trip; PSNR & SSIM metrics |
| 4×4 Viewport tile coding | `app/viewport_tiles.py` | 16-tile spatial grid: high (gaze), medium (guard band), low (periphery) |
| Real-time Opus audio | `app/codecs/audio_codec.py` | libopus 24/64/96/128 kbps mono/stereo round-trip & SNR |
| Rate–distortion analysis | `experiments.py`, `app/experiments_runner.py` | RD curves (Bitrate vs PSNR/SSIM) & size comparisons |
| Viewport-aware / foveated | `app/viewport.py`, `app/viewport_tiles.py` | Bits allocated selectively inside the gaze RoI |
| Bandwidth-adaptive (ABR) | `app/adaptive.py` | EWMA throughput → profile ladder |
| Combined adaptation | `app/adaptive.py`, `experiments/` | Bandwidth budget + viewport-aware tile allocation |

## Quickstart

```bash
uv sync                                    # install dependencies (aiortc, av, pillow, pytest, matplotlib, etc.)
uv run pytest                              # runs all 36 unit and integration tests
uv run python experiments.py               # prints measurement tables, runs benchmark, and generates plots
uv run uvicorn app.server:app --port 8200   # launch the live WebRTC & demo server
```

Then open `http://localhost:8200/viewer` (laptop) and `http://localhost:8200/phone`
(phone, or tunnel URL — see [DEMO.md](DEMO.md)).

### Endpoints (offline study plane):

- `GET /metrics?t=1.0&quality=75` — one frame through every codec (RLE, Huffman, zlib, LZMA, JPEG, Foveated JPEG, H.264, 4×4 Viewport Tiles): bytes, ratios, PSNR, SSIM.
- `GET /stream?bw_kbps=400&yaw=10&frames=30` — adaptive + viewport-aware MJPEG stream walking the profile ladder.
- `GET /audio?profile=low&seconds=1` — FOA audio folded to profile channel count, as WAV.
- `GET /healthz` — server health and active connection status.

## Experimental Evaluation Harness

Running `uv run python experiments.py` executes the full evaluation suite and exports artifacts to `experiments/`:

### Exported CSV Datasets:
1. `experiments/video_conditions_evaluation.csv`: Evaluation of Conditions A, B, C, D across bandwidth levels (250, 500, 1000, 2000, 4000 kbps).
2. `experiments/codecs_rate_distortion_comparison.csv`: Rate-distortion comparison of JPEG, Foveated JPEG, 4×4 Tiled JPEG, and H.264.
3. `experiments/bandwidth_adaptation_timeline.csv`: Dynamic response during sudden congestion drops and recovery.
4. `experiments/audio_codecs_evaluation.csv`: Comparison of Raw float32, int16 PCM, DPCM + zlib, Folded Stereo, and Opus (24–128 kbps).

### Generated Plots:
- `experiments/plot_bitrate_vs_psnr_ssim.png`: Bitrate vs PSNR and SSIM rate-distortion curves.
- `experiments/plot_compressed_size_comparison.png`: Frame size comparison across codecs at comparable visual quality.
- `experiments/plot_viewport_vs_peripheral_quality.png`: Viewport vs peripheral quality across gaze yaw angles (-35° to +35°).
- `experiments/plot_bandwidth_adaptation_drop_recovery.png`: EWMA tracking and profile adaptation under sudden link congestion and recovery.
- `experiments/plot_four_conditions_comparison.png`: Four-condition comparison (A: Baseline, B: Bandwidth-only, C: Viewport-only, D: Combined).
- `experiments/plot_audio_bitrate_vs_snr.png`: Audio bitrate vs SNR across Opus, int16 quantization, and DPCM.

## Four Evaluated Experimental Conditions

1. **Condition A (Fixed baseline)**: Uniform quality (JPEG q=75), fixed resolution (1280×720), no viewport foveation, fixed target budget.
2. **Condition B (Bandwidth adaptation only)**: Profile ladder walks according to observed EWMA network throughput (`ultra`, `high`, `medium`, `low`), but with uniform quality across the entire frame.
3. **Condition C (Viewport adaptation only)**: Gaze-dependent 4×4 tile quality allocation (high in gaze RoI, medium guard band, low periphery), but with a fixed budget regardless of bandwidth.
4. **Condition D (Combined adaptation)**: Bandwidth-aware profile budgeting coupled with viewport-aware 4×4 tile quality allocation.

## Live demo: real WebRTC, phone -> laptop

The adaptive + viewport-aware pipeline runs live over WebRTC:

1. **Phone** (`/phone`, over an HTTPS tunnel) publishes camera + mic via `RTCPeerConnection` and streams touch position as gaze.
2. **Bridge** (aiortc, in this server) runs the adaptive stage per viewer: profile resolution scaling on the invariant 1280×720 canvas, foveation from gaze, FOA channel folding, and shadow metrics sampled every 5th frame.
3. **Viewer** (`/viewer`, on the laptop) receives the adapted VP8/Opus stream and reports WebRTC receiver stats over WebSocket; the EWMA walks the profile ladder.

See **[DEMO.md](DEMO.md)** for step-by-step setup with ngrok or localtunnel.

## Architecture Notes

- `app/codecs/video_codec.py`: PyAV / libx264 encoder and decoder, SSIM calculation, Annex-B packetization, and reusable streaming encoder.
- `app/viewport_tiles.py`: 4×4 grid tiling, gaze RoI intersection, 1-hop Chebyshev guard band, independent tile encoding, gapless canvas reconstruction.
- `app/codecs/audio_codec.py`: PyAV / libopus encoder and decoder for mono and stereo streams across 24–128 kbps with audio SNR measurement.
- `app/adaptive.py`: EWMA throughput estimator, profile ladder picker, frame budget allocation, and overshoot accounting.
- `app/rtc/tracks.py`: aiortc media tracks enforcing the fixed 1280×720 video canvas, frame pacing, and non-blocking shadow metrics.
- `app/rtc/bridge.py` & `app/rtc/signaling.py`: State management and WebSocket signaling for browser trickle ICE.
