# XR Compression Lab: Intelligent Viewport-Aware and Bandwidth-Adaptive Compression for Live XR Streaming

Course research and implementation project for **Advanced Data Compression**: studying classical entropy/dictionary coding, transform coding, **real H.264 continuous video encoding**, **4×4 viewport-aware tile compression with rate–distortion optimization (RDO)**, **lookahead viewport trajectory prediction with saccade fallback**, **360° equirectangular panoramic video streaming**, **real-time Opus multichannel spatial audio**, **EWMA network bandwidth estimation**, and **multi-objective adaptive policies** — measured live on an end-to-end phone-to-laptop WebRTC system.

---

## 1. Media Planes Under Test

The laboratory provides two distinct media sources:

1. **Synthetic Stereoscopic XR Rig (`app/media_source.py`)**:
   - **Video**: Fixed-viewpoint ~100°×62° frustum, Side-by-Side (SBS) stereo frame (left/right eye views with parallax), 1280×720 (2,764,800 bytes raw RGB24).
   - **Audio**: 4-channel First-Order Ambisonics (FOA) B-format field ($W, X, Y, Z$) with an orbiting 440 Hz sound source at 48 kHz (768,000 bytes/s raw float32).
2. **360° Equirectangular Panoramic Source (`app/panoramic.py`)**:
   - **Real Equirectangular Video Demuxing (`PanoramicVideoSource`)**: High-performance PyAV (`av.open`) demuxer reading genuine 2:1 equirectangular MP4 video files with accurate FPS timing and arbitrary temporal seeking.
   - **Procedural Panoramic Video (`ProceduralPanoramicSource`)**: Synthetic 2:1 equirectangular celestial sphere with latitude/longitude grid rings, spherical gaze projection, and repeatable viewer trajectories (*Central fixation*, *Smooth equator panning*, *Sudden 90° saccades*).
3. **Live WebRTC Camera & Microphone Stream (`/phone` → `/viewer`)**:
   - Live mobile capture published via browser `RTCPeerConnection` (VP8 video + Opus audio over SRTP), processed through an adaptive server-side pipeline.

---

## 2. Topic Map & Implementation Reference

| Course Topic | Module | Core Implementation & Theoretical Significance |
|---|---|---|
| **Run-length coding (RLE)** | `app/codecs/lossless.py` | Byte-level count-value run-length encoding; evaluates compressibility limits on synthetic flat regions vs. photographic entropy. |
| **Entropy coding (Huffman)** | `app/codecs/lossless.py` | Canonical binary tree construction, prefix-free variable-length bitstream generation, and entropy bounds. |
| **Dictionary coding (LZ77)** | `app/codecs/lossless.py` | zlib (DEFLATE: LZ77 + Huffman) and LZMA (Markov chain Lempel-Ziv) comparative ratio analysis. |
| **Transform coding (DCT)** | `app/codecs/lossy.py` | 8×8 2D Discrete Cosine Transform + Annex K luminance/chrominance quantization matrices. |
| **Scalar quantization** | `app/codecs/lossy.py`, `spatial_audio.py` | Quality ladder mapping; float32 to uniform int16 quantization (84.6 dB SNR). |
| **Chroma subsampling** | `app/codecs/lossy.py` | 4:4:4 to 4:2:0 YUV chroma decimation and bilinear reconstruction. |
| **Predictive coding (DPCM)** | `app/codecs/spatial_audio.py` | 1st-order temporal difference delta calculation followed by entropy coding. |
| **Multichannel spatial audio** | `app/codecs/spatial_audio.py` | FOA 4-channel $B$-format folding to stereo/mono matching available link budget. |
| **Real Opus audio coding** | `app/codecs/audio_codec.py` | libopus encoder/decoder across 24, 64, 96, 128 kbps (mono/stereo) with true SNR metering. |
| **Real H.264 / AVC video** | `app/codecs/video_codec.py` | Real libx264 Annex-B packetization, intra-refresh IDR frames, SSIM & PSNR calculation, and stateful `H264StreamEncoder`. |
| **4×4 Viewport tile coding** | `app/viewport_tiles.py` | 16-tile spatial partitioning, gaze RoI intersection, 1-hop Chebyshev guard band, and gapless canvas reconstruction. |
| **Knapsack RDO budget controller** | `app/viewport_tiles.py` | Greedy bounded knapsack rate–distortion optimization with importance weighting, calibrated JPEG/H.264 byte estimation, and 6-iteration measurement-driven retry convergence. |
| **Lookahead viewport prediction** | `app/prediction.py` | Kinematic velocity extrapolation across 50–200 ms horizons, dynamic saccade velocity detection (> 180°/s), and confidence-weighted fallback protection. |
| **360° Equirectangular streaming** | `app/panoramic.py` | Real PyAV video reader + procedural source, equirectangular bounding box mapping, and tile boundary artifact MSE measurement. |
| **Bandwidth adaptation (ABR)** | `app/adaptive.py` | EWMA link capacity estimator with asymmetric up/down time-constants, volatility dampening, and profile ladder walks. |

---

## 3. Quickstart

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/Raunak-23/Compressions-in-AR-VR.git
cd Compressions-in-AR-VR

# Run the full automated test suite (75 tests across 11 modules)
python -m pytest tests/ -v

# Run the comprehensive benchmark suite (generates all 8 CSVs and 9 publication-grade plots)
python experiments.py

# Launch the live WebRTC server
python -m uvicorn app.server:app --port 8200
```

Open `http://localhost:8200/viewer` on your laptop and `http://localhost:8200/phone` on your smartphone (see [DEMO.md](DEMO.md) for public HTTPS tunneling instructions).
Or navigate directly to the interactive exploration tools:
- `http://localhost:8200/compare` — **Quality Comparison Slider** (before/after split slider with measured PSNR, SSIM, and size reduction).
- `http://localhost:8200/playground` — **Video Compression Playground** (upload video, offline H.264/M-JPEG encode, synchronized dual players, download).

### REST Endpoints:
- `GET /compare` — Interactive Before/After Compression Comparison Slider interface.
- `GET /playground` — Interactive Video Compression Playground interface.
- `GET /metrics?t=1.0&quality=75` — Single-frame comparative benchmark across all codecs (RLE, Huffman, zlib, LZMA, JPEG, Foveated JPEG, H.264, 4×4 Viewport Tiles): bytes, compression ratio, PSNR, SSIM, and timing.
- `GET /api/compression-methods` — Supported single-image compression algorithms and control specifications.
- `GET /api/video-compression-methods` — Supported video compression codecs and parameter ranges.
- `GET /api/sample-frame?t=1.0` — Synthetic stereoscopic XR frame as base64 JPEG.
- `POST /api/compress-image` — Compress an image or extracted video frame using JPEG, H.264, or 4×4 Viewport Tiles.
- `POST /api/upload-video` — Validate and ingest user video files (up to 100 MB), extracting container metadata.
- `POST /api/compress-video` — Run offline H.264 or M-JPEG compression in a background thread via PyAV.
- `GET /api/job-status/{job_id}` — Query compression job progress and computed results.
- `GET /api/video-frame/{job_id}?which=original|compressed&time_s=0.0` — Extract aligned video frames.
- `GET /api/compare-video-frames/{job_id}?time_s=0.0` — Dual-frame alignment with frame-level PSNR & SSIM.
- `GET /api/serve-video/{job_id}?which=original|compressed` — Stream original or compressed video to HTML5 players.
- `GET /api/download/{job_id}` — Download the compressed video file.
- `DELETE /api/cleanup/{job_id}` — Clean up temporary upload and output artifacts.
- `GET /stream?bw_kbps=400&yaw=10&frames=30` — Adaptive + viewport-aware MJPEG stream walking the profile ladder.
- `GET /audio?profile=low&seconds=1` — FOA audio folded to profile channel count, returned as uncompressed WAV.
- `GET /healthz` — Server health, active WebRTC peers, and system diagnostics.

---

## 4. Five Evaluated Experimental Conditions

1. **Condition A: Fixed Baseline**: Uniform spatial quality (JPEG $q=75$), fixed resolution (1280×720), no foveation or tiling, fixed bitrate.
2. **Condition B: Bandwidth Adaptation Only**: Adaptive profile ladder walk (`ultra`, `high`, `medium`, `low`) matching EWMA throughput, but applying uniform spatial quality across the full frame.
3. **Condition C: Viewport Adaptation Only**: 4×4 gaze-aware tile allocation (high in gaze RoI, medium in guard band, low in periphery) using fixed quality tiers (85/55/25) with fixed budget.
4. **Condition D: Combined Adaptation**: Bandwidth-directed profile budgeting coupled with viewport-aware tile quality allocation.
5. **Condition E: Combined Adaptation with Viewport Prediction**: Lookahead trajectory prediction ($100\text{ ms}$ lead) driving greedy knapsack RDO quality allocation with iterative measurement-driven budget retry convergence.

---

## 5. Live WebRTC Demonstration Dashboard (`/viewer`)

The live `/viewer` dashboard features:
- **Interactive 4×4 Spatial Tile Matrix**: Dynamic 16-cell grid visualizing real-time tile tiers (`HIGH`, `MED`, `LOW`) and quality assignments.
- **Lookahead Minimap Overlay**: Dual bounding box visualization comparing current gaze position (solid cyan box) against kinematic predicted RoI (gold dashed box) with confidence metering.
- **Adaptation Mode Selector**: Live runtime switching between `intelligent_allocator` (Knapsack RDO), `fixed_tiers`, `bandwidth_only`, `viewport_only`, and `uniform`.
- **Lookahead Viewport Prediction Toggle**: Real-time enabling/disabling of velocity-based trajectory extrapolation with saccade fallback.
- **Congestion Simulation Levers**: Instantaneous bandwidth injection buttons (`Drop 300k`, `1000k`, `Recover`) testing sub-second downswitch agility and hysteresis recovery.
- **Dual Hop Telemetry Table**: Clear operational distinction between actual WebRTC wire payload (VP8/Opus over SRTP) and parallel experimental shadow tile metrics (JPEG/H.264 byte budgets and reconstructed PSNR/SSIM).

---

## 6. Comprehensive Benchmark Datasets & Generated Artifacts

Executing `python -m app.experiments_runner` or `python experiments.py` exports genuine, non-fabricated experimental artifacts into `experiments/`:

### Generated Datasets (CSV):
1. `experiments/video_conditions_evaluation.csv`: 5 conditions evaluated across 5 bandwidth levels (250, 500, 1000, 2000, 4000 kbps) with retry counts and peripheral metrics.
2. `experiments/codecs_rate_distortion_comparison.csv`: Rate–distortion evaluation of Single-Frame JPEG, Foveated JPEG, 4×4 Tiled JPEG, and Single-Frame H.264 IDR.
3. `experiments/bandwidth_adaptation_timeline.csv`: Dynamic response during sudden 90% congestion drop (3000 → 300 kbps) and safe EWMA recovery.
4. `experiments/audio_codecs_evaluation.csv`: Float32 raw vs. scalar int16 vs. DPCM+zlib vs. Folded Stereo vs. Opus (24–128 kbps).
5. `experiments/sequence_comparison.csv`: Continuous 30-frame H.264 streaming vs. standalone M-JPEG.
6. `experiments/tile_quality_multi_gaze.csv`: Spatial tile quality across 7 horizontal gaze angles ($-35^\circ$ to $+35^\circ$).
7. `experiments/prediction_evaluation.csv`: 4 trajectories across 4 horizons (50, 100, 150, 200 ms) comparing Strategies A, B, and C with RoI coverage and overhead.
8. `experiments/panoramic_evaluation.csv`: 360° equirectangular video evaluation recording Viewport, Guard Band, Peripheral, and Full-frame PSNR/SSIM.

### Publication-Quality Plots:
- `experiments/plot_bitrate_vs_psnr_ssim.png`: Rate–distortion curves (Bitrate vs. PSNR and SSIM).
- `experiments/plot_compressed_size_comparison.png`: Encoded frame size comparison across codecs.
- `experiments/plot_viewport_vs_peripheral_quality.png`: Viewport vs. peripheral quality across gaze angles.
- `experiments/plot_bandwidth_adaptation_drop_recovery.png`: Link congestion drop and recovery adaptation.
- `experiments/plot_four_conditions_comparison.png` & `plot_five_conditions_comparison.png`: Viewport quality across conditions A–E.
- `experiments/plot_audio_bitrate_vs_snr.png`: Audio bitrate vs. SNR trade-offs.
- `experiments/plot_sequence_comparison.png`: Continuous H.264 sequence vs. MJPEG bitrate and encode time.
- `experiments/plot_prediction_error_vs_horizon.png`: Angular error vs. lookahead horizon across trajectories.
- `experiments/plot_prediction_strategies_comparison.png`: IoU and RoI coverage comparing Strategies A, B, and C.
- `experiments/summary_manifest.json` & `experiments/EXPERIMENT_REPORT.md`: Machine-readable summary and detailed report.

---

## 7. Key Empirical Findings

From `experiments/EXPERIMENT_REPORT.md`:
- **Bitrate Compliance**: Condition E (Intelligent Adaptive Allocator with knapsack optimization and retry loop) converges within available frame budgets under network congestion while maintaining high fidelity in the primary gaze region.
- **Dynamic Congestion Response**: Under sudden 90% bandwidth drops, the EWMA controller downswitches in < 2 frames to avoid queue bufferbloat, while hysteresis prevents flapping during recovery.
- **Lookahead Viewport Prediction**: In smooth pursuit, kinematic prediction achieves $\text{IoU} > 0.95$ with only $4.6\text{–}6.4\ \mu\text{s}$ overhead. Under sudden saccades, unconstrained prediction degrades ($54.5\%$ worse than current gaze), whereas confidence-based fallback limits error to $5.04^\circ$ and guarantees $100\%$ RoI coverage.
- **Audio Efficiency**: Opus stereo encoding achieves $> 40\text{ dB}$ SNR at $64\text{ kbps}$, delivering $> 10\times$ compression over raw PCM while preserving binaural spatial imaging.

---

---

## 8. Interactive Quality Comparison Slider (`/compare`)

The laboratory features a dedicated visual inspection tool at `/compare` for evaluating pixel-level distortion across compression configurations:

- **Viewing Architecture**: Co-located split view overlaying the original source frame and the reconstructed compressed frame on identical display geometry with zero spatial offset.
- **Draggable Vertical Slider**: Smooth mouse, touch, and keyboard-driven divider (`ArrowLeft`, `ArrowRight`, `Home`, `End`) with reset-to-center button.
- **Supported Input Sources**:
  1. *Synthetic Stereoscopic XR Rig*: Arbitrary temporal frames ($t \in [0, 10]\text{ s}$) from the SBS stereo rig.
  2. *User File Upload*: Direct image (`.jpg`, `.png`, `.webp`, `.bmp`, `.tiff`) or video file (`.mp4`, `.webm`, `.mkv`, `.avi`, `.mov`) with automatic frame extraction at timestamp $t$.
  3. *Video Playground Job Integration*: Direct inspection of any video compression job via `?job_id=<id>` with frame extraction and measured job telemetry.
- **Compression Controls**:
  - **JPEG**: Quality slider ($1\text{–}100$) and Chroma Subsampling toggle ($4:2:0$ vs. $4:4:4$).
  - **H.264 / AVC**: Target bitrate budget ($100\text{–}5000\text{ kbps}$) using single-frame IDR slice coding.
  - **4×4 Viewport Tiles**: Viewport, Guard Band, and Peripheral quality ladder ($85/55/25$) with gaze orientation $(\text{yaw}, \text{pitch})$.
- **Telemetry Displayed**:
  - Original and compressed sizes in bytes / KB / MB.
  - Size reduction percentage: $\frac{S_{\text{orig}} - S_{\text{comp}}}{S_{\text{orig}}} \times 100\%$.
  - Compression ratio: $\frac{S_{\text{orig}}}{S_{\text{comp}}}$.
  - Peak Signal-to-Noise Ratio (PSNR) in dB and Structural Similarity (SSIM).
  - Processing / encode roundtrip time in milliseconds.
  - Viewport-specific metrics (Viewport PSNR, Peripheral PSNR, Boundary Seam MSE).

---

## 9. Video Compression Playground (`/playground`)

The video compression playground at `/playground` provides an offline testbed for compressing arbitrary user-uploaded videos:

- **5-Stage Workflow**:
  1. *Upload*: Drag-and-drop or file browser with pre-flight format validation and file size verification (up to $100\text{ MB}$).
  2. *Configure*: Select codec, bitrate budget / quality factor, output frame rate ($5\text{–}60\text{ fps}$), and resolution preset (`original`, `720p`, `480p`, `360p`).
  3. *Process*: Non-blocking asynchronous processing with live percentage indicator and status feedback.
  4. *Results Dashboard*: Measured file sizes, reduction percentage, compression ratio, output duration, average bitrate, sampled PSNR, sampled SSIM, and target bitrate compliance badge.
  5. *Compare & Download*: Side-by-side HTML5 video players with **synchronized playback** (play, pause, seek), direct link to the Quality Comparison slider, and one-click compressed file download.
- **Supported Encoders**:
  - **H.264 / AVC (`libx264`)**: Inter-frame predictive video compression utilizing spatial prediction, motion estimation, and in-loop deblocking via PyAV.
  - **M-JPEG (`mjpeg`)**: Frame-by-frame transform compression without temporal redundancy.
- **Upload Constraints & Safety**:
  - Maximum upload size: $100\text{ MB}$.
  - Supported containers: `.mp4`, `.webm`, `.mkv`, `.avi`, `.mov`.
  - Non-blocking server operations: Long-running video transcoding executes inside thread pools (`asyncio.to_thread`) preserving WebRTC stream responsiveness.
  - Automatic sandbox cleanup: Temporary working files stored in a dedicated temp location with explicit `DELETE /api/cleanup/{job_id}` endpoint.

---

## 10. Verification & Test Suite

All 96 automated unit and integration tests pass cleanly:
```bash
python -m pytest tests/ -v
# ======================= 96 passed, 5 warnings in 54.17s =======================
```
Verified test modules:
- `tests/test_comparison_playground.py`: 21 comprehensive tests covering format validation, image/video compression pipelines, metric formulas, video frame extraction, dual-player synchronization endpoints, and error recovery.
- `tests/test_budget_controller.py`: Feasible/infeasible budgets, priority ordering, peripheral tile downscaling, audit trail.
- `tests/test_metrics_validation.py`: `METRIC_UNAVAILABLE` sentinel verification, CSV formatting, timing independence.
- `tests/test_panoramic.py`: Aspect ratio validation, spherical-to-equirectangular projection, PyAV MP4 video roundtrip.
- `tests/test_prediction.py`: Multi-horizon prediction, saccade detection, confidence fallback, RoI coverage, strategy comparison.
- `tests/test_adaptive_controller.py`: Volatility dampening, fast downswitch, hysteresis, latency throttling.
- `tests/test_audio_codec.py`: Opus mono/stereo roundtrips, bitrate ladder, parameter validation.
- `tests/test_video_codec.py`: libx264 roundtrip, SSIM computation, streaming encoder, numpy input.
- `tests/test_viewport_tiles.py`: Gapless 4×4 grid tiling, Chebyshev guard band, boundary artifact MSE.
- `tests/test_intelligent_allocator.py`: Knapsack RDO solver, 5 allocation strategies, importance weighting.
- `tests/test_compression.py`: Classical lossless/lossy codecs, DCT quantization, chroma subsampling.
- `tests/test_live.py` & `tests/test_live_webrtc.py`: aiortc WebRTC track processing, peer signaling, and full end-to-end streaming.
