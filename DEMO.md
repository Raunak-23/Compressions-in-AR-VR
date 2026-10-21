# Live demo walkthrough — phone → WebRTC → laptop

What you will show: **a real phone streaming its camera over WebRTC to the
laptop**, where the adaptive compression stage runs (profile ladder +
viewport-aware foveation) and the viewer dashboard displays the live metrics —
profile switches, EWMA bandwidth, effective resolution, foveation, shadow JPEG
bytes, stage PSNR, and the real phone→server link throughput.

Transport is genuine WebRTC (VP8 video + Opus audio over SRTP via aiortc).
The ngrok/localtunnel link carries **only** the HTTPS pages and the two
signaling WebSockets; the media itself is peer-to-peer.

**Where compression happens:** the phone compresses first (its browser's
VP8/Opus encoder), the server's adaptive stage re-encodes the adapted frame
(scale / foveation / fps per profile), and the viewer's browser decompresses.
Both hops are metered live — raw → wire on the phone page, wire → raw on the
viewer page. Only one phone publishes at a time; a new one takes over and
kicks the previous publisher.

## 0. One-time setup

```bash
uv sync          # installs aiortc etc.
```

If you want the tunnel: install either

```bash
ngrok            # https://ngrok.com — needs a free account + authtoken
# or nothing for localtunnel (npx fetches it)
```

## 1. Start the server (laptop)

```bash
uv run uvicorn app.server:app --port 8200
```

Check: http://localhost:8200/healthz → `{"ok": true, ...}`

## 2. Expose it (laptop, second terminal — pick one)

```bash
ngrok http 8200
# → copy the https://xxxx.ngrok-free.app URL
```

or

```bash
npx localtunnel --port 8200
# → copy the https://xxxx.loca.lt URL
# (first visit from a new browser asks for a tunnel password:
#  enter your public IP as instructed on that page)
```

## 3. Phone side (the XR rig)

1. On the **phone**, open `https://<tunnel-url>/phone`.
2. Tap **Start streaming** and allow camera + microphone.
3. When the chip shows **live — streaming**, drag your finger across the
   video: that position is the gaze that steers the foveation RoI.

## 4. Viewer side (laptop)

Open **http://localhost:8200/viewer** (no tunnel needed — it reaches the
server directly). Within a couple of seconds:

- The phone's video appears, adaptively encoded.
- The tiles start moving: measured downlink, JPEG shadow bytes, profile badge,
  gaze RoI minimap, and the phone-link throughput in the table.

## 5. Show the adaptation (the actual study)

The viewer's real WebRTC receiver stats drive the EWMA, which walks the
ladder (`ultra → high → medium → low` at 3500/1800/800 kbps). On a loopback
laptop viewer the measured bandwidth is huge, so two demo levers exist in the
dashboard:

1. **Simulated link cap** — drag the cap slider to ~600 kbps: the profile
   badge drops to *low*, the encode resolution drops to 480×270, foveation
   switches on (watch the minimap: sharp box = gaze RoI, blurred periphery),
   audio folds to mono (in FOA mode), and the JPEG shadow curve collapses.
   Drag it back up: the ladder walks back to *ultra*.
2. **Profile dropdown** — force any single rung to show its knobs directly.

A genuinely degraded network works too: open the *viewer* on a second phone
(or any remote browser) over cellular — its measured receiver stats are
low and the ladder downshifts on its own, no slider needed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv run uvicorn ...` → `error: uv trampoline failed to canonicalize script path` | The `.venv` was created at an old folder location. Recreate it: `Remove-Item -Recurse -Force .venv; uv venv; uv sync`. Or launch directly: `.venv\Scripts\python.exe -m uvicorn app.server:app --port 8200`. |
| Opened the tunnel URL on the phone and see the metrics dashboard | The tunnel **root** is the laptop viewer. The banner on that page links to the phone page, or go to `https://<tunnel-url>/phone` directly. |
| Phone page loads but camera blocked | Tunnel URL must be **https** (both ngrok and localtunnel give you one). iPhone: Settings → Safari → Camera = Allow, or use Chrome. |
| Viewer stuck on "Waiting for the phone source" | Check the phone chip says *live*. Check `http://localhost:8200/healthz` shows `phone_connected: true`. |
| Viewer "connecting" forever, never "connected" | Media is peer-to-peer: laptop firewall may block inbound UDP for python.exe — allow it (first-run prompt), or run the viewer on the same machine (loopback, no firewall). |
| Phone on **cellular** fails to connect | Symmetric NAT needs TURN: `set XR_ICE_SERVERS=turn:user@turn-host:3478?transport=udp` before starting uvicorn (aiortc reads username/credential form the same env var format shown in `app/config.py`). Same-WiFi demos don't need this. |
| Choppy video on a weak laptop | Lower the phone's capture resolution in `/phone` (the stage rescales anyway), or keep the cap at *high* instead of *ultra*. |
| localtunnel password page | That's localtunnel's anti-abuse page — type your public IP once, then the app loads. |

## Architecture note (for the report)

- `app/rtc/signaling.py` — WebSocket signaling + aiortc session wiring; parses
  browser trickle ICE; samples `pc.getStats()` on the phone leg for the real
  phone→server throughput.
- `app/rtc/tracks.py` — the adaptive stage as an aiortc source track: profile
  resolution (`scale`), foveation from gaze (`foveated`, `blur`, approximated
  at quarter size for speed), pacing at profile `fps` (never sleeps when the
  pipeline is behind), and shadow metrics (JPEG size / PSNR via
  `app.codecs.lossy`) sampled every 5th frame *off the frame path* so they
  never add latency. Outbound frames are always the fixed 960×540 canvas so
  the VP8 encoder never renegotiates; downshifted profiles ship fewer real
  pixels upscaled, which is precisely the ABR trade. Frames a slow consumer
  skips are counted as `frames_dropped` for that viewer.
- `app/rtc/bridge.py` — the shared frame slots + per-viewer adaptation state
  (EWMA → `pick_profile`, with the demo overrides), unit-tested headlessly.
- The offline study plane (`/metrics`, `/stream`, `/audio`, `experiments.py`)
  is unchanged; the live plane reuses the same codecs and ladder.
