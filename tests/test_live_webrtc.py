"""End-to-end WebRTC loopback: aiortc phone -> bridge -> aiortc viewer.

This exercises the full live pipeline headlessly -- signaling, ICE over
loopback, the adaptive stage, shadow metrics and the profile ladder -- with
synthetic phone media, no browser required.
"""

import asyncio
import fractions
import json

import numpy as np
import pytest

aiortc = pytest.importorskip("aiortc")
websockets = pytest.importorskip("websockets")

from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError, VideoStreamTrack  # noqa: E402
from av import AudioFrame, VideoFrame  # noqa: E402

from app import config  # noqa: E402
from app.server import app  # noqa: E402

PORT = 8209
BASE = f"ws://127.0.0.1:{PORT}"


class PulseVideoTrack(VideoStreamTrack):
    """Synthetic phone camera: a red pulse each frame, 30 fps."""

    kind = "video"

    def __init__(self):
        super().__init__()
        self._n = 0

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        self._n += 1
        arr = np.zeros((180, 320, 3), dtype=np.uint8)
        arr[..., 0] = (self._n * 25) % 256
        arr[40:140, 80:240, 1] = 220
        frame = VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts, frame.time_base = pts, time_base
        return frame


class ToneAudioTrack(AudioStreamTrack):
    """Synthetic phone mic: 960-sample 48 kHz stereo frames at 50 pps."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._pts = 0
        self._next = None

    async def recv(self) -> AudioFrame:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._next is None:
            self._next = now
        if self._next > now:
            await asyncio.sleep(self._next - now)
        self._next = max(loop.time(), self._next) + 0.02
        arr = np.zeros((1, 960 * 2), dtype=np.int16)  # packed stereo
        arr[0, ::2] = 2000
        frame = AudioFrame.from_ndarray(arr, format="s16", layout="stereo")
        frame.sample_rate = 48000
        frame.pts, frame.time_base = self._pts, fractions.Fraction(1, 48000)
        self._pts += 960
        return frame


class Ws:
    """websockets client with a background JSON message pump."""

    def __init__(self, ws):
        self.ws = ws
        self.q: asyncio.Queue = asyncio.Queue()
        self.task = asyncio.create_task(self._pump())

    async def _pump(self):
        try:
            async for raw in self.ws:
                await self.q.put(json.loads(raw))
        except Exception:
            pass

    async def send(self, obj):
        await self.ws.send(json.dumps(obj))

    async def recv(self, type_: str | None = None, timeout: float = 20):
        while True:
            msg = await asyncio.wait_for(self.q.get(), timeout)
            if type_ is None or msg.get("type") == type_:
                return msg

    async def close(self):
        self.task.cancel()
        await self.ws.close()


async def _negotiate(ws, pc, first_msg):
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await ws.send(first_msg | {"sdp": pc.localDescription.sdp})
    answer = await ws.recv("answer")
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type="answer"))
    # server candidates are embedded in the SDP; nothing to trickle back in


async def _run():
    old_servers = config.ICE_SERVERS
    config.ICE_SERVERS = []  # loopback: host candidates only, no external STUN
    server = None
    server_task = None
    phone_ws = viewer_ws = None
    phone_pc = viewer_pc = None
    try:
        import uvicorn

        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning", ws="websockets")
        )
        server_task = asyncio.create_task(server.serve())
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert server.started, "uvicorn did not start"

        # --- phone publishes ------------------------------------------------
        phone_pc = RTCPeerConnection()
        phone_pc.addTrack(PulseVideoTrack())
        phone_pc.addTrack(ToneAudioTrack())
        phone_ws = Ws(await websockets.connect(f"{BASE}/ws/phone"))
        await _negotiate(phone_ws, phone_pc, {"type": "offer", "audio": "mic"})
        await asyncio.sleep(0.1)

        # --- viewer subscribes ----------------------------------------------
        viewer_pc = RTCPeerConnection()
        video_frames: list[VideoFrame] = []
        audio_frames: list[AudioFrame] = []

        @viewer_pc.on("track")
        def on_track(track):
            async def read_video():
                try:
                    while True:
                        video_frames.append(await track.recv())
                except MediaStreamError:
                    pass

            async def read_audio():
                try:
                    while True:
                        audio_frames.append(await track.recv())
                except MediaStreamError:
                    pass

            if track.kind == "video":
                asyncio.create_task(read_video())
            else:
                asyncio.create_task(read_audio())

        viewer_pc.addTransceiver("video", direction="recvonly")
        viewer_pc.addTransceiver("audio", direction="recvonly")
        viewer_ws = Ws(await websockets.connect(f"{BASE}/ws/viewer"))
        await _negotiate(viewer_ws, viewer_pc, {"type": "offer"})

        # --- startup: no measurement yet -> default 'high' -------------------
        m = await viewer_ws.recv("metrics", 15)
        assert m["profile"] == "high" and m["profile_source"] == "default"
        assert m["phone_connected"] is True

        # --- let media flow ---------------------------------------------------
        await asyncio.sleep(2.5)
        assert len(video_frames) >= 10, "viewer should be receiving video"
        assert video_frames[-1].width == config.VIDEO_CANVAS_WIDTH
        assert video_frames[-1].height == config.VIDEO_CANVAS_HEIGHT
        assert len(audio_frames) >= 10, "viewer should be receiving audio"

        # --- real receiver stats -> EWMA -> ladder ---------------------------
        await viewer_ws.send({"type": "stats", "bytes": 125_000, "dt": 0.25})  # 4000 kbps
        m = await viewer_ws.recv("metrics", 5)
        while m["profile_source"] != "measured":
            m = await viewer_ws.recv("metrics", 5)
        assert m["profile"] == "ultra" and m["measured_kbps"] == pytest.approx(4000.0, rel=0.01)

        # --- gaze from the phone reaches the metrics -------------------------
        await phone_ws.send({"type": "gaze", "nx": 0.9, "ny": 0.2})
        m = await viewer_ws.recv("metrics", 5)
        while m["gaze"] != pytest.approx([0.9, 0.2], abs=1e-6):
            m = await viewer_ws.recv("metrics", 5)
        assert m["shadow"]["roi"] is None  # ultra is unfoveated: gaze live, no RoI

        # --- simulated link cap downshifts to 'low' ---------------------------
        await viewer_ws.send({"type": "set_cap", "kbps": 300})
        m = await viewer_ws.recv("metrics", 5)
        while m["profile"] != "low":
            m = await viewer_ws.recv("metrics", 5)
        assert m["profile_source"] == "capped"
        assert m["foveated"] is True and m["fps_target"] == 15
        assert m["jpeg_quality"] == 35

        # low profile: half resolution, foveated -> shadow jpeg bytes present
        await asyncio.sleep(1.0)
        assert m["shadow"]["effective"] == [640, 360]
        assert m["shadow"]["roi"] is not None  # foveation is on at 'low'
    finally:
        config.ICE_SERVERS = old_servers
        for closer in (phone_ws, viewer_ws):
            if closer:
                await closer.close()
        for pc in (phone_pc, viewer_pc):
            if pc:
                await pc.close()
        if server:
            server.should_exit = True
            if server_task:
                try:
                    await asyncio.wait_for(server_task, 10)
                except Exception:
                    pass


def test_live_pipeline_end_to_end():
    asyncio.run(_run())
