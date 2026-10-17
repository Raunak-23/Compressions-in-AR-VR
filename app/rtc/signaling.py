"""WebSocket signaling for the live demo: phone publishes, viewers subscribe.

Media is peer-to-peer WebRTC; the tunnel (ngrok / localtunnel) carries only
HTTPS + these WebSockets. aiortc embeds its own ICE candidates in the SDP it
answers with, so the server never trickles; browsers trickle their candidates
in and they are parsed here.

Message protocol (JSON):
  client -> server: {"type": "offer", "sdp": str, "audio": "mic"|"foa"}  (phone, first)
                    {"type": "offer", "sdp": str}                        (viewer, first)
                    {"type": "ice", "candidate": {...}|None}
                    {"type": "gaze", "nx": float, "ny": float}            (phone)
                    {"type": "stats", "bytes": int, "dt": float}          (viewer)
                    {"type": "set_profile", "profile": "auto"|name}       (viewer)
                    {"type": "set_cap", "kbps": float|null}               (viewer)
  server -> client: {"type": "answer", "sdp": str}
                    {"type": "metrics", ...}                              (viewer, periodic)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from aioice.candidate import Candidate as AioIceCandidate
from aiortc import (
    RTCConfiguration,
    RTCIceCandidate,
    RTCIceServer,
    RTCSessionDescription,
    RTCPeerConnection,
)
from aiortc.mediastreams import MediaStreamError
from aiortc.rtcicetransport import candidate_from_aioice
from fastapi import WebSocket, WebSocketDisconnect

from app import config
from app.rtc.bridge import Bridge, ViewerState
from app.rtc.tracks import FoaAudioTrack, ProcessedVideoTrack, RelayAudioTrack

logger = logging.getLogger(__name__)


def _rtc_config() -> RTCConfiguration:
    return RTCConfiguration(iceServers=[RTCIceServer(urls=u) for u in config.ICE_SERVERS])


def _candidate_from_json(blob) -> RTCIceCandidate | None:
    """Parse a browser trickle candidate (or None = end-of-candidates)."""
    if blob is None:
        return None
    sdp = blob.get("candidate", "")
    if sdp.startswith("a="):
        sdp = sdp[2:]
    if sdp.startswith("candidate:"):
        sdp = sdp[len("candidate:") :]
    cand = candidate_from_aioice(AioIceCandidate.from_sdp(sdp))
    cand.sdpMid = blob.get("sdpMid")
    cand.sdpMLineIndex = blob.get("sdpMLineIndex")
    return cand


async def _safe_add_ice(pc: RTCPeerConnection, blob) -> None:
    try:
        await pc.addIceCandidate(_candidate_from_json(blob))
    except Exception:  # malformed or late candidates are non-fatal
        logger.debug("dropping ICE candidate: %r", blob)


def _to_canvas_image(frame, canvas: tuple[int, int]):
    img = frame.to_image()
    if img.size != canvas:
        img = img.resize(canvas)
    return img


async def _pump_video(bridge: Bridge, track) -> None:
    bridge.phone_connected = True
    try:
        while True:
            frame = await track.recv()
            img = await asyncio.to_thread(_to_canvas_image, frame, (config.VIDEO_CANVAS_WIDTH, config.VIDEO_CANVAS_HEIGHT))
            await bridge.video.publish(img)
    except MediaStreamError:
        pass
    finally:
        bridge.phone_connected = False


async def _pump_audio(bridge: Bridge, track) -> None:
    try:
        while True:
            frame = await track.recv()
            await bridge.audio.publish(frame)
    except MediaStreamError:
        pass


async def phone_endpoint(bridge: Bridge, websocket: WebSocket) -> None:
    """One phone session: receive its offer, relay its media into the bridge."""
    await websocket.accept()
    pc: RTCPeerConnection | None = None
    pumps: list[asyncio.Task] = []
    try:
        first = json.loads(await websocket.receive_text())
        if first.get("type") != "offer":
            return
        mode = first.get("audio", "mic")
        bridge.audio_mode = mode if mode in ("mic", "foa") else "mic"

        pc = RTCPeerConnection(configuration=_rtc_config())
        bridge.phone_pc = pc

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                pumps.append(asyncio.create_task(_pump_video(bridge, track)))
            elif bridge.audio_mode == "mic":
                pumps.append(asyncio.create_task(_pump_audio(bridge, track)))
            # in "foa" mode the phone's audio is intentionally not relayed

        await pc.setRemoteDescription(RTCSessionDescription(sdp=first["sdp"], type="offer"))
        await pc.setLocalDescription(await pc.createAnswer())
        await websocket.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
        logger.info("phone connected (audio mode: %s)", bridge.audio_mode)

        while True:
            msg = json.loads(await websocket.receive_text())
            mtype = msg.get("type")
            if mtype == "ice":
                await _safe_add_ice(pc, msg.get("candidate"))
            elif mtype == "gaze":
                bridge.set_gaze(float(msg["nx"]), float(msg["ny"]))
    except WebSocketDisconnect:
        pass
    finally:
        bridge.phone_pc = None
        bridge.phone_connected = False
        for task in pumps:
            task.cancel()
        if pc is not None:
            try:
                await pc.close()
            except Exception:
                pass
        logger.info("phone disconnected")


def _metrics_snapshot(bridge: Bridge, state: ViewerState) -> dict:
    knobs = state.knobs
    shadow = state.shadow
    canvas_bytes = config.VIDEO_CANVAS_WIDTH * config.VIDEO_CANVAS_HEIGHT * 3
    return {
        "ts": round(time.time(), 3),
        "viewer_id": state.id,
        "profile": state.profile,
        "profile_source": state.profile_source,
        "measured_kbps": round(state.estimator.kbps, 1) if state.has_measurement else None,
        "cap_kbps": state.cap_kbps,
        "jpeg_quality": knobs["jpeg_quality"],
        "scale": knobs["scale"],
        "fps_target": knobs["fps"],
        "foveated": knobs["foveated"],
        "audio_channels": min(knobs["audio_channels"], 2) if bridge.audio_mode == "foa" else 2,
        "gaze": list(bridge.gaze_norm),
        "shadow": {
            "jpeg_bytes": shadow.get("jpeg_bytes"),
            "psnr_db": shadow.get("psnr_db"),
            "raw_bytes": shadow.get("raw_bytes", canvas_bytes),
            "effective": shadow.get("effective"),
            "roi": shadow.get("roi"),
        },
        "frames_sent": state.frames_sent,
        "audio_frames": state.audio_frames,
        "phone_connected": bridge.phone_connected,
        "phone": bridge.phone_stats,
        "audio_mode": bridge.audio_mode,
    }


async def _push_metrics(bridge: Bridge, state: ViewerState, websocket: WebSocket) -> None:
    while True:
        await asyncio.sleep(config.METRICS_INTERVAL_S)
        try:
            await websocket.send_json({"type": "metrics", **_metrics_snapshot(bridge, state)})
        except Exception:
            break  # viewer websocket is gone; the endpoint loop cleans up


async def viewer_endpoint(bridge: Bridge, websocket: WebSocket) -> None:
    """One viewer session: serve the adaptively-encoded stream + metrics."""
    await websocket.accept()
    pc: RTCPeerConnection | None = None
    state: ViewerState | None = None
    video_track = audio_track = None
    metrics_task: asyncio.Task | None = None
    try:
        first = json.loads(await websocket.receive_text())
        if first.get("type") != "offer":
            return
        state = bridge.new_viewer()
        pc = RTCPeerConnection(configuration=_rtc_config())

        video_track = ProcessedVideoTrack(bridge, state)
        audio_track = FoaAudioTrack(bridge, state) if bridge.audio_mode == "foa" else RelayAudioTrack(bridge)
        pc.addTrack(video_track)
        pc.addTrack(audio_track)
        metrics_task = asyncio.create_task(_push_metrics(bridge, state, websocket))

        await pc.setRemoteDescription(RTCSessionDescription(sdp=first["sdp"], type="offer"))
        await pc.setLocalDescription(await pc.createAnswer())
        await websocket.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
        logger.info("viewer %s connected", state.id)

        while True:
            msg = json.loads(await websocket.receive_text())
            mtype = msg.get("type")
            if mtype == "ice":
                await _safe_add_ice(pc, msg.get("candidate"))
            elif mtype == "stats" and state is not None:
                state.report_throughput(int(msg["bytes"]), float(msg["dt"]))
            elif mtype == "set_profile" and state is not None:
                profile = msg.get("profile")
                state.force_profile = None if profile in (None, "auto") else str(profile)
                state.repick_profile()
            elif mtype == "set_cap" and state is not None:
                kbps = msg.get("kbps")
                state.cap_kbps = None if kbps is None else float(kbps)
                state.repick_profile()
    except WebSocketDisconnect:
        pass
    finally:
        if metrics_task is not None:
            metrics_task.cancel()
        if state is not None:
            bridge.drop_viewer(state)
        for track in (video_track, audio_track):
            if track is not None:
                track.stop()
        if pc is not None:
            try:
                await pc.close()
            except Exception:
                pass
        logger.info("viewer disconnected")


async def phone_stats_loop(bridge: Bridge) -> None:
    """Sample the phone connection's real WebRTC stats (server-side view)."""
    prev_bytes: int | None = None
    prev_ts: float | None = None
    while True:
        await asyncio.sleep(1.0)
        pc = bridge.phone_pc
        if pc is None:
            prev_bytes = prev_ts = None
            bridge.phone_stats = {"inbound_kbps": None, "packets_lost": None, "frames_decoded": None}
            continue
        try:
            report = await pc.getStats()
        except Exception:
            continue
        rx_bytes = sum(
            s.bytesReceived for s in report.values() if getattr(s, "type", "") == "transport"
        )
        lost = sum(
            getattr(s, "packetsLost", 0) or 0
            for s in report.values()
            if getattr(s, "type", "") == "inbound-rtp" and getattr(s, "kind", "") == "video"
        )
        now = time.monotonic()
        kbps = None
        if prev_bytes is not None and prev_ts is not None and now > prev_ts:
            kbps = round(max(rx_bytes - prev_bytes, 0) * 8 / 1000 / (now - prev_ts), 1)
        prev_bytes, prev_ts = rx_bytes, now
        bridge.phone_stats = {
            "inbound_kbps": kbps,
            "packets_lost": lost,
            "frames_decoded": bridge.phone_stats.get("frames_decoded"),
        }
