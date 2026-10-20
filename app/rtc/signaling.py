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
                    {"type": "txstats", "kbps": float, "fps": float}      (phone: its own
                                                                           video-only wire rate)
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


async def _pump_video(bridge: Bridge, track, pc) -> None:
    bridge.phone_connected = True
    try:
        while True:
            frame = await track.recv()
            bridge.phone_capture = [frame.width, frame.height]
            bridge.phone_frames += 1
            img = await asyncio.to_thread(_to_canvas_image, frame, (config.VIDEO_CANVAS_WIDTH, config.VIDEO_CANVAS_HEIGHT))
            await bridge.video.publish(img)
    except MediaStreamError:
        pass
    finally:
        if bridge.phone_pc is pc:  # a newer phone may already own the bridge
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
        early_candidates = []
        msg = json.loads(await websocket.receive_text())
        while msg.get("type") == "ice":
            early_candidates.append(msg.get("candidate"))
            msg = json.loads(await websocket.receive_text())
        if msg.get("type") != "offer":
            return
        first = msg
        mode = first.get("audio", "mic")
        bridge.audio_mode = mode if mode in ("mic", "foa") else "mic"

        # Stop the synthetic pump if it was running
        bridge._synthetic_active = False

        pc = RTCPeerConnection(configuration=_rtc_config())
        await bridge.claim_phone(websocket, pc)  # kicks any previous publisher

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                pumps.append(asyncio.create_task(_pump_video(bridge, track, pc)))
            elif bridge.audio_mode == "mic":
                pumps.append(asyncio.create_task(_pump_audio(bridge, track)))
            # in "foa" mode the phone's audio is intentionally not relayed

        await pc.setRemoteDescription(RTCSessionDescription(sdp=first["sdp"], type="offer"))
        await pc.setLocalDescription(await pc.createAnswer())
        await websocket.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
        for cand in early_candidates:
            await _safe_add_ice(pc, cand)
        logger.info("phone connected (audio mode: %s)", bridge.audio_mode)

        while True:
            msg = json.loads(await websocket.receive_text())
            mtype = msg.get("type")
            if mtype == "ice":
                await _safe_add_ice(pc, msg.get("candidate"))
            elif mtype == "gaze":
                bridge.set_gaze(float(msg["nx"]), float(msg["ny"]))
            elif mtype == "txstats":
                # the phone's own video-only wire measurement; aiortc's stats
                # cannot isolate video bytes from the transport total, so the
                # sender-side compression ratio uses this number
                bridge.phone_reported = {"sent_kbps": msg.get("kbps"), "sent_fps": msg.get("fps")}
    except WebSocketDisconnect:
        pass
    finally:
        bridge.release_phone(websocket)  # no-op if a newer phone took over
        # Allow the synthetic pump to resume
        bridge._synthetic_active = False
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
    sample = state.shadow_sample  # JPEG/PSNR etc: sticky between samples
    canvas_bytes = config.VIDEO_CANVAS_WIDTH * config.VIDEO_CANVAS_HEIGHT * 3
    return {
        "ts": round(time.time(), 3),
        "viewer_id": state.id,
        "canvas": [config.VIDEO_CANVAS_WIDTH, config.VIDEO_CANVAS_HEIGHT],
        "profile": state.profile,
        "profile_source": state.profile_source,
        "measured_kbps": round(state.estimator.kbps, 1) if state.has_measurement else None,
        "volatility_ratio": round(state.estimator.volatility_ratio, 3),
        "safe_kbps": round(state.estimator.safe_kbps(), 1),
        "adaptation_reason": getattr(state.controller, "_last_reason", "normal"),
        "adaptation_action": getattr(state.controller, "_last_action", "maintain"),
        "allocation_strategy": getattr(state, "allocation_strategy", "intelligent_allocator"),
        "prediction_enabled": getattr(state, "prediction_enabled", False),
        "cap_kbps": state.cap_kbps,
        "jpeg_quality": knobs["jpeg_quality"],
        "scale": knobs["scale"],
        "fps_target": knobs["fps"],
        "foveated": knobs["foveated"],
        "audio_channels": min(knobs["audio_channels"], 2) if bridge.audio_mode == "foa" else 2,
        "gaze": list(bridge.gaze_norm),
        "shadow": {
            "jpeg_bytes": sample.get("jpeg_bytes"),
            "psnr_db": sample.get("psnr_db"),
            "raw_bytes": shadow.get("raw_bytes", canvas_bytes),
            "effective": shadow.get("effective"),
            "roi": shadow.get("roi"),
            "tile_bytes": sample.get("tile_bytes"),
            "viewport_psnr_db": sample.get("viewport_psnr_db"),
            "guard_band_psnr_db": sample.get("guard_band_psnr_db"),
            "peripheral_psnr_db": sample.get("peripheral_psnr_db"),
            "budget_bytes": sample.get("budget_bytes"),
            "overshoot_bytes": sample.get("overshoot_bytes"),
            "budget_retries": sample.get("budget_retries"),
            "within_budget": sample.get("within_budget", True),
            "boundary_artifact_mse": sample.get("boundary_artifact_mse"),
            "h264_bytes": shadow.get("h264_bytes"),
            "allocation_strategy": shadow.get("allocation_strategy", "intelligent_allocator"),
            "note": shadow.get("note"),
        },
        "frames_sent": state.frames_sent,
        "audio_frames": state.audio_frames,
        "phone_connected": bridge.phone_connected,
        "synthetic_active": getattr(bridge, "_synthetic_active", False),
        "phone": bridge.phone_stats | bridge.phone_reported,
        "audio_mode": bridge.audio_mode,
    }


async def _push_metrics(bridge: Bridge, state: ViewerState, websocket: WebSocket) -> None:
    last_ts = time.monotonic()
    last_fs = last_pf = 0
    while True:
        await asyncio.sleep(config.METRICS_INTERVAL_S)
        now = time.monotonic()
        dt = max(now - last_ts, 1e-6)
        try:
            fs, pf = state.frames_sent, bridge.phone_frames
            d_fs, d_pf = fs - last_fs, pf - last_pf
            last_ts, last_fs, last_pf = now, fs, pf
            snapshot = _metrics_snapshot(bridge, state)
            capture = bridge.phone_capture or [0, 0]
            snapshot["sender"] = {
                "capture": capture,
                "fps_in": round(d_pf / dt, 1),
                "raw_kbps": round(d_pf * capture[0] * capture[1] * 3 * 8 / 1000 / dt, 1) if d_pf else 0.0,
                # video-only wire rate as reported by the phone itself (the
                # server's transport meter mixes in audio + RTCP + retransmits)
                "wire_kbps": bridge.phone_reported.get("sent_kbps"),
            }
            canvas_bytes = config.VIDEO_CANVAS_WIDTH * config.VIDEO_CANVAS_HEIGHT * 3
            snapshot["receiver"] = {
                "fps_actual": round(d_fs / dt, 1),
                "raw_kbps": round(d_fs * canvas_bytes * 8 / 1000 / dt, 1) if d_fs else 0.0,
                "wire_kbps": state.last_report_kbps,
                "frames_dropped": state.frames_dropped,
            }
            await websocket.send_json({"type": "metrics", **snapshot})
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
        early_candidates = []
        msg = json.loads(await websocket.receive_text())
        while msg.get("type") == "ice":
            early_candidates.append(msg.get("candidate"))
            msg = json.loads(await websocket.receive_text())
        if msg.get("type") != "offer":
            return
        first = msg
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
        for cand in early_candidates:
            await _safe_add_ice(pc, cand)
        logger.info("viewer %s connected", state.id)

        while True:
            msg = json.loads(await websocket.receive_text())
            mtype = msg.get("type")
            if mtype == "ice":
                await _safe_add_ice(pc, msg.get("candidate"))
            elif mtype == "stats" and state is not None:
                proc_delay = float(msg.get("proc_delay", 0.0))
                state.report_throughput(int(msg["bytes"]), float(msg["dt"]), processing_delay_s=proc_delay)
            elif mtype == "set_profile" and state is not None:
                profile = msg.get("profile")
                state.force_profile = None if profile in (None, "auto") else str(profile)
                state.repick_profile()
            elif mtype == "set_cap" and state is not None:
                kbps = msg.get("kbps")
                state.cap_kbps = None if kbps is None else float(kbps)
                state.repick_profile()
            elif mtype == "set_mode" and state is not None:
                state.allocation_strategy = str(msg.get("mode", "intelligent_allocator"))
            elif mtype == "set_prediction" and state is not None:
                state.prediction_enabled = bool(msg.get("enabled", False))
            elif mtype == "set_gaze":
                x = float(msg.get("x", 0.5))
                y = float(msg.get("y", 0.5))
                bridge.set_gaze(x, y)
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
