"""Unit tests for the live bridge control logic (no WebRTC transport needed)."""

import asyncio

import pytest

from app import config
from app.rtc.bridge import Bridge, VideoSlot, ViewerState
from app.rtc.signaling import _rtc_config


def test_rtc_config_builds_ice_servers():
    """Regression: the real config lists a STUN URL, and _rtc_config must
    actually construct an RTCIceServer per entry (an empty test list once
    masked a missing import here)."""
    assert config.ICE_SERVERS, "default config should ship at least one ICE server"
    rtc_config = _rtc_config()
    assert [s.urls for s in rtc_config.iceServers] == config.ICE_SERVERS


def test_viewer_profile_precedence():
    state = ViewerState(id="t")
    assert state.profile == "high" and state.profile_source == "default"

    # A healthy link measures in: EWMA -> ladder.
    state.report_throughput(125_000, 0.25)  # 4000 kbps
    assert state.profile == "ultra" and state.profile_source == "measured"

    # Forced demo override wins over measurement...
    state.force_profile = "low"
    state.repick_profile()
    assert state.profile == "low" and state.profile_source == "forced"
    # ...and releasing it falls back to the measurement.
    state.force_profile = None
    state.repick_profile()
    assert state.profile == "ultra"

    # A simulated cap wins over the (huge, loopback) measurement.
    state.cap_kbps = 300
    state.repick_profile()
    assert state.profile == "low" and state.profile_source == "capped"
    state.cap_kbps = None
    state.repick_profile()
    assert state.profile == "ultra"


def test_viewer_degenerate_reports_are_ignored():
    state = ViewerState(id="t")
    state.report_throughput(100_000, 0.0)  # dt<=0 must not poison the estimator
    assert not state.has_measurement and state.profile == "high"


def test_video_slot_fanout_and_skip():
    async def main():
        slot = VideoSlot()
        await slot.publish("frame1")
        await slot.publish("frame2")
        a = asyncio.create_task(slot.wait_for(0))  # fresh consumer: newest wins
        b = asyncio.create_task(slot.wait_for(1))  # stale consumer: catches up
        img_a, seq_a = await asyncio.wait_for(a, 2)
        img_b, seq_b = await asyncio.wait_for(b, 2)
        assert (img_a, seq_a) == ("frame2", 2)
        assert (img_b, seq_b) == ("frame2", 2)

    asyncio.run(main())


def test_bridge_gaze_clamped():
    bridge = Bridge()
    bridge.set_gaze(1.7, -0.4)
    assert bridge.gaze_norm == (1.0, 0.0)
    bridge.set_gaze(0.25, 0.75)
    assert bridge.gaze_norm == (0.25, 0.75)


def test_bridge_phone_takeover():
    """A second publisher kicks the first; a stale session's release is a no-op."""

    class FakePc:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    class FakeWs:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    async def main():
        bridge = Bridge()
        ws1, pc1, ws2, pc2 = FakeWs(), FakePc(), FakeWs(), FakePc()
        await bridge.claim_phone(ws1, pc1)
        assert bridge.phone_ws is ws1 and bridge.phone_connected
        await bridge.claim_phone(ws2, pc2)  # second publisher takes over
        assert pc1.closed and ws1.closed  # first is kicked
        assert bridge.phone_ws is ws2 and bridge.phone_pc is pc2
        bridge.release_phone(ws1)  # stale cleanup must not clear the new owner
        assert bridge.phone_ws is ws2 and bridge.phone_connected
        bridge.release_phone(ws2)
        assert bridge.phone_ws is None and not bridge.phone_connected
        assert bridge.phone_reported["sent_kbps"] is None  # released phone's report cleared

    asyncio.run(main())


def test_process_frame_fast_paths_and_off_path_shadow():
    """Scale-1.0 profiles must not resize at all, foveation must blur the
    periphery, and the JPEG/PSNR shadow must sample off the frame path."""
    import numpy as np
    from PIL import Image

    from app.rtc.tracks import ProcessedVideoTrack, process_frame

    img = Image.fromarray((np.arange(540 * 960 * 3, dtype=np.uint8) % 255).reshape(540, 960, 3))

    out, work, shadow = process_frame(img, config.PROFILES["ultra"], (0.5, 0.5))
    assert work is img  # ultra (scale 1.0): zero resize on the hot path
    assert out.size == (960, 540) and shadow["roi"] is None

    out_m, work_m, shadow_m = process_frame(img, config.PROFILES["medium"], (0.5, 0.5))
    assert out_m.size == (960, 540) and shadow_m["roi"] is not None
    assert work_m.size == (720, 405)
    # periphery (corner, outside the centered RoI) is genuinely blurred
    plain = img.resize((720, 405), Image.Resampling.BILINEAR)
    assert np.asarray(work_m)[5, 5].tolist() != np.asarray(plain)[5, 5].tolist()

    async def sample(profile, work_img):
        bridge = Bridge()
        state = bridge.new_viewer()
        state.force_profile = profile
        track = ProcessedVideoTrack(bridge, state)
        await track._sample_shadow(img, work_img, config.PROFILES[profile])
        return state.shadow_sample

    med_sample = asyncio.run(sample("medium", work_m))
    assert med_sample["jpeg_bytes"] > 0 and med_sample["psnr_db"] is not None
    ultra_sample = asyncio.run(sample("ultra", work))
    assert ultra_sample["jpeg_bytes"] > 0
    assert ultra_sample["psnr_db"] is None  # lossless stage -> infinite PSNR
