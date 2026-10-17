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
