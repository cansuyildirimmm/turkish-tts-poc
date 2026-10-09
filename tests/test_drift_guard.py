"""Selection logic of the drift guard, with synthesis and pitch tracking stubbed out."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tts_engine  # noqa: E402

TAKE_LEN = 4800


class StubTTS:
    sample_rate = 48000
    max_words = 3

    def __init__(self, chunks):
        self.chunks = chunks

    def _clauses(self, text):
        return list(self.chunks)


def run(monkeypatch, floors):
    """floors: {chunk_text: {scale: lowest window Hz (0 = broken take)}}. Returns (wav, info, calls)."""
    calls = []
    take_floor = {}

    def fake_chunk(tts, text, steps, scale):
        calls.append((text, scale))
        wav = np.zeros(TAKE_LEN, dtype=np.float32)
        take_floor[id(wav)] = floors[text][scale]
        return wav

    monkeypatch.setattr(tts_engine, "_synth_chunk", fake_chunk)
    monkeypatch.setattr(tts_engine, "upstream_normalize", lambda t: t)
    monkeypatch.setattr(tts_engine.voice_check, "window_f0_array", lambda wav, sr, w: take_floor[id(wav)])
    monkeypatch.setattr(tts_engine.voice_check, "take_floor", lambda floor, *_: floor)

    wav, info = tts_engine.synthesize(StubTTS(floors), "bir iki üç dört beş", steps=4, drift_guard=True)
    return wav, info, calls


def test_accepts_first_take_above_floor(monkeypatch):
    _, info, calls = run(monkeypatch, {"a": {1.0: 185.0, 0.9: 220.0, 1.1: 230.0, 1.2: 240.0}})
    assert info["chunks"][0]["scale"] == 1.0
    assert calls == [("a", 1.0)]  # at or above the floor: no extra takes


def test_picks_highest_floor_take(monkeypatch):
    wav, info, calls = run(monkeypatch, {"a": {1.0: 100.0, 0.9: 130.0, 1.1: 170.0, 1.2: 150.0},
                                         "b": {1.0: 120.0, 0.9: 180.0, 1.1: 250.0, 1.2: 250.0}})
    assert [c["scale"] for c in info["chunks"]] == [1.1, 0.9]
    assert [c["min_f0_hz"] for c in info["chunks"]] == [170.0, 180.0]
    assert calls == [("a", 1.0), ("a", 0.9), ("a", 1.1), ("a", 1.2), ("b", 1.0), ("b", 0.9)]
    assert len(wav) == 2 * TAKE_LEN + int(0.12 * 48000)  # two takes joined by one gap


def test_broken_take_never_wins(monkeypatch):
    _, info, _ = run(monkeypatch, {"a": {1.0: 0.0, 0.9: 100.0, 1.1: 110.0, 1.2: 120.0}})
    assert info["chunks"][0]["scale"] == 1.2


def test_all_broken_reports_none(monkeypatch):
    _, info, _ = run(monkeypatch, {"a": {1.0: 0.0, 0.9: 0.0, 1.1: 0.0, 1.2: 0.0}})
    assert info["chunks"][0] == {"text": "a", "scale": 1.0, "min_f0_hz": None}


def test_guard_off_is_plain_upstream(monkeypatch):
    class Plain:
        def synthesize(self, text, steps):
            return np.ones(10, dtype=np.float32)

    wav, info = tts_engine.synthesize(Plain(), "metin", steps=4, drift_guard=False)
    assert info == {"drift_guard": False} and len(wav) == 10


def test_take_floor_rejects_broken_takes():
    from voice_check import take_floor
    nan = float("nan")
    assert take_floor([330, 300, nan, 280], 0.5, 200) == 280.0
    assert take_floor([60, nan, nan, nan], 0.5, 200) == 0.0   # mostly unvoiced
    assert take_floor([150, 140, 130], 0.5, 200) == 0.0       # not the target voice
    assert take_floor([], 0.5, 200) == 0.0
