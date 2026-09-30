"""Selection logic of the drift guard, with synthesis and pitch tracking stubbed out."""

import sys
from pathlib import Path

import numpy as np
import pytest

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


def run(monkeypatch, drops):
    """drops: {chunk_text: {scale: pitch drop or None}}. Returns (wav, info, calls)."""
    calls = []
    take_drop = {}

    def fake_chunk(tts, text, steps, scale):
        calls.append((text, scale))
        wav = np.zeros(TAKE_LEN, dtype=np.float32)
        take_drop[id(wav)] = drops[text][scale]
        return wav

    monkeypatch.setattr(tts_engine, "_synth_chunk", fake_chunk)
    monkeypatch.setattr(tts_engine, "upstream_normalize", lambda t: t)
    monkeypatch.setattr(tts_engine.voice_check, "window_f0_array", lambda wav, sr, w: take_drop[id(wav)])
    monkeypatch.setattr(tts_engine.voice_check, "take_score",
                        lambda drop, *_: float("inf") if drop is None else drop)

    wav, info = tts_engine.synthesize(StubTTS(drops), "bir iki üç dört beş", steps=4, drift_guard=True)
    return wav, info, calls


def test_accepts_first_good_take(monkeypatch):
    _, info, calls = run(monkeypatch, {"a": {1.0: 2.0, 0.9: 0.5, 1.1: 0.1, 1.2: 0.0}})
    assert info["chunks"][0]["scale"] == 1.0
    assert calls == [("a", 1.0)]  # below the acceptance threshold: no extra takes


def test_picks_least_drifting_take(monkeypatch):
    wav, info, calls = run(monkeypatch, {"a": {1.0: 9.0, 0.9: 7.0, 1.1: 5.0, 1.2: 6.0},
                                         "b": {1.0: 8.0, 0.9: 3.0, 1.1: 0.0, 1.2: 0.0}})
    assert [c["scale"] for c in info["chunks"]] == [1.1, 0.9]
    assert [c["pitch_drop_st"] for c in info["chunks"]] == [5.0, 3.0]
    assert calls == [("a", 1.0), ("a", 0.9), ("a", 1.1), ("a", 1.2), ("b", 1.0), ("b", 0.9)]
    assert len(wav) == 2 * TAKE_LEN + int(0.12 * 48000)  # two takes joined by one gap


def test_unvoiced_take_never_wins(monkeypatch):
    _, info, _ = run(monkeypatch, {"a": {1.0: None, 0.9: 12.0, 1.1: 11.0, 1.2: 10.0}})
    assert info["chunks"][0]["scale"] == 1.2


def test_guard_off_is_plain_upstream(monkeypatch):
    class Plain:
        def synthesize(self, text, steps):
            return np.ones(10, dtype=np.float32)

    wav, info = tts_engine.synthesize(Plain(), "metin", steps=4, drift_guard=False)
    assert info == {"drift_guard": False} and len(wav) == 10


def test_take_score_rejects_broken_takes():
    from voice_check import take_score
    nan = float("nan")
    assert take_score([330, 300, 280], 0.5, 200) == pytest.approx(12 * np.log2(315 / 280))
    assert take_score([60, nan, nan, nan], 0.5, 200) == float("inf")   # mostly unvoiced
    assert take_score([150, 140, 130], 0.5, 200) == float("inf")       # not the target voice
    assert take_score([], 0.5, 200) == float("inf")
