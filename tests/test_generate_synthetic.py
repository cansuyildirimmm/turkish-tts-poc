"""Synthetic-data generation: take selection, checks and the consistency filter.

Synthesis and pitch tracking are stubbed out; no model is loaded.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import generate_synthetic as gs  # noqa: E402

SR = 48000
TH = gs.GenThresholds()
TEXT = "Stok kartı güncellendi."  # 20 letters


def speech(seconds: float, sr: int = SR, edge_s: float = 0.1) -> np.ndarray:
    """Tone with silent edges, so the speech span is `seconds`."""
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    edge = np.zeros(int(edge_s * sr), dtype=np.float32)
    return np.concatenate([edge, tone, edge])


@pytest.fixture
def f0(monkeypatch):
    """Set the per-window F0 track that the stubbed pitch tracker returns."""
    track = {"windows": [230.0, 228.0, 225.0, 226.0]}
    monkeypatch.setattr(gs.voice_check, "window_f0_array", lambda wav, sr, w: list(track["windows"]))
    return track


def test_assess_accepts_natural_final_fall(f0):
    # 330 -> 186 Hz is ~10 semitones, judged natural by listening: must pass
    f0["windows"] = [330.0, 320.0, 280.0, 186.0]
    m = gs.assess(speech(1.5), SR, TEXT, TH)
    assert m["reasons"] == [] and m["max_drop_st"] > 9 and m["min_f0_hz"] == 186.0


def test_assess_accepts_clean_take(f0):
    m = gs.assess(speech(1.5), SR, TEXT, TH)
    assert m["reasons"] == []
    assert m["opening_hz"] == 229.0
    assert m["letters_per_s"] == pytest.approx(20 / 1.5, abs=0.2)


@pytest.mark.parametrize("windows,code", [
    ([230.0, 230.0, 230.0, 160.0], "low_pitch"),           # below the 180 Hz floor
    ([150.0, 150.0, 150.0, 150.0], "low_opening"),         # not the target voice
    ([230.0, float("nan"), float("nan"), float("nan")], "unvoiced"),
])
def test_assess_rejects_pitch_problems(f0, windows, code):
    f0["windows"] = windows
    codes = [r.split(":")[0] for r in gs.assess(speech(1.5), SR, TEXT, TH)["reasons"]]
    assert code in codes


def test_assess_rejects_rate_duration_and_clipping(f0):
    codes = lambda wav, text=TEXT: [r.split(":")[0] for r in gs.assess(wav, SR, text, TH)["reasons"]]  # noqa: E731
    assert "speaking_rate" in codes(speech(10.0))           # 2 letters/s: text/audio mismatch
    assert "duration" in codes(speech(0.2), "Evet.")
    assert "clipped" in codes(np.clip(speech(1.5) * 10, -1, 1))


def run_generate(monkeypatch, drops):
    """drops: lowest window pitch (Hz) per scale, in scale order. Returns (wav, log fields, scales tried)."""
    tried = []

    def fake_chunk(tts, text, steps, scale):
        tried.append(scale)
        return np.full(10, len(tried), dtype=np.float32)

    def fake_assess(wav, sr, text, th):
        low = drops[int(wav[0]) - 1]
        reasons = [] if low >= th.min_pitch_hz else [f"low_pitch: {low}"]
        return {"min_f0_hz": low, "duration_s": 1.0, "reasons": reasons}

    monkeypatch.setattr(gs.tts_engine, "_synth_chunk", fake_chunk)
    monkeypatch.setattr(gs, "assess", fake_assess)
    tts = type("T", (), {"sample_rate": SR})()
    wav, m = gs.generate_clip(tts, TEXT, 4, TH)
    return wav, m, tried


def test_generate_keeps_first_passing_take(monkeypatch):
    wav, m, tried = run_generate(monkeypatch, [150.0, 200.0, 220.0, 220.0])
    assert tried == [1.0, 0.9]
    assert m["status"] == "accepted" and m["scale"] == 0.9 and m["attempts"] == 2
    assert wav[0] == 2


def test_generate_rejects_and_reports_best_take(monkeypatch):
    wav, m, tried = run_generate(monkeypatch, [120.0, 175.0, 110.0, 160.0])
    assert wav is None
    assert tried == list(TH.scales)
    assert m["status"] == "rejected" and m["scale"] == 0.9 and m["min_f0_hz"] == 175.0
    assert m["attempts"] == len(TH.scales)


def test_clip_ids():
    assert gs.clip_ids("x001", 1) == ["x001"]
    assert gs.clip_ids("x001", 2) == ["x001_1", "x001_2"]


def test_trial_selection_is_deterministic_and_seeded():
    rows = [{"id": f"x{i:03d}"} for i in range(50)]
    a = gs.select_trial(rows, 10, 42)
    assert a == gs.select_trial(list(reversed(rows)), 10, 42)
    assert len(a) == 10 and a != rows[:10]
    assert a != gs.select_trial(rows, 10, 7)


def write_log(out: Path, rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "wavs").mkdir(exist_ok=True)
    with open(out / "generation_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=gs.LOG_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in gs.LOG_FIELDS})
            if r["status"] == "accepted":
                (out / "wavs" / f"{r['clip_id']}.wav").write_bytes(b"x")


def log_row(cid, status, f0_hz, source="synthetic"):
    return {"clip_id": cid, "source": source, "source_id": cid, "category": "erp_long",
            "text": f"metin {cid}", "status": status, "scale": 1.0, "attempts": 1, "duration_s": 2.0,
            "median_f0_hz": f0_hz, "max_drop_st": 1.0, "reasons": "" if status == "accepted" else "low_pitch: 150",
            "compute_s": 10.0}


def test_finalize_excludes_pitch_outliers_and_writes_script(tmp_path):
    rows = [log_row("a", "accepted", 230), log_row("b", "accepted", 232), log_row("c", "accepted", 228),
            log_row("d", "accepted", 180),  # ~4 semitones below the voice median
            log_row("e", "rejected", 230)]
    write_log(tmp_path, rows)
    s = gs.finalize(tmp_path, TH)

    assert s["kept"] == 3 and s["excluded_by_consistency"] == ["d"]
    assert (tmp_path / "excluded" / "d.wav").is_file() and not (tmp_path / "wavs" / "d.wav").exists()
    with open(tmp_path / "script.tsv", encoding="utf-8-sig") as f:
        script = list(csv.DictReader(f, delimiter="\t"))
    assert [r["id"] for r in script] == ["a", "b", "c"]
    assert script[0]["train_text"] == "metin a"
    assert s["reject_reasons"] == {"low_pitch": 1}
    assert json.loads((tmp_path / "generation_summary.json").read_text(encoding="utf-8"))["kept"] == 3


def test_finalize_restores_clip_that_is_consistent_again(tmp_path):
    write_log(tmp_path, [log_row("a", "accepted", 230), log_row("b", "accepted", 231)])
    (tmp_path / "wavs" / "b.wav").unlink()
    (tmp_path / "excluded").mkdir()
    (tmp_path / "excluded" / "b.wav").write_bytes(b"x")
    gs.finalize(tmp_path, TH)
    assert (tmp_path / "wavs" / "b.wav").is_file()


def test_resume_refuses_changed_thresholds(tmp_path):
    gs._check_config(tmp_path, {"thresholds": {"min_pitch_hz": 180.0}})
    gs._check_config(tmp_path, {"thresholds": {"min_pitch_hz": 180.0}})  # same settings: fine
    with pytest.raises(SystemExit):
        gs._check_config(tmp_path, {"thresholds": {"min_pitch_hz": 170.0}})
