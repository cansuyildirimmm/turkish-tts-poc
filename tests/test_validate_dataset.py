import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import validate_dataset as vd  # noqa: E402

GOOD_TEXT = "Sabah erkenden kalkıp işe gitmek için hazırlandım."  # 43 letters


def speech_like(seconds, sr, amp=0.3, seed=0):
    """Harmonic tone with a syllable-rate envelope: loud, voiced, non-silent."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 180 + 20 * rng.random()
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)
    return (amp * sig / np.max(np.abs(sig)) * env).astype(np.float32)


def write_wav(path, speech_s=3.0, sr=48000, lead=0.2, trail=0.2, amp=0.3, channels=1,
              subtype="PCM_24", seed=0, clip=False, noise=1e-4):
    rng = np.random.default_rng(seed + 1000)
    x = np.concatenate([np.zeros(int(lead * sr)), speech_like(speech_s, sr, amp, seed), np.zeros(int(trail * sr))])
    x = x + rng.normal(0, noise, len(x))  # noise floor
    if clip:
        x = np.clip(x * 5, -1, 1)
    if channels == 2:
        x = np.stack([x, x], axis=1)
    sf.write(str(path), x.astype(np.float32), sr, subtype=subtype)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    if not config.FREYATTS_VOCAB_PATH.is_file():
        pytest.skip("FreyaTTS source not installed (see README)")
    root = tmp_path_factory.mktemp("ds")
    wavs = root / "wavs"
    wavs.mkdir()
    texts = {
        "ok1": GOOD_TEXT,
        "ok2": "Toplantı odasındaki projeksiyon cihazı yine çalışmıyor.",
        "short": "Tamam.",
        "long": GOOD_TEXT,
        "clip": GOOD_TEXT + " Bir.",
        "silent": GOOD_TEXT + " İki.",
        "lowsr": GOOD_TEXT + " Üç.",
        "sr44": GOOD_TEXT + " Dört.",
        "stereo": GOOD_TEXT + " Beş.",
        "pcm16": GOOD_TEXT + " Altı.",
        "dup": GOOD_TEXT + " Yedi.",
        "emptytext": "",
        "mojibake": "Ã–ÄŸrenci kaydÄ± oluÅŸturuldu tamamen.",
        "leak": "Tahsilat fişi başarıyla oluşturuldu.",
        "mismatch": GOOD_TEXT * 4,
        "noedge": GOOD_TEXT + " Sekiz.",
        "noisy": GOOD_TEXT + " On bir.",
        "corrupt": GOOD_TEXT + " Dokuz.",
        "missing": GOOD_TEXT + " On.",
    }
    with open(root / "script.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "category", "train_text"])
        for k, v in texts.items():
            w.writerow([k, "test", v])

    write_wav(wavs / "ok1.wav", seed=1)
    write_wav(wavs / "ok2.wav", seed=2)
    write_wav(wavs / "short.wav", speech_s=0.3, lead=0.05, trail=0.05, seed=3)
    write_wav(wavs / "long.wav", speech_s=15, seed=4)
    write_wav(wavs / "clip.wav", clip=True, seed=5)
    sf.write(str(wavs / "silent.wav"), np.zeros(48000 * 3, dtype=np.float32), 48000, subtype="PCM_24")
    write_wav(wavs / "lowsr.wav", sr=16000, seed=6)
    write_wav(wavs / "sr44.wav", sr=44100, seed=7)
    write_wav(wavs / "stereo.wav", channels=2, seed=8)
    write_wav(wavs / "pcm16.wav", subtype="PCM_16", seed=9)
    (wavs / "dup.wav").write_bytes((wavs / "ok1.wav").read_bytes())
    write_wav(wavs / "emptytext.wav", seed=10)
    write_wav(wavs / "mojibake.wav", seed=11)
    write_wav(wavs / "leak.wav", seed=12)
    write_wav(wavs / "mismatch.wav", seed=13)
    write_wav(wavs / "noedge.wav", lead=0.0, seed=14)
    write_wav(wavs / "noisy.wav", noise=0.02, seed=16)
    (wavs / "corrupt.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEgarbage")
    write_wav(wavs / "orphan.wav", seed=15)  # not in script
    return root


@pytest.fixture(scope="module")
def report(dataset):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (dataset / "wavs").iterdir()}
    rep = vd.validate(dataset, dataset / "script.tsv")
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (dataset / "wavs").iterdir()}
    assert before == after, "validation must not modify the dataset"
    return rep


def rec(report, rid):
    return next(r for r in report["records"] if r["id"] == rid)


def has(messages, prefix):
    return any(m.startswith(prefix) for m in messages)


def test_good_records_accepted(report):
    for rid in ("ok1", "ok2"):
        r = rec(report, rid)
        assert r["errors"] == [], r
        assert r["warnings"] == [], r
        assert r["sample_rate"] == 48000 and r["channels"] == 1
        assert 3.0 < r["duration_s"] < 3.5


@pytest.mark.parametrize("rid, prefix", [
    ("short", "too short"),
    ("long", "too long"),
    ("clip", "clipping"),
    ("silent", "no speech detected"),
    ("lowsr", "low sample rate"),
    ("dup", "duplicate audio"),
    ("emptytext", "empty transcript"),
    ("mojibake", "encoding problem"),
    ("leak", "evaluation-set leakage"),
    ("corrupt", "unreadable audio"),
    ("orphan", "no transcript"),
])
def test_errors(report, rid, prefix):
    r = rec(report, rid)
    assert has(r["errors"], prefix), r["errors"]
    assert rid in report["rejected_ids"]


@pytest.mark.parametrize("rid, prefix", [
    ("sr44", "sample rate"),
    ("stereo", "channels"),
    ("pcm16", "bit depth"),
    ("mismatch", "speaking rate"),
    ("noedge", "no leading silence"),
    ("noisy", "low estimated SNR"),
])
def test_warnings_keep_record(report, rid, prefix):
    r = rec(report, rid)
    assert has(r["warnings"], prefix), r["warnings"]
    assert r["errors"] == [], r["errors"]
    assert rid not in report["rejected_ids"]


def test_missing_audio_listed(report):
    assert report["missing_audio"] == ["missing"]


def test_summary_counts(report):
    s = report["summary"]
    assert s["script_entries"] == 19
    assert s["audio_files"] == 19
    assert s["accepted"] + s["rejected"] == s["audio_files"]
    assert s["accepted"] == 8  # ok1, ok2, sr44, stereo, pcm16, mismatch, noedge, noisy


def test_snr_measured_from_edge_silence(report):
    assert rec(report, "ok1")["snr_db"] > 50
    assert rec(report, "noisy")["snr_db"] is None  # noise masks the silent edges
    assert has(rec(report, "noisy")["warnings"], "low estimated SNR: ~")


def test_text_report_renders(report):
    txt = vd.format_text_report(report)
    assert "REJECTED RECORDS" in txt and "corrupt:" in txt and "MISSING AUDIO (1)" in txt
