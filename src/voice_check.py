"""Detect speaker drift (e.g. female voice turning male) via pitch tracking.

FreyaTTS has no speaker conditioning: the voice comes from the noise seed, so
it can drift within an utterance. This measures median F0 per time window and
flags windows whose pitch drops well below the file's own opening pitch.

Usage:
    python src/voice_check.py outputs/base
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np

ANALYSIS_SR = 16000
WINDOW_S = 1.0
# A drop of this many semitones from the opening pitch is flagged as drift.
# Typical female->male difference is ~12 semitones (one octave).
DRIFT_SEMITONES = 5.0


def window_f0_array(y: np.ndarray, sr: int, window_s: float = WINDOW_S) -> list[float]:
    """Median voiced F0 per window (NaN where a window has too little voicing)."""
    if sr != ANALYSIS_SR:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=ANALYSIS_SR)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=450, sr=ANALYSIS_SR)
    hop_s = 512 / ANALYSIS_SR  # librosa.pyin default hop_length
    per_win = int(round(window_s / hop_s))
    out = []
    for i in range(0, len(f0), per_win):
        seg = f0[i:i + per_win][voiced[i:i + per_win]]
        out.append(float(np.median(seg)) if len(seg) >= 5 else float("nan"))
    return out


def pitch_drop(windows: list[float]) -> float | None:
    """Largest drop (semitones) below the opening pitch; None if nothing is voiced."""
    valid = [w for w in windows if not np.isnan(w)]
    if not valid:
        return None
    ref = float(np.median(valid[:2]))  # opening pitch
    return max(12 * np.log2(ref / w) if not np.isnan(w) else 0.0 for w in windows)


def take_floor(windows: list[float], min_voiced_ratio: float, min_opening_hz: float) -> float:
    """Lowest window pitch (Hz) of a take, or 0 if it is mostly unvoiced or not the target voice."""
    valid = [w for w in windows if not np.isnan(w)]
    if not windows or len(valid) / len(windows) < min_voiced_ratio:
        return 0.0
    if float(np.median(valid[:2])) < min_opening_hz:
        return 0.0
    return float(min(valid))


def window_f0(path: Path) -> list[float]:
    y, sr = librosa.load(path, sr=None, mono=True)
    return window_f0_array(y, sr)


def analyze(path: Path) -> dict:
    wins = window_f0(path)
    drop = pitch_drop(wins)
    if drop is None:
        return {"file": path.name, "windows": wins, "ref": float("nan"), "drift": False}
    valid = [w for w in wins if not np.isnan(w)]
    return {"file": path.name, "windows": wins, "ref": float(np.median(valid[:2])),
            "max_drop": drop, "drift": drop >= DRIFT_SEMITONES}


def main() -> int:
    p = argparse.ArgumentParser(description="Flag speaker/pitch drift in WAV files")
    p.add_argument("path", type=Path, help="WAV file or directory")
    args = p.parse_args()
    files = sorted(args.path.glob("*.wav")) if args.path.is_dir() else [args.path]

    flagged = 0
    for f in files:
        r = analyze(f)
        f0s = " ".join("  - " if np.isnan(w) else f"{w:4.0f}" for w in r["windows"])
        mark = "DRIFT" if r["drift"] else "ok"
        flagged += r["drift"]
        print(f"{r['file']:<10} {mark:<5} drop={r.get('max_drop', 0):4.1f}st  F0/s: {f0s}")
    print(f"\n{flagged}/{len(files)} files flagged (drop >= {DRIFT_SEMITONES} semitones from opening pitch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
