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


def window_f0(path: Path) -> list[float]:
    y, _ = librosa.load(path, sr=ANALYSIS_SR, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=450, sr=ANALYSIS_SR)
    hop_s = 512 / ANALYSIS_SR  # librosa.pyin default hop_length
    per_win = int(round(WINDOW_S / hop_s))
    out = []
    for i in range(0, len(f0), per_win):
        seg = f0[i:i + per_win][voiced[i:i + per_win]]
        out.append(float(np.median(seg)) if len(seg) >= 5 else float("nan"))
    return out


def analyze(path: Path) -> dict:
    wins = window_f0(path)
    valid = [w for w in wins if not np.isnan(w)]
    if not valid:
        return {"file": path.name, "windows": wins, "ref": float("nan"), "drift": False}
    ref = float(np.median(valid[:2]))  # opening pitch
    drops = [12 * np.log2(ref / w) if not np.isnan(w) else 0.0 for w in wins]
    return {"file": path.name, "windows": wins, "ref": ref,
            "max_drop": max(drops), "drift": max(drops) >= DRIFT_SEMITONES}


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
