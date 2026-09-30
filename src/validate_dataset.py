"""Validate a single-speaker TTS dataset before training. Read-only.

Dataset layout (see docs/recording_guide.md):
    <dataset-dir>/wavs/<id>.wav
Transcripts come from a TSV with columns `id` and `train_text`
(default: recording/pilot_script.tsv).

Each record gets errors (excluded from training) and warnings (kept, but
reported). Nothing in the dataset is modified, moved or deleted.

Outputs:
    reports/dataset_report.txt   human-readable
    reports/dataset_report.json  machine-readable (per-record details)

Usage:
    python src/validate_dataset.py
    python src/validate_dataset.py --dataset-dir datasets/pilot --script recording/pilot_script.tsv
"""

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

import config
import tts_engine
from evaluation_set import EvalLeakChecker


@dataclass(frozen=True)
class Thresholds:
    expected_sample_rate: int = 48000
    min_sample_rate: int = 22050       # below this the 48 kHz model output would suffer
    # Keep in sync with the --min_s/--max_s passed to FreyaTTS precompute_latents.py
    # (its defaults are 1.0/14.0; 0.5 keeps short prompts like "Tamam.").
    min_duration_s: float = 0.5
    max_duration_s: float = 14.0
    clip_level: float = 0.999          # |sample| at or above this counts as clipped
    max_clipped_fraction: float = 0.001
    min_peak_dbfs: float = -20.0       # quieter than this: recording level too low
    silence_rel_db: float = -40.0      # frame is silence if this far below the loudest frame
    no_speech_dbfs: float = -50.0      # loudest frame below this: file has no speech
    max_edge_silence_s: float = 1.0
    min_edge_silence_s: float = 0.02   # speech starting/ending immediately: likely cut
    min_snr_db: float = 30.0
    min_chars_per_s: float = 5.0       # speaking-rate bounds used to spot text/audio mismatch
    max_chars_per_s: float = 25.0


FRAME_S = 0.02
MOJIBAKE_MARKERS = ("Ã", "Ä", "Å", "Â±", "�", "﻿")


@dataclass
class Record:
    id: str
    audio: str
    text: str
    duration_s: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    subtype: str | None = None
    peak_dbfs: float | None = None
    snr_db: float | None = None
    leading_silence_s: float | None = None
    trailing_silence_s: float | None = None
    chars_per_s: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_transcripts(script_path: Path) -> dict[str, str]:
    with open(script_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows or "id" not in rows[0] or "train_text" not in rows[0]:
        raise ValueError(f"{script_path}: expected TSV columns 'id' and 'train_text'")
    return {r["id"].strip(): (r["train_text"] or "").strip() for r in rows}


def _db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-10))


def analyze_audio(path: Path, rec: Record, th: Thresholds) -> str | None:
    """Fill audio metrics into `rec`; return a hash of the PCM data (for duplicates)."""
    try:
        info = sf.info(str(path))
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:  # unreadable / corrupt file
        rec.errors.append(f"unreadable audio: {type(e).__name__}: {e}")
        return None

    rec.sample_rate, rec.channels, rec.subtype = sr, info.channels, info.subtype
    rec.duration_s = round(len(audio) / sr, 3) if sr else 0.0
    if len(audio) == 0:
        rec.errors.append("empty audio (0 samples)")
        return None

    if sr < th.min_sample_rate:
        rec.errors.append(f"low sample rate: {sr} Hz < {th.min_sample_rate} Hz")
    elif sr != th.expected_sample_rate:
        rec.warnings.append(f"sample rate: {sr} Hz (expected {th.expected_sample_rate})")
    if info.channels != 1:
        rec.warnings.append(f"channels: {info.channels} (expected mono; will be downmixed)")
    if info.subtype not in ("PCM_24", "PCM_32", "FLOAT", "DOUBLE"):
        rec.warnings.append(f"bit depth: {info.subtype} (24-bit recommended)")

    if rec.duration_s < th.min_duration_s:
        rec.errors.append(f"too short: {rec.duration_s:.2f}s < {th.min_duration_s}s")
    elif rec.duration_s > th.max_duration_s:
        rec.errors.append(f"too long: {rec.duration_s:.2f}s > {th.max_duration_s}s")

    clipped = float(np.mean(np.abs(audio) >= th.clip_level))
    if clipped > th.max_clipped_fraction:
        rec.errors.append(f"clipping: {clipped:.2%} of samples at full scale")

    mono = audio.mean(axis=1)
    peak = float(np.max(np.abs(mono)))
    rec.peak_dbfs = round(_db(peak), 1)

    hop = max(1, int(FRAME_S * sr))
    n = len(mono) // hop
    if n == 0:
        rec.errors.append("too short: shorter than one analysis frame")
        return hashlib.sha256(audio.tobytes()).hexdigest()
    frame_db = np.array([_db(float(np.sqrt(np.mean(mono[i * hop:(i + 1) * hop] ** 2)))) for i in range(n)])
    loudest = float(frame_db.max())

    if loudest < th.no_speech_dbfs:
        rec.errors.append(f"no speech detected: loudest frame {loudest:.1f} dBFS")
    else:
        if rec.peak_dbfs < th.min_peak_dbfs:
            rec.warnings.append(f"low level: peak {rec.peak_dbfs} dBFS")
        voiced = np.where(frame_db > loudest + th.silence_rel_db)[0]
        rec.leading_silence_s = round(voiced[0] * FRAME_S, 2)
        rec.trailing_silence_s = round((n - 1 - voiced[-1]) * FRAME_S, 2)
        for name, val in (("leading", rec.leading_silence_s), ("trailing", rec.trailing_silence_s)):
            if val > th.max_edge_silence_s:
                rec.warnings.append(f"long {name} silence: {val:.2f}s")
            elif val < th.min_edge_silence_s:
                rec.warnings.append(f"no {name} silence: speech may be cut off")
        # Noise floor from the leading/trailing silence only: pauses inside speech
        # are not silent enough. Without enough edge silence, SNR is not measured.
        edge = np.concatenate([frame_db[:voiced[0]], frame_db[voiced[-1] + 1:]])
        speech_db = float(np.percentile(frame_db[voiced[0]:voiced[-1] + 1], 90))
        if len(edge) >= 5:
            rec.snr_db = round(speech_db - float(np.median(edge)), 1)
            if rec.snr_db < th.min_snr_db:
                rec.warnings.append(f"low estimated SNR: {rec.snr_db} dB")
        else:
            # No silent edges: either tightly trimmed or noise masks the silence.
            # The quietest frames give an upper bound on the noise floor.
            rough = round(speech_db - float(np.percentile(frame_db, 10)), 1)
            if rough < th.min_snr_db:
                rec.warnings.append(f"low estimated SNR: ~{rough} dB (no silent edges; rough estimate)")

        speech_s = (voiced[-1] - voiced[0] + 1) * FRAME_S
        letters = sum(ch.isalpha() for ch in rec.text)
        if rec.text and speech_s > 0:
            rec.chars_per_s = round(letters / speech_s, 1)
            if not th.min_chars_per_s <= rec.chars_per_s <= th.max_chars_per_s:
                rec.warnings.append(
                    f"speaking rate: {rec.chars_per_s} letters/s outside "
                    f"{th.min_chars_per_s}-{th.max_chars_per_s}: transcript may not match audio")

    return hashlib.sha256(audio.tobytes()).hexdigest()


def check_text(rec: Record, vocab: dict, leak_checker: EvalLeakChecker) -> None:
    if not rec.text:
        rec.errors.append("empty transcript")
        return
    bad = [m for m in MOJIBAKE_MARKERS if m in rec.text]
    if bad:
        rec.errors.append(f"encoding problem in transcript: mojibake markers {bad}")
    unmappable = tts_engine.unmappable_chars(rec.text, vocab)
    if unmappable:
        rec.errors.append(f"unmappable characters: {sorted(unmappable)}")
    leak = leak_checker.leak(rec.text)
    if leak:
        rec.errors.append(f"evaluation-set leakage: {leak}")


def validate(dataset_dir: Path, script_path: Path, th: Thresholds = Thresholds()) -> dict:
    with open(config.FREYATTS_VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)
    leak_checker = EvalLeakChecker()
    transcripts = load_transcripts(script_path)
    wav_dir = dataset_dir / "wavs"
    wavs = {p.stem: p for p in sorted(wav_dir.glob("*.wav"))} if wav_dir.is_dir() else {}

    records: list[Record] = []
    audio_hashes: dict[str, str] = {}
    text_seen: dict[str, str] = {}
    # Script order first, so the earlier take is kept and later copies are flagged;
    # audio files with no script entry come last.
    order = [rid for rid in transcripts if rid in wavs] + sorted(set(wavs) - set(transcripts))
    for rid in order:
        rec = Record(id=rid, audio=str(wavs[rid].relative_to(dataset_dir)), text=transcripts.get(rid, ""))
        if rid not in transcripts:
            rec.errors.append("no transcript for this audio file (id not in script)")
        else:
            check_text(rec, vocab, leak_checker)
        digest = analyze_audio(wavs[rid], rec, th)
        if digest:
            if digest in audio_hashes:
                rec.errors.append(f"duplicate audio: same samples as {audio_hashes[digest]}")
            else:
                audio_hashes[digest] = rid
        key = rec.text.lower()
        if key and rid in transcripts:
            if key in text_seen:
                rec.warnings.append(f"same transcript: as {text_seen[key]}")
            else:
                text_seen[key] = rid
        records.append(rec)

    missing_audio = sorted(set(transcripts) - set(wavs))
    accepted = [r for r in records if not r.errors]
    rejected = [r for r in records if r.errors]
    durations = [r.duration_s for r in records if r.duration_s]
    acc_durations = [r.duration_s for r in accepted]
    letters = Counter(ch for r in accepted for ch in r.text.lower() if ch.isalpha())

    def stats(xs):
        if not xs:
            return {"count": 0}
        return {"count": len(xs), "total_min": round(sum(xs) / 60, 2), "mean_s": round(statistics.mean(xs), 2),
                "median_s": round(statistics.median(xs), 2), "min_s": round(min(xs), 2), "max_s": round(max(xs), 2)}

    summary = {
        "script_entries": len(transcripts),
        "audio_files": len(wavs),
        "missing_audio": len(missing_audio),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "with_warnings": sum(1 for r in accepted if r.warnings),
        "all_audio": stats(durations),
        "accepted_audio": stats(acc_durations),
        "sample_rates": dict(Counter(str(r.sample_rate) for r in records if r.sample_rate)),
        "channels": dict(Counter(str(r.channels) for r in records if r.channels)),
        "subtypes": dict(Counter(r.subtype for r in records if r.subtype)),
        "error_types": dict(Counter(e.split(":")[0] for r in rejected for e in r.errors)),
        "warning_types": dict(Counter(w.split(":")[0] for r in records for w in r.warnings)),
        "turkish_letter_counts": {ch: letters.get(ch, 0) for ch in "çğıöşü"},
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "script": str(script_path),
        "thresholds": asdict(th),
        "summary": summary,
        "missing_audio": missing_audio,
        "rejected_ids": [r.id for r in rejected],
        "records": [asdict(r) for r in records],
    }


def format_text_report(report: dict) -> str:
    s = report["summary"]
    lines = [
        "DATASET VALIDATION REPORT",
        "=" * 60,
        f"Generated : {report['generated_at']}",
        f"Dataset   : {report['dataset_dir']}",
        f"Script    : {report['script']}",
        "",
        "SUMMARY",
        f"  Script entries        : {s['script_entries']}",
        f"  Audio files           : {s['audio_files']}",
        f"  Missing audio         : {s['missing_audio']}",
        f"  Accepted for training : {s['accepted']}  ({s['with_warnings']} with warnings)",
        f"  Rejected              : {s['rejected']}",
    ]
    for label, key in (("All audio", "all_audio"), ("Accepted audio", "accepted_audio")):
        st = s[key]
        if st["count"]:
            lines.append(f"  {label:<22}: {st['total_min']} min, mean {st['mean_s']}s, median {st['median_s']}s, "
                         f"min {st['min_s']}s, max {st['max_s']}s")
    lines += [
        f"  Sample rates          : {s['sample_rates']}",
        f"  Channels              : {s['channels']}",
        f"  Bit depth             : {s['subtypes']}",
        f"  Turkish letters       : {s['turkish_letter_counts']}",
        "",
        "ERROR TYPES (records excluded from training)",
    ]
    lines += [f"  {k}: {v}" for k, v in sorted(s["error_types"].items())] or ["  none"]
    lines += ["", "WARNING TYPES (records kept)"]
    lines += [f"  {k}: {v}" for k, v in sorted(s["warning_types"].items())] or ["  none"]
    lines += ["", "REJECTED RECORDS"]
    rejected = [r for r in report["records"] if r["errors"]]
    lines += [f"  {r['id']}: " + "; ".join(r["errors"]) for r in rejected] or ["  none"]
    lines += ["", "RECORDS WITH WARNINGS"]
    warned = [r for r in report["records"] if r["warnings"] and not r["errors"]]
    lines += [f"  {r['id']}: " + "; ".join(r["warnings"]) for r in warned] or ["  none"]
    lines += ["", f"MISSING AUDIO ({len(report['missing_audio'])})"]
    lines += ["  " + ", ".join(report["missing_audio"])] if report["missing_audio"] else ["  none"]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Validate a TTS dataset (read-only)")
    p.add_argument("--dataset-dir", type=Path, default=config.PILOT_DATASET_DIR)
    p.add_argument("--script", type=Path, default=config.PILOT_SCRIPT_PATH)
    p.add_argument("--report-dir", type=Path, default=config.REPORTS_DIR)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Turkish letters on Windows consoles

    if not (args.dataset_dir / "wavs").is_dir():
        print(f"error: {args.dataset_dir / 'wavs'} not found (see docs/recording_guide.md)", file=sys.stderr)
        return 2

    report = validate(args.dataset_dir, args.script)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    txt = format_text_report(report)
    (args.report_dir / "dataset_report.txt").write_text(txt, encoding="utf-8")
    with open(args.report_dir / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(txt)
    print(f"Reports written to {args.report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
