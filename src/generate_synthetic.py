"""Generate a synthetic single-voice training set from FreyaTTS-small's own output.

Voice-lock fine-tuning data (decision 2026-10-01: no human recording, nothing
paid, no extra model). FreyaTTS-small has no speaker conditioning and drifts
toward a lower voice at clause ends; we keep only takes that hold the voice.

For every clause of every script sentence, the clause is synthesized with the
pipeline's own seed (= the base voice) at duration scales SCALES in order. The
first take that passes every check is kept; if none passes, the clause is
rejected. Rejected audio is not stored. Each decision is appended to
generation_log.csv immediately, so an interrupted run resumes where it stopped.

A finalize pass (run after every generation run) drops clips whose median
pitch is far from the voice's overall median (speaker consistency), moving
them to excluded/, and writes script.tsv for validate_dataset.py / split_dataset.py.

Layout (<out>, default datasets/synthetic):
    wavs/<clip_id>.wav          accepted clips, 48 kHz 24-bit mono
    excluded/<clip_id>.wav      passed generation checks, failed consistency
    generation_log.csv          one row per clip
    generation_config.json      thresholds/model/seed; a resume must match it
    script.tsv                  id, category, source, source_id, train_text
    generation_summary.json

Usage:
    python src/generate_synthetic.py --trial 10   # 10 sentences from each source script
    python src/generate_synthetic.py              # all sentences (resumes)
"""

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

import config
import tts_engine
import voice_check
from evaluation_set import EvalLeakChecker
from validate_dataset import Thresholds as ValidationThresholds

_VT = ValidationThresholds()

SOURCES = {"pilot": config.PILOT_SCRIPT_PATH, "synthetic": config.SYNTHETIC_SCRIPT_PATH}

LOG_FIELDS = ["clip_id", "source", "source_id", "category", "text", "status", "scale", "attempts",
              "duration_s", "opening_hz", "median_f0_hz", "min_f0_hz", "voiced_ratio", "max_drop_st",
              "letters_per_s", "clipped_fraction", "reasons", "compute_s"]


@dataclass(frozen=True)
class GenThresholds:
    scales: tuple = config.DRIFT_GUARD_SCALES
    window_s: float = config.DRIFT_WINDOW_S
    min_opening_hz: float = config.DRIFT_MIN_OPENING_HZ
    min_voiced_ratio: float = config.DRIFT_MIN_VOICED_RATIO
    min_pitch_hz: float = config.DRIFT_MIN_PITCH_HZ
    consistency_st: float = 2.0        # clip median F0 vs the voice's overall median
    min_duration_s: float = _VT.min_duration_s
    max_duration_s: float = _VT.max_duration_s
    min_letters_per_s: float = _VT.min_chars_per_s  # no ASR available: rate is the
    max_letters_per_s: float = _VT.max_chars_per_s  # only text/audio mismatch signal
    clip_level: float = _VT.clip_level
    max_clipped_fraction: float = _VT.max_clipped_fraction


def speech_span_s(wav: np.ndarray, sr: int, silence_rel_db: float = _VT.silence_rel_db) -> float:
    """Length of the region between first and last non-silent 20 ms frame (as in validate_dataset)."""
    hop = max(1, int(0.02 * sr))
    n = len(wav) // hop
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(wav[:n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    voiced = np.where(db > db.max() + silence_rel_db)[0]
    return float((voiced[-1] - voiced[0] + 1) * 0.02) if len(voiced) else 0.0


def assess(wav: np.ndarray, sr: int, text: str, th: GenThresholds) -> dict:
    """Metrics of one take plus the failed checks as 'code: detail' (empty list = accepted)."""
    windows = voice_check.window_f0_array(wav, sr, th.window_s)
    valid = [w for w in windows if not np.isnan(w)]
    duration = len(wav) / sr
    span = speech_span_s(wav, sr)
    letters = sum(ch.isalpha() for ch in text)
    m = {
        "duration_s": round(duration, 3),
        "voiced_ratio": round(len(valid) / len(windows), 3) if windows else 0.0,
        "opening_hz": round(float(np.median(valid[:2])), 1) if valid else None,
        "median_f0_hz": round(float(np.median(valid)), 1) if valid else None,
        "min_f0_hz": round(float(min(valid)), 1) if valid else None,
        "max_drop_st": round(voice_check.pitch_drop(windows), 2) if valid else None,
        "letters_per_s": round(letters / span, 1) if span > 0 else None,
        "clipped_fraction": round(float(np.mean(np.abs(wav) >= th.clip_level)), 5) if len(wav) else 0.0,
    }
    reasons = []
    if not valid or m["voiced_ratio"] < th.min_voiced_ratio:
        reasons.append(f"unvoiced: voiced ratio {m['voiced_ratio']} < {th.min_voiced_ratio}")
    if m["opening_hz"] is not None and m["opening_hz"] < th.min_opening_hz:
        reasons.append(f"low_opening: {m['opening_hz']} Hz < {th.min_opening_hz}")
    if m["min_f0_hz"] is not None and m["min_f0_hz"] < th.min_pitch_hz:
        reasons.append(f"low_pitch: {m['min_f0_hz']} Hz < {th.min_pitch_hz}")
    if not th.min_duration_s <= duration <= th.max_duration_s:
        reasons.append(f"duration: {duration:.2f}s outside {th.min_duration_s}-{th.max_duration_s}")
    if m["letters_per_s"] is None or not th.min_letters_per_s <= m["letters_per_s"] <= th.max_letters_per_s:
        reasons.append(f"speaking_rate: {m['letters_per_s']} letters/s outside "
                       f"{th.min_letters_per_s}-{th.max_letters_per_s}")
    if m["clipped_fraction"] > th.max_clipped_fraction:
        reasons.append(f"clipped: {m['clipped_fraction']}")
    m["reasons"] = reasons
    return m


def _badness(m: dict) -> tuple:
    """Order rejected takes: fewest failed checks, then highest lowest-window pitch."""
    low = m["min_f0_hz"] if m.get("min_f0_hz") is not None else 0.0
    return (len(m["reasons"]), -low)


def generate_clip(tts, text: str, steps: int, th: GenThresholds) -> tuple[np.ndarray | None, dict]:
    """Try each duration scale until a take passes. Returns (wav or None, log fields)."""
    best = None
    for attempt, scale in enumerate(th.scales, 1):
        wav = tts_engine._synth_chunk(tts, text, steps, scale)
        m = assess(wav, tts.sample_rate, text, th)
        m.update(scale=scale, attempts=attempt)
        if not m["reasons"]:
            return wav, {**m, "status": "accepted"}
        if best is None or _badness(m) < _badness(best):
            best = m
    return None, {**best, "attempts": len(th.scales), "status": "rejected"}


def clip_texts(tts, train_text: str) -> list[str]:
    """Model-input clauses of a sentence, split exactly as the pipeline would."""
    t = tts_engine.fit_to_vocab(tts_engine.upstream_normalize(train_text), tts.char_to_id)
    return tts._clauses(t) if len(t.split()) > tts.max_words else [t]


def clip_ids(source_id: str, n: int) -> list[str]:
    return [source_id] if n == 1 else [f"{source_id}_{k}" for k in range(1, n + 1)]


def load_script(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def select_trial(rows: list[dict], n: int, seed: int) -> list[dict]:
    """First n rows by a seeded hash of the id: deterministic, spread over categories."""
    key = lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()  # noqa: E731
    return sorted(rows, key=key)[:n]


def read_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _check_config(out: Path, run_cfg: dict) -> None:
    path = out / "generation_config.json"
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != run_cfg:
            raise SystemExit(f"{path} differs from the current settings; existing clips were judged "
                             "with other thresholds. Use a new --out directory or remove the old one.")
    else:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def finalize(out: Path, th: GenThresholds) -> dict:
    """Speaker-consistency filter, script.tsv and summary from the generation log."""
    log = read_log(out / "generation_log.csv")
    accepted = [r for r in log if r["status"] == "accepted"]
    wavs, excluded_dir = out / "wavs", out / "excluded"
    voice_median = statistics.median(float(r["median_f0_hz"]) for r in accepted) if accepted else None

    kept, excluded = [], []
    for r in accepted:
        dev = 12 * np.log2(float(r["median_f0_hz"]) / voice_median)
        keep = abs(dev) <= th.consistency_st
        src, dst = (excluded_dir, wavs) if keep else (wavs, excluded_dir)
        f = f"{r['clip_id']}.wav"
        if (src / f).is_file():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src / f), str(dst / f))
        (kept if keep else excluded).append({**r, "f0_dev_st": round(float(dev), 2)})

    with open(out / "script.tsv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "category", "source", "source_id", "train_text"], delimiter="\t")
        w.writeheader()
        for r in kept:
            w.writerow({"id": r["clip_id"], "category": r["category"], "source": r["source"],
                        "source_id": r["source_id"], "train_text": r["text"]})

    def stats(xs):
        xs = [float(x) for x in xs if x not in ("", None)]
        return {"n": len(xs), "median": round(statistics.median(xs), 2), "max": round(max(xs), 2)} if xs else {"n": 0}

    per_source = {}
    for src in sorted({r["source"] for r in log}):
        rows = [r for r in log if r["source"] == src]
        k = [r for r in kept if r["source"] == src]
        per_source[src] = {"clips": len(rows), "kept": len(k), "kept_ratio": round(len(k) / len(rows), 3),
                           "kept_minutes": round(sum(float(r["duration_s"]) for r in k) / 60, 2)}
    reasons = Counter()
    for r in log:
        if r["status"] == "rejected":
            reasons.update(x.split(":")[0] for x in r["reasons"].split("; ") if x)
    audio_s = sum(float(r["duration_s"]) for r in kept)
    compute_s = sum(float(r["compute_s"]) for r in log)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clips_total": len(log),
        "accepted_by_generation": len(accepted),
        "excluded_by_consistency": [r["clip_id"] for r in excluded],
        "kept": len(kept),
        "kept_ratio": round(len(kept) / len(log), 3) if log else 0.0,
        "kept_minutes": round(audio_s / 60, 2),
        "voice_median_f0_hz": round(voice_median, 1) if voice_median else None,
        "per_source": per_source,
        "per_category_kept": dict(Counter(r["category"] for r in kept)),
        "scale_of_kept": {str(k): v for k, v in sorted(Counter(r["scale"] for r in kept).items())},
        "attempts_of_kept": {str(k): v for k, v in sorted(Counter(r["attempts"] for r in kept).items())},
        "reject_reasons": dict(reasons.most_common()),
        "min_f0_hz_kept": stats(r["min_f0_hz"] for r in kept),
        "min_f0_hz_rejected_best_take": stats(r["min_f0_hz"] for r in log if r["status"] == "rejected"),
        "max_drop_st_kept": stats(r["max_drop_st"] for r in kept),
        "compute_hours": round(compute_s / 3600, 3),
        "compute_s_per_kept_audio_s": round(compute_s / audio_s, 2) if audio_s else None,
    }
    (out / "generation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Generate filtered synthetic single-voice training data")
    p.add_argument("--out", type=Path, default=config.SYNTHETIC_DATASET_DIR)
    p.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--trial", type=int, default=0, help="only N sentences per source (seeded selection)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()

    th = GenThresholds()
    out = args.out if args.out.is_absolute() else config.PROJECT_ROOT / args.out
    if args.finalize_only:
        print(json.dumps(finalize(out, th), indent=2, ensure_ascii=False))
        return 0

    device = tts_engine.select_device(args.device)
    tts = tts_engine.load_tts(device)
    run_cfg = {"thresholds": asdict(th), "steps": config.INFERENCE_STEPS, "voice_seed": tts.seed,
               "max_words": tts.max_words, "model": config.MODEL_NAME, "model_revision": config.FREYA_REVISION}
    run_cfg = json.loads(json.dumps(run_cfg))  # tuples -> lists, as read back from disk
    _check_config(out, run_cfg)

    leak_checker = EvalLeakChecker()
    log_path = out / "generation_log.csv"
    done = {r["clip_id"] for r in read_log(log_path)}
    (out / "wavs").mkdir(parents=True, exist_ok=True)

    todo = []
    for src in args.sources:
        rows = load_script(SOURCES[src])
        if args.trial:
            rows = select_trial(rows, args.trial, config.SEED)
        for r in rows:
            texts = clip_texts(tts, r["train_text"])
            for cid, text in zip(clip_ids(r["id"], len(texts)), texts):
                if leak_checker.leak(text):
                    raise SystemExit(f"{cid}: text leaks the evaluation set: {text!r}")
                if cid not in done:
                    todo.append((src, r, cid, text))
    print(f"device={device} voice_seed={tts.seed} clips to generate={len(todo)} (already done: {len(done)})")

    new_log = not log_path.is_file()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_log:
            writer.writeheader()
        t_run = time.perf_counter()
        for i, (src, r, cid, text) in enumerate(todo, 1):
            t0 = time.perf_counter()
            wav, m = generate_clip(tts, text, config.INFERENCE_STEPS, th)
            if wav is not None:
                sf.write(out / "wavs" / f"{cid}.wav", wav.astype(np.float32), tts.sample_rate, subtype="PCM_24")
            dt = time.perf_counter() - t0
            writer.writerow({"clip_id": cid, "source": src, "source_id": r["id"], "category": r["category"],
                             "text": text, "status": m["status"], "scale": m["scale"], "attempts": m["attempts"],
                             "duration_s": m["duration_s"], "opening_hz": m["opening_hz"],
                             "median_f0_hz": m["median_f0_hz"], "voiced_ratio": m["voiced_ratio"],
                             "min_f0_hz": m["min_f0_hz"], "max_drop_st": m["max_drop_st"], "letters_per_s": m["letters_per_s"],
                             "clipped_fraction": m["clipped_fraction"], "reasons": "; ".join(m["reasons"]),
                             "compute_s": round(dt, 2)})
            f.flush()
            eta = (time.perf_counter() - t_run) / i * (len(todo) - i) / 60
            print(f"[{i}/{len(todo)}] {cid:<8} {m['status']:<8} scale={m['scale']} tries={m['attempts']} "
                  f"minF0={m['min_f0_hz']} {dt:5.1f}s  ETA {eta:5.1f} min", flush=True)

    s = finalize(out, th)
    print(json.dumps(s, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
