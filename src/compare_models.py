"""Base vs fine-tuned comparison on the unseen ERP evaluation set (FAZ 9).

Both models get the same sentences, the same text normalization and the same
inference parameters. Outputs:

    comparison/base/<id>.wav, comparison/fine_tuned/<id>.wav
    comparison/results.csv         timings + objective pitch-drift metric
    comparison/run_info.json       models, parameters, device, git commit
    evaluation/manual_evaluation.csv   template for listening scores (created
                                       once, never overwritten)

Usage:
    python src/compare_models.py --fine-tuned checkpoints/pilot_sft/best
    python src/compare_models.py                 # base only (before fine-tuning)
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import config
import text_normalizer
import tts_engine
import voice_check
from evaluation_set import EvalSentence, load_sentences

MODELS = ("base", "fine_tuned")
RESULT_FIELDS = [
    # columns required by the FAZ 9 spec
    "sentence_id", "text", "base_audio", "fine_tuned_audio", "base_inference_time", "fine_tuned_inference_time",
    # extra objective information
    "category", "tts_text", "base_duration_s", "fine_tuned_duration_s",
    "base_pitch_drop_st", "fine_tuned_pitch_drop_st",
]
MANUAL_FIELDS = ["sentence_id", "text", "base_naturalness", "fine_tuned_naturalness",
                 "base_pronunciation", "fine_tuned_pronunciation", "preferred_model", "notes"]
MANUAL_EVAL_PATH = config.EVALUATION_DIR / "manual_evaluation.csv"
WARMUP_TEXT = "Merhaba."


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def run_model(model_dir: Path, sentences: list[EvalSentence], texts: dict[str, str], out_dir: Path,
              device: str, steps: int, drift_guard: bool) -> dict[str, dict]:
    """Synthesize every sentence with one model; return per-sentence metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = tts_engine.load_tts(device, model_dir)
    tts_engine.synthesize(tts, WARMUP_TEXT, steps=steps, drift_guard=drift_guard)
    results = {}
    for s in sentences:
        t0 = time.perf_counter()
        wav, _ = tts_engine.synthesize(tts, texts[s.sentence_id], steps=steps, drift_guard=drift_guard)
        elapsed = time.perf_counter() - t0
        path = out_dir / f"{s.sentence_id}.wav"
        tts.save_wav(wav, str(path))
        drop = voice_check.pitch_drop(voice_check.window_f0_array(wav, tts.sample_rate))
        results[s.sentence_id] = {"audio": _rel(path), "time": elapsed,
                                  "duration": len(wav) / tts.sample_rate, "drop": drop}
        print(f"  {out_dir.name:<10} {s.sentence_id}  {elapsed:6.2f}s  drop {drop if drop is None else round(drop, 1)}",
              flush=True)
    del tts
    gc.collect()
    return results


def build_rows(sentences: list[EvalSentence], texts: dict[str, str], results: dict[str, dict]) -> list[dict]:
    rows = []
    for s in sentences:
        row = {"sentence_id": s.sentence_id, "text": s.text, "category": s.category,
               "tts_text": texts[s.sentence_id]}
        for m in MODELS:
            r = results.get(m, {}).get(s.sentence_id)
            row[f"{m}_audio"] = r["audio"] if r else ""
            row[f"{m}_inference_time"] = f"{r['time']:.2f}" if r else ""
            row[f"{m}_duration_s"] = f"{r['duration']:.2f}" if r else ""
            row[f"{m}_pitch_drop_st"] = "" if not r or r["drop"] is None else f"{r['drop']:.1f}"
        rows.append(row)
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Turkish text in Excel
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ensure_manual_template(sentences: list[EvalSentence], path: Path = MANUAL_EVAL_PATH) -> bool:
    """Create the listening-test sheet once. Returns False if it already exists (left untouched)."""
    if path.exists():
        return False
    write_csv(path, MANUAL_FIELDS, [{"sentence_id": s.sentence_id, "text": s.text} for s in sentences])
    return True


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Compare base and fine-tuned models on the evaluation set")
    p.add_argument("--base", type=Path, default=config.FREYA_MODEL_DIR)
    p.add_argument("--fine-tuned", type=Path, help="fine-tuned checkpoint dir (e.g. checkpoints/pilot_sft/best)")
    p.add_argument("--out", type=Path, default=config.PROJECT_ROOT / "comparison")
    p.add_argument("--sentences", type=Path, default=config.EVAL_SENTENCES_PATH)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--steps", type=int, default=config.INFERENCE_STEPS)
    p.add_argument("--drift-guard", action=argparse.BooleanOptionalAction, default=config.DRIFT_GUARD)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    torch.manual_seed(config.SEED)
    device = tts_engine.select_device(args.device)
    sentences = load_sentences(args.sentences)
    with open(config.FREYATTS_VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)
    # One prepared text per sentence, shared by both models.
    texts = {s.sentence_id: tts_engine.fit_to_vocab(text_normalizer.normalize(s.text), vocab)
             for s in sentences}

    model_dirs = {"base": args.base}
    if args.fine_tuned:
        model_dirs["fine_tuned"] = args.fine_tuned
    print(f"Device {device} | steps {args.steps} | drift guard {args.drift_guard} | "
          f"{len(sentences)} sentences | models: {', '.join(model_dirs)}")

    results = {name: run_model(d, sentences, texts, args.out / name, device, args.steps, args.drift_guard)
               for name, d in model_dirs.items()}
    rows = build_rows(sentences, texts, results)
    write_csv(args.out / "results.csv", RESULT_FIELDS, rows)
    with open(args.out / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "models": {k: _rel(v) for k, v in model_dirs.items()}, "device": device,
                   "steps": args.steps, "drift_guard": args.drift_guard, "seed": config.SEED,
                   "sentences": _rel(args.sentences), "git_commit": _git_commit()}, f, indent=2)
    created = ensure_manual_template(sentences)

    print()
    for name, res in results.items():
        times = [r["time"] for r in res.values()]
        audio = sum(r["duration"] for r in res.values())
        drifted = sum(1 for r in res.values() if r["drop"] is not None and r["drop"] >= voice_check.DRIFT_SEMITONES)
        print(f"{name:<10}: mean {sum(times) / len(times):.2f}s/sentence, RTF {sum(times) / audio:.2f}, "
              f"pitch drift >= {voice_check.DRIFT_SEMITONES:g} st in {drifted}/{len(res)}")
    print(f"Results   : {_rel(args.out / 'results.csv')}")
    print(f"Manual    : {_rel(MANUAL_EVAL_PATH)} " + ("(created)" if created else "(exists, left untouched)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
