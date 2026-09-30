"""Synthesize every sentence in the evaluation set to <out-dir>/<id>.wav.

Loads the model once, runs one untimed warm-up (JIT compilation), then times
each sentence. Also writes <out-dir>/results.csv. Fully offline.

Usage:
    python src/generate_eval.py
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

import config
import tts_engine
from evaluation_set import load_sentences

WARMUP_TEXT = "Merhaba."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate WAVs for the evaluation set")
    p.add_argument("--out-dir", type=Path, default=config.BASE_OUTPUT_DIR)
    p.add_argument("--sentences", type=Path, default=config.EVAL_SENTENCES_PATH)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--steps", type=int, default=config.INFERENCE_STEPS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sentences = load_sentences(args.sentences)
    out_dir = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.SEED)
    device = tts_engine.select_device(args.device)
    tts = tts_engine.load_tts(device)
    tts.synthesize(WARMUP_TEXT, steps=args.steps)

    print(f"Model: {config.MODEL_NAME} | device: {device} | steps: {args.steps} | sentences: {len(sentences)}")
    rows = []
    total_audio = total_infer = 0.0
    for s in sentences:
        t0 = time.perf_counter()
        wav = tts.synthesize(s.text, steps=args.steps)
        infer_s = time.perf_counter() - t0

        wav_path = out_dir / f"{s.sentence_id}.wav"
        tts.save_wav(wav, str(wav_path))
        dur = len(wav) / config.SAMPLE_RATE
        total_audio += dur
        total_infer += infer_s
        rows.append({
            "sentence_id": s.sentence_id,
            "category": s.category,
            "text": s.text,
            "model_text": tts_engine.upstream_normalize(s.text),
            "audio": wav_path.name,
            "audio_duration_s": f"{dur:.2f}",
            "inference_time_s": f"{infer_s:.2f}",
            "rtf": f"{infer_s / dur:.2f}",
        })
        print(f"  {s.sentence_id} [{s.category:<14}] {dur:5.2f}s audio  {infer_s:6.2f}s  RTF {infer_s / dur:.2f}")

    # utf-8-sig so Excel shows Turkish characters correctly
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total: {total_audio:.1f}s audio in {total_infer:.1f}s (RTF {total_infer / total_audio:.2f})")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
