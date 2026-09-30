"""Turkish text -> WAV with the pretrained base model.

Runs fully offline: model files are read from models/ (see download_models.py)
and Hugging Face Hub access is disabled, so input text never leaves the machine.

Usage:
    python src/inference.py --text "Yeni öğrenci kaydı oluşturabilirsiniz."
"""

import os

# Hard-disable any Hub network access before huggingface_hub can be imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import sys
import time
from pathlib import Path

import torch

import config
import tts_engine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Turkish TTS PoC inference (base model)")
    p.add_argument("--text", required=True, help="Turkish text to synthesize")
    p.add_argument("--out", type=Path, default=config.DEFAULT_OUTPUT_PATH, help="output WAV path")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--steps", type=int, default=config.INFERENCE_STEPS, help="Euler ODE steps")
    p.add_argument("--no-normalize", action="store_true", help="skip Turkish text normalization")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    text = args.text.strip()
    if not text:
        print("error: --text is empty", file=sys.stderr)
        return 2

    torch.manual_seed(config.SEED)
    device = tts_engine.select_device(args.device)

    t0 = time.perf_counter()
    tts = tts_engine.load_tts(device)
    load_s = time.perf_counter() - t0

    tts_text = tts_engine.prepare_text(tts, text, normalize=not args.no_normalize)
    t0 = time.perf_counter()
    wav = tts.synthesize(tts_text, steps=args.steps)
    infer_s = time.perf_counter() - t0

    out = args.out if args.out.is_absolute() else Path.cwd() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tts.save_wav(wav, str(out))

    duration_s = len(wav) / config.SAMPLE_RATE
    try:
        shown = out.relative_to(config.PROJECT_ROOT)
    except ValueError:
        shown = out

    print(f"Model           : {config.MODEL_NAME}")
    print(f"Device          : {device}")
    print(f"Input text      : {text}")
    print(f"TTS text        : {tts_text}")
    print(f"Audio output    : {shown}")
    print(f"Audio duration  : {duration_s:.2f} s ({config.SAMPLE_RATE} Hz)")
    print(f"Model load time : {load_s:.2f} s")
    print(f"Inference time  : {infer_s:.2f} s (RTF {infer_s / duration_s:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
