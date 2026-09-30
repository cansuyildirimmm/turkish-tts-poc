"""Encode split manifests into AudioVAE latents for fine-tuning. Offline.

Equivalent to FreyaTTS training/precompute_latents.py, but loads the AudioVAE
from models/ (no Hugging Face download), keeps record ids, uses our clip-length
bounds (0.5-14 s, same as validate_dataset.py) and processes all splits.

Input : <dataset-dir>/splits/{train,val,test}.jsonl   (from split_dataset.py)
Output: <dataset-dir>/latents/<split>/shard_00000.pt  + manifest.json

Usage:
    python src/prepare_latents.py              # device auto (cuda if available)
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import soundfile as sf
import torch

import config
import tts_engine
from split_dataset import SPLITS
from validate_dataset import Thresholds

VAE_SAMPLE_RATE = 16000
LATENT_RATE_HZ = 25
SHARD_SIZE = 2000


def load_clip(path: Path):
    """Mono float32 at the VAE sample rate (same as upstream)."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != VAE_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=VAE_SAMPLE_RATE)
    return audio


def _resolve(audio: str) -> Path:
    p = Path(audio)
    return p if p.is_absolute() else config.PROJECT_ROOT / p


def encode_split(vae, manifest: Path, out_dir: Path, device: str, min_s: float, max_s: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("shard_*.pt"):  # stale shards from a previous run of this split
        old.unlink()
    shard, shards, skipped = [], [], []
    total_s = 0.0

    def flush():
        if shard:
            name = f"shard_{len(shards):05d}.pt"
            torch.save(list(shard), out_dir / name)
            shards.append({"file": name, "n": len(shard)})
            shard.clear()

    with open(manifest, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    for e in entries:
        try:
            audio = load_clip(_resolve(e["audio"]))
        except Exception as ex:
            skipped.append({"id": e["id"], "reason": f"unreadable: {type(ex).__name__}"})
            continue
        dur = len(audio) / VAE_SAMPLE_RATE
        if not min_s <= dur <= max_s:
            skipped.append({"id": e["id"], "reason": f"duration {dur:.2f}s outside {min_s}-{max_s}s"})
            continue
        with torch.no_grad():
            wav = torch.from_numpy(audio).to(device).view(1, 1, -1)
            z = vae.encode(wav, VAE_SAMPLE_RATE)  # [1, 64, T]
        shard.append({"id": e["id"], "latent": z.squeeze(0).transpose(0, 1).contiguous().half().cpu(),
                      "text": e["text"]})
        total_s += dur
        if len(shard) >= SHARD_SIZE:
            flush()
    flush()
    info = {"manifest": str(manifest), "kept": len(entries) - len(skipped), "skipped": skipped,
            "total_minutes": round(total_s / 60, 2), "latent_rate_hz": LATENT_RATE_HZ, "shards": shards}
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def main() -> int:
    th = Thresholds()
    p = argparse.ArgumentParser(description="Encode split manifests into AudioVAE latents")
    p.add_argument("--dataset-dir", type=Path, default=config.PILOT_DATASET_DIR)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--min-s", type=float, default=th.min_duration_s)
    p.add_argument("--max-s", type=float, default=th.max_duration_s)
    args = p.parse_args()

    splits_dir = args.dataset_dir / "splits"
    missing = [s for s in SPLITS if not (splits_dir / f"{s}.jsonl").is_file()]
    if missing:
        print(f"error: missing {missing} in {splits_dir}; run src/split_dataset.py first", file=sys.stderr)
        return 2

    device = tts_engine.select_device(args.device)
    vae = tts_engine._load_audio_vae(device)
    for split in SPLITS:
        t0 = time.perf_counter()
        info = encode_split(vae, splits_dir / f"{split}.jsonl", args.dataset_dir / "latents" / split,
                            device, args.min_s, args.max_s)
        print(f"{split:<6}: kept {info['kept']:4d} ({info['total_minutes']:.2f} min), "
              f"skipped {len(info['skipped'])}, {time.perf_counter() - t0:.1f}s")
    print(f"Latents written to {args.dataset_dir / 'latents'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
