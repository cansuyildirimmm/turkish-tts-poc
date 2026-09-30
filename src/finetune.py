"""Single-speaker fine-tuning ("voice lock" SFT) of FreyaTTS-small.

Same objective as FreyaTTS training/sft.py (masked flow-matching + duration
loss, AdamW, warmup + cosine LR, grad clip 1.0), plus what the upstream script
lacks: validation loss, best-checkpoint tracking, epoch logging, early
stopping, CSV logs and exact resume. Plain PyTorch, single device.

Checkpoints are written as <run>/<name>/{model.safetensors, config.json,
training_state.pt}; `best/` and `final/` load like models/freya-tts.

Usage:
    python src/finetune.py --config configs/finetune_pilot.json --plan     # show parameters, no training
    python src/finetune.py --config configs/finetune_pilot.json            # train
    python src/finetune.py --config configs/finetune_pilot.json --resume latest
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import config
import tts_engine
from training_data import FILL_ID, LatentDataset, collate, lr_at

LOG_FIELDS = ["step", "epoch", "lr", "train_cfm", "train_dur", "train_loss",
              "val_cfm", "val_dur", "val_loss", "is_best", "elapsed_s"]
VAL_SEED_OFFSET = 10_000


# ---------------------------------------------------------------- config

def _path(p: str | None) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else config.PROJECT_ROOT / p


def load_config(path: Path, overrides: list[str]) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for item in overrides:
        key, _, value = item.partition("=")
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value
    return cfg


def _arch(cfg: dict) -> dict:
    """Architecture comes from the init checkpoint, never from hand-typed sizes."""
    init = _path(cfg.get("init"))
    if init:
        with open(init / "config.json", encoding="utf-8") as f:
            arch = json.load(f)
    else:
        arch = dict(cfg["arch"])
    return {k: arch[k] for k in ("vocab", "d", "depth", "heads", "ff")}


# ---------------------------------------------------------------- model / data

def build_model(cfg: dict, device: str):
    freyatts = tts_engine._import_freyatts()
    arch = _arch(cfg)
    model = freyatts.model.FreyaDiT(vocab=arch["vocab"], feat=64, d=arch["d"], depth=arch["depth"],
                                    heads=arch["heads"], ff=arch["ff"], fill_id=FILL_ID)
    init = _path(cfg.get("init"))
    if init:
        model.load_state_dict(load_file(init / "model.safetensors"), strict=True)
    return model.to(device), arch


def load_vocab() -> dict:
    with open(config.FREYATTS_VOCAB_PATH, encoding="utf-8") as f:
        return json.load(f)


def epoch_batches(n: int, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    """Seeded shuffle per epoch, drop_last: reproducible and resumable mid-epoch."""
    g = torch.Generator().manual_seed(seed * 1000 + epoch)
    perm = torch.randperm(n, generator=g).tolist()
    return [perm[i:i + batch_size] for i in range(0, n - batch_size + 1, batch_size)]


def _to(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def losses(model, batch: dict, lambda_dur: float):
    cfm = model.cfm_loss(batch["lat"], batch["text"], batch["fmask"], batch["cmask"])
    dur, _ = model.dur_loss(batch["text"], batch["logT"], batch["cmask"])
    return cfm, dur, cfm + lambda_dur * dur


@torch.no_grad()
def evaluate(model, dataset, cfg: dict, device: str, amp: bool) -> dict:
    """Validation loss with fixed noise/timesteps so values are comparable across steps."""
    model.eval()
    totals = {"cfm": 0.0, "dur": 0.0, "n": 0}
    devices = [torch.device(device)] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg["seed"] + VAL_SEED_OFFSET)
        for i in range(0, len(dataset), cfg["batch_size"]):
            items = [dataset[j] for j in range(i, min(i + cfg["batch_size"], len(dataset)))]
            batch = _to(collate(items), device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
                cfm, dur, _ = losses(model, batch, cfg["lambda_dur"])
            totals["cfm"] += float(cfm) * len(items)
            totals["dur"] += float(dur) * len(items)
            totals["n"] += len(items)
    model.train()
    n = max(1, totals["n"])
    cfm, dur = totals["cfm"] / n, totals["dur"] / n
    return {"cfm": cfm, "dur": dur, "loss": cfm + cfg["lambda_dur"] * dur}


# ---------------------------------------------------------------- checkpoints

def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(run_dir: Path, name: str, model, optimizer, state: dict, arch: dict) -> Path:
    final = run_dir / name
    tmp = run_dir / f".{name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}, str(tmp / "model.safetensors"))
    with open(tmp / "config.json", "w", encoding="utf-8") as f:
        json.dump({**arch, "arch": "xattn"}, f, indent=2)
    torch.save({"optimizer": optimizer.state_dict(), "rng": _rng_state(), **state}, tmp / "training_state.pt")
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)  # a crash mid-save never leaves a half-written checkpoint under its real name
    return final


def _step_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))


def find_resume(run_dir: Path, resume: str) -> Path:
    if resume == "latest":
        ckpts = _step_checkpoints(run_dir)
        if not ckpts:
            raise FileNotFoundError(f"no step_* checkpoints in {run_dir}")
        return ckpts[-1]
    return _path(resume)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------- plan

def plan(cfg: dict, device: str, train_n: int, val_n: int, n_params: int, arch: dict, amp: bool) -> str:
    steps_per_epoch = max(1, train_n // cfg["batch_size"])
    eff_batch = cfg["batch_size"] * cfg["grad_accum"]
    epochs = cfg["steps"] * cfg["grad_accum"] / steps_per_epoch
    # Rough VRAM: fp32 weights + grads + AdamW moments, plus bf16 activations.
    state_gb = n_params * (4 + 4 + 8) / 1e9
    act_gb = cfg["batch_size"] * cfg["max_frames"] * arch["d"] * arch["depth"] * 2 * 24 / 1e9
    es = cfg.get("early_stopping") or {}
    return "\n".join([
        "FINE-TUNING PLAN (nothing is trained with --plan)",
        f"  Model             : FreyaTTS-small, init {cfg.get('init')}  ({n_params / 1e6:.1f}M params, "
        f"d={arch['d']} depth={arch['depth']} heads={arch['heads']})",
        f"  Data              : {cfg['data_dir']}",
        f"  Train / val clips : {train_n} / {val_n}",
        f"  Batch size        : {cfg['batch_size']} x grad_accum {cfg['grad_accum']} = {eff_batch}",
        f"  Steps             : {cfg['steps']}  (~{steps_per_epoch} steps/epoch, ~{epochs:.0f} epochs)",
        f"  Learning rate     : {cfg['lr']}  warmup {cfg['warmup']} steps, cosine decay to 5%",
        f"  Optimizer         : AdamW betas={tuple(cfg['betas'])} weight_decay={cfg['weight_decay']} "
        f"grad_clip={cfg['grad_clip']}",
        f"  Loss              : flow-matching + {cfg['lambda_dur']} x duration",
        f"  Precision         : {'bf16 autocast' if amp else 'fp32'} on {device}",
        f"  Log / val / ckpt  : every {cfg['log_every']} / {cfg['val_every']} / {cfg['ckpt_every']} steps "
        f"(keep last {cfg['keep_last']})",
        f"  Early stopping    : " + (f"patience {es['patience']} validations, min_delta {es.get('min_delta', 0)}"
                                    if es.get("patience") else "off"),
        f"  Seed              : {cfg['seed']}",
        f"  VRAM estimate     : ~{state_gb + act_gb:.1f} GB (rough: {state_gb:.1f} optimizer state + "
        f"{act_gb:.1f} activations)",
        f"  Output            : {_path(cfg['output_dir']) / cfg['run_name']}",
    ])


# ---------------------------------------------------------------- train

def train(cfg: dict, device: str, resume: str | None, plan_only: bool = False) -> dict:
    run_dir = _path(cfg["output_dir"]) / cfg["run_name"]
    data_dir = _path(cfg["data_dir"])
    vocab = load_vocab()
    train_ds = LatentDataset(str(data_dir / "train"), vocab, cfg["max_frames"])
    val_ds = LatentDataset(str(data_dir / "val"), vocab, cfg["max_frames"])
    if len(train_ds) < cfg["batch_size"]:
        raise ValueError(f"{len(train_ds)} training clips < batch size {cfg['batch_size']}")
    if len(val_ds) == 0:
        raise ValueError("validation set is empty")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model, arch = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    amp = cfg["precision"] == "bf16" and device.startswith("cuda") and torch.cuda.is_bf16_supported()
    if plan_only:
        print(plan(cfg, device, len(train_ds), len(val_ds), n_params, arch, amp))
        return {"planned": True}

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                                  weight_decay=cfg["weight_decay"], eps=1e-8)
    state = {"step": 0, "epoch": 0, "batch_pos": 0, "best_val": None, "best_step": None, "bad_evals": 0}

    if resume:
        ckpt = find_resume(run_dir, resume)
        model.load_state_dict(load_file(ckpt / "model.safetensors"), strict=True)
        saved = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        _set_rng_state(saved["rng"])
        state = {k: saved[k] for k in state}
        print(f"[resume] {ckpt} at step {state['step']}, epoch {state['epoch']}", flush=True)
    else:
        if _step_checkpoints(run_dir) or (run_dir / "final").exists():
            raise FileExistsError(f"{run_dir} already has checkpoints; use --resume latest or a new run_name")
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "arch": arch, "device": device, "amp_bf16": amp, "params": n_params,
                       "train_clips": len(train_ds), "val_clips": len(val_ds), "git_commit": _git_commit(),
                       "torch": torch.__version__}, f, indent=2)
        with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, LOG_FIELDS).writeheader()

    def log(row: dict) -> None:
        with open(run_dir / "train_log.csv", "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, LOG_FIELDS).writerow({k: row.get(k, "") for k in LOG_FIELDS})

    def batch_stream():
        while True:
            batches = epoch_batches(len(train_ds), cfg["batch_size"], cfg["seed"], state["epoch"])
            while state["batch_pos"] < len(batches):
                idx = batches[state["batch_pos"]]
                state["batch_pos"] += 1
                yield idx
            state["epoch"] += 1
            state["batch_pos"] = 0

    model.train()
    stream = batch_stream()
    window = {"cfm": 0.0, "dur": 0.0, "n": 0}
    t_start = time.time()
    stop_reason = "steps completed"
    es = cfg.get("early_stopping") or {}

    while state["step"] < cfg["steps"]:
        lr = lr_at(state["step"], cfg["warmup"], cfg["steps"], cfg["lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _ in range(cfg["grad_accum"]):
            batch = _to(collate([train_ds[i] for i in next(stream)]), device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
                cfm, dur, loss = losses(model, batch, cfg["lambda_dur"])
            (loss / cfg["grad_accum"]).backward()
            window["cfm"] += cfm.item()
            window["dur"] += dur.item()
            window["n"] += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state["step"] += 1
        step = state["step"]

        if step % cfg["log_every"] == 0:
            n = max(1, window["n"])
            row = {"step": step, "epoch": state["epoch"], "lr": f"{lr:.3e}",
                   "train_cfm": f"{window['cfm'] / n:.5f}", "train_dur": f"{window['dur'] / n:.5f}",
                   "train_loss": f"{(window['cfm'] + cfg['lambda_dur'] * window['dur']) / n:.5f}",
                   "elapsed_s": f"{time.time() - t_start:.1f}"}
            log(row)
            print(f"step {step:6d}/{cfg['steps']}  epoch {state['epoch']:3d}  loss {row['train_loss']}  "
                  f"(cfm {row['train_cfm']} dur {row['train_dur']})  lr {row['lr']}", flush=True)
            window = {"cfm": 0.0, "dur": 0.0, "n": 0}

        is_last = step >= cfg["steps"]
        if step % cfg["val_every"] == 0 or is_last:
            val = evaluate(model, val_ds, cfg, device, amp)
            improved = state["best_val"] is None or val["loss"] < state["best_val"] - es.get("min_delta", 0.0)
            if improved:
                state.update(best_val=val["loss"], best_step=step, bad_evals=0)
                save_checkpoint(run_dir, "best", model, optimizer, state, arch)
                with open(run_dir / "best.json", "w", encoding="utf-8") as f:
                    json.dump({"step": step, "epoch": state["epoch"], "val_loss": val["loss"],
                               "val_cfm": val["cfm"], "val_dur": val["dur"]}, f, indent=2)
            else:
                state["bad_evals"] += 1
            log({"step": step, "epoch": state["epoch"], "lr": f"{lr:.3e}", "val_cfm": f"{val['cfm']:.5f}",
                 "val_dur": f"{val['dur']:.5f}", "val_loss": f"{val['loss']:.5f}", "is_best": int(improved),
                 "elapsed_s": f"{time.time() - t_start:.1f}"})
            print(f"[val] step {step}  val_loss {val['loss']:.5f}  " + ("(best)" if improved else
                  f"(best {state['best_val']:.5f} @ {state['best_step']}, no improvement x{state['bad_evals']})"),
                  flush=True)
            if es.get("patience") and state["bad_evals"] >= es["patience"]:
                stop_reason = f"early stopping: no improvement in {es['patience']} validations"

        if step % cfg["ckpt_every"] == 0 and not is_last:
            save_checkpoint(run_dir, f"step_{step:06d}", model, optimizer, state, arch)
            for old in _step_checkpoints(run_dir)[:-cfg["keep_last"]]:
                shutil.rmtree(old)
        if stop_reason != "steps completed":
            break

    save_checkpoint(run_dir, "final", model, optimizer, state, arch)
    batches_per_epoch = len(train_ds) // cfg["batch_size"]
    summary = {"stop_reason": stop_reason, "final_step": state["step"],
               "epochs_completed": round(state["step"] * cfg["grad_accum"] / batches_per_epoch, 2),
               "best_step": state["best_step"], "best_val_loss": state["best_val"],
               "elapsed_s": round(time.time() - t_start, 1)}
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {stop_reason}; best val {state['best_val']:.5f} at step {state['best_step']} -> {run_dir / 'best'}",
          flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FreyaTTS single-speaker fine-tuning")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--resume", help="'latest' or a checkpoint directory")
    p.add_argument("--plan", action="store_true", help="print parameters and exit without training")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(args.config, args.set)
    device = tts_engine.select_device(args.device)
    train(cfg, device, args.resume, plan_only=args.plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
