"""Training-loop mechanics on a tiny random model (NOT a real fine-tune).

Uses a ~10k-parameter FreyaDiT on random latents, CPU, a few steps: checks
logging, checkpoints, best tracking, early stopping and exact resume.
"""

import csv
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import finetune  # noqa: E402
import training_data  # noqa: E402

TEXTS = ["Merhaba dünya.", "Kayıt tamamlandı.", "Ödeme alındı.", "Rapor hazır.",
         "Fatura gönderildi.", "Yeni öğrenci eklendi.", "Şifre değişti.", "Görüşmek üzere."]


@pytest.fixture(scope="module")
def latents(tmp_path_factory):
    if not config.FREYATTS_VOCAB_PATH.is_file():
        pytest.skip("FreyaTTS source not installed (see README)")
    root = tmp_path_factory.mktemp("latents")
    g = torch.Generator().manual_seed(0)
    for split, n in (("train", 12), ("val", 4)):
        (root / split).mkdir()
        shard = [{"id": f"{split}{i}", "text": TEXTS[i % len(TEXTS)],
                  "latent": torch.randn(20 + 3 * i, 64, generator=g).half()} for i in range(n)]
        torch.save(shard, root / split / "shard_00000.pt")
    return root


def make_cfg(latents, out, **kw):
    cfg = {
        "run_name": "tiny", "init": None, "arch": {"vocab": 92, "d": 32, "depth": 1, "heads": 2, "ff": 64},
        "data_dir": str(latents), "output_dir": str(out), "seed": 42,
        "steps": 6, "batch_size": 4, "grad_accum": 1, "lr": 1e-3, "warmup": 2,
        "betas": [0.9, 0.95], "weight_decay": 0.01, "grad_clip": 1.0, "lambda_dur": 0.1,
        "max_frames": 500, "precision": "bf16",
        "log_every": 1, "val_every": 2, "ckpt_every": 3, "keep_last": 2, "early_stopping": None,
    }
    cfg.update(kw)
    return cfg


def weights(path):
    return load_file(str(path / "model.safetensors"))


def test_full_run_outputs(latents, tmp_path):
    summary = finetune.train(make_cfg(latents, tmp_path), "cpu", resume=None)
    run = tmp_path / "tiny"
    assert summary["final_step"] == 6 and summary["stop_reason"] == "steps completed"
    assert summary["epochs_completed"] == 2  # 12 clips / batch 4 = 3 steps per epoch
    for name in ("best", "final", "step_000003"):
        assert (run / name / "model.safetensors").is_file()
        assert json.loads((run / name / "config.json").read_text())["d"] == 32
    best = json.loads((run / "best.json").read_text())
    assert best["step"] == summary["best_step"]

    rows = list(csv.DictReader(open(run / "train_log.csv", encoding="utf-8")))
    train_rows = [r for r in rows if r["train_loss"]]
    val_rows = [r for r in rows if r["val_loss"]]
    assert [int(r["step"]) for r in train_rows] == [1, 2, 3, 4, 5, 6]
    assert [int(r["step"]) for r in val_rows] == [2, 4, 6]
    assert all(r["lr"] and r["epoch"] != "" for r in rows)
    assert sum(int(r["is_best"]) for r in val_rows) >= 1
    run_cfg = json.loads((run / "run_config.json").read_text())
    assert run_cfg["amp_bf16"] is False  # bf16 autocast only on CUDA


def test_resume_is_exact(latents, tmp_path):
    straight = tmp_path / "straight"
    finetune.train(make_cfg(latents, straight), "cpu", resume=None)

    split = tmp_path / "split"
    cfg = make_cfg(latents, split)
    finetune.train({**cfg, "steps": 6, "ckpt_every": 3}, "cpu", resume=None)
    # simulate a crash after step 3: drop everything saved later, then resume
    import shutil
    shutil.rmtree(split / "tiny" / "final")
    finetune.train(cfg, "cpu", resume="latest")

    a, b = weights(straight / "tiny" / "final"), weights(split / "tiny" / "final")
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_refuses_to_overwrite(latents, tmp_path):
    finetune.train(make_cfg(latents, tmp_path), "cpu", resume=None)
    with pytest.raises(FileExistsError):
        finetune.train(make_cfg(latents, tmp_path), "cpu", resume=None)


def test_early_stopping(latents, tmp_path):
    # lr 0: validation never improves after the first check
    cfg = make_cfg(latents, tmp_path, lr=0.0, steps=20, early_stopping={"patience": 2, "min_delta": 0.0})
    summary = finetune.train(cfg, "cpu", resume=None)
    assert summary["stop_reason"].startswith("early stopping")
    assert summary["final_step"] == 6  # val at 2 (best), 4 (bad 1), 6 (bad 2) -> stop
    assert summary["best_step"] == 2


def test_validation_loss_deterministic(latents):
    vocab = finetune.load_vocab()
    ds = training_data.LatentDataset(str(latents / "val"), vocab, 500)
    cfg = make_cfg(latents, "unused")
    model, _ = finetune.build_model(cfg, "cpu")
    assert finetune.evaluate(model, ds, cfg, "cpu", False) == finetune.evaluate(model, ds, cfg, "cpu", False)


def test_plan_does_not_train(latents, tmp_path, capsys):
    result = finetune.train(make_cfg(latents, tmp_path), "cpu", resume=None, plan_only=True)
    assert result == {"planned": True}
    assert "FINE-TUNING PLAN" in capsys.readouterr().out
    assert not (tmp_path / "tiny").exists()


def test_epoch_batches_reproducible():
    a = finetune.epoch_batches(10, 3, seed=42, epoch=0)
    assert a == finetune.epoch_batches(10, 3, seed=42, epoch=0)
    assert a != finetune.epoch_batches(10, 3, seed=42, epoch=1)
    assert len(a) == 3 and all(len(b) == 3 for b in a)  # drop_last
