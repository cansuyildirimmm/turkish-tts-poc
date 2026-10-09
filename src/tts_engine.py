"""Loads FreyaTTS-small from local files only (no network access).

FreyaTTS decodes audio with the AudioVAE from the `voxcpm` package. The
package's top-level __init__ imports the full 2B VoxCPM2 stack (transformers,
modelscope, funasr, gradio, ...), none of which we need. We register a bare
`voxcpm` namespace so that only `voxcpm.modules.audiovae` gets imported.
"""

import importlib.util
import json
import math
import re
import sys
import types
import unicodedata
from pathlib import Path

import numpy as np
import torch

import config
import text_normalizer
import voice_check


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _register_voxcpm_namespace() -> None:
    if "voxcpm" in sys.modules:
        return
    spec = importlib.util.find_spec("voxcpm")  # locates the package without executing it
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("voxcpm package not installed; see README installation steps")
    pkg = types.ModuleType("voxcpm")
    pkg.__path__ = list(spec.submodule_search_locations)
    pkg.__spec__ = spec
    sys.modules["voxcpm"] = pkg


def _import_freyatts():
    if not (config.FREYATTS_SRC_DIR / "freyatts").is_dir():
        raise FileNotFoundError(
            f"FreyaTTS source not found at {config.FREYATTS_SRC_DIR}; see README installation steps"
        )
    _register_voxcpm_namespace()
    src = str(config.FREYATTS_SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
    import freyatts.pipeline
    return freyatts


def _check_model_files() -> None:
    missing = [mf.local_dir / mf.filename for mf in config.MODEL_FILES
               if not (mf.local_dir / mf.filename).is_file()]
    if missing:
        names = "\n  ".join(str(p.relative_to(config.PROJECT_ROOT)) for p in missing)
        raise FileNotFoundError(f"Missing model files:\n  {names}\nRun: python src/download_models.py")


def _load_audio_vae(device: str):
    _register_voxcpm_namespace()
    from voxcpm.modules.audiovae import AudioVAEConfigV2, AudioVAEV2

    vae = AudioVAEV2(AudioVAEConfigV2())
    ckpt = torch.load(config.AUDIOVAE_DIR / "audiovae.pth", map_location="cpu", weights_only=True)
    # Same non-strict load as upstream freyatts/vae.py, but surface what was skipped.
    result = vae.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    if result.missing_keys:
        print(f"[warn] AudioVAE: {len(result.missing_keys)} missing keys", file=sys.stderr)
    vae = vae.to(device).float().eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def load_tts(device: str, model_dir: Path | None = None):
    """Build the FreyaTTS pipeline from local files.

    `model_dir` holds config.json + model.safetensors: the base model under
    models/ (default) or a fine-tuned checkpoint such as checkpoints/<run>/best.
    """
    _check_model_files()
    model_dir = Path(model_dir) if model_dir else config.FREYA_MODEL_DIR
    for name in ("config.json", "model.safetensors"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"{model_dir / name} not found")
    freyatts = _import_freyatts()
    from safetensors.torch import load_file

    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    model = freyatts.pipeline.FreyaDiT(
        vocab=cfg["vocab"], d=cfg["d"], depth=cfg["depth"], heads=cfg["heads"], ff=cfg["ff"],
    )
    model.load_state_dict(load_file(model_dir / "model.safetensors"), strict=True)
    model = model.to(device).eval()

    vae = _load_audio_vae(device)

    with open(config.FREYATTS_VOCAB_PATH, encoding="utf-8") as f:
        char_to_id = json.load(f)

    return freyatts.pipeline.FreyaTTS(model, vae, char_to_id, device=device)


def upstream_normalize(text: str) -> str:
    """FreyaTTS's own light normalization, applied inside synthesize()."""
    return _import_freyatts().pipeline.normalize(text)


@torch.no_grad()
def _synth_chunk(tts, text: str, steps: int, duration_scale: float) -> np.ndarray:
    """FreyaTTS._synth_one with the predicted frame count scaled by `duration_scale`.

    With duration_scale=1.0 this is identical to upstream. The seed (= voice)
    is always the pipeline's own, so every candidate is the same speaker.
    """
    m = tts.model
    ids = tts._ids(text)
    cmask = torch.ones_like(ids, dtype=torch.bool)
    te = m.text_encode(ids)
    pooled = (te * cmask[..., None].float()).sum(1) / (cmask.sum(1, keepdim=True) + 1e-6)
    frames = int(round(math.exp(float(m.dur(pooled).squeeze(-1))) * duration_scale))
    frames = max(tts.t_floor, ids.shape[1] + 4, min(300, frames))
    latents = m.sample(ids, frames, steps=steps, cmask=cmask, seed=tts.seed)
    return tts.vae.decode(latents.transpose(1, 2).float()).squeeze().float().cpu().numpy()


def synthesize(tts, text: str, steps: int = config.INFERENCE_STEPS,
               drift_guard: bool = config.DRIFT_GUARD) -> tuple[np.ndarray, dict]:
    """Prepared text -> 48 kHz waveform, plus per-chunk details.

    Without the drift guard this is exactly FreyaTTS.synthesize. With it, each
    clause is synthesized at duration scales DRIFT_GUARD_SCALES in order until
    no window falls below DRIFT_MIN_PITCH_HZ; otherwise the take with the
    highest lowest-window pitch is kept. Deterministic: same text, same output.
    """
    if not drift_guard:
        return tts.synthesize(text, steps=steps), {"drift_guard": False}

    t = upstream_normalize(text)
    chunks = tts._clauses(t) if len(t.split()) > tts.max_words else [t]
    gap = np.zeros(int(0.12 * tts.sample_rate), dtype=np.float32)
    parts, chosen = [], []
    for chunk in chunks:
        best = None
        for scale in config.DRIFT_GUARD_SCALES:
            wav = _synth_chunk(tts, chunk, steps, scale)
            windows = voice_check.window_f0_array(wav, tts.sample_rate, config.DRIFT_WINDOW_S)
            floor = voice_check.take_floor(windows, config.DRIFT_MIN_VOICED_RATIO, config.DRIFT_MIN_OPENING_HZ)
            if best is None or floor > best[0]:
                best = (floor, scale, wav)
            if floor >= config.DRIFT_MIN_PITCH_HZ:
                break
        parts += [best[2].astype(np.float32), gap]
        chosen.append({"text": chunk, "scale": best[1],
                       "min_f0_hz": None if best[0] == 0.0 else round(best[0], 1)})
    return np.concatenate(parts[:-1]), {"drift_guard": True, "chunks": chosen}


# Characters missing from the model's 92-symbol vocabulary, mapped to the
# closest symbol it knows. Anything else unknown is decomposed or replaced.
_VOCAB_FALLBACK = {"Ğ": "ğ", "q": "k", "Â": "A", "î": "i", "Î": "İ", "û": "u", "Û": "U",
                   "!": ".", "“": "", "”": "", '"': "", "«": "", "»": "", "–": ",", "—": ","}


def unmappable_chars(text: str, vocab: dict) -> set[str]:
    """Characters fit_to_vocab would have to drop (replace with a space)."""
    return {ch for ch in text
            if ch not in vocab and ch not in _VOCAB_FALLBACK and ch.lower() not in vocab
            and unicodedata.normalize("NFD", ch)[0] not in vocab}


def fit_to_vocab(text: str, vocab: dict) -> str:
    """Replace characters the model cannot encode (it would read them as <UNK>)."""
    out = []
    for ch in text:
        if ch in vocab:
            out.append(ch)
        elif ch in _VOCAB_FALLBACK:
            out.append(_VOCAB_FALLBACK[ch])
        elif ch.lower() in vocab:
            out.append(ch.lower())
        else:
            base = unicodedata.normalize("NFD", ch)[0]
            out.append(base if base in vocab else " ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def prepare_text(tts, text: str, normalize: bool = True) -> str:
    """Document text -> exact string handed to the model."""
    t = text_normalizer.normalize(text) if normalize else text
    return fit_to_vocab(t, tts.char_to_id)
