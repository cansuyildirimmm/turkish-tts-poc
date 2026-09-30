"""Loads FreyaTTS-small from local files only (no network access).

FreyaTTS decodes audio with the AudioVAE from the `voxcpm` package. The
package's top-level __init__ imports the full 2B VoxCPM2 stack (transformers,
modelscope, funasr, gradio, ...), none of which we need. We register a bare
`voxcpm` namespace so that only `voxcpm.modules.audiovae` gets imported.
"""

import importlib.util
import json
import re
import sys
import types
import unicodedata

import torch

import config
import text_normalizer


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


def load_tts(device: str):
    """Build the FreyaTTS pipeline from files under models/."""
    _check_model_files()
    freyatts = _import_freyatts()
    from safetensors.torch import load_file

    with open(config.FREYA_MODEL_DIR / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    model = freyatts.pipeline.FreyaDiT(
        vocab=cfg["vocab"], d=cfg["d"], depth=cfg["depth"], heads=cfg["heads"], ff=cfg["ff"],
    )
    model.load_state_dict(load_file(config.FREYA_MODEL_DIR / "model.safetensors"), strict=True)
    model = model.to(device).eval()

    vae = _load_audio_vae(device)

    with open(config.FREYATTS_SRC_DIR / "freyatts" / "char_vocab.json", encoding="utf-8") as f:
        char_to_id = json.load(f)

    return freyatts.pipeline.FreyaTTS(model, vae, char_to_id, device=device)


def upstream_normalize(text: str) -> str:
    """FreyaTTS's own light normalization, applied inside synthesize()."""
    return _import_freyatts().pipeline.normalize(text)


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
