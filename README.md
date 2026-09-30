# Turkish TTS PoC

Self-hosted Turkish text-to-speech proof of concept. Standalone: no ERP
integration, no external TTS APIs. Inference runs fully offline; input text
never leaves the machine.

**Current phase:** FAZ 4 — Turkish text normalization (no training, no API, no Docker).

## Model

| Component | Source | Pinned revision | License |
|---|---|---|---|
| FreyaTTS-small (183M, acoustic model) | [freyavoice/freya-tts](https://huggingface.co/freyavoice/freya-tts) | `d124e07` | Apache-2.0 |
| AudioVAE2 decoder (only `audiovae.pth`) | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | `32279ef` | Apache-2.0 |
| FreyaTTS inference code | [freyavoiceai/FreyaTTS](https://github.com/freyavoiceai/FreyaTTS) | `146d36c` | Apache-2.0 |

Output: 48 kHz mono WAV, single fixed voice ("Leyla").

> Before production use, the training-data provenance of FreyaTTS-small must be
> confirmed in writing with the model authors (not disclosed in the model card).

## Setup (Windows, PowerShell)

Requires Python 3.11 (64-bit) and Git.

```powershell
cd turkish-tts-poc
py -3.11 -m venv .venv          # or: <path-to-python3.11>\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -r requirements-nodeps.txt

# FreyaTTS inference code (not on PyPI), pinned commit
git clone https://github.com/freyavoiceai/FreyaTTS.git third_party/FreyaTTS
git -C third_party/FreyaTTS checkout 146d36c1cb6660646be57d31339db4eed9315de3

# One-time model download (~1.1 GB, SHA-256 verified) — the only network step
python src/download_models.py
```

On Linux replace `.\.venv\Scripts\Activate.ps1` with `source .venv/bin/activate`.

`requirements.lock.txt` holds the full resolved environment for exact reproduction.

## Usage

```powershell
python src/inference.py --text "Yeni öğrenci kaydı oluşturabilirsiniz."
```

Writes `outputs/base/test.wav` and prints model, device, input text, output
path, audio duration, load time and inference time.

Options:

| Flag | Default | Description |
|---|---|---|
| `--text` | (required) | Turkish text |
| `--out` | `outputs/base/test.wav` | Output WAV path |
| `--device` | `auto` | `auto` picks `cuda` if available, else `cpu` |
| `--steps` | `32` | Flow-matching ODE steps (quality/speed trade-off) |

### Text normalization

Before synthesis, document text is converted to its spoken form by
`src/text_normalizer.py` (the original text is never modified):

| Input | Spoken form sent to TTS |
|---|---|
| `KDV %20'dir.` | `Ka de ve yüzde yirmidir.` |
| `₺12.550,75'tir` | `on iki bin beş yüz elli Türk lirası yetmiş beş kuruştur` |
| `5.200 TL'ye` | `beş bin iki yüz Türk lirasına` |
| `30.09.2026'dır` | `otuz Eylül iki bin yirmi altıdır` |
| `14:30'da` | `on dört otuzda` |
| `%87,5` | `yüzde seksen yedi virgül beş` |
| `TR12 0006 …` | `te re, on iki, sıfır sıfır sıfır altı, …` |
| `CRM`, `e-Fatura` | `si ar em`, `e fatura` |

Rules are deterministic; Turkish suffix harmony after apostrophes is
re-applied when the spoken last word changes (`TL'ye` -> `Türk lirasına`).
Readings of abbreviations and English terms are plain dictionaries at the top
of the module (`ABBREVIATIONS`, `FOREIGN_WORDS`). Use `--no-normalize` on the
CLIs to bypass it. Characters outside the model's 92-symbol vocabulary are
mapped to the nearest known symbol in `tts_engine.fit_to_vocab`.

```powershell
pip install pytest==8.4.2
python -m pytest tests
```

### Evaluation set

```powershell
python src/generate_eval.py
```

Synthesizes every line of `evaluation/sentences.txt` (`id|category|text`) to
`outputs/base/<id>.wav` and writes `outputs/base/results.csv` (text, text the
model actually received, audio duration, inference time, RTF). The model is
loaded once and warmed up before timing.

The evaluation set is **unseen data**: never add these sentences to any
training dataset.

`outputs/base_raw/` holds the FAZ 3 run without normalization
(`--no-normalize --out-dir outputs/base_raw`) for before/after comparison.

### Voice drift check

```powershell
python src/voice_check.py outputs/base
```

Tracks pitch per second and flags files whose pitch drops >= 5 semitones from
the opening (the female voice drifting toward a male-sounding voice). See
*Known limitations*.

## Offline guarantee

- `src/download_models.py` is the only code that contacts Hugging Face.
- `src/inference.py` sets `HF_HUB_OFFLINE=1` before any import and loads all
  weights from `models/` by local path.

## Project layout

```
src/config.py           central paths, pinned revisions, inference params
src/download_models.py  one-time pinned model download
src/tts_engine.py       offline model loader
src/inference.py        CLI: text -> WAV
src/evaluation_set.py   reader for evaluation/sentences.txt
src/generate_eval.py    batch synthesis of the evaluation set
src/text_normalizer.py  Turkish text normalization (numbers, money, dates, abbreviations)
src/voice_check.py      pitch-based voice drift detector
tests/                  unit tests
evaluation/sentences.txt  unseen ERP evaluation sentences
outputs/base/           generated audio (git-ignored)
models/                 model weights (git-ignored)
third_party/FreyaTTS/   upstream inference code (git-ignored)
datasets/ checkpoints/  later phases (git-ignored)
```

## Known limitations

- **Voice drift:** FreyaTTS-small has no speaker conditioning; the voice comes
  from the noise seed. Toward the end of each synthesized segment the pitch
  often drops by up to an octave and the voice can sound male. Shorter
  chunks, more ODE steps, other seeds and duration scaling do not fix it
  reliably (measured with `voice_check.py`).
- CPU speed: RTF ~1.7 on a Ryzen 5 5600H (slower than real time).

## Notes

- `voxcpm` is installed with `--no-deps`: we only use its AudioVAE module, not
  the 2B VoxCPM2 model, so its heavy dependencies (transformers, funasr,
  gradio, ...) are not installed. `src/tts_engine.py` imports the AudioVAE
  without running `voxcpm/__init__.py`.
- The first synthesis in a new process is slower because librosa/numba
  JIT-compile the pitch check used by FreyaTTS's quality retry.
