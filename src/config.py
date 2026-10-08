"""Central configuration for the Turkish TTS PoC.

All paths are resolved relative to the project root so the code runs the same
on Windows and inside a Linux container. Nothing here is machine-specific.
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BASE_OUTPUT_DIR = OUTPUTS_DIR / "base"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"

# Unseen ERP evaluation set. Must never be used as training data.
EVAL_SENTENCES_PATH = EVALUATION_DIR / "sentences.txt"

# Pilot recording (FAZ 5/6): source sentences -> script the talent reads.
RECORDING_DIR = PROJECT_ROOT / "recording"
PILOT_SENTENCES_PATH = RECORDING_DIR / "pilot_sentences.txt"
PILOT_SCRIPT_PATH = RECORDING_DIR / "pilot_script.tsv"
PILOT_DATASET_DIR = PROJECT_ROOT / "datasets" / "pilot"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Synthetic voice-lock data (decision 2026-10-01): FreyaTTS-small's own output,
# filtered for drift. Source scripts: the pilot script plus group B (longer).
SYNTHETIC_SENTENCES_PATH = RECORDING_DIR / "synthetic_sentences.txt"
SYNTHETIC_SCRIPT_PATH = RECORDING_DIR / "synthetic_script.tsv"
SYNTHETIC_DATASET_DIR = PROJECT_ROOT / "datasets" / "synthetic"

# FreyaTTS inference code (not on PyPI) is cloned here at a pinned commit.
FREYATTS_SRC_DIR = THIRD_PARTY_DIR / "FreyaTTS"
FREYATTS_SRC_COMMIT = "146d36c1cb6660646be57d31339db4eed9315de3"
FREYATTS_VOCAB_PATH = FREYATTS_SRC_DIR / "freyatts" / "char_vocab.json"

SEED = 42


@dataclass(frozen=True)
class ModelFile:
    repo_id: str
    revision: str  # pinned Hugging Face commit, never a moving branch
    filename: str
    sha256: str | None  # None for small non-LFS files
    local_dir: Path


# FreyaTTS-small acoustic model (Apache-2.0).
FREYA_REPO_ID = "freyavoice/freya-tts"
FREYA_REVISION = "d124e07493615208f58bdd21d432736849ee4230"
FREYA_MODEL_DIR = MODELS_DIR / "freya-tts"

# Frozen AudioVAE2 decoder from VoxCPM2 (Apache-2.0). Only this file is used,
# not the 2B VoxCPM2 language model.
VOXCPM_REPO_ID = "openbmb/VoxCPM2"
VOXCPM_REVISION = "32279effe8c19989596f05d353d1447f51d9e915"
AUDIOVAE_DIR = MODELS_DIR / "voxcpm2-audiovae"

MODEL_FILES = [
    ModelFile(FREYA_REPO_ID, FREYA_REVISION, "config.json", None, FREYA_MODEL_DIR),
    ModelFile(FREYA_REPO_ID, FREYA_REVISION, "model.safetensors",
              "9e5828ce9eb6aaf197adc5cb098e2e80e8ff301d238add631418c5557fa08b22", FREYA_MODEL_DIR),
    ModelFile(VOXCPM_REPO_ID, VOXCPM_REVISION, "audiovae.pth",
              "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1", AUDIOVAE_DIR),
]

MODEL_NAME = "FreyaTTS-small (freyavoice/freya-tts)"
SAMPLE_RATE = 48000

# Inference parameters. Keep these identical between base and fine-tuned runs.
INFERENCE_STEPS = 32  # Euler ODE steps, the upstream default

# Drift guard: FreyaTTS-small tends to drift toward a lower (male-sounding)
# voice at the end of a clause. When enabled, each clause is re-synthesized
# at these duration scales (same seed = same voice) until the pitch drop is
# acceptable, and the least-drifting take is kept. Costs up to 4x compute.
DRIFT_GUARD = False
DRIFT_GUARD_SCALES = (1.0, 0.9, 1.1, 1.2)
DRIFT_ACCEPT_SEMITONES = 4.0
DRIFT_WINDOW_S = 0.5
# A take only qualifies if it is mostly voiced and opens in the target voice's
# range; otherwise a broken (whispery/noisy) take would look drift-free.
DRIFT_MIN_VOICED_RATIO = 0.5
DRIFT_MIN_OPENING_HZ = 200.0

DEFAULT_OUTPUT_PATH = BASE_OUTPUT_DIR / "test.wav"
