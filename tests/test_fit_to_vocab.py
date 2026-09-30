import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import tts_engine  # noqa: E402


@pytest.fixture(scope="module")
def vocab():
    if not config.FREYATTS_VOCAB_PATH.is_file():
        pytest.skip("FreyaTTS source not installed (see README)")
    with open(config.FREYATTS_VOCAB_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("text, expected", [
    ("Dikkat!", "Dikkat."),                  # '!' is not in the model vocabulary
    ("ĞÜMÜŞ", "ğÜMÜŞ"),                      # uppercase Ğ is missing
    ("Aqua", "Akua"),
    ("hâlâ", "hâlâ"),
    ("“Tamam”", "Tamam"),
    ("Evet — hayır", "Evet , hayır"),
])
def test_fit_to_vocab(vocab, text, expected):
    assert tts_engine.fit_to_vocab(text, vocab) == expected


def test_unmappable(vocab):
    assert tts_engine.unmappable_chars("Merhaba ☺", vocab) == {"☺"}
    assert tts_engine.unmappable_chars("Dikkat!", vocab) == set()
