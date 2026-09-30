"""Build and check the pilot recording script.

Reads recording/pilot_sentences.txt, normalizes each sentence into the exact
spoken form the talent must read (numbers, currency, abbreviations spelled
out), and writes recording/pilot_script.tsv with two text columns:
  read_text   what the talent reads (keeps '!' etc. for intonation)
  train_text  the training transcript: read_text mapped to the model vocabulary
Audio and transcript therefore match by construction.

Checks (non-zero exit on any hard failure):
  - overlap with the unseen evaluation set (exact match or shared 5-word run)
  - duplicate sentences
  - characters with no mapping into the model vocabulary
Also reports estimated duration and letter coverage.

Usage:
    python src/recording_script.py
"""

import csv
import json
import sys
from collections import Counter

import config
import text_normalizer
import tts_engine
from evaluation_set import load_sentences

NGRAM = 5
# Rough Turkish read-speech rate on normalized text, used only for estimates.
CHARS_PER_SECOND = 14.0
MAX_CLIP_S = 14.0  # FreyaTTS precompute_latents.py default --max_s
TURKISH_LETTERS = "abcçdefgğhıijklmnoöprsştuüvyz"


def _words(text: str) -> list[str]:
    return [w.strip(".,;:!?'\"").lower() for w in text.split() if w.strip(".,;:!?'\"")]


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main() -> int:
    with open(config.FREYATTS_VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)

    sentences = load_sentences(config.PILOT_SENTENCES_PATH)
    eval_sentences = load_sentences(config.EVAL_SENTENCES_PATH)
    eval_texts = {text_normalizer.normalize(s.text).lower() for s in eval_sentences}
    eval_grams = set().union(*(_ngrams(text_normalizer.normalize(s.text), NGRAM) for s in eval_sentences))

    rows, errors = [], []
    seen: dict[str, str] = {}
    letters = Counter()
    total_s = 0.0
    for s in sentences:
        read_text = text_normalizer.normalize(s.text)
        train_text = tts_engine.fit_to_vocab(read_text, vocab)
        key = read_text.lower()

        if key in eval_texts:
            errors.append(f"{s.sentence_id}: identical to an evaluation sentence")
        shared = _ngrams(read_text, NGRAM) & eval_grams
        if shared:
            errors.append(f"{s.sentence_id}: shares '{' '.join(sorted(shared)[0])}' with the evaluation set")
        if key in seen:
            errors.append(f"{s.sentence_id}: duplicate of {seen[key]}")
        seen[key] = s.sentence_id
        unmappable = tts_engine.unmappable_chars(read_text, vocab)
        if unmappable:
            errors.append(f"{s.sentence_id}: characters with no vocabulary mapping {sorted(unmappable)}: {read_text!r}")

        est_s = len(read_text) / CHARS_PER_SECOND
        if est_s > MAX_CLIP_S:
            errors.append(f"{s.sentence_id}: estimated {est_s:.1f}s exceeds {MAX_CLIP_S}s clip limit")
        total_s += est_s
        letters.update(ch for ch in read_text.lower().replace("İ", "i") if ch in TURKISH_LETTERS)
        rows.append({
            "id": s.sentence_id,
            "category": s.category,
            "source_text": s.text,
            "read_text": read_text,
            "train_text": train_text,
            "est_seconds": f"{est_s:.1f}",
        })

    with open(config.PILOT_SCRIPT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    per_cat = Counter(r["category"] for r in rows)
    print(f"Sentences          : {len(rows)}  " + ", ".join(f"{k}={v}" for k, v in per_cat.items()))
    print(f"Estimated duration : {total_s / 60:.1f} min of speech (excluding pauses/retakes)")
    rare = [f"{ch}={letters[ch]}" for ch in TURKISH_LETTERS if letters[ch] < 20]
    missing = [ch for ch in TURKISH_LETTERS if letters[ch] == 0]
    print(f"Letter coverage    : {len(TURKISH_LETTERS) - len(missing)}/{len(TURKISH_LETTERS)} letters"
          + (f"; missing: {''.join(missing)}" if missing else "")
          + (f"; rare (<20): {', '.join(rare)}" if rare else ""))
    print(f"Script written     : {config.PILOT_SCRIPT_PATH.relative_to(config.PROJECT_ROOT)}")
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors:
            print("  - " + e)
        return 1
    print("Checks             : OK (no evaluation overlap, no duplicates, all characters mappable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
