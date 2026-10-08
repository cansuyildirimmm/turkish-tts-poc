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

Also used for the synthetic-data source sentences (group B):
    python src/recording_script.py --sentences recording/synthetic_sentences.txt         --out recording/synthetic_script.tsv --also-check recording/pilot_sentences.txt

Usage:
    python src/recording_script.py
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import config
import text_normalizer
import tts_engine
from evaluation_set import EvalLeakChecker, load_sentences

# Rough Turkish read-speech rate on normalized text, used only for estimates.
CHARS_PER_SECOND = 14.0
COMMA_PAUSE_S = 0.35  # readers pause at commas (e.g. between IBAN digit groups)
MAX_CLIP_S = 14.0  # FreyaTTS precompute_latents.py default --max_s
TURKISH_LETTERS = "abcçdefgğhıijklmnoöprsştuüvyz"


def main() -> int:
    p = argparse.ArgumentParser(description="Normalize and check a sentence list into a script TSV")
    p.add_argument("--sentences", type=Path, default=config.PILOT_SENTENCES_PATH)
    p.add_argument("--out", type=Path, default=config.PILOT_SCRIPT_PATH)
    p.add_argument("--also-check", type=Path, action="append", default=[],
                   help="other sentence file whose sentences and ids must not be duplicated")
    args = p.parse_args()

    with open(config.FREYATTS_VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)

    sentences = load_sentences(args.sentences)
    leak_checker = EvalLeakChecker()

    rows, errors = [], []
    seen: dict[str, str] = {}
    other_ids = set()
    for other in args.also_check:
        for s in load_sentences(other):
            seen[text_normalizer.normalize(s.text).lower()] = f"{other.name}:{s.sentence_id}"
            other_ids.add(s.sentence_id)
    letters = Counter()
    total_s = 0.0
    for s in sentences:
        read_text = text_normalizer.normalize(s.text)
        train_text = tts_engine.fit_to_vocab(read_text, vocab)
        key = read_text.lower()
        if s.sentence_id in other_ids:
            errors.append(f"{s.sentence_id}: id already used in --also-check file")

        leak = leak_checker.leak(read_text)
        if leak:
            errors.append(f"{s.sentence_id}: {leak}")
        if key in seen:
            errors.append(f"{s.sentence_id}: duplicate of {seen[key]}")
        seen[key] = s.sentence_id
        unmappable = tts_engine.unmappable_chars(read_text, vocab)
        if unmappable:
            errors.append(f"{s.sentence_id}: characters with no vocabulary mapping {sorted(unmappable)}: {read_text!r}")

        est_s = len(read_text) / CHARS_PER_SECOND + COMMA_PAUSE_S * read_text.count(",")
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

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
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
    print(f"Script written     : {args.out}")
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors:
            print("  - " + e)
        return 1
    print("Checks             : OK (no evaluation overlap, no duplicates, all characters mappable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
