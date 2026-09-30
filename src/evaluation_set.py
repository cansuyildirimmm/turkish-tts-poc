"""Reader for evaluation/sentences.txt (format: id|category|text)."""

from dataclasses import dataclass
from pathlib import Path

import config


@dataclass(frozen=True)
class EvalSentence:
    sentence_id: str
    category: str
    text: str


def load_sentences(path: Path = config.EVAL_SENTENCES_PATH) -> list[EvalSentence]:
    sentences = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 2)
            if len(parts) != 3 or not all(p.strip() for p in parts):
                raise ValueError(f"{path.name}:{lineno}: expected 'id|category|text', got {line!r}")
            sid, category, text = (p.strip() for p in parts)
            if sid in seen:
                raise ValueError(f"{path.name}:{lineno}: duplicate id {sid}")
            seen.add(sid)
            sentences.append(EvalSentence(sid, category, text))
    return sentences
