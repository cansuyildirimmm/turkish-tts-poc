"""Reader for evaluation/sentences.txt (format: id|category|text)."""

from dataclasses import dataclass
from pathlib import Path

import config


@dataclass(frozen=True)
class EvalSentence:
    sentence_id: str
    category: str
    text: str


def _words(text: str) -> list[str]:
    return [w.strip(".,;:!?'\"").lower() for w in text.split() if w.strip(".,;:!?'\"")]


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


class EvalLeakChecker:
    """Detects training text that leaks the unseen evaluation set.

    Compares normalized text: an identical sentence, or any shared run of
    `n` consecutive words, counts as a leak.
    """

    def __init__(self, n: int = 5, path: Path = config.EVAL_SENTENCES_PATH):
        import text_normalizer

        self._normalize = text_normalizer.normalize
        self.n = n
        normalized = [self._normalize(s.text) for s in load_sentences(path)]
        self._texts = {t.lower() for t in normalized}
        self._grams = set().union(*(ngrams(t, n) for t in normalized))

    def leak(self, text: str) -> str | None:
        """Return a reason string if `text` leaks the evaluation set, else None."""
        t = self._normalize(text)
        if t.lower() in self._texts:
            return "identical to an evaluation sentence"
        shared = ngrams(t, self.n) & self._grams
        if shared:
            return f"shares '{' '.join(sorted(shared)[0])}' with the evaluation set"
        return None


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
