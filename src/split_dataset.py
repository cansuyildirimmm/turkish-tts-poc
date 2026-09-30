"""Reproducible train / validation / test split of a validated TTS dataset.

Only records that pass validate_dataset.py (no errors) are used; validation is
re-run here so the split never relies on a stale report. Leakage guards:
  - recordings of the same transcript (re-takes) always land in the same split
  - any text leaking the evaluation set aborts the split
Within each script category, transcript groups are ordered by a seeded hash
and the first ones go to test, then validation. The result is deterministic
for a given seed and stays mostly stable when new recordings are added.

Outputs (in <dataset-dir>/splits/):
    train.jsonl, val.jsonl, test.jsonl   {"id", "audio", "text"} per line, the
                                         manifest format FreyaTTS expects;
                                         audio paths are relative to the project root
    split_info.json                      seed, ratios, counts, durations, ids

Usage:
    python src/split_dataset.py
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import config
import validate_dataset
from evaluation_set import EvalLeakChecker

SPLITS = ("train", "val", "test")


def _order_key(seed: int, group: str) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()


def _n_holdout(n_groups: int, ratio: float) -> int:
    # At least one held-out group per category once the category can spare it.
    if n_groups < 3 or ratio <= 0:
        return 0
    return max(1, round(n_groups * ratio))


def assign_splits(records: list[dict], categories: dict[str, str], seed: int,
                  val_ratio: float, test_ratio: float) -> dict[str, list[dict]]:
    """Group records by transcript, stratify by category, assign groups to splits."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["text"].lower()].append(r)

    by_category: dict[str, list[str]] = defaultdict(list)
    for key, recs in groups.items():
        by_category[categories.get(recs[0]["id"], "unknown")].append(key)

    out = {name: [] for name in SPLITS}
    for cat in sorted(by_category):
        keys = sorted(by_category[cat], key=lambda k: _order_key(seed, k))
        n_test = _n_holdout(len(keys), test_ratio)
        n_val = _n_holdout(len(keys) - n_test, val_ratio)
        for i, key in enumerate(keys):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            out[split].extend(groups[key])
    for name in SPLITS:
        out[name].sort(key=lambda r: r["id"])
    return out


def _load_categories(script_path: Path) -> dict[str, str]:
    with open(script_path, encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r.get("category", "unknown") for r in csv.DictReader(f, delimiter="\t")}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_split(dataset_dir: Path, script_path: Path, seed: int = config.SEED,
                val_ratio: float = 0.05, test_ratio: float = 0.05) -> dict:
    report = validate_dataset.validate(dataset_dir, script_path)
    accepted = [r for r in report["records"] if not r["errors"]]
    if not accepted:
        raise ValueError("no records passed validation; nothing to split")

    leak_checker = EvalLeakChecker()
    leaks = [r["id"] for r in accepted if leak_checker.leak(r["text"])]
    if leaks:  # validation already rejects these; this guards against future changes
        raise ValueError(f"evaluation-set leakage in accepted records: {leaks}")

    splits = assign_splits(accepted, _load_categories(script_path), seed, val_ratio, test_ratio)

    def summary(recs):
        cats = defaultdict(int)
        cat_map = _load_categories(script_path)
        for r in recs:
            cats[cat_map.get(r["id"], "unknown")] += 1
        return {"records": len(recs), "minutes": round(sum(r["duration_s"] for r in recs) / 60, 2),
                "categories": dict(sorted(cats.items()))}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "ratios": {"val": val_ratio, "test": test_ratio},
        "method": "group by transcript, stratify by category, seeded sha256 ordering",
        "dataset_dir": str(dataset_dir),
        "script": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "validation": {k: report["summary"][k] for k in ("audio_files", "accepted", "rejected", "missing_audio")},
        "rejected_ids": report["rejected_ids"],
        "summary": {name: summary(recs) for name, recs in splits.items()},
        "ids": {name: [r["id"] for r in recs] for name, recs in splits.items()},
        "_records": splits,
    }


def _manifest_path(dataset_dir: Path, record: dict) -> str:
    audio = (dataset_dir / record["audio"]).resolve()
    try:
        return audio.relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:  # dataset outside the project (e.g. tests): keep it usable
        return audio.as_posix()


def write_split(split: dict, dataset_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in split["_records"].items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps({"id": r["id"], "audio": _manifest_path(dataset_dir, r), "text": r["text"]},
                                   ensure_ascii=False) + "\n")
    info = {k: v for k, v in split.items() if k != "_records"}
    with open(out_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main() -> int:
    p = argparse.ArgumentParser(description="Reproducible train/val/test split")
    p.add_argument("--dataset-dir", type=Path, default=config.PILOT_DATASET_DIR)
    p.add_argument("--script", type=Path, default=config.PILOT_SCRIPT_PATH)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not (args.dataset_dir / "wavs").is_dir():
        print(f"error: {args.dataset_dir / 'wavs'} not found (see docs/recording_guide.md)", file=sys.stderr)
        return 2
    if not (0 <= args.val_ratio < 1 and 0 <= args.test_ratio < 1 and args.val_ratio + args.test_ratio < 1):
        print("error: ratios must be in [0, 1) and sum to < 1", file=sys.stderr)
        return 2

    split = build_split(args.dataset_dir, args.script, args.seed, args.val_ratio, args.test_ratio)
    out_dir = args.dataset_dir / "splits"
    write_split(split, args.dataset_dir, out_dir)

    v = split["validation"]
    print(f"Validation : {v['accepted']} accepted, {v['rejected']} rejected, {v['missing_audio']} missing audio")
    print(f"Seed       : {split['seed']}  ratios val={args.val_ratio} test={args.test_ratio}")
    for name in SPLITS:
        s = split["summary"][name]
        print(f"{name:<6}: {s['records']:4d} records, {s['minutes']:6.2f} min  {s['categories']}")
    print(f"Written    : {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
