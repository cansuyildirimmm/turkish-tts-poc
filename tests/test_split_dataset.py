import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import split_dataset as sd  # noqa: E402
from audio_helpers import write_wav  # noqa: E402

PER_CATEGORY = {"general": 20, "erp": 15, "number": 10, "short": 10, "question": 5}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    if not config.FREYATTS_VOCAB_PATH.is_file():
        pytest.skip("FreyaTTS source not installed (see README)")
    with open(config.PILOT_SCRIPT_PATH, encoding="utf-8-sig", newline="") as f:
        script = list(csv.DictReader(f, delimiter="\t"))
    rows = []
    for cat, n in PER_CATEGORY.items():
        rows += [r for r in script if r["category"] == cat][:n]

    root = tmp_path_factory.mktemp("split_ds")
    (root / "wavs").mkdir()
    entries = [(r["id"], r["category"], r["train_text"]) for r in rows]
    entries.append(("g001b", "general", rows[0]["train_text"]))   # re-take of g001
    entries.append(("clipped", "general", "Bu kayıt kırpılmış olduğu için reddedilmeli."))
    with open(root / "script.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "category", "train_text"])
        w.writerows(entries)
    for i, (rid, _, text) in enumerate(entries):
        letters = sum(ch.isalpha() for ch in text)
        write_wav(root / "wavs" / f"{rid}.wav", speech_s=max(0.6, letters / 12), seed=i, clip=(rid == "clipped"))
    return root


@pytest.fixture(scope="module")
def split(dataset):
    return sd.build_split(dataset, dataset / "script.tsv", seed=42)


def test_disjoint_and_complete(split):
    ids = {name: set(split["ids"][name]) for name in sd.SPLITS}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    total = ids["train"] | ids["val"] | ids["test"]
    assert len(total) == split["validation"]["accepted"] == sum(PER_CATEGORY.values()) + 1


def test_rejected_records_excluded(split):
    assert "clipped" in split["rejected_ids"]
    assert all("clipped" not in split["ids"][name] for name in sd.SPLITS)


def test_retakes_share_a_split(split):
    home = [name for name in sd.SPLITS if "g001" in split["ids"][name]]
    assert home and "g001b" in split["ids"][home[0]]


def test_every_category_held_out(split):
    for cat, n in PER_CATEGORY.items():
        if n >= 3:
            assert split["summary"]["test"]["categories"].get(cat, 0) >= 1, cat
        if n >= 4:
            assert split["summary"]["val"]["categories"].get(cat, 0) >= 1, cat
    assert split["summary"]["train"]["records"] > split["summary"]["val"]["records"] + split["summary"]["test"]["records"]


def test_deterministic(dataset, split):
    again = sd.build_split(dataset, dataset / "script.tsv", seed=42)
    assert again["ids"] == split["ids"]
    other = sd.build_split(dataset, dataset / "script.tsv", seed=7)
    assert other["ids"] != split["ids"]


def test_stable_when_data_grows():
    recs = [{"id": f"x{i:03d}", "text": f"cümle {i}"} for i in range(200)]
    cats = {r["id"]: "general" for r in recs}
    small = sd.assign_splits(recs[:100], cats, 42, 0.05, 0.05)
    big = sd.assign_splits(recs, cats, 42, 0.05, 0.05)
    moved = {r["id"] for r in small["train"]} - {r["id"] for r in big["train"]}
    # Growth may pull a few train items into the larger holdouts, never the reverse.
    assert len(moved) <= 10
    assert not ({r["id"] for r in small["test"]} - {r["id"] for r in big["test"] + big["val"]})


def test_written_manifests(dataset, split, tmp_path):
    sd.write_split(split, dataset, tmp_path)
    for name in sd.SPLITS:
        lines = (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(split["ids"][name])
        for line in lines:
            item = json.loads(line)
            assert set(item) == {"id", "audio", "text"}
            assert Path(item["audio"]).is_file()
            assert item["text"]
    info = json.loads((tmp_path / "split_info.json").read_text(encoding="utf-8"))
    assert info["seed"] == 42 and "_records" not in info
    assert info["ids"] == split["ids"]
