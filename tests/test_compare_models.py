import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import compare_models as cm  # noqa: E402
from evaluation_set import EvalSentence  # noqa: E402

SENTENCES = [EvalSentence("01", "erp", "KDV %20'dir."), EvalSentence("02", "normal", "Tamam.")]
TEXTS = {"01": "Ka de ve yüzde yirmidir.", "02": "Tamam."}


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_results_csv_has_spec_columns_first(tmp_path):
    results = {"base": {"01": {"audio": "comparison/base/01.wav", "time": 1.234, "duration": 2.0, "drop": 6.25},
                        "02": {"audio": "comparison/base/02.wav", "time": 0.5, "duration": 1.0, "drop": None}}}
    rows = cm.build_rows(SENTENCES, TEXTS, results)
    cm.write_csv(tmp_path / "results.csv", cm.RESULT_FIELDS, rows)
    out = read(tmp_path / "results.csv")
    assert list(out[0].keys())[:6] == ["sentence_id", "text", "base_audio", "fine_tuned_audio",
                                       "base_inference_time", "fine_tuned_inference_time"]
    assert out[0]["text"] == "KDV %20'dir."          # original text, not the normalized one
    assert out[0]["tts_text"] == "Ka de ve yüzde yirmidir."
    assert out[0]["base_inference_time"] == "1.23" and out[0]["base_pitch_drop_st"] == "6.2"
    assert out[1]["base_pitch_drop_st"] == ""        # nothing voiced
    assert out[0]["fine_tuned_audio"] == "" and out[0]["fine_tuned_inference_time"] == ""  # base-only run


def test_manual_template_created_once(tmp_path):
    path = tmp_path / "manual_evaluation.csv"
    assert cm.ensure_manual_template(SENTENCES, path) is True
    rows = read(path)
    assert list(rows[0].keys()) == cm.MANUAL_FIELDS
    assert [r["sentence_id"] for r in rows] == ["01", "02"]
    assert all(r["base_naturalness"] == "" for r in rows)

    # the user fills in scores; a later run must not touch them
    rows[0]["base_naturalness"] = "4"
    cm.write_csv(path, cm.MANUAL_FIELDS, rows)
    assert cm.ensure_manual_template(SENTENCES, path) is False
    assert read(path)[0]["base_naturalness"] == "4"
