from pathlib import Path

from check_data_leak import MAX_BYTES, check, main


def rules(root: Path, files: list[str]) -> set[str]:
    return {v.rule for v in check(root, files)}


def write(root: Path, rel: str, text: str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return rel


def test_rule_a_tracked_path_under_data_dir(tmp_path):
    rel = write(tmp_path, "data/raw/notes.txt", "x")
    assert rules(tmp_path, [rel]) == {"data-dir"}


def test_rule_b_oversize_data_file(tmp_path):
    rel = "results/big.json"
    (tmp_path / "results").mkdir()
    (tmp_path / rel).write_bytes(b" " * (MAX_BYTES + 1))
    assert rules(tmp_path, [rel]) == {"oversize"}


def test_rule_b_small_data_file_passes(tmp_path):
    rel = write(tmp_path, "results/small.json", "{}")
    assert rules(tmp_path, [rel]) == set()


def test_rule_c_csv_with_text_columns(tmp_path):
    rel = write(tmp_path, "samples/train.csv", "id,prompt,response_a,response_b\n1,a,b,c\n")
    assert rules(tmp_path, [rel]) == {"text-columns-csv"}


def test_rule_c_tsv_with_text_column(tmp_path):
    rel = write(tmp_path, "samples/train.tsv", "id\tresponse_a\n1\tb\n")
    assert rules(tmp_path, [rel]) == {"text-columns-csv"}


def test_rule_d_jsonl_with_text_keys(tmp_path):
    rel = write(tmp_path, "samples/rows.jsonl", '{"id": 1, "prompt": "hi"}\n')
    assert rules(tmp_path, [rel]) == {"text-columns-jsonl"}


def test_rule_d_skips_unparseable_lines_to_first_record(tmp_path):
    rel = write(tmp_path, "samples/rows.jsonl", 'not json\n{"response_b": "x"}\n')
    assert rules(tmp_path, [rel]) == {"text-columns-jsonl"}


def test_rule_e_prediction_file_allowed_any_order(tmp_path):
    rel = write(
        tmp_path, "predictions/p.csv", "winner_tie,id,winner_model_b,winner_model_a\n1,0.3,0.3,0.4\n"
    )
    assert rules(tmp_path, [rel]) == set()


def test_rule_e_prediction_header_plus_text_column_not_exempt(tmp_path):
    rel = write(
        tmp_path,
        "predictions/p.csv",
        "id,winner_model_a,winner_model_b,winner_tie,prompt\n1,0.3,0.3,0.4,hi\n",
    )
    assert rules(tmp_path, [rel]) == {"text-columns-csv"}


def test_rule_e_oversize_prediction_file_still_flagged(tmp_path):
    header = "id,winner_model_a,winner_model_b,winner_tie\n"
    rel = write(tmp_path, "predictions/p.csv", header + "1" * (MAX_BYTES + 1))
    assert rules(tmp_path, [rel]) == {"oversize"}


def test_clean_repo_passes(tmp_path, capsys):
    files = [
        write(tmp_path, "README.md", "hello"),
        write(tmp_path, "results/ledger.jsonl", ""),
        write(tmp_path, "predictions/p.csv", "id,winner_model_a,winner_model_b,winner_tie\n"),
    ]
    assert check(tmp_path, files) == []
    assert main(["--root", str(tmp_path)], files=files) == 0
    assert "ok: 3 tracked files checked" in capsys.readouterr().out


def test_main_nonzero_on_violation(tmp_path, capsys):
    files = [write(tmp_path, "data/x.csv", "a\n")]
    assert main(["--root", str(tmp_path)], files=files) == 1
    assert "data-dir" in capsys.readouterr().out
