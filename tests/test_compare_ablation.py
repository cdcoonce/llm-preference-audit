import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compare_ablation", ROOT / "kaggle/rung5_deberta/compare_ablation.py"
)
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)


def _frames():
    rng = np.random.default_rng(0)
    n = 400
    rows = pd.DataFrame(
        {
            "id": np.arange(n),
            "y": rng.integers(0, 3, n),
            "fold": np.arange(n) % 2,
            "group": np.arange(n) // 2,
            "words_a": rng.integers(1, 200, n),
            "words_b": rng.integers(1, 200, n),
        }
    )
    P = rng.dirichlet([2, 2, 2], n)
    main = pd.DataFrame(P, columns=ca.PROB).assign(id=rows["id"])
    return rows, main


def test_compare_fold_identical_runs_have_zero_diff():
    rows, main = _frames()
    out = ca.compare_fold(main, main[main["id"] % 2 == 0], rows, 0, n_boot=200)
    assert out["n"] == 200 and out["log_loss_main"] == out["log_loss_ablation"]
    assert out["main_better_by"]["diff"] == 0
    assert set(out["verbosity_main"]) == set(out["verbosity_ablation"])


def test_compare_fold_better_main_has_positive_diff_and_checks_ids():
    rows, main = _frames()
    y = rows["y"].to_numpy()
    sharp = np.full((len(rows), 3), 0.1)
    sharp[np.arange(len(rows)), y] = 0.8
    main = pd.DataFrame(sharp, columns=ca.PROB).assign(id=rows["id"])
    abl = main[main["id"] % 2 == 0].copy()
    abl[ca.PROB] = 1 / 3
    out = ca.compare_fold(main, abl, rows, 0, n_boot=200)
    assert out["log_loss_main"] < out["log_loss_ablation"]
    assert out["main_better_by"]["lo"] > 0
    with pytest.raises(ValueError, match="exactly"):
        ca.compare_fold(main, abl.iloc[:-1], rows, 0, n_boot=50)
