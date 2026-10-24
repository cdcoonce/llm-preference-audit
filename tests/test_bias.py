import numpy as np
import pytest

from llm_preference_audit.bias import identical_rows_readout, longer_wins_readout


def test_longer_wins_readout():
    y = np.array([0, 1, 2, 0, 1])
    wa = np.array([10, 10, 5, 3, 7])
    wb = np.array([5, 20, 9, 3, 2])  # row 2 is a tie label, row 3 has equal lengths
    P = np.array(
        [[0.6, 0.2, 0.2], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4], [0.5, 0.3, 0.2], [0.6, 0.3, 0.1]]
    )
    r = longer_wins_readout(y, P, wa, wb)
    assert r["n"] == 3
    assert r["human_longer_share"] == pytest.approx(2 / 3)  # row 4: b won but a is longer
    assert r["model_longer_share"] == pytest.approx(1.0)


def test_identical_rows_readout():
    P = np.array([[0.4, 0.4, 0.2], [0.5, 0.3, 0.2], [0.3, 0.5, 0.2]])
    r = identical_rows_readout(P, np.array([True, False, False]))
    assert r["n"] == 1 and r["share_favours_a"] == 0.0 and r["max_abs_p_a_minus_p_b"] == 0.0
    assert identical_rows_readout(P, np.zeros(3, bool))["n"] == 0
