import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from conftest import COLS, LABELS
from matcher import ExpressionMatcher


def test_match_is_one_to_one(model_dir, person_df):
    m = ExpressionMatcher(str(model_dir))
    pred, s, p = m.match(person_df)
    assert sorted(pred.tolist()) == list(range(10))
    assert s.shape == (10, 10) and np.allclose(p.sum(axis=1), 1)


def test_match_ignores_row_order_of_photos(model_dir, person_df):
    m = ExpressionMatcher(str(model_dir))
    pred, _, _ = m.match(person_df)
    perm = np.random.RandomState(4).permutation(10)
    pred2, _, _ = m.match(person_df.iloc[perm].reset_index(drop=True))
    assert pred2.tolist() == pred[perm].tolist()


def test_needs_ten_photos(model_dir, person_df):
    with pytest.raises(ValueError):
        ExpressionMatcher(str(model_dir)).match(person_df.iloc[:9])


def test_rejects_nan_features(model_dir, person_df):
    person_df.loc[0, COLS[0]] = np.nan
    with pytest.raises(ValueError):
        ExpressionMatcher(str(model_dir)).match(person_df)


def test_bad_model_files_fail_loudly(tmp_path):
    W = np.zeros((10, 156), dtype="float32")
    W[0, 0] = np.nan
    save_file({"weights": W}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(dict(labels=LABELS, feature_columns=COLS, rank_mode="minmax")))
    with pytest.raises(ValueError):
        ExpressionMatcher(str(tmp_path))
    save_file({"weights": np.zeros((9, 156), dtype="float32")}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError):
        ExpressionMatcher(str(tmp_path))


def test_score_matches_grader_formula(model_dir, person_df):
    m = ExpressionMatcher(str(model_dir))
    raw = person_df[COLS].to_numpy(float)
    rk = person_df[COLS].rank(method="average").to_numpy()
    rk = (rk - 1) / 9
    expected = np.hstack([raw, rk]) @ m.w.T
    _, s, _ = m.match(person_df)
    assert np.allclose(s, expected)
