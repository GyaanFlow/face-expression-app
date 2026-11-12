import json

import numpy as np
import pandas as pd

import submission as sub
from conftest import COLS, LABELS


def good(w=None):
    return sub.to_json(np.random.RandomState(2).randn(10, 156) if w is None else w)


def failed(checks):
    return [m for ok, m in checks if not ok]


def test_valid_file_passes_and_is_small():
    t = good()
    assert not failed(sub.validate(t))
    assert len(t.encode()) < 64 * 1024


def test_rejects_wrong_keys_version_and_shape():
    assert failed(sub.validate('{"version":"linear-rank-v1"}'))
    assert failed(sub.validate('{"version":"v2","weights":[]}'))
    assert failed(sub.validate(json.dumps({"version": "linear-rank-v1", "weights": [[0.0] * 155] * 10})))
    assert failed(sub.validate(json.dumps({"version": "linear-rank-v1", "weights": [[0.0] * 156] * 9})))
    assert failed(sub.validate(json.dumps({"version": "linear-rank-v1", "weights": [[0.0] * 156] * 10, "extra": 1})))


def test_rejects_non_standard_json():
    row = ",".join(["0"] * 155)
    assert failed(sub.validate('{"version":"linear-rank-v1","weights":[[NaN,%s]%s]}' % (row, ",[" + row + ",0]" * 9)))
    assert failed(sub.validate('{"version":"linear-rank-v1","weights":[[Infinity,%s]]}' % row))
    assert failed(sub.validate('{"version":"linear-rank-v1","version":"linear-rank-v1","weights":[]}'))
    assert failed(sub.validate("not json"))
    assert failed(sub.validate(""))
    assert failed(sub.validate("[1,2,3]"))


def test_rejects_non_numbers_and_bools():
    w = [[0.0] * 156 for _ in range(10)]
    w[3][4] = "1"
    assert failed(sub.validate(json.dumps({"version": "linear-rank-v1", "weights": w})))
    w[3][4] = True
    assert failed(sub.validate(json.dumps({"version": "linear-rank-v1", "weights": w})))
    w[3][4] = None
    assert failed(sub.validate(json.dumps({"version": "linear-rank-v1", "weights": w})))


def test_rejects_oversize():
    w = [[1.2345678901234567e-300] * 156 for _ in range(10)]
    big = json.dumps({"version": "linear-rank-v1", "weights": w}, indent=8)
    assert len(big.encode()) > 64 * 1024
    assert failed(sub.validate(big))


def test_accepts_bom():
    assert not failed(sub.validate(b"\xef\xbb\xbf" + good().encode()))


def test_to_json_refuses_nan():
    w = np.zeros((10, 156))
    w[0, 0] = np.nan
    try:
        sub.to_json(w)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_roundtrip_detects_changes():
    W = np.random.RandomState(3).randn(10, 156)
    t = sub.to_json(W)
    assert sub.roundtrip_ok(t, W)
    assert not sub.roundtrip_ok(sub.to_json(W[::-1]), W)
    assert not sub.roundtrip_ok("garbage", W)


def make_people(n, seed=0, label_case=False):
    rng = np.random.RandomState(seed)
    frames = []
    protos = rng.randn(10, 78)
    for p in range(n):
        x = protos + 0.4 * rng.randn(10, 78)
        df = pd.DataFrame(x, columns=COLS)
        df["person_id"] = f"{p:02d}"
        df["intended_expression"] = [l.upper() if label_case else l for l in rng.permutation(LABELS)]
        # make the signal depend on the label so training can learn something
        for i in range(10):
            df.loc[i, COLS] = protos[LABELS.index(df.loc[i, "intended_expression"].lower())] + 0.4 * rng.randn(78)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_prepare_people_dedupes_and_skips_bad():
    a = make_people(12)
    b = a[a["person_id"].isin(["00", "01"])].copy()
    b["person_id"] = ["x0"] * 10 + ["x1"] * 10
    broken = make_people(1, seed=9)
    broken = broken.iloc[:8]
    broken["person_id"] = "bad"
    X, y, g, info = sub.prepare_people([a, pd.concat([b, broken])], COLS, LABELS)
    assert info["dup"] == 2 and info["bad"] == 1 and info["kept"] == 12
    assert X.shape == (120, 156) and set(y) == set(range(10))


def test_prepare_people_labels_are_case_insensitive():
    X, y, g, info = sub.prepare_people([make_people(5, label_case=True)], COLS, LABELS)
    assert info["kept"] == 5


def test_training_learns_and_produces_valid_json():
    X, y, g, info = sub.prepare_people([make_people(30)], COLS, LABELS)
    m, sd = sub.cv_ridge(X, y, g, 1000.0, folds=5, repeats=2)
    assert m > 0.5
    W = sub.fit_ridge(X, y)
    t = sub.to_json(W)
    assert not failed(sub.validate(t)) and sub.roundtrip_ok(t, W, X[:10])


def test_cv_needs_enough_people():
    X, y, g, info = sub.prepare_people([make_people(3)], COLS, LABELS)
    try:
        sub.cv_ridge(X, y, g)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
