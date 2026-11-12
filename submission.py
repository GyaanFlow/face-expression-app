"""Build and validate the grader file: {"version": "linear-rank-v1", "weights": 10 x 156}."""
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

VERSION = "linear-rank-v1"
MAX_BYTES = 64 * 1024


def to_json(weights, digits=9):
    W = np.asarray(weights, dtype=float)
    if not np.isfinite(W).all():
        raise ValueError("weights contain NaN or infinity")
    rows = [[float(f"{v:.{digits}g}") for v in r] for r in W]
    return json.dumps({"version": VERSION, "weights": rows}, separators=(",", ":"))


class _Bad(ValueError):
    pass


def _strict_load(raw):
    """JSON parse that rejects what a browser JSON.parse would reject: NaN, Infinity, duplicate keys."""
    def no_const(name):
        raise _Bad(f"{name} is not allowed in JSON")

    def no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise _Bad("duplicate keys")
        return dict(pairs)

    return json.loads(raw, parse_constant=no_const, object_pairs_hook=no_dupes)


def validate(text, n_rows=10, n_cols=156):
    """Returns a list of (ok, message). Mirrors the rules on the submission page."""
    out = []
    raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    out.append((0 < len(raw) <= MAX_BYTES, f"size {len(raw) / 1024:.1f} KiB (limit 64 KiB)"))
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        obj = _strict_load(raw)
    except Exception as ex:
        out.append((False, f"not valid JSON ({str(ex)[:70]})"))
        return out
    out.append((True, "valid JSON"))
    if not isinstance(obj, dict):
        out.append((False, "top level must be an object"))
        return out
    out.append((set(obj.keys()) == {"version", "weights"}, f"keys are exactly version and weights (found {sorted(obj.keys())})"))
    out.append((obj.get("version") == VERSION, f'version is "{VERSION}"'))
    w = obj.get("weights")
    if not isinstance(w, list) or len(w) != n_rows:
        out.append((False, f"weights has {n_rows} rows (found {len(w) if isinstance(w, list) else 'none'})"))
        return out
    out.append((True, f"weights has {n_rows} rows"))
    shape_ok = all(isinstance(r, list) and len(r) == n_cols for r in w)
    out.append((shape_ok, f"every row has {n_cols} numbers"))
    if shape_ok:
        flat = [v for r in w for v in r]
        nums = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in flat)
        out.append((nums, "every entry is a number"))
        if nums:
            out.append((all(math.isfinite(v) for v in flat), "every number is finite"))
    return out


def roundtrip_ok(text, W, X=None):
    """Re-reads the JSON text and checks it ranks photos like the in-memory weights do."""
    try:
        W2 = np.array(_strict_load(text.encode("utf-8"))["weights"], dtype=float)
    except Exception:
        return False
    if W2.shape != np.asarray(W).shape:
        return False
    rng = np.random.RandomState(0)
    Xt = X if X is not None else rng.randn(10, W2.shape[1])
    a, b = Xt @ np.asarray(W).T, Xt @ W2.T
    ra, ca = linear_sum_assignment(-a)
    rb, cb = linear_sum_assignment(-b)
    return bool(np.array_equal(ca, cb)) and bool(np.allclose(a, b, rtol=1e-4, atol=1e-3 * (np.abs(a).max() + 1e-9)))


def all_ok(checks):
    return all(c[0] for c in checks)


def make_inputs(df, cols):
    raw = df[cols].to_numpy(float)
    rk = df[cols].rank(method="average").to_numpy()
    return np.hstack([raw, (rk - 1) / (len(df) - 1)])


def prepare_people(frames, cols, labels):
    """frames: list of dataframes with person_id, intended_expression and the feature columns.
    Drops people that are exact copies of an earlier person. Returns (X, y, g, info)."""
    seen, Xs, ys, gs, info = [], [], [], [], dict(listed=0, kept=0, dup=0, bad=0)
    lab = {l: i for i, l in enumerate(labels)}
    for fi in range(len(frames)):
        df = frames[fi]
        df = df.copy()
        df["person_id"] = df["person_id"].astype(str)
        df["intended_expression"] = df["intended_expression"].astype(str).str.strip().str.lower()
        for pid in list(dict.fromkeys(df["person_id"])):
            g = df[df["person_id"] == pid]
            info["listed"] += 1
            if len(g) != len(labels) or sorted(g["intended_expression"]) != sorted(labels):
                info["bad"] += 1
                continue
            sig = {hashlib.md5(np.round(r, 6).tobytes()).hexdigest() for r in g[cols].to_numpy(float)}
            if any(len(sig & s) >= 5 for s in seen):
                info["dup"] += 1
                continue
            seen.append(sig)
            g = g.reset_index(drop=True)
            Xs.append(make_inputs(g, cols))
            ys.append(g["intended_expression"].map(lab).to_numpy())
            gs.append(np.full(len(g), f"{fi}:{pid}"))
            info["kept"] += 1
    if not Xs:
        return None, None, None, info
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), info


def fit_ridge(X, y, alpha=1000.0):
    sc = StandardScaler().fit(X)
    W = Ridge(alpha=alpha).fit(sc.transform(X), np.eye(10)[y] - 0.1).coef_
    return W / sc.scale_


def match_acc(S, y, g):
    hit = 0
    for p in np.unique(g):
        k = np.where(g == p)[0]
        r, c = linear_sum_assignment(-S[k])
        hit += int((c == y[k][r]).sum())
    return hit / len(y)


def cv_ridge(X, y, g, alpha=1000.0, folds=5, repeats=3):
    people = np.unique(g)
    if len(people) < 4:
        raise ValueError("need at least 4 people for cross validation")
    nf = max(2, min(folds, len(people)))
    accs = []
    for r in range(repeats):
        rng = np.random.RandomState(r)
        ps = people.copy()
        rng.shuffle(ps)
        fold = {p: i % nf for i, p in enumerate(ps)}
        f = np.array([fold[p] for p in g])
        S = np.zeros((len(y), 10))
        for k in range(nf):
            tr, te = f != k, f == k
            S[te] = X[te] @ fit_ridge(X[tr], y[tr], alpha).T
        accs.append(match_acc(S, y, g))
    return float(np.mean(accs)), float(np.std(accs))
