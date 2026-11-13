"""Loads the linear-rank-v1 weights from the Hub and matches expressions to photos."""
import json
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file
from scipy.optimize import linear_sum_assignment

log = logging.getLogger("matcher")


class ExpressionMatcher:
    def __init__(self, repo_or_dir):
        p = Path(repo_or_dir)
        if not p.is_dir():
            p = Path(snapshot_download(repo_or_dir, allow_patterns=["config.json", "model.safetensors", "metrics.json", "reports/*.csv"],
                                       token=os.environ.get("HF_TOKEN") or None))
        self.cfg = json.loads((p / "config.json").read_text())
        self.w = load_file(str(p / "model.safetensors"))["weights"].astype(np.float64)
        self.cols = list(self.cfg["feature_columns"])
        self.labels = list(self.cfg["labels"])
        self.cv = self.cfg.get("cv_accuracy")
        self.n_people = self.cfg.get("n_people")
        self.dir = p
        self.chosen = self.cfg.get("chosen", {}) or {}
        self.trained = self.cfg.get("trained")
        self.metrics = {}
        self.reports = {}
        try:
            if (p / "metrics.json").exists():
                self.metrics = json.loads((p / "metrics.json").read_text())
        except Exception:
            log.warning("metrics.json unreadable, model insights will be limited")
        for name in ("leaderboard", "confusion", "per_class"):
            f = p / "reports" / f"{name}.csv"
            if f.exists():
                try:
                    self.reports[name] = pd.read_csv(f)
                except Exception:
                    pass
        if self.w.shape != (len(self.labels), 2 * len(self.cols)):
            raise ValueError(f"weights {self.w.shape} do not match config")
        if not np.isfinite(self.w).all():
            raise ValueError("model weights contain NaN or infinity")
        if len(set(self.cols)) != len(self.cols) or len(self.labels) != 10:
            raise ValueError("config has duplicate feature columns or the wrong number of labels")
        log.info("model loaded: %s, cv %s", self.cfg.get("chosen", {}).get("name"), self.cv)

    def inputs(self, df):
        raw = df[self.cols].to_numpy(float)
        rk = df[self.cols].rank(method="average").to_numpy()
        if self.cfg.get("rank_mode") == "minmax":
            rk = (rk - 1) / (len(df) - 1)
        else:
            rk = rk / len(df)
        return np.hstack([raw, rk])

    def match(self, df):
        """df holds the 10 photos of one person. Returns (labels per photo, score matrix, probs)."""
        if len(df) != len(self.labels):
            raise ValueError(f"need exactly {len(self.labels)} photos of one person, got {len(df)}")
        if not np.isfinite(df[self.cols].to_numpy(float)).all():
            raise ValueError("feature values must be finite numbers")
        s = self.inputs(df) @ self.w.T
        r, c = linear_sum_assignment(-s)
        pred = np.empty(len(df), dtype=int)
        pred[r] = c
        # confidence: softmax of each photo's scores after removing the photo-wide offset
        z = s - s.mean(axis=1, keepdims=True)
        z = z / (z.std() + 1e-9)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return pred, s, p
