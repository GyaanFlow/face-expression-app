"""Photo -> the 78 Py-Feat features used in training (51 blendshapes, 20 AUs, 7 emotions)."""
import logging
import threading

import numpy as np
import pandas as pd

log = logging.getLogger("extractor")
_lock = threading.Lock()
_det = None


def _detector():
    global _det
    if _det is None:
        from feat import Detectorv2
        log.info("loading Py-Feat Detectorv2 (first call downloads weights)")
        _det = Detectorv2(device="cpu")
    return _det


def _largest(fex):
    box = [c for c in fex.columns if c in ("FaceRectWidth", "FaceRectHeight")]
    if len(fex) == 1 or len(box) < 2:
        return fex.iloc[[0]]
    area = fex["FaceRectWidth"] * fex["FaceRectHeight"]
    return fex.iloc[[int(np.argmax(area.to_numpy()))]]


def extract(paths, wanted_cols):
    """Returns (df with wanted columns, list of problems). One row per path that had a face."""
    rows, problems = [], []
    with _lock:
        det = _detector()
        for i, p in enumerate(paths):
            try:
                fex = det.detect(p, data_type="image", progress_bar=False)
                fex = fex.dropna(subset=[c for c in fex.columns if c in ("FaceRectX",)]) if "FaceRectX" in fex.columns else fex
                if len(fex) == 0:
                    problems.append((i, "no face found"))
                    continue
                f = _largest(fex)
                parts = {}
                for pref, part in (("bs_", f.blendshapes), ("au_", f.aus), ("emotion_", f.emotions)):
                    for c in part.columns:
                        parts[pref + str(c)] = float(part[c].iloc[0])
                row = {c: parts.get(c) for c in wanted_cols}
                miss = [c for c, v in row.items() if v is None]
                if miss:
                    raise KeyError(f"detector did not return {miss[:3]}")
                row["_idx"] = i
                rows.append(row)
            except Exception as ex:
                log.exception("photo %s failed", i)
                problems.append((i, f"{type(ex).__name__}: {str(ex)[:80]}"))
    return pd.DataFrame(rows), problems
