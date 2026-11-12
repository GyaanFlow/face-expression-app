"""Photo -> the 78 Py-Feat features used in training (51 blendshapes, 20 AUs, 7 emotions)."""
import logging
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("extractor")
_lock = threading.Lock()
_det = None


def _writable_resources():
    """Py-Feat downloads model weights into its own package folder, which is read only on
    Streamlit Cloud. Point it at a writable copy in the temp dir instead."""
    import feat
    import feat.utils.io as fio

    src = Path(feat.__path__[0]) / "resources"
    if os.access(src, os.W_OK):
        return
    dst = Path(os.environ.get("FEAT_RESOURCE_DIR") or tempfile.gettempdir()) / "feat_resources"
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    path = str(dst)

    def get_resource_path():
        return path

    fio.get_resource_path = get_resource_path
    for name, mod in list(sys.modules.items()):
        if name.startswith("feat") and hasattr(mod, "get_resource_path"):
            mod.get_resource_path = get_resource_path
    log.info("py-feat resources redirected to %s", path)


def _detector():
    global _det
    if _det is None:
        _writable_resources()
        from feat import Detectorv2
        _writable_resources()
        log.info("loading Py-Feat Detectorv2 (first call downloads weights)")
        # identity_model=None skips the face-identity network, which we do not use
        _det = Detectorv2(device="cpu", identity_model=None)
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
