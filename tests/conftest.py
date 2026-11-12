import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LABELS = ["neutral", "happy", "sad", "angry", "surprised", "contempt", "restrained-amusement", "scepticism", "uncertainty", "hopelessness"]
COLS = [f"bs_f{i}" for i in range(51)] + [f"au_AU{i}" for i in range(20)] + [f"emotion_e{i}" for i in range(7)]


@pytest.fixture()
def model_dir(tmp_path):
    rng = np.random.RandomState(0)
    W = rng.randn(10, 156).astype("float32")
    save_file({"weights": W}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(dict(labels=LABELS, feature_columns=COLS, rank_mode="minmax", cv_accuracy=0.69, n_people=58)))
    return tmp_path


@pytest.fixture()
def person_df():
    rng = np.random.RandomState(1)
    import pandas as pd
    df = pd.DataFrame(rng.rand(10, 78), columns=COLS)
    df["person_id"] = "01"
    df["photo_id"] = [f"01-{i:02d}" for i in range(10)]
    df["intended_expression"] = LABELS
    return df
