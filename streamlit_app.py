import io
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Face Expression Matcher", page_icon="🎭", layout="wide")

try:
    for k in ("HF_TOKEN", "MODEL_REPO"):
        if k in st.secrets:
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass

from matcher import ExpressionMatcher
from ui import CSS, cards_html, heatmap_html, thumb_uri

logging.basicConfig(level=logging.INFO)
REPO = os.environ.get("MODEL_REPO", "23f1000805/face-expression-v3")
EX = Path(__file__).parent / "examples"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


@st.cache_resource(show_spinner="Loading model from the Hub...")
def get_model():
    return ExpressionMatcher(REPO)


try:
    M = get_model()
except Exception as ex:
    st.error(f"Could not load the model from {REPO}. If the repo is private, add HF_TOKEN in the app secrets. ({type(ex).__name__}: {str(ex)[:150]})")
    st.stop()
N = len(M.labels)
acc = (M.cv or 0) * 100

with st.sidebar:
    st.markdown("### Model")
    st.markdown(f"[{REPO}](https://huggingface.co/{REPO})")
    st.metric("Held-out accuracy", f"{acc:.1f}%", help="Matching accuracy on people the model never saw. Chance is 10%.")
    st.caption(f"{M.n_people} unique people, {N} expressions, {len(M.cols)} features")
    st.divider()
    st.caption("Photos are processed in memory and deleted after each run. Nothing is stored.")
    st.caption("Hardest pairs: sad vs hopelessness, scepticism vs uncertainty.")

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f"""<div class="hero"><h1>Face Expression Matcher</h1>
<p>Give it ten photos of one person and it works out which of ten expressions each photo shows.</p>
<div class="pills"><span>10 expressions</span><span>Py-Feat features</span><span>Linear model from the Hub</span>
<span>Held-out accuracy {acc:.0f}%</span></div></div>""", unsafe_allow_html=True)


def example_sets():
    return sorted(d.name for d in EX.iterdir() if d.is_dir()) if EX.exists() else []


def example_csvs():
    return sorted(p.name for p in EX.glob("*.csv")) if EX.exists() else []


def show_result(title, names, pred, p, s, uris, truths):
    labels = [M.labels[k] for k in pred]
    confs = [float(p[i, pred[i]]) for i in range(len(pred))]
    st.markdown(f"### {title}")
    if truths:
        hit = sum(1 for a, b in zip(labels, truths) if a == b)
        st.markdown(f'<div class="acc">{hit * 100 // len(labels)}% of this person\'s photos match their label</div>', unsafe_allow_html=True)
    st.markdown(cards_html(names, labels, confs, uris, truths), unsafe_allow_html=True)
    with st.expander("Score of every photo for every expression"):
        st.markdown(heatmap_html(s, names, M.labels), unsafe_allow_html=True)
    return labels, confs


tab_photo, tab_csv, tab_about = st.tabs(["Test with photos", "Test with features csv", "About the model"])

# photos
MAX_SIDE = 1024
MAX_MB = 15


def prepare(src):
    """Decode, fix rotation, shrink. Returns (md5, path, thumb_uri) or raises ValueError with a readable message."""
    import hashlib
    import tempfile
    from PIL import Image, ImageOps, UnidentifiedImageError
    if isinstance(src, str):
        raw = Path(src).read_bytes()
        name = Path(src).name
    else:
        raw = src.getvalue()
        name = src.name
    if len(raw) > MAX_MB * 1024 * 1024:
        raise ValueError(f"{name} is larger than {MAX_MB} MB")
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise ValueError(f"{name} is not a readable image")
    if min(im.size) < 80:
        raise ValueError(f"{name} is too small ({im.size[0]}x{im.size[1]})")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    im.save(f, "JPEG", quality=92)
    f.close()
    return hashlib.md5(raw).hexdigest(), f.name, thumb_uri(f.name)


@st.cache_data(show_spinner=False, max_entries=300)
def features_for(md5, path):
    from extractor import extract
    df, bad = extract([path], M.cols)
    if bad:
        raise RuntimeError(bad[0][1])
    return df.drop(columns="_idx").iloc[0].to_dict()


with tab_photo:
    left, right = st.columns([3, 2])
    with left:
        up = st.file_uploader(f"Upload {N} photos of one person", type=["jpg", "jpeg", "png", "webp"],
                              accept_multiple_files=True, help="One photo per expression, face clearly visible, one person only.")
    with right:
        sets = example_sets()
        if sets:
            pick = st.selectbox("Or use an example set", sets)
            if st.button("Use this example"):
                st.session_state["ex_photos"] = [str(p) for p in sorted((EX / pick).iterdir()) if p.suffix.lower() in IMG_EXT][:N]
        if st.session_state.get("ex_photos") and st.button("Clear example"):
            st.session_state.pop("ex_photos")
            st.rerun()
    sources = list(up) if up else st.session_state.get("ex_photos", [])
    if sources:
        st.caption(f"{len(sources)} of {N} photos selected")
    go = st.button("Match expressions", type="primary", disabled=not sources, key="go_photos")
    if go:
        if len(sources) != N:
            st.warning(f"Add exactly {N} photos of the same person (you have {len(sources)}). One photo per expression.")
        else:
            prepared, problems, temp = [], [], []
            for i in range(len(sources)):
                try:
                    prepared.append(prepare(sources[i]))
                    temp.append(prepared[-1][1])
                except ValueError as ex:
                    problems.append(str(ex))
            try:
                if problems:
                    st.warning("Fix these photos and try again: " + "; ".join(problems))
                else:
                    try:
                        import feat  # noqa: F401
                    except Exception as ex:
                        st.error(f"The photo reader (py-feat) is not available in this deployment ({type(ex).__name__}). The features csv tab still works.")
                        st.stop()
                    rows, failed = [], []
                    bar = st.progress(0.0, text="Reading expressions from the photos. The first run downloads the model, so give it a minute.")
                    for i in range(N):
                        try:
                            rows.append(features_for(prepared[i][0], prepared[i][1]))
                        except Exception as ex:
                            failed.append(f"photo {i + 1}: {str(ex)[:90]}")
                            rows.append(None)
                        bar.progress((i + 1) / N, text=f"Photo {i + 1} of {N}")
                    bar.empty()
                    if failed:
                        st.warning("Could not use every photo (" + "; ".join(failed) + "). Use clear, front facing photos with one visible face.")
                    else:
                        df = pd.DataFrame(rows)
                        if not np.isfinite(df[M.cols].to_numpy(float)).all():
                            st.error("The photo reader returned invalid values. Try different photos.")
                        else:
                            pred, s, p = M.match(df)
                            names = [f"Photo {i + 1}" for i in range(N)]
                            labels, confs = show_result("Result", names, pred, p, s, [x[2] for x in prepared], None)
                            out = pd.DataFrame({"photo": names, "predicted_expression": labels, "confidence": np.round(confs, 3)})
                            st.download_button("Download predictions csv", out.to_csv(index=False), "predictions.csv", "text/csv", key="dl_photo")
            finally:
                for t in temp:
                    try:
                        os.remove(t)
                    except OSError:
                        pass

# csv
with tab_csv:
    c1, c2 = st.columns([3, 2])
    with c1:
        upc = st.file_uploader("features.csv (78 feature columns, 10 rows per person, optional person_id)", type=["csv"])
    with c2:
        csvs = example_csvs()
        if csvs:
            pc = st.selectbox("Or use an example", csvs)
            if st.button("Use this csv"):
                st.session_state["ex_csv"] = str(EX / pc)
    src = upc if upc else st.session_state.get("ex_csv")
    go2 = st.button("Match expressions", type="primary", disabled=src is None, key="go_csv")
    if go2:
        try:
            df = pd.read_csv(src)
        except Exception as ex:
            st.error(f"Could not read that file: {ex}")
            st.stop()
        miss = [c for c in M.cols if c not in df.columns]
        if miss:
            st.error(f"The csv is missing {len(miss)} feature columns, for example {miss[:3]}.")
            st.stop()
        if len(df) > 3000:
            st.error("That file has more than 3000 rows. Split it into smaller files.")
            st.stop()
        df[M.cols] = df[M.cols].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(df[M.cols].to_numpy(float)).all():
            st.error("Some feature values are empty or not numbers. Check the file.")
            st.stop()
        if "person_id" not in df.columns:
            df["person_id"] = "1"
        if "photo_id" not in df.columns:
            df["photo_id"] = [f"row {i + 1}" for i in range(len(df))]
        df["person_id"] = df["person_id"].astype(str)
        tables, hits, total, shown = [], 0, 0, 0
        for pid in list(dict.fromkeys(df["person_id"])):
            g = df[df["person_id"] == pid].reset_index(drop=True)
            if len(g) != N:
                st.warning(f"Person {pid} has {len(g)} rows, needs {N}. Skipped.")
                continue
            pred, s, p = M.match(g)
            truths = g["intended_expression"].tolist() if "intended_expression" in g.columns else None
            names = g["photo_id"].astype(str).tolist()
            if shown < 8:
                labels, confs = show_result(f"Person {pid}", names, pred, p, s, None, truths)
                shown += 1
            else:
                labels = [M.labels[k] for k in pred]
                confs = [float(p[i, pred[i]]) for i in range(N)]
            t = pd.DataFrame({"person_id": pid, "photo_id": names, "predicted_expression": labels, "confidence": np.round(confs, 3)})
            if truths:
                t["intended_expression"] = truths
                hits += sum(1 for a, b in zip(labels, truths) if a == b)
                total += N
            tables.append(t)
        if tables:
            out = pd.concat(tables, ignore_index=True)
            if total:
                st.info(f"{hits / total * 100:.1f}% matched on {len(tables)} people. If these people were in training, this is optimistic.")
            st.download_button("Download predictions csv", out.to_csv(index=False), "predictions.csv", "text/csv", key="dl_csv")

with tab_about:
    st.markdown(f"""
### How it works
1. Each photo goes through Py-Feat (Detectorv2), which reads 51 facial movements, 20 action units and 7 emotion scores.
2. The model compares those numbers with the person's other nine photos (each value is also ranked within the person) and scores the ten expressions with a linear model.
3. Each expression is given to exactly one photo, choosing the assignment with the best total score.

### Model
- Weights: [{REPO}](https://huggingface.co/{REPO})
- Honest accuracy on people the model never saw: **{acc:.1f}%** (about {M.n_people} unique people, 10 expressions, chance is 10%).
- The hardest pairs are sad vs hopelessness and scepticism vs uncertainty.

### Use it well
- Use ten photos of the same person, one per expression, face clearly visible.
- Photos are processed in memory and not stored.
""")
