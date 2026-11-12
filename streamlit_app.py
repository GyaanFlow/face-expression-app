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
import submission as sub
from ui import CSS, cards_html, footer_html, heatmap_html, legend_html, pretty, steps_html, strip_html, thumb_uri, tiles_html

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
    st.divider()
    st.markdown("Built by [Gaurav Tomar](https://www.linkedin.com/in/gaurav-tomar-630b2a316)")

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f"""<div class="hero"><h1>Face Expression Matcher</h1>
<p>Give it ten photos of one person. It reads each face with Py-Feat and works out which of ten expressions every photo shows, using each expression exactly once.</p>
<div class="pills"><span>10 expressions</span><span>Py-Feat Detectorv2</span><span>Linear model on the Hugging Face Hub</span>
<span>Runs privately, nothing stored</span></div></div>""", unsafe_allow_html=True)
st.markdown(tiles_html([(f"{acc:.1f}%", "held-out accuracy (chance is 10%)"), (str(M.n_people), "unique people in training"),
                        (str(N), "expressions"), (str(len(M.cols)), "Py-Feat features per photo")]), unsafe_allow_html=True)


def example_sets():
    return sorted(d.name for d in EX.iterdir() if d.is_dir()) if EX.exists() else []


def example_csvs():
    return sorted(p.name for p in EX.glob("*.csv")) if EX.exists() else []


def show_result(title, names, pred, p, s, uris, truths):
    labels = [M.labels[k] for k in pred]
    confs = [float(p[i, pred[i]]) for i in range(len(pred))]
    alts = []
    for i in range(len(pred)):
        q = p[i].copy()
        q[pred[i]] = -1
        j = int(np.argmax(q))
        alts.append((M.labels[j], float(p[i, j])))
    st.markdown(f"### {title}")
    items = [(f"{np.mean(confs) * 100:.0f}%", "average confidence")]
    low = [names[i] for i in range(len(confs)) if confs[i] < 0.2]
    items.append((str(len(low)), "low confidence photos"))
    if truths:
        hit = sum(1 for a, b in zip(labels, truths) if a == b)
        items.insert(0, (f"{hit * 100 // len(labels)}%", "match the labelled expression"))
    st.markdown(strip_html(items), unsafe_allow_html=True)
    st.markdown(cards_html(names, labels, confs, uris, truths, alts), unsafe_allow_html=True)
    with st.expander("Score of every photo for every expression"):
        st.caption("Green means the model leans towards that expression for that photo, red means against. The final answer is the best one-to-one assignment, so a photo can get its second choice when another photo needs its first.")
        st.markdown(heatmap_html(s, names, M.labels), unsafe_allow_html=True)
    return labels, confs


tab_sub, tab_photo, tab_csv, tab_guide, tab_about = st.tabs(["Get submission JSON", "Test with photos", "Test with features csv", "Guide and FAQ", "About the model"])

# submission
def show_checks(checks):
    for ok, msg in checks:
        st.markdown(("&#9989; " if ok else "&#10060; ") + msg)


with tab_sub:
    st.markdown('<div class="callout">The assignment page accepts <b>only a JSON file</b> with two keys, <code>version</code> and <code>weights</code> (10 rows of 156 numbers). Everything on this tab produces exactly that file.</div>', unsafe_allow_html=True)
    st.markdown(steps_html([("Pick a route", "Route A downloads the ready trained model. Route B trains a model on your own features.csv files."),
                            ("Check it", "The app runs the same checks as the submission page: keys, size, shape and finite numbers."),
                            ("Download and submit", "Download the JSON, choose it on the assignment page, press Check, then Save.")]), unsafe_allow_html=True)
    st.markdown("### Route A: download the trained model")
    st.caption(f"Trained on {M.n_people} unique people. Honest accuracy on unseen people: {acc:.1f}%. This is the safest choice.")
    jtext = sub.to_json(M.w)
    checks = sub.validate(jtext, N, 2 * len(M.cols))
    checks.append((sub.roundtrip_ok(jtext, M.w), "self test: the file gives the same matching as the loaded model"))
    show_checks(checks)
    st.download_button("Download model.json", jtext, "model.json", "application/json", type="primary",
                       disabled=not sub.all_ok(checks), key="dl_model_json")
    with st.expander("Preview the first row"):
        st.code(jtext[:420] + " ...", language="json")
    st.divider()
    st.markdown("### Route B: train on your own features.csv")
    st.caption("Upload one or more files from your assignment (columns person_id, intended_expression and the 78 features). Exact duplicate people across files are removed. With only one ZIP (about 30 people) expect lower accuracy than route A.")
    ups = st.file_uploader("features.csv files", type=["csv"], accept_multiple_files=True, key="train_csvs")
    alpha = st.select_slider("Regularisation strength (ridge alpha)", options=[300.0, 1000.0, 3000.0], value=1000.0,
                             help="1000 was the best setting in the notebook. Higher is safer with few people.")
    if st.button("Train and build JSON", disabled=not ups, key="train_go"):
        frames, problem = [], None
        for f in ups:
            try:
                d = pd.read_csv(f)
            except Exception as ex:
                problem = f"Could not read {f.name}: {ex}"
                break
            need = ["person_id", "intended_expression"] + M.cols
            miss = [c for c in need if c not in d.columns]
            if miss:
                problem = f"{f.name} is missing columns, for example {miss[:3]}"
                break
            d[M.cols] = d[M.cols].apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(d[M.cols].to_numpy(float)).all():
                problem = f"{f.name} has empty or non numeric feature values"
                break
            frames.append(d)
        if problem:
            st.error(problem)
        else:
            if sum(len(d) for d in frames) > 5000:
                st.error("That is more than 5000 rows in total. Upload fewer files.")
                st.stop()
            X, y, g, info = sub.prepare_people(frames, M.cols, M.labels)
            st.info(f"{info['listed']} people listed, {info['dup']} duplicates removed, {info['bad']} skipped (need 10 rows with all 10 expressions), {info['kept']} used.")
            if X is None or info["kept"] < 8:
                st.error("Need at least 8 usable people to train. Add more files.")
            else:
                with st.spinner("Cross validating and training..."):
                    m, sd = sub.cv_ridge(X, y, g, alpha)
                    W = sub.fit_ridge(X, y, alpha)
                text = sub.to_json(W)
                chk = sub.validate(text, N, 2 * len(M.cols))
                chk.append((sub.roundtrip_ok(text, W), "self test: the file gives the same matching as the trained model"))
                st.session_state["own_json"] = (text, chk, m, sd, info["kept"])
    own = st.session_state.get("own_json")
    if own:
        text, chk, m, sd, kept = own
        st.markdown(f"**Estimated accuracy on unseen people: {m * 100:.1f}% (plus or minus {sd * 100:.1f})** from {kept} unique people.")
        if m * 100 < acc - 1.5:
            st.warning(f"This is below route A ({acc:.1f}%). Route A was trained on more people, so it is likely the better submission.")
        show_checks(chk)
        st.download_button("Download my_model.json", text, "my_model.json", "application/json", disabled=not sub.all_ok(chk), key="dl_own_json")
    st.divider()
    st.markdown("### Check any JSON before you submit")
    c1, c2 = st.columns(2)
    with c1:
        vf = st.file_uploader("Choose a JSON file", type=["json"], key="val_file")
    with c2:
        vt = st.text_area("or paste it here", height=110, key="val_text", placeholder='{"version":"linear-rank-v1","weights":[ ... ]}')
    vsrc = vf.getvalue().decode("utf-8", "replace") if vf else (vt.strip() or None)
    if vsrc:
        vc = sub.validate(vsrc, N, 2 * len(M.cols))
        show_checks(vc)
        if sub.all_ok(vc):
            st.success("Valid. A valid file earns 0 marks now. The bonus marks come from its accuracy on unseen people after the project closes.")
        else:
            st.error("Fix the items marked with a cross before you submit.")

# photos
MAX_SIDE = 1024
MAX_MB = 15


def prepare(src):
    """Decode, fix rotation, shrink. Returns (md5, path, thumb_uri) or raises ValueError with a readable message."""
    import hashlib
    import tempfile
    from PIL import Image, ImageOps
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
    except Exception:
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
def features_for(md5, _path):
    # _path starts with an underscore so Streamlit leaves it out of the cache key (the temp file name changes every run)
    from extractor import extract
    df, bad = extract([_path], M.cols)
    if bad:
        raise RuntimeError(bad[0][1])
    return df.drop(columns="_idx").iloc[0].to_dict()


with tab_photo:
    st.markdown(steps_html([("Collect 10 photos", "One person, ten different expressions, one photo each. Clear, front facing, one face, good light."),
                            ("Upload or pick an example", "Drop all ten files in together, or load a ready made example set."),
                            ("Press Match expressions", "The first run downloads the face model, so wait about a minute. After that it takes seconds.")]), unsafe_allow_html=True)
    st.markdown("**The ten expressions it can assign**" + legend_html(M.labels), unsafe_allow_html=True)
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
    st.markdown('<div class="callout">Use this tab when you already have the 78 Py-Feat numbers per photo, for example the <b>features.csv</b> from the assignment. It needs no face model, so it is instant.</div>', unsafe_allow_html=True)
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

with tab_guide:
    st.markdown("## How to get a good result")
    st.markdown(steps_html([("Same person", "All ten photos must be of one person. Mixing people breaks the comparison, because every photo is ranked against the other nine."),
                            ("Ten different expressions", "The model hands out each of the ten expressions exactly once, so the set should really cover ten different looks."),
                            ("Clear faces", "Front facing, one face, eyes and mouth visible, no heavy shadows, no masks, no tiny images.")]), unsafe_allow_html=True)
    with st.expander("Photo requirements", expanded=True):
        st.markdown("""
- File types: jpg, jpeg, png, webp. Each file up to 15 MB. Very large photos are shrunk automatically.
- Exactly 10 photos per run.
- If a photo has several faces, the biggest face is used.
- A photo with no detectable face is reported by number so you can swap it.
""")
    with st.expander("Features csv format"):
        st.markdown(f"""
- One row per photo, ten rows per person.
- The {len(M.cols)} feature columns must be present with these names: 51 columns starting `bs_`, 20 starting `au_`, 7 starting `emotion_`.
- Optional columns: `person_id` (needed if the file has several people), `photo_id`, `intended_expression`.
- With `intended_expression` present, every card shows match or mismatch.
""")
        st.code(", ".join(M.cols[:6]) + ", ...", language="text")
    with st.expander("Reading the results"):
        st.markdown("""
- **Chip and bar:** the expression given to that photo and how strongly the model favours it compared with the other nine expressions for the same photo.
- **Low:** under 20 percent confidence. Treat that photo as a coin flip between its top choices.
- **Next likely:** the model's second choice for that photo. It is useful for the confusable pairs.
- **Heatmap:** raw scores for every photo and expression. The shown labels are the best one-to-one assignment, not just each row's maximum.
""")
    with st.expander("Frequently asked questions"):
        st.markdown(f"""
**Why exactly ten photos?** The model was built for a task where each person has ten photos and each expression appears once. It uses the other nine photos as a reference for the person, so a single photo cannot be judged alone.

**How accurate is it?** About {acc:.0f}% of photos get the right expression on people the model never saw. Chance is 10%. It is much better at happy and surprised than at sad vs hopelessness or scepticism vs uncertainty, which look almost the same on a face.

**Does it read real emotions?** No. It matches the expression a person was asked to show. It says nothing about how anyone feels.

**Are my photos stored?** No. They are decoded in memory, written to a temporary file for the face reader, and deleted at the end of the run.

**The photo tab is slow or crashed.** The first run downloads the face model, which takes about a minute. On a small free server the face reader can run out of memory. The features csv tab always works.

**Can I use it on a single person with fewer expressions?** Not meaningfully. Add photos until there are ten.
""")

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

st.markdown(footer_html(), unsafe_allow_html=True)
