import hashlib
import io
import logging
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Face Expression Matcher", page_icon="🎭", layout="wide",
                   menu_items={"About": "Face Expression Matcher by Gaurav Tomar - https://www.linkedin.com/in/gaurav-tomar-630b2a316"})

try:
    for k in ("HF_TOKEN", "MODEL_REPO"):
        if k in st.secrets:
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass

import submission as sub
from matcher import ExpressionMatcher
from ui import (CSS, cards_html, footer_html, heatmap_html, legend_html, steps_html, strip_html,
                thumb_uri, tiles_html)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")
REPO = os.environ.get("MODEL_REPO", "23f1000805/face-expression-v3")
EX = Path(__file__).parent / "examples"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
MAX_SIDE, MAX_MB = 1024, 15
CHART_BLUE = "#2a78d6"


@st.cache_resource(show_spinner="Loading the model from the Hugging Face Hub...", ttl=6 * 3600)
def get_model():
    return ExpressionMatcher(REPO)


def guarded(title):
    """Decorator: show a friendly error with details instead of crashing the whole page."""
    def wrap(fn):
        def inner(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as ex:
                if type(ex).__name__ in ("StopException", "RerunException"):
                    raise
                log.exception("%s failed", title)
                st.error(f"{title} failed: {type(ex).__name__}. Try again, or use another tab.")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc()[-2500:])
        return inner
    return wrap


try:
    M = get_model()
except Exception as ex:
    st.error(f"Could not load the model from {REPO}. If the repo is private, add HF_TOKEN in the app secrets. ({type(ex).__name__}: {str(ex)[:150]})")
    if st.button("Try again"):
        get_model.clear()
        st.rerun()
    st.stop()

N = len(M.labels)
ACC = (M.cv or 0) * 100
MET = M.metrics or {}
TEACH = MET.get("teacher_accuracy", M.chosen.get("teacher_accuracy"))


# ---------- example data ----------
@st.cache_data(show_spinner=False)
def example_index():
    """photo md5 -> (feature dict, label) from every examples/*.csv that has a photo_md5 column."""
    idx = {}
    if not EX.exists():
        return idx
    for f in EX.glob("*.csv"):
        try:
            d = pd.read_csv(f, dtype=str)
        except Exception:
            continue
        if "photo_md5" not in d.columns or any(c not in d.columns for c in M.cols):
            continue
        for _, r in d.iterrows():
            try:
                feats = {c: float(r[c]) for c in M.cols}
            except Exception:
                continue
            idx[str(r["photo_md5"])] = (feats, r.get("intended_expression"))
    return idx


def example_sets():
    return sorted(d.name for d in EX.iterdir() if d.is_dir()) if EX.exists() else []


def example_csvs():
    return sorted(p.name for p in EX.glob("*.csv")) if EX.exists() else []


def set_photos(name):
    d = EX / name
    return [str(p) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT][:N]


# ---------- shared rendering ----------
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
    items = [(f"{np.mean(confs) * 100:.0f}%", "average confidence"),
             (str(sum(1 for c in confs if c < 0.2)), "low confidence photos")]
    if truths and all(t in M.labels for t in truths):
        hit = sum(1 for a, b in zip(labels, truths) if a == b)
        items.insert(0, (f"{hit}/{len(labels)}", "match the labelled expression"))
    else:
        truths = None
    st.markdown(strip_html(items), unsafe_allow_html=True)
    st.markdown(cards_html(names, labels, confs, uris, truths, alts), unsafe_allow_html=True)
    with st.expander("Score of every photo for every expression"):
        st.caption("Green leans towards that expression, red against. The labels are the best one-to-one assignment, so a photo can get its second choice when another photo needs its first.")
        st.markdown(heatmap_html(s, names, M.labels), unsafe_allow_html=True)
    return labels, confs


def prepare(src):
    from PIL import Image, ImageOps
    if isinstance(src, str):
        raw, name = Path(src).read_bytes(), Path(src).name
    else:
        raw, name = src.getvalue(), src.name
    if not raw:
        raise ValueError(f"{name} is empty")
    if len(raw) > MAX_MB * 1024 * 1024:
        raise ValueError(f"{name} is larger than {MAX_MB} MB")
    try:
        Image.MAX_IMAGE_PIXELS = 60_000_000
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
    return dict(name=name, md5=hashlib.md5(raw).hexdigest(), path=f.name, uri=thumb_uri(f.name))


@st.cache_data(show_spinner=False, max_entries=300)
def features_for(md5, _path):
    from extractor import extract
    df, bad = extract([_path], M.cols)
    if bad:
        raise RuntimeError(bad[0][1])
    return df.drop(columns="_idx").iloc[0].to_dict()


def run_photos(sources, key):
    if len(sources) != N:
        st.warning(f"Add exactly {N} photos of the same person (you have {len(sources)}). One photo per expression.")
        return
    prepared, problems = [], []
    for s in sources:
        try:
            prepared.append(prepare(s))
        except ValueError as ex:
            problems.append(str(ex))
    try:
        if problems:
            st.warning("Fix these photos and try again: " + "; ".join(problems))
            return
        if len({x["md5"] for x in prepared}) < N:
            st.warning("Some photos are identical copies. Use ten different photos.")
            return
        cache = example_index()
        rows, truths, failed = [None] * N, [None] * N, []
        need = [i for i in range(N) if prepared[i]["md5"] not in cache]
        for i in range(N):
            if prepared[i]["md5"] in cache:
                rows[i], truths[i] = cache[prepared[i]["md5"]]
        if need:
            try:
                import feat  # noqa: F401
            except Exception as ex:
                st.error(f"The photo reader (py-feat) is not available here ({type(ex).__name__}). The features csv tab and the person_a demo still work.")
                return
            with st.status(f"Reading {len(need)} photos with Py-Feat. The first run downloads the face model (about a minute).", expanded=True) as stat:
                bar = st.progress(0.0)
                for j, i in enumerate(need):
                    try:
                        rows[i] = features_for(prepared[i]["md5"], prepared[i]["path"])
                    except Exception as ex:
                        failed.append(f"photo {i + 1} ({prepared[i]['name']}): {str(ex)[:90]}")
                    bar.progress((j + 1) / len(need), text=f"Photo {j + 1} of {len(need)}")
                stat.update(label="Photos read" if not failed else "Some photos failed", state="complete" if not failed else "error", expanded=False)
        else:
            st.toast("Known example photos, using their saved features (instant).")
        if failed:
            st.warning("Could not use every photo: " + "; ".join(failed) + ". Use clear, front facing photos with one visible face.")
            return
        df = pd.DataFrame(rows)
        if not np.isfinite(df[M.cols].to_numpy(float)).all():
            st.error("The photo reader returned invalid values. Try different photos.")
            return
        pred, s, p = M.match(df)
        names = [x["name"] for x in prepared]
        known = truths if all(t is not None for t in truths) else None
        labels, confs = show_result("Result", names, pred, p, s, [x["uri"] for x in prepared], known)
        if known:
            st.caption("These are the example photos, which were part of the training data, so this result is better than you should expect on a new person.")
        out = pd.DataFrame({"photo": names, "predicted_expression": labels, "confidence": np.round(confs, 3)})
        if known:
            out["intended_expression"] = known
        c1, c2 = st.columns(2)
        c1.download_button("Download results (csv)", out.to_csv(index=False), "predictions.csv", "text/csv", key=f"dlc_{key}")
        c2.download_button("Download results (json)", out.to_json(orient="records", indent=1), "predictions.json", "application/json", key=f"dlj_{key}")
    finally:
        for x in prepared:
            try:
                os.remove(x["path"])
            except OSError:
                pass


def run_csv(src, key):
    try:
        df = pd.read_csv(src)
    except Exception as ex:
        st.error(f"Could not read that file: {ex}")
        return
    miss = [c for c in M.cols if c not in df.columns]
    if miss:
        st.error(f"The csv is missing {len(miss)} feature columns, for example {miss[:3]}.")
        return
    if len(df) > 3000:
        st.error("That file has more than 3000 rows. Split it into smaller files.")
        return
    df[M.cols] = df[M.cols].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(df[M.cols].to_numpy(float)).all():
        st.error("Some feature values are empty or not numbers. Check the file.")
        return
    if "person_id" not in df.columns:
        df["person_id"] = "1"
    if "photo_id" not in df.columns:
        df["photo_id"] = [f"row {i + 1}" for i in range(len(df))]
    df["person_id"] = df["person_id"].astype(str)
    if "intended_expression" in df.columns:
        df["intended_expression"] = df["intended_expression"].astype(str).str.strip().str.lower()
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
    if not tables:
        return
    out = pd.concat(tables, ignore_index=True)
    if shown < len(tables):
        st.caption(f"Showing cards for the first {shown} of {len(tables)} people. The download has all of them.")
    if total:
        st.info(f"{hits / total * 100:.1f}% matched on {len(tables)} people. If these people were in training, this is optimistic.")
    st.download_button("Download predictions csv", out.to_csv(index=False), "predictions.csv", "text/csv", key=f"dl_{key}")


# ---------- layout ----------
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Live model")
    st.metric("Accuracy on unseen people", f"{ACC:.1f}%", help="Nested cross validation: matching accuracy on people the model never saw. Chance is 10%.")
    card = f"<b>{M.chosen.get('name', 'linear model')}</b><br>{M.n_people} unique people · {N} expressions · {len(M.cols)} features"
    if TEACH is not None:
        card += f"<br>Photo teacher tried: {TEACH * 100:.0f}% (not used)" if not M.chosen.get("used_teacher") else f"<br>Distilled from a photo teacher ({TEACH * 100:.0f}%)"
    if M.trained:
        card += f"<br>Trained {M.trained}"
    card += f'<br><a href="https://huggingface.co/{REPO}" target="_blank">{REPO}</a>'
    st.markdown(f'<div class="mcard">{card}</div>', unsafe_allow_html=True)
    if st.button("Reload latest model", help="Fetch the newest weights from the Hub, for example after retraining.", width="stretch"):
        get_model.clear()
        example_index.clear()
        st.toast("Reloading the model...")
        time.sleep(0.3)
        st.rerun()
    st.divider()
    st.caption("Photos are processed in memory and deleted after each run. Nothing is stored.")
    st.markdown("Built by [Gaurav Tomar](https://www.linkedin.com/in/gaurav-tomar-630b2a316)")

st.markdown(f"""<div class="hero"><h1>Face Expression Matcher</h1>
<p>Ten photos of one person in, one of ten expressions per photo out, each expression used exactly once. Plus a one-click, grader-ready submission JSON.</p>
<div class="pills"><span>10 expressions</span><span>Py-Feat Detectorv2</span><span>Model on the Hugging Face Hub</span><span>Private: nothing stored</span></div></div>""",
            unsafe_allow_html=True)
st.markdown(tiles_html([(f"{ACC:.1f}%", "accuracy on unseen people (chance 10%)"), (str(M.n_people), "unique people in training"),
                        (str(N), "expressions"), (str(len(M.cols)), "Py-Feat features per photo")]), unsafe_allow_html=True)

tab_start, tab_sub, tab_photo, tab_csv, tab_ins, tab_guide = st.tabs(
    ["Start here", "Get submission JSON", "Test with photos", "Test with features csv", "Model insights", "Guide and FAQ"])

# start here
with tab_start:
    has_a = (EX / "person_a").is_dir()
    st.markdown(f"""<div class="guest"><h3>New here? Try it in 30 seconds</h3>
<p>The app ships with an example person, <b>person_a</b>, with ten photos and the expression each was meant to show.</p>
<ol><li>Press <b>Run the person_a demo</b> below. It runs instantly because the example's features are saved.</li>
<li>Each card shows the photo, the expression the model gave it, its confidence, the second choice, and a green <b>match</b> or red <b>expected ...</b> against the true label.</li>
<li>Open <b>Score of every photo for every expression</b> to see why: the model scores all 100 photo and expression pairs, then hands out each expression once.</li>
<li>Try the other tabs: <b>Test with photos</b> (your own ten photos), <b>Test with features csv</b> (pick person_a.csv), and <b>Get submission JSON</b>.</li></ol></div>""",
                unsafe_allow_html=True)
    if has_a:
        go_demo = st.button("Run the person_a demo", type="primary", key="demo_go")
        if go_demo:
            guarded("The demo")(run_photos)(set_photos("person_a"), "demo")
    else:
        st.info("The person_a example folder is not in this deployment. Add examples/person_a with ten photos to enable the demo.")
    st.markdown("**The ten expressions**" + legend_html(M.labels), unsafe_allow_html=True)
    st.markdown(steps_html([("Submit for the assignment", "Get submission JSON gives the exact file the assignment page accepts, already checked."),
                            ("Test with your own photos", "Ten photos of one person, one per expression. The first run downloads the face model."),
                            ("Understand the model", "Model insights shows accuracy per expression, the learning curve and the confusion matrix.")]),
                unsafe_allow_html=True)

# submission
def show_checks(checks):
    for ok, msg in checks:
        st.markdown(("&#9989; " if ok else "&#10060; ") + msg)


@guarded("Building the submission")
def tab_submission():
    st.markdown('<div class="callout">The assignment page accepts <b>only a JSON file</b> with two keys, <code>version</code> and <code>weights</code> (10 rows of 156 numbers). Everything on this tab produces exactly that file.</div>', unsafe_allow_html=True)
    st.markdown("### Route A: download the trained model <span class='badge'>recommended</span>", unsafe_allow_html=True)
    st.caption(f"Trained on {M.n_people} unique people. Accuracy on unseen people: {ACC:.1f}%.")
    jtext = sub.to_json(M.w)
    checks = sub.validate(jtext, N, 2 * len(M.cols))
    checks.append((sub.roundtrip_ok(jtext, M.w), "self test: the file gives the same matching as the live model"))
    with st.expander("Checks passed" if sub.all_ok(checks) else "Some checks failed", expanded=not sub.all_ok(checks)):
        show_checks(checks)
    st.download_button("Download model.json", jtext, "model.json", "application/json", type="primary",
                       disabled=not sub.all_ok(checks), key="dl_model_json")
    st.caption("Then on the assignment page: Choose File, select model.json, press Check, then Save.")
    st.divider()
    st.markdown("### Route B: train on your own features.csv")
    st.caption("Upload one or more files (person_id, intended_expression and the 78 features). Duplicate people are removed. With one zip (about 30 people) expect lower accuracy than route A.")
    ups = st.file_uploader("features.csv files", type=["csv"], accept_multiple_files=True, key="train_csvs")
    alpha = st.select_slider("Regularisation strength (ridge alpha)", options=[300.0, 1000.0, 3000.0], value=1000.0)
    if st.button("Train and build JSON", disabled=not ups, key="train_go"):
        frames = []
        for f in ups:
            d = pd.read_csv(f)
            need = ["person_id", "intended_expression"] + M.cols
            miss = [c for c in need if c not in d.columns]
            if miss:
                st.error(f"{f.name} is missing columns, for example {miss[:3]}")
                return
            d[M.cols] = d[M.cols].apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(d[M.cols].to_numpy(float)).all():
                st.error(f"{f.name} has empty or non numeric feature values")
                return
            frames.append(d)
        if sum(len(d) for d in frames) > 5000:
            st.error("More than 5000 rows in total. Upload fewer files.")
            return
        X, y, g, info = sub.prepare_people(frames, M.cols, M.labels)
        st.info(f"{info['listed']} people listed, {info['dup']} duplicates removed, {info['bad']} skipped, {info['kept']} used.")
        if X is None or info["kept"] < 8:
            st.error("Need at least 8 usable people to train. Add more files.")
            return
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
        st.markdown(f"**Estimated accuracy on unseen people: {m * 100:.1f}% (plus or minus {sd * 100:.1f})** from {kept} people.")
        if m * 100 < ACC - 1.5:
            st.warning(f"Below route A ({ACC:.1f}%). Route A was trained on more people and is likely the better submission.")
        show_checks(chk)
        st.download_button("Download my_model.json", text, "my_model.json", "application/json", disabled=not sub.all_ok(chk), key="dl_own_json")
    st.divider()
    st.markdown("### Check any JSON before you submit")
    c1, c2 = st.columns(2)
    vf = c1.file_uploader("Choose a JSON file", type=["json"], key="val_file")
    vt = c2.text_area("or paste it here", height=110, key="val_text", placeholder='{"version":"linear-rank-v1","weights":[ ... ]}')
    vsrc = vf.getvalue().decode("utf-8", "replace") if vf else (vt.strip() or None)
    if vsrc:
        vc = sub.validate(vsrc, N, 2 * len(M.cols))
        show_checks(vc)
        if sub.all_ok(vc):
            st.success("Valid. A valid file earns 0 marks now; the bonus comes from its accuracy on unseen people after the project closes.")
        else:
            st.error("Fix the items marked with a cross before you submit.")


with tab_sub:
    tab_submission()

# photos
with tab_photo:
    st.markdown(steps_html([("Collect 10 photos", "One person, ten different expressions, one photo each. Clear, front facing, one face."),
                            ("Upload or pick an example", "Drop all ten files at once, or load an example set."),
                            ("Press Match expressions", "The first run downloads the face model, about a minute. Example photos are instant.")]),
                unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    with left:
        up = st.file_uploader(f"Upload {N} photos of one person", type=["jpg", "jpeg", "png", "webp"],
                              accept_multiple_files=True, key="photo_up")
    with right:
        sets = example_sets()
        if sets:
            pick = st.selectbox("Or use an example set", sets, key="ex_pick")
            b1, b2 = st.columns(2)
            if b1.button("Use this example", key="ex_use", width="stretch"):
                st.session_state["ex_photos"] = set_photos(pick)
            if st.session_state.get("ex_photos") and b2.button("Clear", key="ex_clear", width="stretch"):
                st.session_state.pop("ex_photos")
                st.rerun()
    sources = list(up) if up else st.session_state.get("ex_photos", [])
    if sources:
        st.caption(f"{len(sources)} of {N} photos selected" + ("" if up else " (example set)"))
        if not up and sources:
            st.image([s for s in sources[:N]], width=88)
    if st.button("Match expressions", type="primary", disabled=not sources, key="go_photos"):
        guarded("Matching the photos")(run_photos)(sources, "photos")

# csv
with tab_csv:
    st.markdown('<div class="callout">Use this when you already have the 78 Py-Feat numbers per photo, for example the <b>features.csv</b> from the assignment. No face model needed, so it is instant.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    upc = c1.file_uploader("features.csv (78 feature columns, 10 rows per person, optional person_id)", type=["csv"], key="csv_up")
    csvs = example_csvs()
    if csvs:
        pc = c2.selectbox("Or use an example", csvs, key="csv_pick")
        if c2.button("Use this csv", key="csv_use"):
            st.session_state["ex_csv"] = str(EX / pc)
    src = upc if upc else st.session_state.get("ex_csv")
    if src is not None and not upc:
        st.caption(f"Example selected: {Path(src).name}")
    if st.button("Match expressions", type="primary", disabled=src is None, key="go_csv"):
        guarded("Matching the csv")(run_csv)(src, "csv")


# insights
@guarded("Model insights")
def tab_insights():
    import altair as alt
    nested = MET.get("nested_cv", M.cv)
    ci = MET.get("bootstrap_ci95")
    items = [(f"{(nested or 0) * 100:.1f}%", "nested CV accuracy (the fair number)")]
    if MET.get("best_candidate_cv"):
        items.append((f"{MET['best_candidate_cv'] * 100:.1f}%", f"best candidate CV ({MET.get('best_candidate', '')})"))
    if ci:
        items.append((f"{ci[0] * 100:.0f}-{ci[1] * 100:.0f}%", "95% interval over people"))
    if MET.get("train_accuracy"):
        items.append((f"{MET['train_accuracy'] * 100:.0f}%", "on training people (not the real score)"))
    st.markdown(strip_html(items), unsafe_allow_html=True)

    pc = MET.get("per_class")
    if pc:
        d = pd.DataFrame({"expression": list(pc.keys()), "accuracy": [float(v) for v in pc.values()]}).sort_values("accuracy", ascending=False)
        d["label"] = (d["accuracy"] * 100).round(0).astype(int).astype(str) + "%"
        st.markdown("#### Accuracy per expression")
        st.caption("Held-out people. Each person shows every expression once, so this is the share of times the model got that expression right.")
        base = alt.Chart(d).encode(y=alt.Y("expression:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                                   x=alt.X("accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", grid=True, gridOpacity=0.25), title=None),
                                   tooltip=[alt.Tooltip("expression:N"), alt.Tooltip("accuracy:Q", format=".0%")])
        bars = base.mark_bar(color=CHART_BLUE, cornerRadiusEnd=4, size=18)
        text = base.mark_text(align="left", dx=6, fontSize=12).encode(text="label:N")
        st.altair_chart((bars + text).properties(height=330), width="stretch")

    lc = MET.get("learning_curve")
    if lc:
        d = pd.DataFrame(lc)
        st.markdown("#### Does more data help?")
        st.caption("Accuracy on held-out people as the number of training people grows. A rising line means more unique people would still improve the model.")
        ch = alt.Chart(d).encode(x=alt.X("people:Q", title="training people"),
                                 y=alt.Y("acc:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="%", gridOpacity=0.25), title=None),
                                 tooltip=[alt.Tooltip("people:Q", title="people"), alt.Tooltip("acc:Q", format=".1%", title="accuracy")])
        st.altair_chart((ch.mark_line(color=CHART_BLUE, strokeWidth=2) + ch.mark_point(color=CHART_BLUE, filled=True, size=80)).properties(height=260),
                        width="stretch")

    conf = M.reports.get("confusion")
    if conf is not None and conf.shape[1] == 11:
        st.markdown("#### Confusion matrix")
        st.caption("Rows: true expression. Columns: what the model gave. The off-diagonal hot spots are the pairs it mixes up.")
        cm = conf.set_index(conf.columns[0])
        st.dataframe(cm.style.background_gradient(cmap="Blues", axis=None), width="stretch")
    lb = M.reports.get("leaderboard")
    if lb is not None:
        with st.expander("Model leaderboard (repeated person-grouped CV)"):
            st.dataframe(lb, width="stretch", hide_index=True)
    if TEACH is not None:
        st.info(f"A fine-tuned photo teacher (ConvNeXt + CLIP) reached {TEACH * 100:.0f}% on unseen people. "
                + ("Its knowledge was distilled into the live model." if M.chosen.get("used_teacher") else "It did not improve the linear model enough, so the live model uses the 78 features only."))
    if not MET:
        st.caption("This model was uploaded without metrics.json, so only the headline numbers are shown.")


with tab_ins:
    tab_insights()

# guide
with tab_guide:
    st.markdown("## How to get a good result")
    st.markdown(steps_html([("Same person", "All ten photos must be of one person. Every photo is ranked against the other nine."),
                            ("Ten different expressions", "Each of the ten expressions is handed out exactly once."),
                            ("Clear faces", "Front facing, one face, eyes and mouth visible, no heavy shadows, no tiny images.")]),
                unsafe_allow_html=True)
    with st.expander("Guest walkthrough with person_a", expanded=True):
        st.markdown("""
1. **Start here** tab: press **Run the person_a demo**. You should see ten cards, most with a green *match*.
2. **Test with photos** tab: pick *person_a* in **Or use an example set**, press **Use this example**, then **Match expressions**. Same result, through the photo path.
3. **Test with features csv** tab: pick *person_a.csv*, press **Use this csv**, then **Match expressions**. Same person from the raw numbers.
4. **Get submission JSON** tab: press **Download model.json** and paste it into the checker at the bottom to see every check pass.

person_a was part of the training data, so expect a better score here than on a brand new person.
""")
    with st.expander("Photo requirements"):
        st.markdown(f"- jpg, jpeg, png or webp, up to {MAX_MB} MB each; large photos are shrunk automatically\n- exactly 10 photos, all different\n- with several faces, the biggest face is used\n- a photo with no detectable face is reported by name")
    with st.expander("Features csv format"):
        st.markdown(f"- one row per photo, ten rows per person\n- the {len(M.cols)} feature columns: 51 starting `bs_`, 20 starting `au_`, 7 starting `emotion_`\n- optional: `person_id`, `photo_id`, `intended_expression`")
        st.code(", ".join(M.cols[:6]) + ", ...", language="text")
    with st.expander("Reading the results"):
        st.markdown("- **Chip and bar:** the expression given to the photo and how strongly the model prefers it over the other nine\n- **low:** under 20% confidence, close to a coin flip\n- **next likely:** the second choice, useful for the confusable pairs\n- **Heatmap:** raw scores; labels come from the best one-to-one assignment")
    with st.expander("Frequently asked questions"):
        st.markdown(f"""
**Why exactly ten photos?** The model uses the other nine photos as the reference for the person, and gives out each expression once.

**How accurate is it?** About {ACC:.0f}% on people it never saw (chance 10%). Best on happy and surprised; weakest on sad vs hopelessness and scepticism vs uncertainty.

**Does it read real emotions?** No. It matches the expression a person was asked to show.

**Are photos stored?** No. They live in memory and a temporary file that is deleted after the run.

**The photo tab is slow or failed.** The first run downloads the face model. On the free server it can run out of memory; the demo, the csv tab and the JSON tab always work.

**The model was retrained. How do I see it?** Press **Reload latest model** in the sidebar.
""")

st.markdown(footer_html(), unsafe_allow_html=True)
