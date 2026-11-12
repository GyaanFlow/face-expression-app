# Face Expression Matcher

Streamlit app for the TDS P1 face-expression bonus question. Built by [Gaurav Tomar](https://www.linkedin.com/in/gaurav-tomar-630b2a316).

Tabs
- Get submission JSON: download the trained model as the grader file (linear-rank-v1), train your own from features.csv files, and validate any JSON.
- Test with photos: ten photos of one person in, one expression per photo out (needs py-feat).
- Test with features csv: the same from the 78 Py-Feat columns.

Run locally
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Tests (no py-feat or torch needed)
    pip install pytest numpy pandas scipy scikit-learn pillow safetensors huggingface_hub
    pytest -q

Settings
- MODEL_REPO (default 23f1000805/face-expression-v3) and HF_TOKEN (only for a private repo) go in Streamlit secrets or environment variables.
- See DEPLOY.md for Streamlit Community Cloud steps.
