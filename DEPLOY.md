Deploy on Streamlit Community Cloud (free)

1. Put these files in a new GitHub repo (public is simplest, a private repo also works if you connect it):
   streamlit_app.py, matcher.py, extractor.py, ui.py, requirements.txt, packages.txt, .streamlit/config.toml
   (optional) examples/ built with make_examples.py. Do not commit secrets.
2. Go to share.streamlit.io, sign in with GitHub, press Create app, pick the repo, branch main, main file streamlit_app.py.
3. Open Advanced settings: choose Python 3.11 (3.12 also fine). If your model repo 23f1000805/face-expression-v3 is private, paste this in Secrets:
   HF_TOKEN = "hf_xxx"
   MODEL_REPO = "23f1000805/face-expression-v3"
4. Deploy. The first build takes several minutes (torch and py-feat). The first photo run downloads the Py-Feat weights.

Memory: the free tier has limited RAM. If the app restarts when you press Match expressions on photos, the photo reader ran out of memory. The features csv tab keeps working. Then either reboot the app, or run the photo tab elsewhere.

Examples: python make_examples.py <unzipped data folder> 3 creates examples/person_XX folders and csv files. A public repo shows them to everyone, so only use data you may publish.
