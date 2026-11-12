"""HTML pieces for the result cards."""
import base64
import html
import io

import numpy as np
from PIL import Image, ImageOps

COLORS = {
    "neutral": "#64748b", "happy": "#f59e0b", "sad": "#3b82f6", "angry": "#ef4444", "surprised": "#a855f7",
    "contempt": "#14b8a6", "restrained-amusement": "#ec4899", "scepticism": "#84cc16", "uncertainty": "#f97316",
    "hopelessness": "#6366f1",
}

CSS = """
<style>
.block-container{max-width:1150px;padding-top:3.4rem}
.hero{padding:28px 30px;border-radius:18px;background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%);color:#fff;margin-bottom:14px}
.hero h1{margin:0 0 6px;font-size:30px;color:#fff;padding:0}
.hero p{margin:0;opacity:.92;font-size:15px}
.pills{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap}
.pills span{background:rgba(255,255,255,.18);padding:4px 12px;border-radius:999px;font-size:12.5px}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:10px 0 14px}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}
.fx{background:var(--secondary-background-color);border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:10px}
.fx img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:10px;display:block}
.noimg{aspect-ratio:1;display:flex;align-items:center;justify-content:center;border-radius:10px;background:rgba(128,128,128,.12);font-size:13px;opacity:.7}
.cap{font-size:12px;opacity:.65;margin:6px 0 2px}
.chip{display:inline-block;color:#fff;font-weight:600;font-size:13px;padding:3px 10px;border-radius:999px;margin-top:8px}
.meter{height:6px;background:rgba(128,128,128,.2);border-radius:6px;margin-top:8px;overflow:hidden}
.meter i{display:block;height:100%}
.pct{font-size:11.5px;opacity:.65;margin-top:4px}
.ok{color:#16a34a;font-weight:600;font-size:12px;margin-top:6px}.bad{color:#dc2626;font-weight:600;font-size:12px;margin-top:6px}
.acc{display:inline-block;padding:7px 12px;border-radius:10px;background:rgba(79,70,229,.14);font-size:13.5px;margin:4px 0}
.hm{overflow-x:auto}.hm table{border-collapse:collapse;font-size:12px;width:100%}
.hm th,.hm td{padding:5px 7px;text-align:center;border:1px solid rgba(128,128,128,.25)}
.hm th{background:rgba(128,128,128,.12);white-space:nowrap}
</style>
"""


def pretty(s):
    return s.replace("-", " ").title()


def thumb_uri(file_or_path, px=240):
    im = ImageOps.exif_transpose(Image.open(file_or_path)).convert("RGB")
    im = ImageOps.fit(im, (px, px), method=Image.BILINEAR, centering=(0.5, 0.4))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def card(title, label, conf, uri=None, truth=None):
    col = COLORS.get(label, "#64748b")
    img = f'<img src="{uri}" alt="">' if uri else f'<div class="noimg">{html.escape(title)}</div>'
    cap = f'<div class="cap">{html.escape(title)}</div>' if uri else ""
    tr = ""
    if truth is not None:
        tr = '<div class="ok">match</div>' if truth == label else f'<div class="bad">expected {html.escape(pretty(truth))}</div>'
    return (f'<div class="fx">{img}{cap}<div class="chip" style="background:{col}">{html.escape(pretty(label))}</div>'
            f'<div class="meter"><i style="width:{conf * 100:.0f}%;background:{col}"></i></div>'
            f'<div class="pct">{conf * 100:.0f}% confidence</div>{tr}</div>')


def cards_html(names, labels, confs, uris=None, truths=None):
    out = ""
    for i in range(len(labels)):
        out += card(names[i], labels[i], confs[i], uris[i] if uris else None, truths[i] if truths else None)
    return f'<div class="grid">{out}</div>'


def heatmap_html(s, rows, label_names):
    z = s - s.mean(axis=1, keepdims=True)
    z = z / (np.abs(z).max() + 1e-9)
    head = "".join(f"<th>{html.escape(pretty(l))}</th>" for l in label_names)
    body = ""
    for i in range(len(s)):
        tds = ""
        for j in range(len(label_names)):
            a = abs(z[i, j]) * 0.75
            c = "34,197,94" if z[i, j] > 0 else "239,68,68"
            tds += f'<td style="background:rgba({c},{a:.2f})">{s[i, j]:.2f}</td>'
        body += f"<tr><th>{html.escape(rows[i])}</th>{tds}</tr>"
    return f'<div class="hm"><table><tr><th></th>{head}</tr>{body}</table></div>'
