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
.hero{background-size:200% 200%;animation:flow 14s ease infinite;box-shadow:0 10px 30px rgba(79,70,229,.25)}
@keyframes flow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0 6px}
@media(max-width:800px){.tiles{grid-template-columns:repeat(2,1fr)}}
.tile{background:var(--secondary-background-color);border:1px solid rgba(128,128,128,.22);border-radius:14px;padding:14px 16px}
.tile b{display:block;font-size:24px;line-height:1.1}.tile span{font-size:12.5px;opacity:.65}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:8px 0 14px}
@media(max-width:800px){.steps{grid-template-columns:1fr}}
.step{border-radius:14px;padding:16px;border:1px solid rgba(128,128,128,.22);background:var(--secondary-background-color);position:relative}
.step .n{position:absolute;top:-12px;left:14px;background:linear-gradient(135deg,#4f46e5,#db2777);color:#fff;width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px}
.step h4{margin:6px 0 4px;font-size:15px}.step p{margin:0;font-size:13px;opacity:.75;line-height:1.45}
.strip{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 10px}
.strip div{padding:8px 14px;border-radius:12px;background:rgba(79,70,229,.1);font-size:13.5px}
.strip b{font-size:16px}
.alt{font-size:11.5px;opacity:.6;margin-top:3px}
.low{display:inline-block;font-size:10.5px;padding:1px 7px;border-radius:999px;background:rgba(245,158,11,.2);color:#b45309;margin-left:6px;font-weight:600}
.legend{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 4px}
.legend span{font-size:12px;padding:3px 10px;border-radius:999px;color:#fff;font-weight:600}
.callout{border-left:4px solid #4f46e5;background:rgba(79,70,229,.08);padding:10px 14px;border-radius:0 10px 10px 0;font-size:14px;margin:8px 0}
.callout.warn{border-color:#f59e0b;background:rgba(245,158,11,.1)}
.foot{margin-top:34px;padding:18px 20px;border-radius:16px;text-align:center;border:1px solid rgba(128,128,128,.22);background:var(--secondary-background-color);font-size:13.5px}
.foot a{color:#4f46e5;font-weight:600;text-decoration:none}.foot a:hover{text-decoration:underline}
.foot small{display:block;opacity:.6;margin-top:4px}
.guest{border-radius:18px;padding:20px 22px;margin:6px 0 12px;border:1px solid rgba(79,70,229,.35);
 background:linear-gradient(135deg,rgba(79,70,229,.10),rgba(219,39,119,.08))}
.guest h3{margin:0 0 6px}.guest p{margin:0 0 8px;opacity:.85;font-size:14.5px}
.guest ol{margin:6px 0 0 18px;padding:0;font-size:14px;line-height:1.7}
.badge{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;padding:3px 9px;border-radius:999px;
 background:rgba(22,163,74,.15);color:#15803d;margin-left:8px;vertical-align:middle}
.badge.warn{background:rgba(245,158,11,.18);color:#b45309}
.mcard{border-radius:14px;padding:12px 14px;border:1px solid rgba(128,128,128,.22);background:var(--secondary-background-color);font-size:13px;line-height:1.6}
.mcard b{font-size:13.5px}
.fx{transition:transform .15s ease,box-shadow .15s ease}.fx:hover{transform:translateY(-3px);box-shadow:0 8px 22px rgba(0,0,0,.12)}
.tile{transition:transform .15s ease}.tile:hover{transform:translateY(-2px)}
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


def card(title, label, conf, uri=None, truth=None, alt=None):
    col = COLORS.get(label, "#64748b")
    img = f'<img src="{uri}" alt="">' if uri else f'<div class="noimg">{html.escape(title)}</div>'
    cap = f'<div class="cap">{html.escape(title)}</div>' if uri else ""
    tr = ""
    if truth is not None:
        tr = '<div class="ok">match</div>' if truth == label else f'<div class="bad">expected {html.escape(pretty(truth))}</div>'
    low = '<span class="low">low</span>' if conf < 0.2 else ""
    altt = f'<div class="alt">next likely: {html.escape(pretty(alt[0]))} {alt[1] * 100:.0f}%</div>' if alt else ""
    return (f'<div class="fx">{img}{cap}<div class="chip" style="background:{col}">{html.escape(pretty(label))}</div>'
            f'<div class="meter"><i style="width:{conf * 100:.0f}%;background:{col}"></i></div>'
            f'<div class="pct">{conf * 100:.0f}% confidence{low}</div>{altt}{tr}</div>')


def cards_html(names, labels, confs, uris=None, truths=None, alts=None):
    out = ""
    for i in range(len(labels)):
        out += card(names[i], labels[i], confs[i], uris[i] if uris else None, truths[i] if truths else None, alts[i] if alts else None)
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


def tiles_html(items):
    t = "".join(f'<div class="tile"><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>' for a, b in items)
    return f'<div class="tiles">{t}</div>'


def steps_html(steps):
    t = ""
    for i in range(len(steps)):
        t += f'<div class="step"><div class="n">{i + 1}</div><h4>{html.escape(steps[i][0])}</h4><p>{html.escape(steps[i][1])}</p></div>'
    return f'<div class="steps">{t}</div>'


def legend_html(labels):
    t = "".join(f'<span style="background:{COLORS.get(l, "#64748b")}">{html.escape(pretty(l))}</span>' for l in labels)
    return f'<div class="legend">{t}</div>'


def strip_html(items):
    t = "".join(f"<div><b>{html.escape(a)}</b> {html.escape(b)}</div>" for a, b in items)
    return f'<div class="strip">{t}</div>'


def footer_html():
    return ('<div class="foot">Built by <a href="https://www.linkedin.com/in/gaurav-tomar-630b2a316" target="_blank" rel="noopener">Gaurav Tomar</a>'
            ' &middot; <a href="https://www.linkedin.com/in/gaurav-tomar-630b2a316" target="_blank" rel="noopener">Connect on LinkedIn</a>'
            '<small>Facial features by Py-Feat. Linear model and weights hosted on the Hugging Face Hub. '
            'Predictions describe the intended expression in a photo, not anyone\'s real feelings.</small></div>')
