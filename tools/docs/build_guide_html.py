"""Build a single self-contained HTML (sidebar TOC, embedded images, no external deps) from a Markdown guide.

  python tools/docs/build_guide_html.py                          # COMPLETE_USER_GUIDE_EN.md -> .html
  python tools/docs/build_guide_html.py SERVER_USER_MANUAL_EN.md # any guide written in the same style

Conventions the Markdown must follow:
  * one H1 (becomes the hero title), H2 = parts, H3 = numbered sections (they form the sidebar);
  * images are referenced as manual_images/... relative to the repo root and are embedded as data URIs
    (so the HTML can be copied to any computer on its own); a paragraph with image(s) followed by an
    italic paragraph becomes a captioned figure ("screen_*" images are drawn phone-sized);
  * callouts are blockquotes starting with **Note.**, **Tip.** or **Warning.**;
  * anchors are GitHub-style slugs; the script checks that every #link and every "section N" reference
    resolves to a real heading and exits non-zero otherwise.
Needs: pip install markdown
"""
import base64, re, sys
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[2]
src = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "COMPLETE_USER_GUIDE_EN.md")
out = src.with_suffix(".html")

def slugify(value, sep):
    value = re.sub(r"[^\w\- ]", "", value).strip().lower()
    return value.replace(" ", sep)

text = src.read_text(encoding="utf-8")
# adjacent blockquotes would merge into one box: separate them with an HTML comment
src_text = text
text = re.sub(r"(\n> [^\n]*)\n\n(> \*\*(?:Note|Tip|Warning)\.)",
              lambda mm: mm.group(1) + "\n\n<!-- -->\n\n" + mm.group(2), text)
md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                       extension_configs={"toc": {"slugify": slugify}})
body = md.convert(text)

# ---- hero: pull the H1 out ---------------------------------------------------
m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
title = re.sub(r"<[^>]+>", "", m.group(1))
body = body.replace(m.group(0), "", 1)

# ---- figures: <p><img..>[<img..>]</p><p><em>caption</em></p> ------------------
def figure(m):
    imgs, cap = m.group(1), m.group(2)
    kind = "screens" if "screen_" in imgs else ("photos" if "manual_images/" in imgs and "ui/" not in imgs else "diagram")
    return f'<figure class="fig {kind}"><div class="imgrow">{imgs}</div><figcaption>{cap}</figcaption></figure>'
body = re.sub(r"<p>((?:<img[^>]*>\s*)+)</p>\s*<p><em>(.*?)</em></p>", figure, body, flags=re.S)

# ---- callouts -------------------------------------------------------------------
def callout(m):
    kind = m.group(1).lower()
    cls = {"note": "note", "tip": "tip", "warning": "warn"}[kind]
    return f'<div class="{cls}"><p><strong>{m.group(1)}.</strong>{m.group(2)}'
body = re.sub(r"<blockquote>\s*<p><strong>(Note|Tip|Warning)\.</strong>(.*?)(?=</blockquote>)",
              lambda mm: callout(mm) + "</div>", body, flags=re.S)
# the regex above leaves "</p>" inside the captured text; close any blockquote remnants
body = re.sub(r"</div>\s*</blockquote>", "</div>", body)

# ---- wrap tables --------------------------------------------------------------
body = body.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")

# ---- embed images ---------------------------------------------------------------
MIME = {".svg": "image/svg+xml", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
def embed(m):
    rel = m.group(1)
    p = ROOT / rel
    data = base64.b64encode(p.read_bytes()).decode()
    return f'src="data:{MIME[p.suffix.lower()]};base64,{data}"'
body, n_img = re.subn(r'src="(manual_images/[^"]+)"', embed, body)
logo = "data:image/svg+xml;base64," + base64.b64encode((ROOT / "manual_images/ui/app_logo.svg").read_bytes()).decode()

# ---- sidebar TOC: h2 (parts) with nested h3 (sections) ---------------------------
heads = re.findall(r'<h([23]) id="([^"]+)">(.*?)</h\1>', body, re.S)
nav, open_ul = [], False
for lvl, hid, txt in heads:
    t = re.sub(r"<[^>]+>", "", txt)
    if lvl == "2":
        if open_ul:
            nav.append("</ul>"); open_ul = False
        nav.append(f'<a class="part" href="#{hid}">{t}</a><ul>'); open_ul = True
    else:
        nav.append(f'<li><a href="#{hid}">{t}</a></li>')
if open_ul:
    nav.append("</ul>")
nav_html = "\n".join(nav)

# ---- checks ------------------------------------------------------------------------
ids = set(re.findall(r'id="([^"]+)"', body))
bad_links = sorted({h for h in re.findall(r'href="#([^"]+)"', body) if h not in ids})
numbers = set(re.findall(r'<h[234] id="[^"]+">(\d+(?:\.\d+)?)\.? ', body))
plain = re.sub(r"<[^>]+>", " ", re.sub(r"<pre.*?</pre>", lambda mm: mm.group(0), body, flags=re.S))
refs = set(re.findall(r"\bsection (\d+(?:\.\d+)?)", re.sub(r'src="data:[^"]+"', "", src_text)))
bad_refs = sorted(r for r in refs if r not in numbers)
leftover = re.findall(r'(?:src|href)="(?!#|data:|http)[^"]+"', body)
ext = sorted(set(re.findall(r'(?:src|href)="(http[^"]+)"', body)))

css = """
:root{--bg:#fff;--fg:#1d1d1f;--mute:#6e6e73;--line:#e5e5ea;--card:#f5f5f7;--blue:#0a84ff;--green:#1f9d4d;--radius:16px;color-scheme:light}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:16px}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;-webkit-font-smoothing:antialiased}
.layout{display:grid;grid-template-columns:300px minmax(0,1fr);min-height:100vh}
nav.toc{position:sticky;top:0;height:100vh;overflow:auto;padding:26px 18px;border-right:1px solid var(--line);background:var(--card)}
nav.toc .brand{display:flex;align-items:center;gap:10px;font-weight:700;margin-bottom:14px;line-height:1.25;font-size:15px}
nav.toc .brand img{width:34px;height:34px;border:0;border-radius:8px}
nav.toc ul{list-style:none;margin:0 0 8px;padding:0}
nav.toc a{display:block;padding:4px 10px;border-radius:8px;color:var(--fg);text-decoration:none;font-size:13.5px;line-height:1.35}
nav.toc a.part{font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);padding-top:12px}
nav.toc a:hover{background:#e8e8ed}
main{padding:0 clamp(20px,5vw,64px) 80px;max-width:1160px;width:100%}
.hero{margin:0 calc(-1 * clamp(20px,5vw,64px)) 36px;padding:52px clamp(20px,5vw,64px) 42px;color:#fff;background:radial-gradient(1200px 400px at 85% -10%,rgba(191,90,242,.55),transparent 60%),linear-gradient(135deg,#0a2a66,#0a84ff 60%,#5e5ce6)}
.hero img{width:84px;height:84px;border:0;border-radius:20px;box-shadow:0 10px 30px rgba(0,0,0,.3)}
.hero h1{font-size:clamp(30px,4.6vw,46px);line-height:1.12;margin:20px 0 10px;letter-spacing:-.02em}
.hero p{font-size:18px;max-width:780px;margin:0;opacity:.95}
h2{font-size:30px;line-height:1.2;margin:80px 0 14px;letter-spacing:-.015em;padding:10px 0 0;border-top:3px solid var(--blue)}
h3{font-size:23px;margin:54px 0 8px;padding-top:12px;border-top:1px solid var(--line)}
h4{font-size:18px;margin:30px 0 6px}
h5{font-size:15px;margin:22px 0 6px;color:var(--mute);text-transform:uppercase;letter-spacing:.06em}
p{margin:10px 0} a{color:var(--blue)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.9em;background:var(--card);padding:2px 6px;border-radius:6px}
pre{background:#14161a;color:#e6e8ec;border-radius:12px;padding:16px 20px;overflow:auto;font:13.5px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin:16px 0}
pre code{background:none;padding:0;font-size:inherit;color:inherit;white-space:pre}
table{border-collapse:collapse;width:100%;margin:0;font-size:14.5px}
.tablewrap{overflow-x:auto;margin:16px 0}
th,td{border:1px solid var(--line);padding:8px 12px;text-align:left;vertical-align:top}
th{background:var(--card);font-weight:650}
td code{white-space:nowrap}
.note,.tip,.warn{border-left:5px solid;border-radius:10px;padding:10px 16px;margin:18px 0}
.note{background:#eef5ff;border-color:var(--blue)} .tip{background:#eefaf1;border-color:#34c759} .warn{background:#fff6e8;border-color:#ff9500}
.note p,.tip p,.warn p{margin:4px 0}
blockquote{margin:16px 0;padding:6px 16px;border-left:4px solid var(--line);background:var(--card);color:#3a3a3c;border-radius:0 10px 10px 0}
img{max-width:100%;height:auto}
figure.fig{margin:26px 0;text-align:center}
figure.fig figcaption{font-size:14px;color:var(--mute);margin-top:8px}
figure.diagram .imgrow{background:#fff;border:1px solid var(--line);border-radius:var(--radius);padding:14px}
figure.diagram img{width:100%;display:block}
figure.screens .imgrow{display:flex;flex-wrap:wrap;gap:34px;justify-content:center}
figure.screens img{width:268px;filter:drop-shadow(0 16px 22px rgba(0,0,0,.28))}
figure.photos img{border:1px solid var(--line);border-radius:8px}
hr{border:0;border-top:1px solid var(--line);margin:34px 0}
.foot{margin-top:60px;color:var(--mute);font-size:13.5px;border-top:1px solid var(--line);padding-top:14px}
@media (max-width:980px){.layout{display:block}nav.toc{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}}
@media print{nav.toc{display:none}.layout{display:block}main{max-width:none;padding:0 12px}.hero{margin:0 -12px 20px}
h2{break-after:avoid;margin-top:40px}h3,h4{break-after:avoid}figure,table,pre,.note,.tip,.warn{break-inside:avoid}
pre{white-space:pre-wrap;word-break:break-word}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style></head>
<body><div class="layout">
<nav class="toc"><div class="brand"><img src="{logo}" alt=""><span>{title}</span></div>
{nav_html}
</nav>
<main>
<header class="hero"><img src="{logo}" alt="Object Scanner"><h1>{title}</h1>
<p>Scan an object or a room with an iPhone or iPad, train a 3D Gaussian Splatting model on a GPU server, and look at it from every side.
One guide for the app, the server and the way they work together.</p></header>
{body}
<div class="foot">This document needs no internet: styles and illustrations are embedded in the file. To get a PDF, open it in a browser and choose Print, then Save as PDF.</div>
</main></div></body></html>
"""
out.write_text(html, encoding="utf-8")
print(f"wrote {out.name}  {out.stat().st_size/1024:.0f} KB  images embedded={n_img}")
print("broken anchors:", bad_links or "none")
print("unknown 'section N' refs:", bad_refs or "none")
print("non-embedded relative refs:", leftover or "none")
print("external refs:", ext or "none")
sys.exit(1 if (bad_links or bad_refs or leftover) else 0)
