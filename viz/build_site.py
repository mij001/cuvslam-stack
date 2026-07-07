#!/usr/bin/env python3
"""build_site.py — assemble every result this repo produces into one browsable
static HTML page (site/index.html).

Auto-discovers report directories (reports/, profiling/reports/, results/):
for each it renders the figures (figs/*.png — generate them first with
viz/make_figures.py), the machine-readable tables (*.csv, *.tsv), and the
written summary (*.md). All links are relative, so the page works opened
directly as a file OR served by dashboard/serve.py.

Usage:  python3 viz/build_site.py       # writes site/index.html
"""
from __future__ import annotations

import csv
import html
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "site", "index.html")
MAX_TABLE_ROWS = 25

CSS = """
:root { --bg:#fafafa; --card:#fff; --ink:#212121; --sub:#616161; --acc:#1565c0;
        --ok:#2e7d32; --line:#e0e0e0; }
* { box-sizing:border-box; }
body { margin:0; font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
       color:var(--ink); background:var(--bg); }
header { background:linear-gradient(120deg,#0d47a1,#1976d2); color:#fff;
         padding:26px 34px; }
header h1 { margin:0 0 4px; font-size:24px; }
header p  { margin:0; opacity:.85; font-size:14px; }
nav { position:sticky; top:0; background:#fff; border-bottom:1px solid var(--line);
      padding:8px 30px; z-index:5; overflow-x:auto; white-space:nowrap; }
nav a { color:var(--acc); text-decoration:none; margin-right:18px; font-size:13.5px; }
nav a:hover { text-decoration:underline; }
main { max-width:1180px; margin:0 auto; padding:26px 22px 60px; }
section { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:22px 26px; margin-bottom:26px; box-shadow:0 1px 3px rgba(0,0,0,.05); }
section h2 { margin:0 0 2px; font-size:19px; }
.sub { color:var(--sub); font-size:13px; margin:0 0 14px; }
.figgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
           gap:14px; margin:12px 0; }
.figgrid a { display:block; border:1px solid var(--line); border-radius:8px;
             overflow:hidden; background:#fff; }
.figgrid img { width:100%; height:auto; display:block; }
.figcap { font-size:12px; color:var(--sub); padding:6px 10px;
          border-top:1px solid var(--line); }
table { border-collapse:collapse; font-size:12.5px; width:100%; margin:8px 0; }
th,td { border:1px solid var(--line); padding:4px 8px; text-align:left; }
th { background:#f5f5f5; position:sticky; top:44px; }
tr:nth-child(even) td { background:#fafafa; }
details { margin:10px 0; }
summary { cursor:pointer; color:var(--acc); font-size:13.5px; }
.mdbox { background:#f7f9fc; border:1px solid var(--line); border-radius:8px;
         padding:12px 16px; font-size:13.5px; overflow-x:auto; }
.mdbox pre { white-space:pre-wrap; margin:0; font-size:12.5px; }
.badge { display:inline-block; background:#e3f2fd; color:#0d47a1; border-radius:12px;
         padding:1px 10px; font-size:11.5px; margin-left:8px; vertical-align:middle; }
.tablewrap { max-height:420px; overflow:auto; border:1px solid var(--line);
             border-radius:8px; }
footer { text-align:center; color:var(--sub); font-size:12.5px; padding:18px; }
"""


def rel(path):
    """repo path -> href relative to site/ (one level down from the root)."""
    return "../" + os.path.relpath(path, ROOT).replace(os.sep, "/")


def anchor(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def render_table(path):
    delim = "\t" if path.endswith(".tsv") else ","
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh, delimiter=delim))
    if not rows:
        return ""
    head, body = rows[0], rows[1:]
    if len(head) > 14:  # very wide tables stay downloadable, not rendered
        return (f'<p class="sub">{html.escape(os.path.basename(path))}: '
                f'{len(body)} rows × {len(head)} cols — '
                f'<a href="{rel(path)}">open raw</a></p>')

    def tr(cells, tag="td"):
        return "<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>"

    shown = body[:MAX_TABLE_ROWS]
    t = ['<div class="tablewrap"><table>', tr(head, "th")]
    t += [tr(r) for r in shown]
    t.append("</table></div>")
    more = (f'<p class="sub">showing {len(shown)} of {len(body)} rows — '
            f'<a href="{rel(path)}">full file</a></p>') if len(body) > len(shown) else ""
    label = html.escape(os.path.basename(path)) + f' <span class="badge">{len(body)} rows</span>'
    return f"<details><summary>table: {label}</summary>{''.join(t)}{more}</details>"


def render_md(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    title = ""
    m = re.match(r"#\s*(.+)", text)
    if m:
        title = m.group(1)
    return (f"<details><summary>notes: {html.escape(os.path.basename(path))}"
            f"{' — ' + html.escape(title) if title else ''}</summary>"
            f'<div class="mdbox"><pre>{html.escape(text)}</pre></div></details>')


def collect(dirpath):
    """figures / tables / notes inside one report dir (figs/ + data/ + top level)."""
    figs, tables, notes = [], [], []
    for base, _dirs, files in os.walk(dirpath):
        depth = os.path.relpath(base, dirpath).count(os.sep)
        if depth > 1:
            continue
        for f in sorted(files):
            p = os.path.join(base, f)
            if f.endswith(".png") and f"{os.sep}figs" in base:
                figs.append(p)
            elif f.endswith((".csv", ".tsv")):
                tables.append(p)
            elif f.endswith(".md"):
                notes.append(p)
    return figs, tables, notes


def first_paragraph(md_paths):
    for p in md_paths:
        if os.path.basename(p).upper().startswith(("SUMMARY", "NOTES", "README")):
            text = open(p, encoding="utf-8", errors="replace").read()
            body = re.sub(r"^#.*\n", "", text).strip()
            para = body.split("\n\n")[0].replace("\n", " ")
            para = re.sub(r"[*_`#>]", "", para)
            return (para[:340] + "…") if len(para) > 340 else para
    return ""


def section_html(title, dirpath, blurb=""):
    figs, tables, notes = collect(dirpath)
    if not (figs or tables or notes):
        return "", None
    intro = blurb or first_paragraph(notes)
    parts = [f'<section id="{anchor(title)}">',
             f"<h2>{html.escape(title)}"
             f'<span class="badge">{len(figs)} figures · {len(tables)} tables</span></h2>',
             f'<p class="sub">{html.escape(os.path.relpath(dirpath, ROOT))}'
             + (f" — {html.escape(intro)}" if intro else "") + "</p>"]
    if figs:
        parts.append('<div class="figgrid">')
        for f in figs:
            cap = os.path.splitext(os.path.basename(f))[0].replace("_", " ")
            parts.append(f'<a href="{rel(f)}" target="_blank"><img loading="lazy" '
                         f'src="{rel(f)}" alt="{html.escape(cap)}">'
                         f'<div class="figcap">{html.escape(cap)}</div></a>')
        parts.append("</div>")
    parts += [render_table(t) for t in tables]
    parts += [render_md(n) for n in notes]
    parts.append("</section>")
    return "\n".join(parts), title


def main():
    sections = []

    # headline reports first (newest → oldest), then the per-device deep dives
    ordered = [
        ("Substrate candidacy — GPU / CPU / PiM / ISP per kernel + dynamics",
         "reports/2026-07-07_substrate"),
        ("Accuracy matrix — 141 runs vs the cuVSLAM paper",
         "reports/2026-07-07_accuracy_full"),
        ("Profiler neutrality — nsys / ncu / NVBit",
         "reports/2026-07-07_profiler_neutrality"),
        ("Profiling coverage — 192 feature-toggle variants",
         "reports/2026-07-07_profiling_coverage"),
        ("Accuracy validation (initial 104-run matrix)",
         "profiling/reports/2026-07-06_accuracy"),
        ("Attribution campaign — 27 sequences, per-kernel data structures",
         "profiling/reports/2026-07-05_attribution_campaign"),
        ("Memory attribution (TaggedAllocator + NVTX)",
         "profiling/reports/2026-07-04_attribution"),
        ("Full-scale characterization campaign — 27 sequences",
         "profiling/reports/2026-07-04_campaign"),
        ("Slice 3 — memory locality (NVBit traces)",
         "profiling/reports/2026-07-04_slice3_locality"),
        ("PiM placement model", "results"),
        ("Matrix synthesis (cross-dataset)", "profiling/reports/2026-07-03_matrix_synthesis"),
        ("Device report — TUM office, RTX 2000 Ada",
         "profiling/reports/2026-07-03_tum_office_rtx2000ada"),
        ("Device report — KITTI 06, RTX 2000 Ada",
         "profiling/reports/2026-07-03_kitti06_rtx2000ada"),
        ("Device report — TUM-VI corridor1, RTX 2000 Ada",
         "profiling/reports/2026-07-03_tumvi_corridor1_rtx2000ada"),
        ("Device report — TUM office, MX450",
         "profiling/reports/2026-07-02_tum_office_mx450"),
    ]
    toc = []
    for title, sub in ordered:
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        htm, name = section_html(title, d)
        if htm:
            sections.append(htm)
            toc.append(f'<a href="#{anchor(title)}">{html.escape(title)}</a>')

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>cuvslam-stack — results</title><style>{CSS}</style></head>
<body>
<header><h1>cuvslam-stack — results</h1>
<p>GPU memory characterization of NVIDIA cuVSLAM for PiM/ISP research —
every report, figure and table this repo produces.
Regenerate with <code>viz/make_figures.py</code> + <code>viz/build_site.py</code>.</p>
</header>
<nav>{''.join(toc)}</nav>
<main>
{chr(10).join(sections)}
</main>
<footer>built by viz/build_site.py — figures are PNG counterparts of the committed CSV/TSV artifacts</footer>
</body></html>"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(page)
    print(f"[site] {os.path.relpath(OUT, ROOT)}  ({len(sections)} sections)")


if __name__ == "__main__":
    main()
