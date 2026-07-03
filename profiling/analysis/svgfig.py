"""svgfig.py — dependency-free SVG charts for headless analysis.

Deliberately stdlib-only: the analysis layer must run on any headless box (dev
laptop, workstation, CI) without pip installs, so figures are hand-emitted SVG.
GitHub / any browser renders them; they diff cleanly in git.

Provides just the three chart types the characterization report needs:
  hbar        — horizontal bars (stage time share, per-stage bandwidth)
  stacked_hbar— stacked horizontal bars (stall breakdown)
  roofline    — log-log scatter with bandwidth/compute ceilings
"""
from __future__ import annotations

import math
from html import escape

# stage palette (colorblind-safe-ish); '.other' grey
PALETTE = ["#4477aa", "#66ccee", "#228833", "#ccbb44", "#ee6677",
           "#aa3377", "#bbbbbb", "#999933", "#882255", "#44aa99"]

FONT = 'font-family="DejaVu Sans, Helvetica, Arial, sans-serif"'


def _header(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}">\n'
            f'<rect width="{w}" height="{h}" fill="white"/>\n')


def _txt(x, y, s, size=12, anchor="start", color="#222", bold=False, rotate=None):
    weight = ' font-weight="bold"' if bold else ""
    rot = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" {FONT}{weight}{rot}>{escape(str(s))}</text>\n')


def _fmt(v):
    if v == 0:
        return "0"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.3g}"
    return f"{v:.2g}"


def hbar(path, title, labels, values, unit="", colors=None, annotations=None,
         width=760):
    """Horizontal bar chart. annotations: optional per-bar right-side text."""
    n = len(labels)
    bar_h, gap, left, right, top = 26, 10, 230, 110, 48
    h = top + n * (bar_h + gap) + 30
    vmax = max([v for v in values if v == v] + [1e-12])
    span = width - left - right
    s = _header(width, h)
    s += _txt(width / 2, 24, title, 15, "middle", bold=True)
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = top + i * (bar_h + gap)
        col = (colors[i] if colors else PALETTE[i % len(PALETTE)])
        w = 0 if val != val else max(1.0, span * val / vmax)
        s += _txt(left - 8, y + bar_h * 0.7, lab, 12, "end")
        s += f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{col}" rx="2"/>\n'
        note = annotations[i] if annotations else f"{_fmt(val)} {unit}".strip()
        s += _txt(left + w + 6, y + bar_h * 0.7, note, 11)
    s += "</svg>\n"
    open(path, "w").write(s)


def stacked_hbar(path, title, row_labels, series_labels, matrix, unit="",
                 width=860):
    """Stacked horizontal bars. matrix[row][series] = value."""
    n = len(row_labels)
    bar_h, gap, left, right, top = 24, 8, 250, 30, 70
    h = top + n * (bar_h + gap) + 40
    vmax = max([sum(v for v in row if v == v) for row in matrix] + [1e-12])
    span = width - left - right
    s = _header(width, h)
    s += _txt(width / 2, 22, title, 15, "middle", bold=True)
    # legend
    lx = left
    for j, sl in enumerate(series_labels):
        col = PALETTE[j % len(PALETTE)]
        s += f'<rect x="{lx}" y="34" width="12" height="12" fill="{col}" rx="2"/>\n'
        s += _txt(lx + 16, 44, sl, 11)
        lx += 16 + 7.2 * len(sl) + 18
    for i, (lab, row) in enumerate(zip(row_labels, matrix)):
        y = top + i * (bar_h + gap)
        s += _txt(left - 8, y + bar_h * 0.7, lab, 11.5, "end")
        x = float(left)
        for j, val in enumerate(row):
            if val != val or val <= 0:
                continue
            w = span * val / vmax
            col = PALETTE[j % len(PALETTE)]
            s += (f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" '
                  f'fill="{col}"><title>{escape(series_labels[j])}: {_fmt(val)} {unit}</title></rect>\n')
            x += w
        s += _txt(x + 5, y + bar_h * 0.7, f"{_fmt(sum(v for v in row if v == v))} {unit}".strip(), 10.5)
    s += "</svg>\n"
    open(path, "w").write(s)


def roofline(path, title, points, dram_gbps, fp32_tflops, l2_gbps=None,
             width=820, height=560):
    """Log-log roofline.

    points: list of (label, ai_flop_per_byte, gflops, color, radius)
    Ceilings: DRAM bandwidth (GB/s), optional L2 bandwidth, FP32 peak (TFLOP/s).
    """
    left, right, top, bottom = 70, 210, 56, 56
    px_w, px_h = width - left - right, height - top - bottom
    peak_gf = fp32_tflops * 1000.0

    ais = [p[1] for p in points if p[1] > 0] or [0.1, 10]
    x_min = 10 ** math.floor(math.log10(min(min(ais), 0.05)))
    x_max = 10 ** math.ceil(math.log10(max(max(ais), peak_gf / dram_gbps * 4)))
    perfs = [p[2] for p in points if p[2] > 0] or [1, peak_gf]
    y_min = 10 ** math.floor(math.log10(min(perfs)))
    y_max = 10 ** math.ceil(math.log10(peak_gf * 1.5))

    def X(v):
        return left + px_w * (math.log10(v) - math.log10(x_min)) / (math.log10(x_max) - math.log10(x_min))

    def Y(v):
        return top + px_h * (1 - (math.log10(v) - math.log10(y_min)) / (math.log10(y_max) - math.log10(y_min)))

    s = _header(width, height)
    s += _txt(width / 2 - right / 2, 24, title, 15, "middle", bold=True)

    # gridlines (decades)
    d = int(math.log10(x_min))
    while 10 ** d <= x_max:
        v = 10 ** d
        s += (f'<line x1="{X(v):.1f}" y1="{top}" x2="{X(v):.1f}" y2="{top+px_h}" '
              f'stroke="#e5e5e5"/>\n')
        s += _txt(X(v), top + px_h + 18, f"1e{d}" if abs(d) > 2 else _fmt(v), 10.5, "middle")
        d += 1
    d = int(math.log10(y_min))
    while 10 ** d <= y_max:
        v = 10 ** d
        s += (f'<line x1="{left}" y1="{Y(v):.1f}" x2="{left+px_w}" y2="{Y(v):.1f}" '
              f'stroke="#e5e5e5"/>\n')
        s += _txt(left - 6, Y(v) + 4, f"1e{d}" if abs(d) > 3 else _fmt(v), 10.5, "end")
        d += 1

    def ceiling(bw_gbps, label, color):
        # memory roof: perf = AI * BW until it hits the compute roof
        knee = peak_gf / bw_gbps
        x0, y0 = max(x_min, y_min / bw_gbps), None
        y0 = x0 * bw_gbps
        pts = f"{X(x0):.1f},{Y(y0):.1f} {X(min(knee, x_max)):.1f},{Y(min(knee, x_max)*bw_gbps):.1f}"
        out = f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>\n'
        mid_ai = math.sqrt(x0 * min(knee, x_max))
        out += _txt(X(mid_ai) + 4, Y(mid_ai * bw_gbps) - 8, label, 11, color=color, rotate=-32)
        return out

    s += ceiling(dram_gbps, f"DRAM {_fmt(dram_gbps)} GB/s", "#cc3311")
    if l2_gbps:
        s += ceiling(l2_gbps, f"L2 {_fmt(l2_gbps)} GB/s", "#ee7733")
    # compute roof
    knee = peak_gf / dram_gbps
    s += (f'<line x1="{X(max(knee, x_min)):.1f}" y1="{Y(peak_gf):.1f}" '
          f'x2="{left+px_w}" y2="{Y(peak_gf):.1f}" stroke="#0077bb" stroke-width="2"/>\n')
    s += _txt(left + px_w - 4, Y(peak_gf) - 8, f"FP32 peak {_fmt(peak_gf)} GFLOP/s", 11,
              "end", "#0077bb")

    # points + side legend
    ly = top + 6
    for label, ai, gf, color, r in points:
        if ai <= 0 or gf <= 0:
            continue
        s += (f'<circle cx="{X(ai):.1f}" cy="{Y(gf):.1f}" r="{r}" fill="{color}" '
              f'fill-opacity="0.85" stroke="#333" stroke-width="0.6">'
              f'<title>{escape(label)}: AI={_fmt(ai)} FLOP/B, {_fmt(gf)} GFLOP/s</title></circle>\n')
        s += f'<circle cx="{left+px_w+16}" cy="{ly}" r="5" fill="{color}"/>\n'
        s += _txt(left + px_w + 26, ly + 4, label[:30], 10.5)
        ly += 17

    # axes + labels
    s += (f'<rect x="{left}" y="{top}" width="{px_w}" height="{px_h}" fill="none" '
          f'stroke="#888"/>\n')
    s += _txt(left + px_w / 2, height - 14, "Arithmetic intensity (FLOP / DRAM byte)", 12.5, "middle")
    s += _txt(18, top + px_h / 2, "Performance (GFLOP/s)", 12.5, "middle", rotate=-90)
    s += "</svg>\n"
    open(path, "w").write(s)


def scatter(path, title, points, x_label, y_label, x_max=None, y_max=None,
            legend=None, width=820, height=520):
    """Linear scatter. points: (label, x, y, color, radius). legend: [(label,color)]."""
    left, right, top, bottom = 70, 230, 56, 56
    px_w, px_h = width - left - right, height - top - bottom
    xs = [p[1] for p in points if p[1] == p[1]] or [1.0]
    ys = [p[2] for p in points if p[2] == p[2]] or [1.0]
    x_max = x_max or max(xs) * 1.15
    y_max = y_max or max(ys) * 1.15

    def X(v):
        return left + px_w * min(v, x_max) / x_max

    def Y(v):
        return top + px_h * (1 - min(v, y_max) / y_max)

    s = _header(width, height)
    s += _txt(width / 2 - right / 2, 24, title, 15, "middle", bold=True)
    for i in range(6):
        xv, yv = x_max * i / 5, y_max * i / 5
        s += f'<line x1="{X(xv):.1f}" y1="{top}" x2="{X(xv):.1f}" y2="{top+px_h}" stroke="#e5e5e5"/>\n'
        s += f'<line x1="{left}" y1="{Y(yv):.1f}" x2="{left+px_w}" y2="{Y(yv):.1f}" stroke="#e5e5e5"/>\n'
        s += _txt(X(xv), top + px_h + 18, _fmt(xv), 10.5, "middle")
        s += _txt(left - 6, Y(yv) + 4, _fmt(yv), 10.5, "end")
    for label, x, y, color, r in points:
        if x != x or y != y:
            continue
        s += (f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{r}" fill="{color}" '
              f'fill-opacity="0.85" stroke="#333" stroke-width="0.6">'
              f'<title>{escape(label)}: x={_fmt(x)}, y={_fmt(y)}</title></circle>\n')
    ly = top + 6
    for lab, color in (legend or []):
        s += f'<rect x="{left+px_w+16}" y="{ly-9}" width="11" height="11" fill="{color}" rx="2"/>\n'
        s += _txt(left + px_w + 32, ly, lab[:34], 10.5)
        ly += 17
    s += f'<rect x="{left}" y="{top}" width="{px_w}" height="{px_h}" fill="none" stroke="#888"/>\n'
    s += _txt(left + px_w / 2, height - 14, x_label, 12.5, "middle")
    s += _txt(18, top + px_h / 2, y_label, 12.5, "middle", rotate=-90)
    s += "</svg>\n"
    open(path, "w").write(s)


def stage_color(stage: str, order: list[str]) -> str:
    try:
        return PALETTE[order.index(stage) % len(PALETTE)]
    except ValueError:
        return "#bbbbbb"
