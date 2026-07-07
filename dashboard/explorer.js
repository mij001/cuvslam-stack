/* explorer.js — the interactive evidence explorer.
 *
 * Renders one profiled run's summary.json as the bottleneck->improvement
 * reasoning chain the studies used, four linked panels:
 *   1 stage time bar   WHERE the GPU time goes (click a stage to filter)
 *   2 kernel list      WHO dominates (to 90% cumulative; limiter + substrate)
 *   3 evidence panel   WHY — the selected kernel's metrics vs the decision
 *                      thresholds, its dominant stall, classifier rationale,
 *                      and the substrate verdict that follows
 *   4 roofline         WHERE it sits vs the memory wall (hover = numbers)
 *
 * No dependencies; every number shown is straight from summary.json, which
 * every adapter's run produces identically (see profiling/analysis/
 * summarize_run.py for the schema).
 */
/* eslint-disable */
(function () {
  const $ = s => document.querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  const STAGE_COLORS = ["#1565c0", "#6a1b9a", "#2e7d32", "#ef6c00", "#00838f",
                        "#c62828", "#5d4037", "#455a64"];
  const LIMITER_COLORS = {
    "G1-bandwidth": "#ef6c00", "G2-coalescing": "#c62828", "G3-cache": "#6a1b9a",
    "G4-latency": "#00838f", "G5-compute": "#2e7d32", "G6-atomics": "#ad1457",
    "G7-dependency": "#455a64", "G8-transfer": "#5d4037",
  };
  // decision thresholds — the same cut lines the classifier/study used
  const THRESH = [
    ["dram_sol_pct", "DRAM utilisation", "%", 60, "≥60% ⇒ DRAM-bandwidth-bound", 100],
    ["sectors_per_req", "sectors / request", "", 8, "≤8 coalesced · ≥16 scattered gather", 32],
    ["occupancy_pct", "occupancy", "%", 25, "<25% ⇒ latency/dependency limited", 100],
    ["lfmr", "L2 miss ratio (LFMR)", "", 0.5, "≥0.5 ⇒ cache-defeating, DRAM-visible", 1],
    ["mpki", "misses / kilo-instr", "", 30, "≥30 ⇒ memory-intensive (DAMOV)", 130],
  ];

  let DATA = null, selStage = null, selKernel = null;

  function el(tag, attrs, text) {
    const e = tag === "svg" || tag === "rect" || tag === "circle" || tag === "line" || tag === "text" || tag === "g"
      ? document.createElementNS(NS, tag) : document.createElement(tag);
    for (const k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  // ── panel 1: stage bar ────────────────────────────────────────────────────
  function drawStages() {
    const host = $("#xp-stages"); host.innerHTML = "";
    const W = host.clientWidth || 900, H = 64;
    const svg = el("svg", { width: "100%", height: H, viewBox: `0 0 ${W} ${H}` });
    let x = 0;
    const stages = DATA.stages.filter(s => s.share_pct > 0);
    stages.forEach((s, i) => {
      const w = Math.max(2, W * s.share_pct / 100);
      const r = el("rect", { x, y: 6, width: w - 1, height: 30, rx: 3,
        fill: STAGE_COLORS[i % STAGE_COLORS.length],
        opacity: (selStage && selStage !== s.name) ? 0.25 : 0.92, cursor: "pointer" });
      r.addEventListener("click", () => { selStage = selStage === s.name ? null : s.name; render(); });
      const tip = el("title", {}, `${s.name} — ${s.share_pct}% of GPU time (${s.time_ms} ms, ${s.n_kernels} kernels)\nclick to filter the kernel list`);
      r.appendChild(tip); svg.appendChild(r);
      if (w > 56) svg.appendChild(el("text", { x: x + 5, y: 26, fill: "#fff",
        "font-size": 11 }, `${s.name} ${s.share_pct}%`));
      x += w;
    });
    svg.appendChild(el("text", { x: 0, y: 54, fill: "#666", "font-size": 11 },
      selStage ? `filtered: ${selStage} — click the segment again to clear`
               : "GPU time by pipeline stage — click a segment to focus it"));
    host.appendChild(svg);
  }

  // ── panel 2: dominant kernels ────────────────────────────────────────────
  function drawKernels() {
    const host = $("#xp-kernels"); host.innerHTML = "";
    let ks = DATA.kernels.filter(k => !selStage || k.stage === selStage);
    const totalShare = ks.reduce((a, k) => a + k.share_pct, 0) || 1;
    // top rows until 90% of the (filtered) time, rest folded into a tail row
    let cum = 0; const rows = []; let tail = { n: 0, share: 0 };
    for (const k of ks) {
      if (cum < 0.9 * totalShare && rows.length < 12) { rows.push(k); cum += k.share_pct; }
      else { tail.n++; tail.share += k.share_pct; }
    }
    const maxShare = rows[0] ? rows[0].share_pct : 1;
    for (const k of rows) {
      const row = el("div", { class: "xp-row" + (selKernel === k.name ? " xp-sel" : "") });
      row.addEventListener("click", () => { selKernel = k.name; render(); });
      const bar = el("div", { class: "xp-bar" });
      bar.appendChild(el("div", { class: "xp-fill",
        style: `width:${100 * k.share_pct / maxShare}%` }));
      const lim = el("span", { class: "xp-chip",
        style: `background:${LIMITER_COLORS[k.limiter] || "#607d8b"}` }, k.limiter);
      const sub = el("span", { class: "xp-chip xp-sub" },
        (k.substrate || "?").split("—")[0].trim().slice(0, 26));
      const name = el("span", { class: "xp-name", title: k.name },
        k.name.length > 34 ? k.name.slice(0, 33) + "…" : k.name);
      const pct = el("span", { class: "xp-pct" }, k.share_pct.toFixed(1) + "%");
      row.append(pct, bar, name, lim, sub);
      host.appendChild(row);
    }
    if (tail.n) host.appendChild(el("div", { class: "xp-row xp-tail" },
      `… ${tail.n} more kernels, ${tail.share.toFixed(1)}% combined`));
    if (!selKernel && rows[0]) { selKernel = rows[0].name; drawEvidence(); drawRoofline(); }
  }

  // ── panel 3: evidence ────────────────────────────────────────────────────
  function drawEvidence() {
    const host = $("#xp-evidence"); host.innerHTML = "";
    const k = DATA.kernels.find(x => x.name === selKernel);
    if (!k) { host.textContent = "select a kernel"; return; }
    host.appendChild(el("div", { class: "xp-evtitle" }, k.name));
    host.appendChild(el("div", { class: "xp-evsub" },
      `${k.stage} · ${k.time_ms} ms (${k.share_pct}% of GPU time) · limiter ${k.limiter}` +
      (k.evidence.dominant_stall ? ` · dominant stall: ${k.evidence.dominant_stall}` : "")));
    for (const [key, label, unit, cut, note, max] of THRESH) {
      const v = k.evidence[key];
      if (v == null) continue;
      const wrap = el("div", { class: "xp-metric" });
      wrap.appendChild(el("div", { class: "xp-mlabel" },
        `${label}: ${v}${unit}  `));
      const bar = el("div", { class: "xp-mbar", title: note });
      const fillw = Math.min(100, 100 * v / max);
      bar.appendChild(el("div", { class: "xp-mfill", style: `width:${fillw}%` }));
      bar.appendChild(el("div", { class: "xp-mcut", style: `left:${100 * cut / max}%`,
        title: `threshold ${cut}${unit} — ${note}` }));
      wrap.appendChild(bar);
      wrap.appendChild(el("div", { class: "xp-mnote" }, note));
      host.appendChild(wrap);
    }
    if (k.rationale) host.appendChild(el("div", { class: "xp-rat" },
      "classifier rationale: " + k.rationale));
    const verdict = el("div", { class: "xp-verdict" });
    verdict.appendChild(el("b", {}, "⇒ best substrate: "));
    verdict.appendChild(el("span", {}, k.substrate || "?"));
    verdict.appendChild(el("div", { class: "xp-mnote" },
      `PiM affinity: ${k.pim_affinity} — the metrics above are the evidence trail for this verdict`));
    host.appendChild(verdict);
  }

  // ── panel 4: roofline ────────────────────────────────────────────────────
  function drawRoofline() {
    const host = $("#xp-roofline"); host.innerHTML = "";
    const pts = DATA.kernels.filter(k => k.roofline && k.roofline.ai > 0 && k.roofline.gflops > 0);
    if (!pts.length) { host.textContent = "no roofline data in this run"; return; }
    const W = host.clientWidth || 440, H = 250, m = 42;
    const svg = el("svg", { width: "100%", height: H, viewBox: `0 0 ${W} ${H}` });
    const lx = v => Math.log10(v), xs = pts.map(p => lx(p.roofline.ai)), ys = pts.map(p => lx(p.roofline.gflops));
    const x0 = Math.min(...xs) - 0.3, x1 = Math.max(...xs) + 0.3;
    const y0 = Math.min(...ys) - 0.3, y1 = Math.max(...ys) + 0.3;
    const X = v => m + (W - m - 10) * (lx(v) - x0) / (x1 - x0);
    const Y = v => H - 26 - (H - 44) * (lx(v) - y0) / (y1 - y0);
    // gridlines at decades
    for (let d = Math.ceil(x0); d <= Math.floor(x1); d++) {
      svg.appendChild(el("line", { x1: X(10 ** d), y1: 8, x2: X(10 ** d), y2: H - 26, stroke: "#eee" }));
      svg.appendChild(el("text", { x: X(10 ** d) - 8, y: H - 12, "font-size": 10, fill: "#888" }, `1e${d}`));
    }
    for (let d = Math.ceil(y0); d <= Math.floor(y1); d++) {
      svg.appendChild(el("line", { x1: m, y1: Y(10 ** d), x2: W - 8, y2: Y(10 ** d), stroke: "#eee" }));
      svg.appendChild(el("text", { x: 2, y: Y(10 ** d) + 3, "font-size": 10, fill: "#888" }, `1e${d}`));
    }
    svg.appendChild(el("text", { x: W / 2 - 60, y: H - 2, "font-size": 10.5, fill: "#555" },
      "arithmetic intensity (FLOP / DRAM byte, log)"));
    svg.appendChild(el("text", { x: 2, y: 8, "font-size": 10.5, fill: "#555" }, "GFLOP/s (log)"));
    for (const p of pts) {
      const isSel = p.name === selKernel;
      const c = el("circle", { cx: X(p.roofline.ai), cy: Y(p.roofline.gflops),
        r: Math.max(3, Math.min(11, 3 + Math.sqrt(p.share_pct) * 2)),
        fill: isSel ? "#c62828" : (LIMITER_COLORS[p.limiter] || "#607d8b"),
        opacity: isSel ? 1 : 0.55, stroke: isSel ? "#000" : "none", cursor: "pointer" });
      c.appendChild(el("title", {},
        `${p.name}\nAI ${p.roofline.ai} FLOP/B · ${p.roofline.gflops} GFLOP/s\n` +
        `${p.share_pct}% of GPU time · ${p.limiter}\nlow-AI + low-GFLOP/s corner = memory-wall territory (PiM candidates)`));
      c.addEventListener("click", () => { selKernel = p.name; render(); });
      svg.appendChild(c);
    }
    host.appendChild(svg);
  }

  function render() { drawStages(); drawKernels(); drawEvidence(); drawRoofline(); }

  async function loadRun(src) {
    DATA = await (await fetch(`/api/summary?src=${encodeURIComponent(src)}`)).json();
    selStage = null; selKernel = null;
    $("#xp-meta").textContent =
      `${DATA.workload} · ${DATA.kernels.length} kernels · adapter ${DATA.adapter}` +
      (DATA.qor ? ` · QoR ${JSON.stringify(DATA.qor)}` : "");
    render();
  }

  async function init() {
    const list = await (await fetch("/api/summaries")).json();
    const sel = $("#xp-run");
    sel.innerHTML = list.map(s =>
      `<option value="${s.source}">${s.workload}</option>`).join("");
    sel.addEventListener("change", () => loadRun(sel.value));
    if (list.length) loadRun(list[0].source);
    else $("#xp-meta").textContent = "no profiled runs yet — run one above (step 3)";
  }
  window.addEventListener("DOMContentLoaded", init);
})();
