/* explorer.js — the interactive evidence explorer.
 *
 * Renders one profiled run's summary.json as the bottleneck->improvement
 * reasoning chain the studies used, four linked panels:
 *   1 stage time bar   WHERE the GPU time goes (click a stage to filter)
 *   2 kernel list      WHO dominates (to 90% cumulative; limiter + substrate)
 *   3 evidence panel   WHY — metrics vs decision thresholds, the two-pass
 *                      attribution join (shared / register-spill "DRAM
 *                      scratch" / global + data structure), taxonomy
 *                      stability + k-means, and the substrate verdict
 *   4 roofline         WHERE it sits vs the ROOF (BW slope + compute peak
 *                      from the device's hw descriptor; hover = numbers)
 *
 * No dependencies; every number comes from summary.json — the standard
 * schema every adapter's run produces (profiling/analysis/summarize_run.py).
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
    "memory-leaning": "#ef6c00", "compute-leaning": "#2e7d32",
  };
  const THRESH = [
    ["dram_sol_pct", "DRAM utilisation", "%", 60, "≥60% ⇒ DRAM-bandwidth-bound", 100],
    ["sectors_per_req", "sectors / request", "", 8, "≤8 coalesced · ≥16 scattered gather", 32],
    ["occupancy_pct", "occupancy", "%", 25, "<25% ⇒ latency/dependency limited", 100],
    ["lfmr", "L2 miss ratio (LFMR)", "", 0.5, "≥0.5 ⇒ cache-defeating, DRAM-visible", 1],
    ["mpki", "misses / kilo-instr", "", 30, "≥30 ⇒ memory-intensive (DAMOV)", 130],
  ];

  let LIST = [], DATA = null, selStage = null, selKernel = null;

  function el(tag, attrs, text) {
    const svgTags = ["svg", "rect", "circle", "line", "text", "g", "title", "polyline"];
    const e = svgTags.includes(tag)
      ? document.createElementNS(NS, tag) : document.createElement(tag);
    for (const k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  // ── panel 1: stage bar ────────────────────────────────────────────────────
  function drawStages() {
    const host = $("#xp-stages"); host.innerHTML = "";
    const stages = (DATA.stages || []).filter(s => s.share_pct > 0);
    if (!stages.length) {
      host.appendChild(el("div", { class: "xp-note" },
        "no stage timeline in this cell (windowed ncu capture) — kernel shares below are within the capture window"));
      return;
    }
    const W = host.clientWidth || 900, H = 64;
    const svg = el("svg", { width: "100%", height: H, viewBox: `0 0 ${W} ${H}` });
    let x = 0;
    stages.forEach((s, i) => {
      const w = Math.max(2, W * s.share_pct / 100);
      const r = el("rect", { x, y: 6, width: w - 1, height: 30, rx: 3,
        fill: STAGE_COLORS[i % STAGE_COLORS.length],
        opacity: (selStage && selStage !== s.name) ? 0.25 : 0.92, cursor: "pointer" });
      r.addEventListener("click", () => { selStage = selStage === s.name ? null : s.name; render(); });
      r.appendChild(el("title", {}, `${s.name} — ${s.share_pct}% of GPU time (${s.time_ms} ms, ${s.n_kernels} kernels)\nclick to filter the kernel list`));
      svg.appendChild(r);
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
    // legend
    const leg = el("div", { class: "xp-legend" });
    const seen = new Set(rows.map(k => k.limiter));
    for (const l of seen) leg.appendChild(el("span", { class: "xp-chip",
      style: `background:${LIMITER_COLORS[l] || "#607d8b"}` }, l));
    leg.appendChild(el("span", { class: "xp-note" }, " limiter classes present"));
    host.appendChild(leg);
    if (!selKernel || !DATA.kernels.some(k => k.name === selKernel)) {
      selKernel = rows[0] ? rows[0].name : null;
    }
  }

  // ── panel 3: evidence ────────────────────────────────────────────────────
  function metricBar(host, label, v, unit, cut, note, max) {
    const wrap = el("div", { class: "xp-metric" });
    wrap.appendChild(el("div", { class: "xp-mlabel" }, `${label}: ${v}${unit}`));
    const bar = el("div", { class: "xp-mbar", title: note });
    bar.appendChild(el("div", { class: "xp-mfill", style: `width:${Math.min(100, 100 * v / max)}%` }));
    if (cut != null) bar.appendChild(el("div", { class: "xp-mcut",
      style: `left:${100 * cut / max}%`, title: `threshold ${cut}${unit}` }));
    wrap.appendChild(bar);
    wrap.appendChild(el("div", { class: "xp-mnote" }, note));
    host.appendChild(wrap);
  }

  function drawEvidence() {
    const host = $("#xp-evidence"); host.innerHTML = "";
    const k = DATA.kernels.find(x => x.name === selKernel);
    if (!k) { host.textContent = "select a kernel"; return; }
    host.appendChild(el("div", { class: "xp-evtitle" }, k.name));
    host.appendChild(el("div", { class: "xp-evsub" },
      `${k.stage !== "?" ? k.stage + " · " : ""}${k.time_ms} ms (${k.share_pct}%` +
      `${DATA.stages && DATA.stages.length ? " of GPU time" : " of capture window"}) · limiter ${k.limiter}` +
      (k.evidence.dominant_stall ? ` · dominant stall: ${k.evidence.dominant_stall}` : "")));

    for (const [key, label, unit, cut, note, max] of THRESH) {
      const v = k.evidence[key];
      if (v != null) metricBar(host, label, v, unit, cut, note, max);
    }
    if (k.sm_sol_pct != null) metricBar(host, "compute (SM) utilisation", k.sm_sol_pct,
      "%", null, "vs DRAM utilisation above — the higher one leans the verdict", 100);

    // the two-pass attribution join: shared / register-spill "scratch" / global
    const at = k.study && k.study.attribution;
    if (at && (at.shared_pct != null)) {
      const seg = el("div", { class: "xp-metric" });
      seg.appendChild(el("div", { class: "xp-mlabel" },
        "memory-space composition (NVBit trace × TaggedAllocator two-pass join)"));
      const bar = el("div", { class: "xp-stack" });
      const parts = [["shared (on-chip)", at.shared_pct, "#4db6ac"],
                     ["register-spill scratch → DRAM", at.spill_pct, "#ffb74d"],
                     ["global (data structures)", at.global_pct, "#7986cb"]];
      for (const [lab, pct, color] of parts) {
        if (!pct) continue;
        const d = el("div", { class: "xp-seg", style: `width:${pct}%;background:${color}`,
          title: `${lab}: ${pct}%` });
        if (pct > 14) d.textContent = `${lab} ${pct}%`;
        bar.appendChild(d);
      }
      seg.appendChild(bar);
      seg.appendChild(el("div", { class: "xp-mnote" },
        `dominant data structure: ${at.data_structure || "?"} ` +
        `(${at.tag_agreement_pct}% agreement across 27 sequences)` +
        (at.spill_pct >= 50 ? " — DRAM traffic is mostly SPILL, not data: a register-pressure fix, not a memory-system fix" : "")));
      host.appendChild(seg);
    }
    // taxonomy stability + k-means validation
    const tx = k.study && k.study.taxonomy_stability;
    const km = k.study && k.study.kmeans;
    if (tx || km) host.appendChild(el("div", { class: "xp-rat" },
      (tx ? `taxonomy: modal ${tx.modal_class} over ${tx.n_seq} sequences (${tx.verdict})` : "") +
      (km ? `${tx ? " · " : ""}k-means cluster ${km.cluster_k8} (k=8; sweep favours k=${km.sweep_best_k})` : "")));
    if (k.rationale) host.appendChild(el("div", { class: "xp-rat" },
      "classifier rationale: " + k.rationale));

    const verdict = el("div", { class: "xp-verdict" });
    verdict.appendChild(el("b", {}, "⇒ best substrate: "));
    verdict.appendChild(el("span", {}, k.substrate || "?"));
    verdict.appendChild(el("div", { class: "xp-mnote" },
      `PiM affinity: ${k.pim_affinity} — the evidence above is the trail to this verdict`));
    host.appendChild(verdict);
  }

  // ── panel 4: roofline (with the roof) ────────────────────────────────────
  function drawRoofline() {
    const host = $("#xp-roofline"); host.innerHTML = "";
    const pts = DATA.kernels.filter(k => k.roofline && k.roofline.ai > 0 && k.roofline.gflops > 0);
    const peaks = DATA.device_peaks;
    if (!pts.length) {
      host.appendChild(el("div", { class: "xp-note" },
        "no per-kernel FLOP/byte data in this cell (quick ncu window) — open a study run for the roofline"));
      return;
    }
    const W = host.clientWidth || 900, H = 300, m = 46;
    const svg = el("svg", { width: "100%", height: H, viewBox: `0 0 ${W} ${H}` });
    const lx = Math.log10;
    let xs = pts.map(p => lx(p.roofline.ai)), ys = pts.map(p => lx(p.roofline.gflops));
    let x0 = Math.min(...xs) - 0.3, x1 = Math.max(...xs) + 0.5;
    let y0 = Math.min(...ys) - 0.3, y1 = Math.max(...ys) + 0.3;
    if (peaks) {  // make room for the roof
      const ridge = peaks.peak_gflops / peaks.dram_gbps;
      x1 = Math.max(x1, lx(ridge) + 0.4);
      y1 = Math.max(y1, lx(peaks.peak_gflops) + 0.15);
    }
    const X = v => m + (W - m - 12) * (lx(v) - x0) / (x1 - x0);
    const Y = v => H - 30 - (H - 52) * (lx(v) - y0) / (y1 - y0);
    for (let d = Math.ceil(x0); d <= Math.floor(x1); d++) {
      svg.appendChild(el("line", { x1: X(10 ** d), y1: 10, x2: X(10 ** d), y2: H - 30, stroke: "#eee" }));
      svg.appendChild(el("text", { x: X(10 ** d) - 10, y: H - 16, "font-size": 10, fill: "#888" }, `1e${d}`));
    }
    for (let d = Math.ceil(y0); d <= Math.floor(y1); d++) {
      svg.appendChild(el("line", { x1: m, y1: Y(10 ** d), x2: W - 10, y2: Y(10 ** d), stroke: "#eee" }));
      svg.appendChild(el("text", { x: 4, y: Y(10 ** d) + 3, "font-size": 10, fill: "#888" }, `1e${d}`));
    }
    // THE ROOF: bandwidth slope up to the ridge, then the compute peak
    if (peaks) {
      const ridgeAI = peaks.peak_gflops / peaks.dram_gbps;
      const aiA = 10 ** x0, aiB = Math.min(ridgeAI, 10 ** x1);
      svg.appendChild(el("line", {
        x1: X(aiA), y1: Y(aiA * peaks.dram_gbps),
        x2: X(aiB), y2: Y(aiB * peaks.dram_gbps),
        stroke: "#c62828", "stroke-width": 2 }));
      if (ridgeAI < 10 ** x1) {
        svg.appendChild(el("line", {
          x1: X(ridgeAI), y1: Y(peaks.peak_gflops),
          x2: X(10 ** x1), y2: Y(peaks.peak_gflops),
          stroke: "#c62828", "stroke-width": 2 }));
      }
      const lbl = el("text", { x: X(aiA) + 6, y: Y(aiA * peaks.dram_gbps) - 8,
        "font-size": 10.5, fill: "#c62828", transform: "" },
        `DRAM roof ${peaks.dram_gbps} GB/s (${peaks.basis})`);
      svg.appendChild(lbl);
      svg.appendChild(el("text", { x: W - 250, y: Y(peaks.peak_gflops) - 6,
        "font-size": 10.5, fill: "#c62828" },
        `compute peak ${Math.round(peaks.peak_gflops)} GFLOP/s (${peaks.basis})`));
      svg.appendChild(el("title", {}, ""));
    }
    svg.appendChild(el("text", { x: W / 2 - 80, y: H - 4, "font-size": 10.5, fill: "#555" },
      "arithmetic intensity (FLOP / DRAM byte, log)"));
    svg.appendChild(el("text", { x: 4, y: 10, "font-size": 10.5, fill: "#555" }, "GFLOP/s (log)"));
    for (const p of pts) {
      const isSel = p.name === selKernel;
      const c = el("circle", { cx: X(p.roofline.ai), cy: Y(p.roofline.gflops),
        r: Math.max(3, Math.min(11, 3 + Math.sqrt(p.share_pct) * 2)),
        fill: isSel ? "#c62828" : (LIMITER_COLORS[p.limiter] || "#607d8b"),
        opacity: isSel ? 1 : 0.55, stroke: isSel ? "#000" : "none", cursor: "pointer" });
      const gap = peaks ? ` · ${Math.round(100 * p.roofline.gflops / Math.min(peaks.peak_gflops, p.roofline.ai * peaks.dram_gbps))}% of its roof` : "";
      c.appendChild(el("title", {},
        `${p.name}\nAI ${p.roofline.ai} FLOP/B · ${p.roofline.gflops} GFLOP/s${gap}\n` +
        `${p.share_pct}% of GPU time · ${p.limiter}\nbelow the sloped roof-left = memory-wall territory (PiM candidates)`));
      c.addEventListener("click", () => { selKernel = p.name; render(); });
      svg.appendChild(c);
    }
    host.appendChild(svg);
  }

  function render() { drawStages(); drawKernels(); drawEvidence(); drawRoofline(); }

  async function loadRun(src) {
    DATA = await (await fetch(`/api/summary?src=${encodeURIComponent(src)}`)).json();
    selStage = null; selKernel = null;
    const q = DATA.qor ? Object.entries(DATA.qor).map(([k, v]) => `${k}=${v}`).join(" ") : "";
    $("#xp-meta").innerHTML =
      `<b>${DATA.workload}</b> · ${DATA.kernels.length} kernels · adapter ${DATA.adapter}` +
      (DATA.note ? ` · ${DATA.note}` : "") +
      (q ? `<br><span class="xp-qor">QoR: ${q}</span>` : "");
    render();
  }

  async function init() {
    LIST = await (await fetch("/api/summaries")).json();
    const sel = $("#xp-run"), filter = $("#xp-filter");
    function fill(qtext) {
      const q = (qtext || "").toLowerCase();
      const hits = LIST.filter(s => s.workload.toLowerCase().includes(q));
      sel.innerHTML = hits.map(s =>
        `<option value="${s.source}">${s.group === "study" ? "★ " : ""}${s.workload}</option>`).join("");
      $("#xp-count").textContent = `${hits.length}/${LIST.length} runs`;
      return hits;
    }
    const first = fill("");
    sel.addEventListener("change", () => loadRun(sel.value));
    filter.addEventListener("input", () => {
      const hits = fill(filter.value);
      if (hits.length) { sel.value = hits[0].source; loadRun(sel.value); }
    });
    if (first.length) loadRun(first[0].source);
    else $("#xp-meta").textContent = "no profiled runs yet — run one above (step 3)";
  }
  window.addEventListener("DOMContentLoaded", init);
})();
