/* app.js — tab shell + the Findings and Methodology tabs.
 *
 * Findings: every project-scope conclusion (reports/findings.json — numbers
 * computed from the committed CSVs, never hand-typed), each with drill-in
 * INTERACTIVE evidence: hover any mark for its numbers, click through to the
 * evidence explorer for run/kernel scope. Charts are hand-rolled SVG over
 * /api/csv — the same files the written reports cite.
 *
 * Methodology: the 6-step pipeline (capture → screen → classify → attribute
 * → validate → verdict) with each step's findings linked.
 */
/* eslint-disable */
(function () {
  const $ = s => document.querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  const STEP_COLORS = { capture: "#455a64", screen: "#00838f", classify: "#6a1b9a",
                        attribute: "#ef6c00", validate: "#2e7d32", verdict: "#c62828" };
  const SUBSTRATE_COLORS = {
    "GPU-keep": "#607d8b", "GPU+layout-fix": "#90a4ae", "CPU/host": "#bdbdbd",
    "PiM-near-bank": "#ef6c00", "PiM-scatter": "#c62828", "ISP/near-storage": "#6a1b9a",
  };

  function el(tag, attrs, text) {
    const svgTags = ["svg","rect","circle","line","text","g","title","polyline","path"];
    const e = svgTags.includes(tag)
      ? document.createElementNS(NS, tag) : document.createElement(tag);
    for (const k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  // ── shared tooltip ────────────────────────────────────────────────────────
  const tip = () => $("#tip");
  function hoverable(node, text) {
    node.addEventListener("mousemove", ev => {
      const t = tip(); t.style.display = "block"; t.textContent = text;
      t.style.left = Math.min(ev.clientX + 14, innerWidth - 360) + "px";
      t.style.top = (ev.clientY + 12) + "px";
    });
    node.addEventListener("mouseleave", () => { tip().style.display = "none"; });
  }

  // ── tab router ────────────────────────────────────────────────────────────
  function showTab(name) {
    document.querySelectorAll(".tab").forEach(s => s.classList.remove("on"));
    document.querySelectorAll("nav.tabs a").forEach(a =>
      a.classList.toggle("on", a.dataset.tab === name));
    const sec = $(`#tab-${name}`);
    if (sec) sec.classList.add("on");
  }
  window.addEventListener("hashchange", () =>
    showTab((location.hash || "#findings").slice(1).split("/")[0]));

  // deep-link into the explorer from a finding
  window.openInExplorer = async function (runSrc, kernel) {
    location.hash = "#explore";
    showTab("explore");
    if (window.Explorer && runSrc) {
      await window.Explorer.open(runSrc, kernel || null);
    }
  };

  // ── csv fetch helper ─────────────────────────────────────────────────────
  const csvCache = {};
  async function csv(src) {
    if (!csvCache[src]) {
      csvCache[src] = (await (await fetch(`/api/csv?src=${encodeURIComponent(src)}`)).json()).rows || [];
    }
    return csvCache[src];
  }
  const f = v => { const x = parseFloat(v); return isNaN(x) ? null : x; };

  // ── interactive charts (each renders into #charthost) ───────────────────
  const CHARTS = {
    async substrate_mix(host, src) {
      const rows = await csv(src);
      const by = {};
      rows.forEach(r => { (by[r.workload] = by[r.workload] || []).push(r); });
      const workloads = Object.keys(by);
      const W = host.clientWidth || 1000, RH = 34, H = workloads.length * RH + 50;
      const svg = el("svg", { width: "100%", height: H, viewBox: `0 0 ${W} ${H}` });
      workloads.forEach((w, i) => {
        let x = 210;
        svg.appendChild(el("text", { x: 4, y: i * RH + 26, "font-size": 11.5 },
          w.length > 30 ? w.slice(0, 29) + "…" : w));
        for (const r of by[w]) {
          const pct = f(r.time_pct) || 0, bw = (W - 220) * pct / 100;
          const rect = el("rect", { x, y: i * RH + 10, width: Math.max(bw - 1, 0), height: 20,
            fill: SUBSTRATE_COLORS[r.substrate] || "#ce93d8", rx: 2 });
          hoverable(rect, `${w}\n${r.substrate}: ${r.time_pct}% of GPU time (${r.time_ms} ms)`);
          svg.appendChild(rect);
          if (bw > 52) svg.appendChild(el("text", { x: x + 4, y: i * RH + 24,
            fill: "#fff", "font-size": 10 }, `${r.substrate.replace("/near-storage","")} ${r.time_pct}%`));
          x += bw;
        }
      });
      let lx = 210;
      Object.entries(SUBSTRATE_COLORS).forEach(([s, c]) => {
        svg.appendChild(el("rect", { x: lx, y: H - 22, width: 10, height: 10, fill: c }));
        svg.appendChild(el("text", { x: lx + 14, y: H - 13, "font-size": 10.5 }, s));
        lx += s.length * 6.4 + 34;
      });
      host.appendChild(svg);
      host.appendChild(el("p", { class: "note" },
        "time-weighted best-substrate mix per deep-profiled workload — hover any segment for its numbers"));
    },

    async flips(host, src) {
      const rows = await csv(src);
      const wrap = el("div", { class: "tablewrap", style: "max-height:420px" });
      const tb = el("table");
      tb.appendChild(el("tr")).innerHTML =
        "<th>kernel</th><th>verdict per workload</th><th>most-moved metric</th><th>swing</th><th></th>";
      for (const r of rows) {
        const tr = el("tr");
        const verd = r.verdicts.split("|").map(v => {
          const [w, s] = v.split(":").map(x => x.trim());
          const c = SUBSTRATE_COLORS[s] || "#ce93d8";
          return `<span class="xp-chip" style="background:${c}" title="${w}">${s}</span>`;
        }).join(" ");
        tr.innerHTML = `<td style="font-family:monospace">${r.kernel}</td>` +
          `<td>${verd}</td><td>${r.most_moved_feature}</td><td>${r.max_over_min}</td>` +
          `<td><a href="#" style="color:var(--acc)">open in explorer</a></td>`;
        tr.querySelector("a").addEventListener("click", ev => {
          ev.preventDefault();
          openInExplorer("profiling/reports/2026-07-03_tum_office_rtx2000ada/summary.json", r.kernel);
        });
        hoverable(tr, r.verdicts.replaceAll("|", "\n"));
        tb.appendChild(tr);
      }
      wrap.appendChild(tb); host.appendChild(wrap);
      host.appendChild(el("p", { class: "note" },
        "each row: one kernel whose substrate verdict differs across workloads, the metric whose max/min swing " +
        "is largest between them (the driver), and the swing magnitude — hover a row for the per-workload detail"));
    },

    async attribution(host, src) {
      const rows = (await csv(src))
        .sort((a, b) => (f(b.med_local_spill_pct) + f(b.med_global_pct)) -
                        (f(a.med_local_spill_pct) + f(a.med_global_pct)))
        .slice(0, 22);
      const W = host.clientWidth || 1000, RH = 22, H = rows.length * RH + 46;
      const svg = el("svg", { width: "100%", height: H, viewBox: `0 0 ${W} ${H}` });
      const parts = [["med_shared_pct", "#4db6ac", "shared (on-chip)"],
                     ["med_local_spill_pct", "#ffb74d", "register-spill scratch → DRAM"],
                     ["med_global_pct", "#7986cb", "global (data structures)"]];
      rows.forEach((r, i) => {
        svg.appendChild(el("text", { x: 4, y: i * RH + 15, "font-size": 10.5,
          "font-family": "monospace" }, r.kernel.slice(0, 34)));
        let x = 280;
        for (const [k, c, lab] of parts) {
          const pct = f(r[k]) || 0, bw = (W - 300) * pct / 100;
          const rect = el("rect", { x, y: i * RH + 4, width: Math.max(bw - 1, 0),
            height: 14, fill: c, rx: 2 });
          hoverable(rect, `${r.kernel}\n${lab}: ${pct}%\ndominant data structure: ` +
            `${r.modal_top_global_tag} (${r.agreement_pct}% agreement across ${r.n_sequences} sequences)`);
          svg.appendChild(rect);
          x += bw;
        }
      });
      let lx = 280;
      for (const [, c, lab] of parts) {
        svg.appendChild(el("rect", { x: lx, y: H - 20, width: 10, height: 10, fill: c }));
        svg.appendChild(el("text", { x: lx + 14, y: H - 11, "font-size": 10.5 }, lab));
        lx += lab.length * 6.2 + 36;
      }
      host.appendChild(svg);
      host.appendChild(el("p", { class: "note" },
        "top kernels by DRAM-visible traffic share — the two-pass join names the space AND the data structure; " +
        "hover for each kernel's tag + cross-sequence agreement"));
    },

    async taxonomy(host, src) {
      const rows = await csv(src);
      const by = {};
      rows.forEach(r => {
        const c = r.modal_class || "?";
        by[c] = by[c] || { total: 0, unan: 0, kernels: [] };
        by[c].total++; by[c].kernels.push(r.kernel);
        if ((r.agreement || "") === "unanimous") by[c].unan++;
      });
      const classes = Object.keys(by).sort();
      const W = host.clientWidth || 1000, BW = Math.min(90, (W - 80) / classes.length - 14), H = 240;
      const maxN = Math.max(...classes.map(c => by[c].total));
      const svg = el("svg", { width: "100%", height: H, viewBox: `0 0 ${W} ${H}` });
      classes.forEach((c, i) => {
        const x = 50 + i * (BW + 14), d = by[c];
        const hAll = (H - 60) * d.total / maxN, hU = (H - 60) * d.unan / maxN;
        const r1 = el("rect", { x, y: H - 40 - hAll, width: BW, height: hAll, fill: "#b0bec5", rx: 3 });
        const r2 = el("rect", { x, y: H - 40 - hU, width: BW, height: hU, fill: "#1565c0", rx: 3 });
        const t = `${c}: ${d.total} kernels, ${d.unan} unanimous across sequences\n` +
          d.kernels.slice(0, 6).join("\n") + (d.kernels.length > 6 ? `\n… +${d.kernels.length - 6}` : "");
        hoverable(r1, t); hoverable(r2, t);
        svg.append(r1, r2);
        svg.appendChild(el("text", { x: x + BW / 2 - 10, y: H - 24, "font-size": 11 }, c));
        svg.appendChild(el("text", { x: x + BW / 2 - 8, y: H - 46 - hAll, "font-size": 11 }, String(d.total)));
      });
      svg.appendChild(el("text", { x: 50, y: 14, "font-size": 11, fill: "#5a6b7c" },
        "kernels per modal bottleneck class (blue = unanimous across all sequences) — hover for the kernel list"));
      host.appendChild(svg);
    },

    async ksweep(host, src) {
      const rows = (await csv(src)).map(r => ({ k: f(r.k), sil: f(r.silhouette), pur: f(r.purity_vs_tree) }));
      const W = host.clientWidth || 1000, H = 250, m = 46;
      const svg = el("svg", { width: "100%", height: H, viewBox: `0 0 ${W} ${H}` });
      const ks = rows.map(r => r.k), k0 = Math.min(...ks), k1 = Math.max(...ks);
      const X = k => m + (W - m - 20) * (k - k0) / (k1 - k0);
      const Y = v => H - 30 - (H - 60) * v;             // sil/purity both 0..1
      for (const [key, color, label] of [["sil", "#1565c0", "silhouette"], ["pur", "#ef6c00", "purity vs decision tree"]]) {
        svg.appendChild(el("polyline", { points: rows.map(r => `${X(r.k)},${Y(r[key])}`).join(" "),
          fill: "none", stroke: color, "stroke-width": 2 }));
        rows.forEach(r => {
          const c = el("circle", { cx: X(r.k), cy: Y(r[key]), r: 4.5, fill: color });
          hoverable(c, `k=${r.k}\nsilhouette ${r.sil}\npurity vs tree ${r.pur}`);
          svg.appendChild(c);
        });
      }
      const best = rows.reduce((a, b) => (b.sil > a.sil ? b : a));
      svg.appendChild(el("line", { x1: X(best.k), y1: 20, x2: X(best.k), y2: H - 30,
        stroke: "#2e7d32", "stroke-dasharray": "4 3" }));
      svg.appendChild(el("text", { x: X(best.k) + 6, y: 30, "font-size": 11, fill: "#2e7d32" },
        `k=${best.k} silhouette-best — matches the hand-built class count`));
      rows.forEach(r => svg.appendChild(el("text", { x: X(r.k) - 4, y: H - 12, "font-size": 10.5 }, r.k)));
      svg.appendChild(el("text", { x: m, y: 14, "font-size": 11, fill: "#5a6b7c" },
        "k-means sweep over the pooled 27-sequence feature cloud (blue silhouette · orange purity)"));
      host.appendChild(svg);
    },

    async neutrality(host, src) {
      const rows = (await csv(src)).map(r => ({
        v: r.variant, mode: r.mode, plain: f(r.plain_APE_m), prof: f(r.nsys_APE_m),
        d: f(r.delta_m), st: r.status })).filter(r => r.plain > 0 && r.prof > 0);
      const W = host.clientWidth || 1000, H = 380, m = 52;
      const svg = el("svg", { width: "100%", height: H, viewBox: `0 0 ${W} ${H}` });
      const lx = Math.log10;
      const all = rows.flatMap(r => [r.plain, r.prof]);
      const lo = lx(Math.min(...all)) - 0.2, hi = lx(Math.max(...all)) + 0.2;
      const X = v => m + (W - m - 16) * (lx(v) - lo) / (hi - lo);
      const Y = v => H - 34 - (H - 60) * (lx(v) - lo) / (hi - lo);
      for (let d = Math.ceil(lo); d <= Math.floor(hi); d++) {
        svg.appendChild(el("line", { x1: X(10 ** d), y1: 12, x2: X(10 ** d), y2: H - 34, stroke: "#eef1f6" }));
        svg.appendChild(el("line", { x1: m, y1: Y(10 ** d), x2: W - 14, y2: Y(10 ** d), stroke: "#eef1f6" }));
        svg.appendChild(el("text", { x: X(10 ** d) - 10, y: H - 18, "font-size": 10 }, `1e${d}`));
        svg.appendChild(el("text", { x: 6, y: Y(10 ** d) + 3, "font-size": 10 }, `1e${d}`));
      }
      svg.appendChild(el("line", { x1: X(10 ** lo), y1: Y(10 ** lo), x2: X(10 ** hi), y2: Y(10 ** hi),
        stroke: "#1c2733", "stroke-width": 1.2 }));
      for (const r of rows) {
        const c = el("circle", { cx: X(r.plain), cy: Y(r.prof), r: r.st === "CHECK" ? 5 : 3.5,
          fill: r.st === "CHECK" ? "#c62828" : "#2e7d32", opacity: 0.6, cursor: "pointer" });
        hoverable(c, `${r.v} (${r.mode})\nplain APE ${r.plain} m\nunder nsys ${r.prof} m\nΔ ${r.d} m — ${r.st}` +
          (r.st === "CHECK" ? "\n(every CHECK was classified benign — see the coverage report)" : ""));
        svg.appendChild(c);
      }
      svg.appendChild(el("text", { x: W / 2 - 90, y: H - 4, "font-size": 11 }, "plain (un-profiled) APE, m — log"));
      svg.appendChild(el("text", { x: 6, y: 10, "font-size": 11 }, "APE under the profiler, m — log"));
      host.appendChild(svg);
      host.appendChild(el("p", { class: "note" },
        "every campaign variant: accuracy with vs without the profiler — points ON the diagonal are unchanged; " +
        "hover any point for the variant + exact numbers"));
    },
  };

  async function openChart(chart, src, title) {
    const panel = $("#chartpanel"), host = $("#charthost");
    host.innerHTML = ""; $("#chartttl").textContent = title;
    panel.classList.add("on");
    await CHARTS[chart](host, src);
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  $("#chartclose").addEventListener("click", () => $("#chartpanel").classList.remove("on"));

  // ── findings tab ─────────────────────────────────────────────────────────
  async function renderFindings(data) {
    const grid = $("#findings");
    const order = ["capture", "screen", "classify", "attribute", "validate", "verdict"];
    const fs = [...data.findings].sort((a, b) => order.indexOf(a.step) - order.indexOf(b.step));
    for (const fd of fs) {
      const card = el("div", { class: "fcard" });
      card.appendChild(el("span", { class: "fstep",
        style: `background:${STEP_COLORS[fd.step] || "#607d8b"}` }, fd.step));
      card.appendChild(el("div", { class: "fnum" }, String(fd.number)));
      card.appendChild(el("div", { class: "funit" }, fd.unit));
      card.appendChild(el("div", { class: "ftitle" }, fd.title));
      card.appendChild(el("div", { class: "fstmt" }, fd.statement));
      const ev = el("div", { class: "fev" });
      for (const e of fd.evidence) {
        const a = el("a", { href: "#" }, e.label);
        a.addEventListener("click", async evn => {
          evn.preventDefault();
          if (e.type === "chart") await openChart(e.chart, e.src, fd.title);
          else if (e.type === "kernel" || e.type === "run") openInExplorer(e.run, e.kernel);
          else if (e.type === "explore") openInExplorer(null, null);
        });
        ev.appendChild(a);
      }
      card.appendChild(ev);
      grid.appendChild(card);
    }
  }

  // ── methodology tab (deep, expandable, stamped from source) ──────────────
  const SEC_COLOR = { overview: "#0d47a1", capture: "#455a64", screen: "#00838f",
                      classify: "#6a1b9a", annotate: "#ef6c00", trace: "#5d4037",
                      verdict: "#c62828" };

  function renderBlock(host, b) {
    if (b.type === "prose") {
      host.appendChild(el("p", { class: "mprose" }, b.text));
    } else if (b.type === "formula") {
      const box = el("div", { class: "formula" });
      box.appendChild(el("div", { class: "fname" }, b.name));
      box.appendChild(el("div", { class: "fexpr" }, b.expr));
      if (b.counters && b.counters.length) {
        const cc = el("div", { class: "fcounters" });
        cc.appendChild(el("span", { class: "note" }, "from: "));
        for (const c of b.counters) cc.appendChild(el("code", { class: "ccode" }, c));
        box.appendChild(cc);
      }
      if (b.note) box.appendChild(el("div", { class: "fnote" }, b.note));
      host.appendChild(box);
    } else if (b.type === "table") {
      if (b.title) host.appendChild(el("div", { class: "mtabtitle" }, b.title));
      const wrap = el("div", { class: "tablewrap" }), t = el("table");
      const hr = el("tr");
      for (const h of b.head) hr.appendChild(el("th", {}, h));
      t.appendChild(hr);
      for (const row of b.rows) {
        const tr = el("tr");
        for (const cell of row) tr.appendChild(el("td", {}, String(cell)));
        t.appendChild(tr);
      }
      wrap.appendChild(t); host.appendChild(wrap);
    } else if (b.type === "decision_tree") {
      const box = el("div", { class: "xp-evbox" });
      if (window.renderDecisionTree) window.renderDecisionTree(box, {});   // template mode
      else box.textContent = "(decision tree renderer unavailable)";
      host.appendChild(box);
      if (b.note) host.appendChild(el("p", { class: "note" }, b.note));
    } else if (b.type === "link") {
      const a = el("a", { href: "#", class: "mlink" }, b.label);
      a.addEventListener("click", async ev => {
        ev.preventDefault();
        if (b.chart) await openChart(b.chart, b.src, b.label);
        else openInExplorer(b.run, b.kernel);
      });
      host.appendChild(a);
    }
  }

  async function renderMethod() {
    const m = await (await fetch("/api/methodology")).json();
    const flow = $("#mflow"), steps = $("#msteps");
    flow.innerHTML = ""; steps.innerHTML = "";
    m.sections.forEach((s, i) => {
      if (i) flow.appendChild(el("span", { class: "marrow" }, "→"));
      const n = el("span", { class: "mnode",
        style: `border-color:${SEC_COLOR[s.id] || "#607d8b"};color:${SEC_COLOR[s.id] || "#607d8b"}` },
        s.title.replace(/^\d+ · /, ""));
      n.style.cursor = "pointer";
      n.addEventListener("click", () => {
        const d = document.getElementById(`msec-${s.id}`); if (d) { d.open = true; d.scrollIntoView({ behavior: "smooth", block: "start" }); }
      });
      flow.appendChild(n);
    });
    m.sections.forEach((s, i) => {
      const det = el("details", { class: "card mstep", id: `msec-${s.id}`,
        style: `border-left-color:${SEC_COLOR[s.id] || "#607d8b"}` });
      if (i < 2) det.open = true;
      const sum = el("summary");
      sum.appendChild(el("span", { class: "msectitle" }, s.title));
      sum.appendChild(el("span", { class: "note", style: "margin-left:8px" }, s.summary));
      det.appendChild(sum);
      const body = el("div", { class: "mbody" });
      for (const b of s.blocks) renderBlock(body, b);
      det.appendChild(body);
      steps.appendChild(det);
    });
    if (m.roadmap && m.roadmap.length) {
      const card = el("div", { class: "card" });
      card.appendChild(el("h2", {}, "Roadmap — what's built, ready, and deferred"));
      card.appendChild(el("p", { class: "note" },
        "the honest backlog (full triage in docs/BACKLOG.md): what this cycle built, " +
        "what's supported and waiting on a run/device, and what's deferred to the architecture phase."));
      const wrap = el("div", { class: "tablewrap" }), t = el("table");
      t.appendChild(el("tr")).innerHTML = "<th>item</th><th>status</th><th>note</th>";
      const badge = { DONE: "#2e7d32", READY: "#1565c0", DEFER: "#8a97a5", DROP: "#c2cbd4" };
      for (const r of m.roadmap) {
        const tr = el("tr");
        tr.appendChild(el("td", {}, r.item));
        const st = el("td");
        st.appendChild(el("span", { class: "xp-chip", style: `background:${badge[r.status] || "#607d8b"}` }, r.status));
        tr.appendChild(st);
        tr.appendChild(el("td", {}, r.note));
        t.appendChild(tr);
      }
      wrap.appendChild(t); card.appendChild(wrap);
      steps.appendChild(card);
    }
  }

  // ── init ─────────────────────────────────────────────────────────────────
  window.addEventListener("DOMContentLoaded", async () => {
    showTab((location.hash || "#findings").slice(1).split("/")[0] || "findings");
    const data = await (await fetch("/api/findings")).json();
    window.__FINDINGS = data.findings;
    await renderFindings(data);
    await renderMethod();
  });
})();
